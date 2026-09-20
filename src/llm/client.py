"""Resilient multi-provider LLM client supporting OpenRouter (Qwen3.8 27B & Gemma4 31B) and Google Gemini with automatic fallback."""
import json
import logging
import re
import time
from typing import Any, Literal
import httpx

from src.config import llm_config, LLMConfig

logger = logging.getLogger(__name__)


def clean_model_text(text: str) -> str:
    """Clean internal monologue or scratchpad formatting from thinking models."""
    cleaned = text.strip()
    # Check for thinking scratchpad block patterns (e.g., "* Question: ...\n* Constraint: ...")
    if re.search(r"^\*\s+Question:.*?\n\*\s+Constraint:", cleaned, re.DOTALL):
        parts = re.split(r"\n\s*\n", cleaned)
        non_scratchpad_parts = []
        for part in parts:
            p_strip = part.strip()
            if not (p_strip.startswith("*   Question:") or p_strip.startswith("*   Constraint:") or p_strip.startswith("*   Option")):
                non_scratchpad_parts.append(part)
        if non_scratchpad_parts:
            cleaned = "\n\n".join(non_scratchpad_parts).strip()
    return cleaned


class LLMClient:
    """Production-grade LLM client supporting OpenRouter and Google Gemini with automatic fallback."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or llm_config
        self._genai = None

        if self.config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                self._genai = genai
            except Exception as e:
                logger.warning("Failed to configure Google Generative AI: %s", e)

        self._http_client = httpx.Client(timeout=45.0)

    def _get_model_chain(self, tier: Literal["primary", "complex", "flash"]) -> list[str]:
        """Construct priority model fallback chain based on requested tier."""
        if tier in ("complex", "flash"):
            chain = [self.config.complex_model, self.config.primary_model] + self.config.fallback_models
        else:
            chain = [self.config.primary_model, self.config.complex_model] + self.config.fallback_models

        seen = set()
        deduped = []
        for m in chain:
            if m and m not in seen:
                seen.add(m)
                deduped.append(m)
        return deduped

    def _generate_openrouter(
        self,
        model_name: str,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
    ) -> str:
        """Call OpenRouter API."""
        if not self.config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # Cap max_tokens to prevent 402 credit limit errors on large context models
        capped_tokens = min(max_output_tokens or 2048, 2048)

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": capped_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/BITS/Assingment_AZ",
            "X-Title": "Intelligent Knowledge Assistant",
        }

        url = f"{self.config.openrouter_base_url.rstrip('/')}/chat/completions"
        response = self._http_client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter returned status {response.status_code}: {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter response contained no choices: {data}")

        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        # If content is empty but reasoning is present (thinking models)
        if not content and "reasoning" in msg:
            content = msg["reasoning"] or ""

        return content.strip()

    def _generate_gemini(
        self,
        model_name: str,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
    ) -> str:
        """Call Google Gemini API."""
        if self._genai is None:
            raise ValueError("Google GenAI is not configured.")

        # Normalize model name for Google API
        normalized_model = model_name
        if not normalized_model.startswith("models/"):
            normalized_model = f"models/{normalized_model.split('/')[-1]}"

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        kwargs: dict[str, Any] = {"model_name": normalized_model, "generation_config": generation_config}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        model = self._genai.GenerativeModel(**kwargs)
        res = model.generate_content(prompt)
        if res and res.text:
            return res.text.strip()
        raise RuntimeError(f"Empty response from Gemini model {model_name}")

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        model_tier: Literal["primary", "complex", "flash"] = "primary",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[str, str]:
        """Generate response text with automatic model fallback across candidate models."""
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_output_tokens or self.config.max_output_tokens
        model_chain = self._get_model_chain(model_tier)

        last_error: Exception | None = None

        for model_name in model_chain:
            try:
                # Route to OpenRouter if model has provider prefix (e.g., 'qwen/', 'google/') or if OpenRouter key is set
                if self.config.openrouter_api_key and ("/" in model_name and not model_name.startswith("models/")):
                    logger.debug("Generating via OpenRouter with model: %s", model_name)
                    text = self._generate_openrouter(
                        model_name=model_name,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temp,
                        max_output_tokens=tokens,
                    )
                    return text, model_name

                # Otherwise route to Google Gemini
                if self.config.gemini_api_key:
                    logger.debug("Generating via Gemini with model: %s", model_name)
                    text = self._generate_gemini(
                        model_name=model_name,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temp,
                        max_output_tokens=tokens,
                    )
                    return text, model_name

            except Exception as e:
                logger.warning(
                    "Model %s failed with error: %s. Attempting next fallback in chain...",
                    model_name,
                    e,
                )
                last_error = e
                time.sleep(0.5)

        raise RuntimeError(
            f"All models in fallback chain failed. Last error: {last_error}"
        ) from last_error

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        model_tier: Literal["primary", "complex", "flash"] = "primary",
    ) -> tuple[dict[str, Any] | list[Any], str]:
        """Generate and parse structured JSON output with fallback cleanup."""
        json_system = (
            (system_instruction + "\n" if system_instruction else "")
            + "CRITICAL: You MUST output valid JSON. If you include reasoning, the final part of your response MUST be the complete, valid JSON object or array."
        )

        raw_text, model_used = self.generate(
            prompt=prompt,
            system_instruction=json_system,
            model_tier=model_tier,
            temperature=0.0,
        )

        cleaned = raw_text.strip()

        # 1. Direct parse attempt
        try:
            return json.loads(cleaned), model_used
        except Exception:
            pass

        # 2. Extract from markdown code fences ```json ... ```
        fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        for block in reversed(fence_matches):
            try:
                return json.loads(block.strip()), model_used
            except Exception:
                continue

        # 3. Search for outermost JSON object {...} or array [...] from the end
        last_brace = cleaned.rfind("}")
        if last_brace != -1:
            open_count = 0
            for i in range(last_brace, -1, -1):
                if cleaned[i] == "}":
                    open_count += 1
                elif cleaned[i] == "{":
                    open_count -= 1
                    if open_count == 0:
                        candidate = cleaned[i : last_brace + 1]
                        try:
                            return json.loads(candidate), model_used
                        except Exception:
                            pass

        last_bracket = cleaned.rfind("]")
        if last_bracket != -1:
            open_count = 0
            for i in range(last_bracket, -1, -1):
                if cleaned[i] == "]":
                    open_count += 1
                elif cleaned[i] == "[":
                    open_count -= 1
                    if open_count == 0:
                        candidate = cleaned[i : last_bracket + 1]
                        try:
                            return json.loads(candidate), model_used
                        except Exception:
                            pass

        # 4. Fallback to complex tier if primary JSON extraction failed
        if model_tier != "complex" and model_tier != "flash":
            logger.info("JSON parsing failed on model %s. Retrying with complex model tier...", model_used)
            return self.generate_json(prompt, system_instruction, model_tier="complex")

        raise ValueError(f"Could not parse valid JSON from model response: {cleaned[:300]}")
