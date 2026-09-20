"""Evaluation benchmark runner assessing routing accuracy, answer grounding, abstention, and latency."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console utf-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag.service import KnowledgeAssistantService

logging.basicConfig(level=logging.WARNING)


def run_evaluation(dataset_path: Path, output_path: Path) -> None:
    print("=" * 80)
    print(" Intelligent Knowledge Assistant - Evaluation Benchmark")
    print("=" * 80)
    print(f"Loading evaluation dataset from: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} evaluation cases.")
    print("-" * 80)

    service = KnowledgeAssistantService()
    results = []

    correct_routing = 0
    correct_abstentions = 0
    grounded_answers = 0
    total_latency_simple = 0.0
    count_simple = 0
    total_latency_agentic = 0.0
    count_agentic = 0

    for i, test in enumerate(test_cases, 1):
        qid = test["id"]
        q = test["question"]
        expected_wf = test["expected_workflow"]
        category = test["category"]
        should_abstain = test["should_abstain"]

        print(f"[{i}/{len(test_cases)}] Evaluating {qid} ({category})...")
        print(f"    Query: '{q}'")

        start = time.time()
        try:
            res = service.answer_query(q)
            latency = time.time() - start
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Metrics evaluation
        routing_match = (res.workflow == expected_wf)
        if routing_match:
            correct_routing += 1

        abstained_correctly = (res.abstained == should_abstain)
        if should_abstain:
            if res.abstained:
                correct_abstentions += 1
        elif not res.abstained and res.verification.status in ("supported", "partially_supported"):
            grounded_answers += 1

        if res.workflow == "simple_rag":
            total_latency_simple += latency
            count_simple += 1
        else:
            total_latency_agentic += latency
            count_agentic += 1

        record = {
            "id": qid,
            "category": category,
            "question": q,
            "expected_workflow": expected_wf,
            "actual_workflow": res.workflow,
            "routing_match": routing_match,
            "router_score": res.routing.total_score,
            "should_abstain": should_abstain,
            "abstained": res.abstained,
            "abstained_correctly": abstained_correctly,
            "verification_status": res.verification.status,
            "confidence": res.confidence,
            "sources_count": len(res.sources),
            "latency_seconds": round(latency, 2),
            "answer_snippet": res.answer[:200].replace("\n", " ") + "...",
        }
        results.append(record)

        match_str = "[YES]" if routing_match else "[NO]"
        print(f"    -> Workflow: {res.workflow} (Expected: {expected_wf}) | Score: {res.routing.total_score} | Match: {match_str}")
        print(f"    -> Verification: {res.verification.status} | Sources: {len(res.sources)} | Latency: {latency:.2f}s\n")

        # Sleep briefly between queries to avoid bursting free-tier limits
        time.sleep(2.0)

    # Compute Summary Statistics
    total_evals = len(results)
    routing_acc = (correct_routing / total_evals) * 100 if total_evals else 0.0
    unanswerable_cases = [t for t in test_cases if t["should_abstain"]]
    abstention_rate = (correct_abstentions / len(unanswerable_cases)) * 100 if unanswerable_cases else 100.0
    answerable_cases = [t for t in test_cases if not t["should_abstain"]]
    grounding_rate = (grounded_answers / len(answerable_cases)) * 100 if answerable_cases else 0.0
    avg_latency_simple = (total_latency_simple / count_simple) if count_simple else 0.0
    avg_latency_agentic = (total_latency_agentic / count_agentic) if count_agentic else 0.0

    summary = {
        "total_test_cases": total_evals,
        "routing_accuracy_pct": round(routing_acc, 1),
        "abstention_accuracy_pct": round(abstention_rate, 1),
        "grounding_faithfulness_pct": round(grounding_rate, 1),
        "avg_latency_simple_rag_sec": round(avg_latency_simple, 2),
        "avg_latency_agentic_sec": round(avg_latency_agentic, 2),
        "detailed_results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(" BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases      : {total_evals}")
    print(f"Routing Accuracy      : {routing_acc:.1f}% ({correct_routing}/{total_evals})")
    print(f"Abstention Accuracy   : {abstention_rate:.1f}% ({correct_abstentions}/{len(unanswerable_cases)})")
    print(f"Grounding / Faithfulness: {grounding_rate:.1f}% ({grounded_answers}/{len(answerable_cases)})")
    print(f"Avg Latency (Simple)  : {avg_latency_simple:.2f}s (n={count_simple})")
    print(f"Avg Latency (Agentic) : {avg_latency_agentic:.2f}s (n={count_agentic})")
    print(f"Report saved to       : {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG & Agentic evaluation benchmark.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "scripts" / "eval_dataset.json"),
        help="Path to evaluation questions dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "evaluation_results.json"),
        help="Path to save evaluation summary",
    )
    args = parser.parse_args()
    run_evaluation(Path(args.dataset), Path(args.output))
