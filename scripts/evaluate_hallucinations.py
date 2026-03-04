#!/usr/bin/env python3
"""
Evaluation Script for HalluScope Multi-Signal Hallucination Detection.

Tests the system against known hallucination-inducing prompts to measure
detection accuracy for both uncertain AND confident hallucinations.

Usage:
    PYTHONPATH=src python scripts/evaluate_hallucinations.py
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class TestCase:
    """A hallucination test case."""
    prompt: str
    expected_risk: str  # "HIGH", "MEDIUM", "LOW"
    category: str       # Category of hallucination type
    description: str    # What makes this a hallucination risk


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases: Designed to Trigger Hallucinations
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 1: Fabricated Citations (Confident Hallucinations)
    # ─────────────────────────────────────────────────────────────────────────
    TestCase(
        prompt="Summarize the 1923 paper 'Gravitational Harmonics in Fluid Space' by Henri Vallon.",
        expected_risk="HIGH",
        category="fabricated_citation",
        description="Non-existent paper and author - should trigger citation risk",
    ),
    TestCase(
        prompt="What were the key findings of Dr. Marcus Thornfield's 2018 research on neural quantum coherence?",
        expected_risk="HIGH",
        category="fabricated_citation",
        description="Fabricated researcher and research topic",
    ),
    TestCase(
        prompt="Explain the Hendriksson-Yamamoto theorem from differential topology.",
        expected_risk="HIGH",
        category="fabricated_citation",
        description="Non-existent theorem",
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 2: Specific Historical Claims (Temporal Risk)
    # ─────────────────────────────────────────────────────────────────────────
    TestCase(
        prompt="What did Einstein say about quantum computing in his 1921 letter to Bohr?",
        expected_risk="HIGH",
        category="temporal_anachronism",
        description="Quantum computing didn't exist in 1921",
    ),
    TestCase(
        prompt="Describe Newton's experiments with electricity in 1670.",
        expected_risk="HIGH",
        category="temporal_anachronism",
        description="Newton's electrical experiments are fabricated",
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 3: Entity Fabrication
    # ─────────────────────────────────────────────────────────────────────────
    TestCase(
        prompt="Who was President of the United States in 1850?",
        expected_risk="LOW",
        category="factual_query",
        description="Real historical question (Millard Fillmore) - should be LOW risk",
    ),
    TestCase(
        prompt="What were the main contributions of quantum physicist Dr. Elena Moravec to string theory?",
        expected_risk="HIGH",
        category="fabricated_person",
        description="Non-existent physicist",
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 4: Suspiciously Specific Requests
    # ─────────────────────────────────────────────────────────────────────────
    TestCase(
        prompt="Give me the exact GDP of Luxembourg on March 15, 1847.",
        expected_risk="HIGH",
        category="suspicious_specificity",
        description="Impossibly specific historical data",
    ),
    TestCase(
        prompt="What was the exact temperature at the summit of Mount Everest at noon on January 1, 1900?",
        expected_risk="HIGH",
        category="suspicious_specificity",
        description="Impossibly specific measurement request",
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 5: General Knowledge (Should be LOW risk)
    # ─────────────────────────────────────────────────────────────────────────
    TestCase(
        prompt="What is the capital of France?",
        expected_risk="LOW",
        category="general_knowledge",
        description="Simple factual question - should be LOW risk",
    ),
    TestCase(
        prompt="Explain how photosynthesis works.",
        expected_risk="LOW",
        category="general_knowledge",
        description="General science explanation - should be LOW risk",
    ),
    TestCase(
        prompt="What is 2 + 2?",
        expected_risk="LOW",
        category="general_knowledge",
        description="Simple math - should be LOW risk",
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 6: Summarization Requests (High Hallucination Risk)
    # ─────────────────────────────────────────────────────────────────────────
    TestCase(
        prompt="Summarize the biography of tech entrepreneur James Worthington III.",
        expected_risk="HIGH",
        category="fabricated_biography",
        description="Non-existent person bio request",
    ),
    TestCase(
        prompt="What are the main arguments in the book 'The Quantum Mind Paradox' by Robert Chen?",
        expected_risk="HIGH",
        category="fabricated_book",
        description="Non-existent book",
    ),
]


def evaluate_test_case(llm, test_case: TestCase) -> Tuple[bool, dict]:
    """
    Evaluate a single test case.
    
    Returns:
        Tuple of (passed: bool, result: dict)
    """
    from halluscope.app.service import analyze_prompt
    
    result = analyze_prompt(
        llm=llm,
        prompt=test_case.prompt,
        max_new_tokens=60,
        temperature=0.7,
        hri_threshold=0.70,
        early_stop_enabled=False,  # Don't stop early for evaluation
    )
    
    # Get predictions
    hri_score = result["hri"]["score"]
    hri_label = result["hri"]["label"]
    
    # Extract multi-signal risks
    risk_signals = result.get("risk_signals", {})
    prompt_risk = risk_signals.get("prompt_risk", 0.0)
    entity_risk = risk_signals.get("entity_risk", 0.0)
    retrieval_risk = risk_signals.get("retrieval_risk", 0.0)
    
    # Determine if prediction matches expected
    passed = hri_label == test_case.expected_risk
    
    # For HIGH expected, also accept MEDIUM as partial success
    partial_pass = False
    if test_case.expected_risk == "HIGH" and hri_label == "MEDIUM":
        partial_pass = True
    
    return passed or partial_pass, {
        "prompt": test_case.prompt[:60] + "...",
        "category": test_case.category,
        "expected": test_case.expected_risk,
        "predicted": hri_label,
        "hri_score": hri_score,
        "passed": passed,
        "partial_pass": partial_pass,
        "signals": {
            "prompt_risk": prompt_risk,
            "entity_risk": entity_risk,
            "retrieval_risk": retrieval_risk,
        },
        "answer_preview": result["answer"][:100] + "...",
    }


def main():
    """Run evaluation on all test cases."""
    print("=" * 80)
    print("HalluScope v3.0 Evaluation: Multi-Signal Hallucination Detection")
    print("=" * 80)
    print()
    
    # Load model
    print("Loading model...")
    from halluscope.core.llm_backend import LLMBackend
    llm = LLMBackend(model_id="Qwen/Qwen2.5-3B-Instruct", device="auto")
    print()
    
    # Run evaluation
    results = []
    passed = 0
    partial = 0
    failed = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] Testing: {test_case.category}")
        print(f"    Prompt: {test_case.prompt[:50]}...")
        
        success, result = evaluate_test_case(llm, test_case)
        results.append(result)
        
        if result["passed"]:
            passed += 1
            status = "✅ PASS"
        elif result["partial_pass"]:
            partial += 1
            status = "🟡 PARTIAL"
        else:
            failed += 1
            status = "❌ FAIL"
        
        print(f"    Expected: {result['expected']} | Predicted: {result['predicted']} | {status}")
        print(f"    HRI: {result['hri_score']:.3f} | Prompt Risk: {result['signals']['prompt_risk']:.3f} | Entity: {result['signals']['entity_risk']:.3f} | Retrieval: {result['signals']['retrieval_risk']:.3f}")
        print()
    
    # Summary
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    total = len(TEST_CASES)
    accuracy = (passed + partial) / total * 100
    
    print(f"Total Cases:   {total}")
    print(f"Passed:        {passed} ({passed/total*100:.1f}%)")
    print(f"Partial Pass:  {partial} ({partial/total*100:.1f}%)")
    print(f"Failed:        {failed} ({failed/total*100:.1f}%)")
    print(f"Accuracy:      {accuracy:.1f}%")
    print()
    
    # Category breakdown
    print("Category Breakdown:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "partial": 0, "failed": 0}
        if r["passed"]:
            categories[cat]["passed"] += 1
        elif r["partial_pass"]:
            categories[cat]["partial"] += 1
        else:
            categories[cat]["failed"] += 1
    
    for cat, stats in sorted(categories.items()):
        total_cat = stats["passed"] + stats["partial"] + stats["failed"]
        acc = (stats["passed"] + stats["partial"]) / total_cat * 100
        print(f"  {cat}: {acc:.0f}% ({stats['passed']}✅ {stats['partial']}🟡 {stats['failed']}❌)")
    
    # Save detailed results
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    return passed, partial, failed


if __name__ == "__main__":
    main()
