"""
benchmark_vividness.py — MemoryBench evaluation harness for VividnessMem

Tests VividnessMem against MemoryBench datasets using fully automatic metrics
(no LLM-as-judge needed). Runs two conditions:
  1. no_memory  — Gemma 3 12B answers test questions cold
  2. vividness  — Same LLM, but with VividnessMem storing training dialog
                  feedback and enriching test prompts via resonate()

Datasets tested (all have automatic metrics):
  - WritingPrompts  (METEOR)
  - LexEval-*       (ROUGE-L)
  - JRE-L           (Rouge-L, BERTScore-F1, CLI, FKGL, DCRS)

Usage:
  python benchmark_vividness.py --dataset WritingPrompts --max-train 50 --max-test 20
  python benchmark_vividness.py --dataset JRE-L --max-train 50 --max-test 20

Results are saved to benchmark_results/<dataset>_<condition>.json
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Pre-import torch to avoid DLL loading issues on Windows
# (must happen before bert_score / evaluate)
try:
    import torch  # noqa: F401
except ImportError:
    pass

from tqdm import tqdm

# ── Setup paths ──────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
MEMORYBENCH_DIR = WORKSPACE / "MemoryBench"
STANDALONE_MEM = WORKSPACE / "standalone memory"
RESULTS_DIR = WORKSPACE / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(MEMORYBENCH_DIR))
sys.path.insert(0, str(STANDALONE_MEM))

from VividnessMem import VividnessMem

# ── LLM via llama-cpp ────────────────────────────────────────────────────
MODEL_PATH = r"D:\AiStuff\google_gemma-3-12b-it-Q4_K_M.gguf"
CTX_SIZE = 8192
MAX_TOKENS = 1024  # shorter for benchmark — we want concise answers


def load_llm():
    """Load Gemma 3 12B via llama-cpp-python."""
    from llama_cpp import Llama
    print(f"Loading model from {MODEL_PATH}…")
    t0 = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=CTX_SIZE,
        n_gpu_layers=40,
        verbose=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return llm


def generate(llm, messages: list[dict], max_tokens: int = MAX_TOKENS) -> str:
    """Generate a response, trimming history if needed to fit context."""
    # Estimate tokens (~3.5 chars per token)
    total_chars = sum(len(m.get("content", "")) for m in messages)
    budget = int((CTX_SIZE - max_tokens - 200) * 3.5)

    # Trim from the middle if too long
    while total_chars > budget and len(messages) > 2:
        messages = [messages[0]] + messages[2:]
        total_chars = sum(len(m.get("content", "")) for m in messages)

    # If still too long, truncate the user message
    if total_chars > budget:
        excess = total_chars - budget
        last = messages[-1]
        last["content"] = last["content"][: max(200, len(last["content"]) - excess)]

    try:
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  Generation error: {e}")
        return ""


# ── Memory helpers ───────────────────────────────────────────────────────

def store_dialog_into_memory(mem: VividnessMem, dialog: list[dict],
                             feedback: list[dict], dataset_name: str):
    """Parse a training dialog + implicit feedback and store into VividnessMem.

    Strategy:
    - For each assistant response, check the corresponding feedback
    - High satisfaction (>=8): store as positive reflection
    - Low satisfaction (<=4): store as negative reflection with what to improve
    - Medium (5-7): store if there are implicit actions (corrections)
    """
    # Map round numbers to feedback
    fb_map = {}
    if feedback:
        for fb in feedback:
            fb_map[fb.get("round", 0)] = fb

    round_num = 0
    for i, msg in enumerate(dialog):
        if msg.get("role") == "assistant":
            round_num += 1
            content = msg.get("content", "")
            if not content or len(content) < 20:
                continue

            fb = fb_map.get(round_num, {})
            satisfaction = fb.get("satisfaction_score", 5)
            actions = fb.get("implicit_actions", [])

            # Get the preceding user question for context
            user_q = ""
            if i > 0 and dialog[i - 1].get("role") == "user":
                user_q = dialog[i - 1].get("content", "")[:200]

            if satisfaction >= 8:
                # Positive: remember what worked
                reflection = (
                    f"When asked about '{user_q[:100]}', my response was well-received "
                    f"(satisfaction {satisfaction}/10). "
                    f"Key approach: {content[:200]}"
                )
                mem.add_self_reflection(
                    content=reflection,
                    emotion="confident",
                    importance=min(10, satisfaction),
                    source=f"benchmark-{dataset_name}",
                    why_saved=f"High satisfaction ({satisfaction}) feedback from training",
                )
            elif satisfaction <= 4:
                # Negative: remember what didn't work
                action_text = ""
                if actions:
                    action_text = f" User corrections: {', '.join(str(a) for a in actions[:3])}"
                reflection = (
                    f"When asked about '{user_q[:100]}', my response was poorly received "
                    f"(satisfaction {satisfaction}/10).{action_text} "
                    f"I need to improve: {content[:150]}"
                )
                mem.add_self_reflection(
                    content=reflection,
                    emotion="frustrated",
                    importance=max(5, 10 - satisfaction),
                    source=f"benchmark-{dataset_name}",
                    why_saved=f"Low satisfaction ({satisfaction}) — learning from mistake",
                )
            elif actions:
                # Medium with corrections: note the feedback
                action_text = ", ".join(str(a) for a in actions[:3])
                reflection = (
                    f"For question '{user_q[:100]}', I received corrections: {action_text}. "
                    f"Satisfaction was {satisfaction}/10. My response: {content[:150]}"
                )
                mem.add_self_reflection(
                    content=reflection,
                    emotion="reflective",
                    importance=6,
                    source=f"benchmark-{dataset_name}",
                    why_saved="User corrections during training",
                )


def build_enriched_prompt(mem: VividnessMem, user_prompt: str,
                          max_resonance: int = 5) -> str:
    """Enrich a test prompt with resonated memories from VividnessMem."""
    resonant = mem.resonate(user_prompt, limit=max_resonance)
    if not resonant:
        return user_prompt

    memory_lines = []
    for r in resonant:
        tag = f" ({r.emotion})" if r.emotion else ""
        memory_lines.append(f"— {r.content[:300]}{tag}")

    memory_block = "\n".join(memory_lines)
    enriched = (
        f"{user_prompt}\n\n"
        f"[Relevant experience from past interactions:]\n{memory_block}\n\n"
        f"Use the above experience to inform your response where relevant."
    )
    return enriched


# ── Dataset loading ──────────────────────────────────────────────────────

def load_dataset(dataset_name: str, eval_mode: bool = True):
    """Load a single MemoryBench dataset."""
    # Need to be in MemoryBench dir for imports to work
    orig_dir = os.getcwd()
    os.chdir(str(MEMORYBENCH_DIR))
    try:
        from memorybench import load_memory_bench
        dataset = load_memory_bench("single", dataset_name, eval_mode=eval_mode)
        return dataset
    finally:
        os.chdir(orig_dir)


# ── Benchmark runner ─────────────────────────────────────────────────────

def run_benchmark(dataset_name: str, max_train: int = 100,
                  max_test: int = 50, skip_no_memory: bool = False):
    """Run the full benchmark: no-memory baseline + VividnessMem condition."""

    print(f"\n{'='*70}")
    print(f"  MEMORYBENCH EVALUATION: {dataset_name}")
    print(f"  Max training dialogs: {max_train}")
    print(f"  Max test samples: {max_test}")
    print(f"{'='*70}\n")

    # Load LLM
    llm = load_llm()

    # Load dataset (non-eval for training data, eval for testing)
    print("Loading dataset for training phase…")
    orig_dir = os.getcwd()
    os.chdir(str(MEMORYBENCH_DIR))
    try:
        from memorybench import load_memory_bench
        dataset_train = load_memory_bench("single", dataset_name, eval_mode=False)
        print("Loading dataset for evaluation phase…")
        dataset_eval = load_memory_bench("single", dataset_name, eval_mode=True)
    finally:
        os.chdir(orig_dir)

    train_data = list(dataset_train.dataset["train"])[:max_train]
    test_data = list(dataset_eval.dataset["test"])[:max_test]

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # ── Condition 1: No memory baseline ──
    if not skip_no_memory:
        print(f"\n{'─'*50}")
        print("  CONDITION 1: No Memory (baseline)")
        print(f"{'─'*50}")

        no_mem_predictions = []
        for data in tqdm(test_data, desc="No-memory predictions"):
            test_idx = data["test_idx"]
            input_prompt = data.get("input_prompt", "")
            if not input_prompt and "input_chat_messages" in data:
                input_prompt = data["input_chat_messages"][-1]["content"]

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
                {"role": "user", "content": input_prompt},
            ]
            response = generate(llm, messages)
            no_mem_predictions.append({
                "test_idx": test_idx,
                "response": response,
                "dataset": dataset_name,
            })

        # Save predictions
        no_mem_file = RESULTS_DIR / f"{dataset_name}_no_memory_predictions.json"
        with open(no_mem_file, "w", encoding="utf-8") as f:
            json.dump(no_mem_predictions, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(no_mem_predictions)} no-memory predictions")

    # ── Condition 2: VividnessMem ──
    print(f"\n{'─'*50}")
    print("  CONDITION 2: VividnessMem")
    print(f"{'─'*50}")

    # Create fresh VividnessMem instance
    mem_dir = RESULTS_DIR / f"{dataset_name}_vividmem_data"
    mem_dir.mkdir(exist_ok=True)
    mem = VividnessMem(
        data_dir=str(mem_dir),
    )

    # Store training dialogs into memory
    print("Storing training dialogs into VividnessMem…")
    stored_count = 0
    for data in tqdm(train_data, desc="Memorizing training data"):
        dialog = data.get("dialog", [])
        feedback = data.get("implicit_feedback", [])
        if dialog:
            store_dialog_into_memory(mem, dialog, feedback, dataset_name)
            stored_count += 1
    mem.save()
    print(f"Stored feedback from {stored_count} dialogs → "
          f"{len(mem.self_reflections)} self-reflections in VividnessMem")

    # Run VividnessMem-enriched predictions
    vivid_predictions = []
    resonance_hits = 0
    for data in tqdm(test_data, desc="VividnessMem predictions"):
        test_idx = data["test_idx"]
        input_prompt = data.get("input_prompt", "")
        if not input_prompt and "input_chat_messages" in data:
            input_prompt = data["input_chat_messages"][-1]["content"]

        # Enrich with VividnessMem resonance
        enriched = build_enriched_prompt(mem, input_prompt)
        if enriched != input_prompt:
            resonance_hits += 1

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately. Use any relevant experience provided to improve your response."},
            {"role": "user", "content": enriched},
        ]
        response = generate(llm, messages)
        vivid_predictions.append({
            "test_idx": test_idx,
            "response": response,
            "dataset": dataset_name,
        })

    vivid_file = RESULTS_DIR / f"{dataset_name}_vividness_predictions.json"
    with open(vivid_file, "w", encoding="utf-8") as f:
        json.dump(vivid_predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(vivid_predictions)} VividnessMem predictions")
    print(f"Resonance hits: {resonance_hits}/{len(test_data)} "
          f"({100*resonance_hits/max(1,len(test_data)):.0f}%)")

    # ── Evaluate both conditions ──
    print(f"\n{'─'*50}")
    print("  EVALUATION")
    print(f"{'─'*50}")

    results = {}

    os.chdir(str(MEMORYBENCH_DIR))
    try:
        from memorybench import evaluate, summary_results

        if not skip_no_memory:
            print("\nEvaluating no-memory baseline…")
            no_mem_details = evaluate("single", dataset_name, no_mem_predictions)
            no_mem_summary = summary_results("single", dataset_name,
                                              no_mem_predictions, no_mem_details)
            results["no_memory"] = {
                "summary": no_mem_summary["summary"],
                "details": [
                    {"test_idx": d["test_idx"], "metrics": d["metrics"]}
                    for d in no_mem_details
                ],
            }
            print(f"  No-memory summary: {no_mem_summary['summary']}")

        print("\nEvaluating VividnessMem condition…")
        vivid_details = evaluate("single", dataset_name, vivid_predictions)
        vivid_summary = summary_results("single", dataset_name,
                                         vivid_predictions, vivid_details)
        results["vividness"] = {
            "summary": vivid_summary["summary"],
            "details": [
                {"test_idx": d["test_idx"], "metrics": d["metrics"]}
                for d in vivid_details
            ],
        }
        print(f"  VividnessMem summary: {vivid_summary['summary']}")
    finally:
        os.chdir(orig_dir)

    # ── Report ──
    print(f"\n{'='*70}")
    print(f"  RESULTS: {dataset_name}")
    print(f"{'='*70}")

    report = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "Gemma-3-12B-IT-Q4_K_M",
            "ctx_size": CTX_SIZE,
            "max_tokens": MAX_TOKENS,
            "max_train": max_train,
            "max_test": max_test,
        },
        "memory_stats": {
            "self_reflections": len(mem.self_reflections),
            "resonance_hits": resonance_hits,
            "resonance_rate": resonance_hits / max(1, len(test_data)),
        },
        "results": {},
    }

    if not skip_no_memory and "no_memory" in results:
        report["results"]["no_memory"] = results["no_memory"]["summary"]
        print("\n  No Memory (baseline):")
        for k, v in results["no_memory"]["summary"].items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    report["results"]["vividness"] = results["vividness"]["summary"]
    print("\n  VividnessMem:")
    for k, v in results["vividness"]["summary"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    if not skip_no_memory and "no_memory" in results:
        print("\n  Delta (VividnessMem − baseline):")
        for k in results["vividness"]["summary"]:
            if k in results["no_memory"]["summary"]:
                v_val = results["vividness"]["summary"][k]
                n_val = results["no_memory"]["summary"][k]
                if isinstance(v_val, (int, float)) and isinstance(n_val, (int, float)):
                    delta = v_val - n_val
                    sign = "+" if delta >= 0 else ""
                    print(f"    {k}: {sign}{delta:.4f}")
                    report["results"].setdefault("delta", {})[k] = delta

    # Save full report
    report_file = RESULTS_DIR / f"{dataset_name}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to {report_file}")

    return report


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MemoryBench evaluation harness for VividnessMem")
    parser.add_argument("--dataset", type=str, default="WritingPrompts",
                        help="Dataset name (WritingPrompts, JRE-L, LexEval-Summarization, etc.)")
    parser.add_argument("--max-train", type=int, default=50,
                        help="Max training dialogs to store in memory")
    parser.add_argument("--max-test", type=int, default=20,
                        help="Max test samples to evaluate")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the no-memory baseline")
    args = parser.parse_args()

    run_benchmark(
        dataset_name=args.dataset,
        max_train=args.max_train,
        max_test=args.max_test,
        skip_no_memory=args.skip_baseline,
    )


if __name__ == "__main__":
    main()
