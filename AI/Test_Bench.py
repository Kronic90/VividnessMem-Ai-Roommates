"""
Test_Bench.py — Quick reusable benchmark for VividnessMem + VELMA

Runs Mem2ActBench with 3 conditions:
  1. no_memory   — LLM answers cold
  2. vividness   — VividnessMem only
  3. velma       — VELMA only

Same LLM, same temperature, same prompts.  Just the memory systems differ.

Usage:
  python Test_Bench.py                       # 100 items, all 3 conditions
  python Test_Bench.py --max-eval 400        # full 400
  python Test_Bench.py --condition vividness  # single condition
  python Test_Bench.py --max-eval 50         # super quick smoke test
"""

import sys, os, json, time, re, argparse, shutil, tempfile, random
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
if not hasattr(np.ndarray, '__class_getitem__'):
    np.ndarray.__class_getitem__ = classmethod(lambda cls, *args: cls)
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
STANDALONE_MEM = WORKSPACE / "standalone memory"
MEM2ACT_DIR = WORKSPACE / "Mem2ActBench_repo"
RESULTS_DIR = WORKSPACE / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(STANDALONE_MEM))

# ── LLM ──────────────────────────────────────────────────────────────────
MODEL_PATH = r"D:\AiStuff\google_gemma-3-12b-it-Q4_K_M.gguf"
CTX_SIZE = 8192
MAX_TOKENS = 512

_llm_cache = None

def load_llm():
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache
    from llama_cpp import Llama
    print(f"Loading LLM: {Path(MODEL_PATH).name}")
    t0 = time.time()
    _llm_cache = Llama(
        model_path=MODEL_PATH,
        n_ctx=CTX_SIZE,
        n_gpu_layers=48,
        verbose=False,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return _llm_cache


def generate(llm, messages, max_tokens=MAX_TOKENS, temperature=0.0):
    total_chars = sum(len(m.get("content", "")) for m in messages)
    budget = int((CTX_SIZE - max_tokens - 200) * 3.5)
    while total_chars > budget and len(messages) > 2:
        messages = [messages[0]] + messages[2:]
        total_chars = sum(len(m.get("content", "")) for m in messages)
    if total_chars > budget:
        last = messages[-1]
        last["content"] = last["content"][:max(200, len(last["content"]) - (total_chars - budget))]
    try:
        resp = llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [gen error] {e}")
        return ""


# ── Data Loading ─────────────────────────────────────────────────────────

def load_qa_dataset():
    path = MEM2ACT_DIR / "Mem2ActBench" / "qa_dataset.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def build_session_index():
    path = MEM2ACT_DIR / "Mem2ActBench" / "toolmem_conversation.jsonl"
    idx = {}
    for line in path.open(encoding="utf-8"):
        sess = json.loads(line)
        for oci in sess.get("original_conversation_ids", []):
            idx[oci] = sess
    return idx


def get_sessions_for_qa(qa, si):
    seen, out = set(), []
    for src in qa["source_conversation_ids"]:
        s = si.get(src)
        if s and s["session_id"] not in seen:
            seen.add(s["session_id"])
            out.append(s)
    return out


# ── Store Functions ──────────────────────────────────────────────────────

def store_sessions_vividness(vmem, sessions):
    for sess in sessions:
        for turn in sess["turns"]:
            role = turn["role"]
            content = (turn.get("content", "") or "").strip()
            if role == "user" and content:
                vmem.add_self_reflection(
                    content=f"User said: {content[:300]}",
                    emotion="neutral", importance=6,
                    source="conversation",
                    why_saved="User statement from past conversation",
                )
            tool_calls = turn.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    fact = f"Used tool {name} with args: {json.dumps(args)}"
                    vmem.add_self_reflection(
                        content=fact, emotion="neutral", importance=7,
                        source="tool_usage",
                        why_saved="Tool call from past interaction",
                    )


def store_sessions_velma(smem, sessions):
    for sess in sessions:
        for ti, turn in enumerate(sess["turns"]):
            content = (turn.get("content", "") or "").strip()
            if turn["role"] == "user" and content and len(content) > 20:
                smem.add_lesson(
                    title=f"User stated: {content[:50]}",
                    trigger=content[:200],
                    tags=set(),
                    strategy=f"Remember: {content[:200]}",
                )
            tool_calls = turn.get("tool_calls")
            if not tool_calls:
                continue
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                else:
                    args = args_raw if isinstance(args_raw, dict) else {}
                if not isinstance(args, dict):
                    args = {}
                trigger = ""
                for j in range(ti - 1, -1, -1):
                    if sess["turns"][j]["role"] == "user":
                        trigger = (sess["turns"][j].get("content", "") or "")[:200]
                        break
                if not name or not trigger:
                    continue
                val_tags = set()
                for v in args.values():
                    if isinstance(v, (str, int, float, bool)):
                        vs = str(v).lower()
                        val_tags.add(vs)
                        if len(vs) < 4:
                            val_tags.add(f"param_{vs}")
                smem.add_lesson(
                    title=f"Use {name} for: {trigger[:60]}",
                    trigger=trigger,
                    tags={name.lower()} | val_tags,
                    strategy=f"Call {name} with arguments: {json.dumps(args)}",
                )
                for key, val in args.items():
                    val_str = str(val).strip()
                    if not val_str or val_str in ("", "null", "none"):
                        continue
                    arg_tags = {name.lower(), key.lower(), val_str.lower()}
                    if len(val_str) < 4:
                        arg_tags.add(f"param_{val_str.lower()}")
                    smem.add_lesson(
                        title=f"Arg: {name}.{key}={val_str}",
                        trigger=f"{name} {key} {val_str} {trigger}"[:300],
                        tags=arg_tags,
                        strategy=f"For tool {name}, set {key}='{val_str}'. Context: {trigger[:150]}",
                    )


# ── Retrieve Functions ───────────────────────────────────────────────────

_RESOLVE_PROMPT = """You are a memory lookup assistant. The user's query contains vague or implicit references like "that stock", "the usual", "my favorite", "back home", etc.

Given the query and the memory context below, identify EVERY specific entity, value, or name the user is implicitly referring to.

Return ONLY a comma-separated list of the resolved values. No explanation.
If nothing needs resolving, return: NONE

Examples:
- Query: "Any fresh headlines on the crypto I track?" → Bitcoin
- Query: "breeds from back home" → american
- Query: "that graphics library we talked about" → Paper.js
- Query: "the search giant" → GOOGL
- Query: "What's new with that stock?" Memory mentions AAPL → AAPL"""

_IMPLICIT_MARKERS = frozenset({
    "that", "those", "the", "same", "usual", "always", "usually",
    "my", "our", "we", "typical", "regular", "normal",
    "again", "previous", "earlier", "back",
})


def _resolve_implicit_refs(llm, query, memory_context):
    """Ask Gemma to resolve implicit references in the query."""
    if not memory_context:
        return []
    msgs = [
        {"role": "system", "content": _RESOLVE_PROMPT},
        {"role": "user", "content": f"Memory Context:\n{memory_context[:2000]}\n\nQuery: {query}\n\nResolved values:"},
    ]
    try:
        raw = generate(llm, msgs, max_tokens=100, temperature=0.0)
        raw = raw.strip().strip('"').strip()
        if not raw or raw.upper() == "NONE":
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]
    except Exception:
        return []


def retrieve_vividness(vmem, query, tool_schema, llm=None):
    tool_name = tool_schema.get("name", "")
    tool_desc = tool_schema.get("description", "")[:200]
    search = f"{query} {tool_name} {tool_desc}"

    # Build a simple llm_fn wrapper for VividnessMem's semantic bridging
    llm_fn = None
    if llm:
        def llm_fn(prompt):
            return generate(llm, [{"role": "user", "content": prompt}],
                            max_tokens=120, temperature=0.0)

    # Standard resonance retrieval (with LLM semantic bridging when available)
    memories = vmem.resonate(search, limit=10, llm_fn=llm_fn)

    # Tool-first: also scan for memories mentioning this tool by name
    seen_contents = set()
    if memories:
        for m in memories:
            seen_contents.add(m.content)
    else:
        memories = []

    if tool_name:
        tool_lower = tool_name.lower()
        for mem in vmem.self_reflections:
            if mem.content in seen_contents:
                continue
            if tool_lower in mem.content.lower():
                memories.append(mem)
                seen_contents.add(mem.content)

    # ── Parameter-aware retrieval (D) ──
    # For each schema parameter, do a targeted search to find memories
    # containing that specific argument value
    params = tool_schema.get("parameters", {}).get("properties", {})
    if params:
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")[:100]
            # Search using param name + description + query context
            param_search = f"{query} {param_name} {param_desc} {tool_name}"
            param_mems = vmem.resonate(param_search, limit=5)
            if param_mems:
                for m in param_mems:
                    if m.content not in seen_contents:
                        memories.append(m)
                        seen_contents.add(m.content)
            # Also direct text scan for parameter name in memories
            param_lower = param_name.lower()
            for mem in vmem.self_reflections:
                if mem.content in seen_contents:
                    continue
                if param_lower in mem.content.lower():
                    memories.append(mem)
                    seen_contents.add(mem.content)

    if not memories:
        return ""

    lines = ["## Retrieved Memories (VividnessMem)"]
    for m in memories[:20]:
        lines.append(f"- {m.content}")
    initial_ctx = "\n".join(lines)

    # Gemma-powered implicit reference resolution
    query_words = set(query.lower().split())
    has_implicit = bool(query_words & _IMPLICIT_MARKERS)
    if has_implicit and llm:
        resolved = _resolve_implicit_refs(llm, query, initial_ctx)
        if resolved:
            for term in resolved:
                extra = vmem.resonate(f"{term} {tool_name}", limit=5)
                if extra:
                    for m in extra:
                        if m.content not in seen_contents:
                            lines.append(f"- {m.content}")
                            seen_contents.add(m.content)
                # Also direct text scan for resolved term
                term_lower = term.lower()
                for mem in vmem.self_reflections:
                    if mem.content in seen_contents:
                        continue
                    if term_lower in mem.content.lower():
                        lines.append(f"- {mem.content}")
                        seen_contents.add(mem.content)

    return "\n".join(lines) if lines else ""


def retrieve_velma(smem, query, tool_schema, llm=None):
    tool_name = tool_schema.get("name", "")
    results = {}

    # Tool-first: direct tag scan for all lessons tagged with this tool
    if tool_name:
        tool_tag = tool_name.lower().strip()
        for idx, lesson in enumerate(smem.lessons):
            if lesson.superseded_by:
                continue
            if tool_tag in lesson.tag_words:
                results[lesson.title] = (lesson, 1.0)

    s1 = f"{query} {tool_name} {tool_schema.get('description', '')[:200]}"
    for lesson, score in smem.retrieve(s1, limit=8):
        if lesson.title not in results:
            results[lesson.title] = (lesson, score)
    params = tool_schema.get("parameters", {}).get("properties", {})
    if params:
        parts = []
        for pk, pv in params.items():
            parts.append(pk)
            d = pv.get("description", "")
            if d:
                parts.append(d[:60])
        s2 = f"{query} {' '.join(parts)}"
        for lesson, score in smem.retrieve(s2, limit=6):
            if lesson.title not in results:
                results[lesson.title] = (lesson, score)
    tn = tool_schema.get("name", "")
    if tn:
        for lesson, score in smem.retrieve(tn, limit=5):
            if lesson.title not in results:
                results[lesson.title] = (lesson, score)

    if not results:
        return ""

    ranked = sorted(results.values(), key=lambda x: x[1], reverse=True)[:20]
    initial_lines = ["## Retrieved Procedural Knowledge (VELMA)"]
    for lesson, score in ranked:
        initial_lines.append(f"- {lesson.current_strategy[:250]}")
    initial_ctx = "\n".join(initial_lines)

    # Gemma-powered implicit reference resolution
    query_words = set(query.lower().split())
    has_implicit = bool(query_words & _IMPLICIT_MARKERS)
    if has_implicit and llm and results:
        resolved = _resolve_implicit_refs(llm, query, initial_ctx)
        if resolved:
            for term in resolved:
                term_search = f"{term} {tool_name}"
                for lesson, score in smem.retrieve(term_search, limit=5):
                    if lesson.title not in results:
                        results[lesson.title] = (lesson, score + 0.1)
                # Direct text scan for resolved term
                term_lower = term.lower()
                for lesson in smem.lessons:
                    if lesson.superseded_by or lesson.title in results:
                        continue
                    lt = f"{lesson.title} {lesson.trigger} {lesson.current_strategy}".lower()
                    if term_lower in lt:
                        results[lesson.title] = (lesson, 0.9)
            # Re-rank with new results
            ranked = sorted(results.values(), key=lambda x: x[1], reverse=True)[:20]
            initial_lines = ["## Retrieved Procedural Knowledge (VELMA)"]
            for lesson, score in ranked:
                initial_lines.append(f"- {lesson.current_strategy[:250]}")

    return "\n".join(initial_lines)


# ── Prompt & Parse ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI assistant that helps users by calling tools. Given a user query, past memory context, and a target tool schema, you must generate the correct tool call with filled-in arguments.

CRITICAL RULES:
1. You must respond ONLY with a JSON object in this exact format:
   {"name": "<tool_name>", "arguments": {<key>: <value>, ...}}
2. The "name" field MUST be EXACTLY the tool name from the Target Tool Schema. Never invent tool names.
3. ALWAYS use EXACT values from the memory context when filling arguments.
   - If memory says the tool was called with keyword='Bitcoin', use 'Bitcoin' — NOT 'crypto'.
   - If memory says text='Hytale', use 'Hytale' — NOT 'sandbox RPG'.
   - If memory says symbol='GOOGL', use 'GOOGL' — NOT 'Google' or 'GOOG'.
   - If memory says location='Monaco Grand Prix', use 'Monaco Grand Prix' — NOT 'Monaco'.
   - Copy the COMPLETE value string from memory. NEVER truncate, abbreviate, or paraphrase.
   - NEVER echo or paraphrase the user's query as an argument value.
   - NEVER substitute your own knowledge for values that appear in memory context.
4. When a user refers to something implicitly ("that stock", "the usual", "my favorite"),
   look up the SPECIFIC value from memory context and use it exactly as stored.
5. Include ALL parameters from the schema, both required AND optional:
   - If a parameter has a default value in the schema, ALWAYS include it.
   - If a parameter has an enum, pick the most appropriate value.
   - For parameters not mentioned in memory, use the schema's default or the most sensible value.
6. Scan memory context carefully for EVERY piece of information that could fill a parameter.
   Different memories may contain different argument values — check them all.

Do NOT include any other text, explanation, or markdown formatting. Just the raw JSON object."""


def build_prompt(query, tool_schema, memory_context=""):
    schema_text = json.dumps(tool_schema, indent=2, ensure_ascii=False)
    user = f"Target Tool Schema:\n{schema_text}\n\n"
    if memory_context:
        user += f"{memory_context}\n\n"
    user += f"User Query: {query}\n\nGenerate the tool call JSON:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_tool_call(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    match = re.search(r'\{[^{}]*"name"\s*:.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ── Metrics ──────────────────────────────────────────────────────────────

def normalize_value(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        # Normalize numeric types: 0.0 == 0, 10.0 == 10
        if float(v) == int(float(v)):
            return str(int(float(v)))
        return str(v)
    if isinstance(v, str):
        # Try parsing as number for comparison (e.g. "10" vs 10)
        try:
            fv = float(v.strip())
            if fv == int(fv):
                return str(int(fv))
            return str(fv)
        except (ValueError, TypeError):
            pass
        return v.strip().lower()
    return json.dumps(v, sort_keys=True).lower()


def compute_tool_accuracy(pred, gold):
    if pred is None:
        return 0.0
    if pred.get("name", "").strip().lower() != gold["name"].strip().lower():
        return 0.0
    for k, v in gold.get("arguments", {}).items():
        if k not in pred.get("arguments", {}):
            return 0.0
        if normalize_value(pred["arguments"][k]) != normalize_value(v):
            return 0.0
    return 1.0


def compute_f1(pred, gold):
    if pred is None:
        return 0.0, 0.0, 0.0
    gold_pairs = {(k, normalize_value(v)) for k, v in gold.get("arguments", {}).items()}
    pred_pairs = {(k, normalize_value(v)) for k, v in (pred.get("arguments", {}) or {}).items()}
    if not gold_pairs and not pred_pairs:
        return 1.0, 1.0, 1.0
    if not pred_pairs or not gold_pairs:
        return 0.0, 0.0, 0.0
    tp = len(gold_pairs & pred_pairs)
    prec = tp / len(pred_pairs)
    rec = tp / len(gold_pairs)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_bleu1(pred, gold):
    if pred is None:
        return 0.0
    gold_args = gold.get("arguments", {})
    pred_args = pred.get("arguments", {}) if pred else {}
    gold_tok, pred_tok = [], []
    for k, v in sorted(gold_args.items()):
        gold_tok.extend(str(k).lower().split())
        gold_tok.extend(str(v).lower().split())
    for k, v in sorted(pred_args.items()):
        pred_tok.extend(str(k).lower().split())
        pred_tok.extend(str(v).lower().split())
    if not gold_tok or not pred_tok:
        return 0.0
    gc = Counter(gold_tok)
    pc = Counter(pred_tok)
    clipped = sum(min(pc[w], gc[w]) for w in pc)
    prec = clipped / sum(pc.values())
    bp = min(1.0, len(pred_tok) / len(gold_tok))
    return bp * prec


# ── Runner ───────────────────────────────────────────────────────────────

def run_condition(llm, qa_items, session_index, condition, max_eval):
    print(f"\n{'='*60}")
    print(f"  Mem2ActBench -- {condition}")
    print(f"  Items: {min(max_eval, len(qa_items))}")
    print(f"{'='*60}\n")

    metrics = {"tool_accuracy": [], "f1": [], "precision": [], "recall": [], "bleu1": []}
    level_metrics = {}
    predictions = []

    items = qa_items[:max_eval]
    for i, qa in enumerate(tqdm(items, desc=condition)):
        query = qa["query"]
        gold = qa["tool_call"]
        schema = qa["target_tool_schema"]
        level = qa["complexity_metadata"]["level"]
        sessions = get_sessions_for_qa(qa, session_index)

        memory_ctx = ""

        if condition == "vividness":
            tmp = tempfile.mkdtemp(prefix="tb_viv_")
            try:
                from VividnessMem import VividnessMem
                vmem = VividnessMem(data_dir=os.path.join(tmp, "v"))
                store_sessions_vividness(vmem, sessions)
                vmem.save()
                memory_ctx = retrieve_vividness(vmem, query, schema, llm=llm)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        elif condition == "velma":
            tmp = tempfile.mkdtemp(prefix="tb_vel_")
            try:
                from VELMA import VELMA
                smem = VELMA(data_dir=os.path.join(tmp, "s"))
                store_sessions_velma(smem, sessions)
                smem.save()
                memory_ctx = retrieve_velma(smem, query, schema, llm=llm)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        msgs = build_prompt(query, schema, memory_ctx)
        raw = generate(llm, msgs)
        pred = parse_tool_call(raw)

        ta = compute_tool_accuracy(pred, gold)
        prec, rec, f1 = compute_f1(pred, gold)
        b1 = compute_bleu1(pred, gold)

        metrics["tool_accuracy"].append(ta)
        metrics["f1"].append(f1)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["bleu1"].append(b1)

        if level not in level_metrics:
            level_metrics[level] = {"tool_accuracy": [], "f1": [], "bleu1": []}
        level_metrics[level]["tool_accuracy"].append(ta)
        level_metrics[level]["f1"].append(f1)
        level_metrics[level]["bleu1"].append(b1)

        predictions.append({
            "qa_id": qa["qa_id"], "level": level, "query": query,
            "gold_tool": gold["name"], "gold_args": gold["arguments"],
            "pred_raw": raw[:500], "pred_parsed": pred,
            "tool_accuracy": ta, "f1": f1, "bleu1": b1,
        })

        if (i + 1) % 25 == 0:
            avg_ta = sum(metrics["tool_accuracy"]) / len(metrics["tool_accuracy"])
            avg_f1 = sum(metrics["f1"]) / len(metrics["f1"])
            print(f"  [{i+1}/{len(items)}] TA={avg_ta:.3f} F1={avg_f1:.3f}")

    n = len(metrics["tool_accuracy"])
    report = {
        "condition": condition, "n_evaluated": n,
        "timestamp": datetime.now().isoformat(),
        "overall": {k: sum(v)/n for k, v in metrics.items()},
        "per_level": {},
    }
    for lvl, lm in sorted(level_metrics.items()):
        ln = len(lm["tool_accuracy"])
        report["per_level"][lvl] = {
            "n": ln,
            "tool_accuracy": sum(lm["tool_accuracy"]) / ln,
            "f1": sum(lm["f1"]) / ln,
            "bleu1": sum(lm["bleu1"]) / ln,
        }
    return report, predictions


# ── Display ──────────────────────────────────────────────────────────────

def print_report(report):
    o = report["overall"]
    print(f"\n{'-'*50}")
    print(f"  {report['condition'].upper()} -- {report['n_evaluated']} items")
    print(f"{'-'*50}")
    for k in ["tool_accuracy", "f1", "precision", "recall", "bleu1"]:
        print(f"  {k:<16}: {o[k]:.4f}")
    print(f"\n  Per-Level:")
    for lvl, lm in sorted(report["per_level"].items()):
        print(f"    {lvl} (n={lm['n']}): TA={lm['tool_accuracy']:.3f}  F1={lm['f1']:.3f}  BLEU-1={lm['bleu1']:.3f}")


def print_comparison(reports, prev=None):
    print(f"\n{'='*75}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*75}")
    header = f"{'Metric':<18}"
    for r in reports:
        header += f" {r['condition']:>12}"
    if prev:
        header += "  |  prev_vivid  prev_velma"
    print(header)
    print("-" * len(header))
    for m in ["tool_accuracy", "f1", "precision", "recall", "bleu1"]:
        row = f"  {m:<16}"
        for r in reports:
            row += f" {r['overall'][m]:>12.4f}"
        if prev:
            pv = prev.get("vividness", {}).get(m, 0)
            ps = prev.get("velma", {}).get(m, 0)
            row += f"  |  {pv:>10.4f}  {ps:>10.4f}"
        print(row)

    all_lvls = sorted(set(l for r in reports for l in r["per_level"]))
    for lvl in all_lvls:
        print(f"\n  {lvl}:")
        for m in ["tool_accuracy", "f1", "bleu1"]:
            row = f"    {m:<14}"
            for r in reports:
                lm = r["per_level"].get(lvl, {})
                row += f" {lm.get(m, 0):>12.4f}"
            print(row)


# ── Previous Benchmark Results (pre-improvement baseline) ───────────────
PREV_RESULTS = {
    "no_memory":  {"tool_accuracy": 0.053, "f1": 0.118, "precision": 0.118, "recall": 0.118, "bleu1": 0.273},
    "vividness":  {"tool_accuracy": 0.355, "f1": 0.439, "precision": 0.439, "recall": 0.439, "bleu1": 0.579},
    "velma":      {"tool_accuracy": 0.340, "f1": 0.436, "precision": 0.436, "recall": 0.436, "bleu1": 0.574},
}


# ── Main ─────────────────────────────────────────────────────────────────

def print_averaged_results(all_seed_reports):
    """Print averaged metrics across multiple seed runs."""
    # all_seed_reports: {condition: [list of reports per seed]}
    print(f"\n{'='*70}")
    print(f"  AVERAGED RESULTS ACROSS {len(list(all_seed_reports.values())[0])} SEEDS")
    print(f"{'='*70}")
    for cond, reports in all_seed_reports.items():
        n_seeds = len(reports)
        avg = {}
        for m in ["tool_accuracy", "f1", "precision", "recall", "bleu1"]:
            vals = [r["overall"][m] for r in reports]
            avg[m] = sum(vals) / n_seeds
            avg[m + "_std"] = (sum((v - avg[m])**2 for v in vals) / n_seeds) ** 0.5
        print(f"\n  {cond.upper()} (n={reports[0]['n_evaluated']} x {n_seeds} seeds)")
        print(f"  {'-'*50}")
        for m in ["tool_accuracy", "f1", "precision", "recall", "bleu1"]:
            vals = [r["overall"][m] for r in reports]
            print(f"    {m:<16}: {avg[m]:.4f} +/- {avg[m+'_std']:.4f}  ({', '.join(f'{v:.3f}' for v in vals)})")

        # Per-level averaged
        all_lvls = sorted(set(l for r in reports for l in r["per_level"]))
        print(f"\n    Per-Level:")
        for lvl in all_lvls:
            ta_vals = [r["per_level"].get(lvl, {}).get("tool_accuracy", 0) for r in reports]
            f1_vals = [r["per_level"].get(lvl, {}).get("f1", 0) for r in reports]
            b1_vals = [r["per_level"].get(lvl, {}).get("bleu1", 0) for r in reports]
            n_items = reports[0]["per_level"].get(lvl, {}).get("n", "?")
            ta_avg = sum(ta_vals) / n_seeds
            f1_avg = sum(f1_vals) / n_seeds
            b1_avg = sum(b1_vals) / n_seeds
            print(f"      {lvl} (n~{n_items}): TA={ta_avg:.3f}  F1={f1_avg:.3f}  BLEU-1={b1_avg:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Quick benchmark: VividnessMem + VELMA")
    parser.add_argument("--condition", default="all",
                        choices=["no_memory", "vividness", "velma", "all"])
    parser.add_argument("--max-eval", type=int, default=100,
                        help="Number of QA items to evaluate (default 100, max 400)")
    parser.add_argument("--no-compare-prev", action="store_true",
                        help="Don't show previous results in comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds for multi-run averaging (e.g. --seeds 42 123 7 99 256)")
    args = parser.parse_args()

    print("Loading Mem2ActBench data...")
    qa_items = load_qa_dataset()
    session_index = build_session_index()
    print(f"  {len(qa_items)} QA items, {len(session_index)} sessions indexed")

    llm = load_llm()

    if args.condition == "all":
        conds = ["no_memory", "vividness", "velma"]
    else:
        conds = [args.condition]

    seeds = args.seeds or [None]  # None = no shuffle (original order)
    all_seed_reports = {c: [] for c in conds}  # condition -> [reports]

    for seed_idx, seed in enumerate(seeds):
        if seed is not None:
            print(f"\n{'#'*60}")
            print(f"  SEED {seed} (run {seed_idx+1}/{len(seeds)})")
            print(f"{'#'*60}")
            shuffled = list(qa_items)
            random.seed(seed)
            random.shuffle(shuffled)
        else:
            shuffled = qa_items

        all_reports = []
        for cond in conds:
            t0 = time.time()
            report, preds = run_condition(llm, shuffled, session_index, cond, args.max_eval)
            elapsed = time.time() - t0
            all_reports.append(report)
            all_seed_reports[cond].append(report)
            print_report(report)
            print(f"  Time: {elapsed:.0f}s")

            # Save results
            seed_tag = f"_s{seed}" if seed is not None else ""
            tag = datetime.now().strftime("%Y%m%d_%H%M")
            rp = RESULTS_DIR / f"TestBench_{cond}{seed_tag}_{tag}.json"
            rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"  Saved: {rp.name}")

        if len(all_reports) > 1:
            prev = None if args.no_compare_prev else PREV_RESULTS
            print_comparison(all_reports, prev=prev)

    # Print averaged results if multi-seed
    if len(seeds) > 1:
        print_averaged_results(all_seed_reports)

    print("\nDone!")


if __name__ == "__main__":
    main()
