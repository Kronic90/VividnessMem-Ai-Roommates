"""
Diagnose VividnessMem benchmark failures.
Runs 5 items per level (L1-L4) and shows:
- What memories were stored
- What was retrieved  
- What the LLM generated
- What the gold answer was
"""
import sys, os, json, tempfile, shutil, time
from pathlib import Path

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE / "standalone memory"))

from Test_Bench import (
    load_llm, load_qa_dataset, build_session_index, get_sessions_for_qa,
    store_sessions_vividness, retrieve_vividness, build_prompt, generate,
    parse_tool_call, compute_tool_accuracy, compute_f1
)

def main():
    llm = load_llm()
    qa_items = load_qa_dataset()
    session_index = build_session_index()

    # Group by level
    by_level = {}
    for qa in qa_items:
        lvl = qa["complexity_metadata"]["level"]
        by_level.setdefault(lvl, []).append(qa)

    print(f"Total items: {len(qa_items)}")
    for lvl in sorted(by_level.keys()):
        print(f"  {lvl}: {len(by_level[lvl])} items")

    # Sample from each level
    for lvl in sorted(by_level.keys()):
        items = by_level[lvl][:5]  # first 5 per level
        print(f"\n{'='*80}")
        print(f"  LEVEL {lvl} — {len(items)} sample items")
        print(f"{'='*80}")

        for qi, qa in enumerate(items):
            query = qa["query"]
            gold = qa["tool_call"]
            schema = qa["target_tool_schema"]
            sessions = get_sessions_for_qa(qa, session_index)

            tmp = tempfile.mkdtemp(prefix="diag_")
            try:
                from VividnessMem import VividnessMem
                vmem = VividnessMem(data_dir=os.path.join(tmp, "v"))
                store_sessions_vividness(vmem, sessions)
                vmem.save()

                n_memories = len(vmem.self_reflections)
                
                # Retrieve
                memory_ctx = retrieve_vividness(vmem, query, schema, llm=llm)
                
                # Generate
                msgs = build_prompt(query, schema, memory_ctx)
                raw = generate(llm, msgs)
                pred = parse_tool_call(raw)
                
                ta = compute_tool_accuracy(pred, gold)
                _, _, f1 = compute_f1(pred, gold)
                
                status = "PASS" if ta == 1.0 else "FAIL"
                
                print(f"\n--- Item {qi+1} [{status}] TA={ta:.0f} F1={f1:.2f} ---")
                print(f"  Query: {query}")
                print(f"  Gold: {json.dumps(gold)[:300]}")
                print(f"  Pred: {json.dumps(pred) if pred else 'NONE'}")
                if pred and not ta:
                    print(f"  Raw LLM: {raw[:300]}")
                print(f"  Sessions: {len(sessions)}, Stored memories: {n_memories}")
                
                # Show retrieved context (abbreviated)
                if memory_ctx:
                    ctx_lines = memory_ctx.split('\n')
                    print(f"  Retrieved ({len(ctx_lines)-1} memories):")
                    for cl in ctx_lines[1:6]:  # first 5 memories
                        print(f"    {cl[:150]}")
                    if len(ctx_lines) > 6:
                        print(f"    ... +{len(ctx_lines)-6} more")
                else:
                    print(f"  Retrieved: NOTHING")
                
                # If failed, show WHY
                if not ta and pred:
                    gold_name = gold.get("name", "")
                    pred_name = pred.get("name", "")
                    if pred_name != gold_name:
                        print(f"  REASON: Wrong tool name (pred={pred_name}, gold={gold_name})")
                    else:
                        # Compare args
                        gold_args = gold.get("arguments", {})
                        pred_args = pred.get("arguments", {})
                        for k in set(list(gold_args.keys()) + list(pred_args.keys())):
                            gv = str(gold_args.get(k, "")).lower().strip()
                            pv = str(pred_args.get(k, "")).lower().strip()
                            if gv != pv:
                                print(f"  REASON: Arg '{k}' mismatch: pred='{pv[:80]}' gold='{gv[:80]}'")
                elif not pred:
                    print(f"  REASON: Failed to parse prediction")
                    print(f"  Raw: {raw[:300]}")
                    
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
