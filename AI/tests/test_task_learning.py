"""
test_task_learning.py -- Test whether the AIs actually learn from doing tasks.

Tests the LIVE task memory data from real conversations.
Checks: retrieval quality, search relevance, learning progression,
        keyword extraction, vividness dynamics, and system behavior.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import from the actual project
from task_memory import (
    AriaTaskMemory,
    RexTaskMemory,
    TaskEntry,
    parse_task_memory_tags,
    _score_relevance,
)

DATA_ROOT = Path(__file__).parent.parent / "ai_dialogue_data"

pass_count = 0
fail_count = 0
failures = []


def check(label, condition, detail=""):
    global pass_count, fail_count
    if condition:
        print(f"  [PASS] {label}")
        pass_count += 1
    else:
        print(f"  [FAIL] {label}")
        if detail:
            print(f"         {detail}")
        fail_count += 1
        failures.append((label, detail))


print("=" * 76)
print("  TASK LEARNING TEST")
print("  Do the AIs actually learn from doing tasks?")
print("  Testing against LIVE task memory data")
print("=" * 76)

# Load the live data
aria_tm = AriaTaskMemory()
rex_tm = RexTaskMemory()

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 1: Live Data Exists -- Both AIs Have Task Memories")
print("=" * 76)

aria_total = len(aria_tm.entries)
rex_total = len(rex_tm.entries)
rex_core = len(rex_tm.core)
rex_archival = len(rex_tm.archival)

print(f"\n  Aria: {aria_total} task memories")
print(f"  Rex:  {rex_total} task memories ({rex_core} core, {rex_archival} archival)")

check("Aria has task memories", aria_total > 0, f"Found {aria_total}")
check("Rex has task memories", rex_total > 0, f"Found {rex_total}")
check("Rex has core memories (high-importance learnings)", rex_core > 0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 2: Memory Types -- Auto-captured vs Deliberate Learning")
print("=" * 76)

aria_auto = [e for e in aria_tm.entries if e.task_type == "auto"]
aria_deliberate = [e for e in aria_tm.entries if e.task_type != "auto"]
rex_auto = [e for e in rex_tm.entries if e.task_type == "auto"]
rex_deliberate = [e for e in rex_tm.entries if e.task_type != "auto"]

print(f"\n  Aria auto-captured (file ops, code results): {len(aria_auto)}")
print(f"  Aria deliberate (LLM-tagged reflections):     {len(aria_deliberate)}")
print(f"  Rex auto-captured:  {len(rex_auto)}")
print(f"  Rex deliberate:     {len(rex_deliberate)}")

check("Aria has BOTH auto and deliberate memories",
      len(aria_auto) > 0 and len(aria_deliberate) > 0)
check("Auto-captured entries have low importance (noise filter)",
      all(e.importance <= 5 for e in aria_auto),
      f"Max auto imp: {max(e.importance for e in aria_auto) if aria_auto else 'N/A'}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 3: Deliberate Reflections Have Substance")
print("=" * 76)

print("\n  Aria's deliberate task reflections:")
for e in aria_deliberate[:5]:
    print(f"    imp={e.importance} | {e.summary[:70]}...")
    if e.reflection:
        print(f"    -> {e.reflection[:70]}...")

check("Deliberate entries have reflections",
      sum(1 for e in aria_deliberate if e.reflection) > len(aria_deliberate) * 0.5,
      f"{sum(1 for e in aria_deliberate if e.reflection)}/{len(aria_deliberate)} have reflections")
check("Deliberate entries have higher importance than auto",
      (sum(e.importance for e in aria_deliberate) / max(len(aria_deliberate), 1)) >
      (sum(e.importance for e in aria_auto) / max(len(aria_auto), 1)),
      f"Deliberate avg: {sum(e.importance for e in aria_deliberate)/max(len(aria_deliberate),1):.1f}, "
      f"Auto avg: {sum(e.importance for e in aria_auto)/max(len(aria_auto),1):.1f}")
check("Deliberate entries have meaningful keywords",
      sum(1 for e in aria_deliberate if len(e.keywords) >= 2) > len(aria_deliberate) * 0.5)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 4: Search Retrieval -- Can They Find What They Learned?")
print("=" * 76)

# Aria's organic search (vividness-ranked)
aria_sim = aria_tm.search("simulation Aetheria society")
aria_ethics = aria_tm.search("cognitive bias conformity totalitarianism")
aria_random = aria_tm.search("banana underwater basketball")

print(f"\n  Aria search('simulation Aetheria society'): {len(aria_sim)} results")
for e in aria_sim:
    print(f"    viv={e.vividness:.1f} imp={e.importance} | {e.summary[:60]}...")
print(f"\n  Aria search('cognitive bias conformity'): {len(aria_ethics)} results")
for e in aria_ethics:
    print(f"    viv={e.vividness:.1f} imp={e.importance} | {e.summary[:60]}...")
print(f"\n  Aria search('banana underwater basketball'): {len(aria_random)} results")

check("Aria finds simulation-related memories",
      len(aria_sim) > 0)
check("Most simulation results are about simulations",
      sum(1 for e in aria_sim if "simul" in e.summary.lower() or "aetheria" in e.summary.lower()) >= len(aria_sim) * 0.5,
      f"{sum(1 for e in aria_sim if 'simul' in e.summary.lower() or 'aetheria' in e.summary.lower())}/{len(aria_sim)} match")
check("No false positives for random query", len(aria_random) == 0)

# Rex's keyword search (archival)
rex_sim = rex_tm.search("simulation")
rex_climate = rex_tm.search("climate Aetheria")
rex_random = rex_tm.search("banana underwater basketball")

print(f"\n  Rex search('simulation'): {len(rex_sim)} results")
for e in rex_sim:
    print(f"    imp={e.importance} | {e.summary[:60]}...")
print(f"\n  Rex search('climate Aetheria'): {len(rex_climate)} results")
print(f"  Rex search('banana underwater basketball'): {len(rex_random)} results")

check("Rex finds simulation memories", len(rex_sim) > 0)
check("Rex random query returns few/no results", len(rex_random) <= 1,
      f"Got {len(rex_random)} results (keyword overlap in code snippets is expected)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 5: Vividness Curve -- Do Important Experiences Stay Vivid?")
print("=" * 76)

# Sort by vividness and check if high-importance entries rank higher
all_vivid = sorted(aria_tm.entries, key=lambda e: e.vividness, reverse=True)
top5 = all_vivid[:5]
bottom5 = all_vivid[-5:]

top5_avg_imp = sum(e.importance for e in top5) / len(top5)
bottom5_avg_imp = sum(e.importance for e in bottom5) / len(bottom5)

print(f"\n  Top 5 by vividness:")
for e in top5:
    age_days = (datetime.now() - datetime.fromisoformat(e.timestamp)).days
    print(f"    viv={e.vividness:.1f} imp={e.importance} age={age_days}d acc={e._access_count} | {e.summary[:55]}...")

print(f"\n  Bottom 5 by vividness:")
for e in bottom5:
    age_days = (datetime.now() - datetime.fromisoformat(e.timestamp)).days
    print(f"    viv={e.vividness:.1f} imp={e.importance} age={age_days}d acc={e._access_count} | {e.summary[:55]}...")

print(f"\n  Top 5 avg importance:    {top5_avg_imp:.1f}")
print(f"  Bottom 5 avg importance: {bottom5_avg_imp:.1f}")

check("Top-vividness entries have higher importance than bottom",
      top5_avg_imp > bottom5_avg_imp,
      f"Top avg: {top5_avg_imp:.1f}, Bottom avg: {bottom5_avg_imp:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 6: Access Patterns -- Are Memories Being Retrieved?")
print("=" * 76)

accessed = [e for e in aria_tm.entries if e._access_count > 0]
never_accessed = [e for e in aria_tm.entries if e._access_count == 0]
max_acc = max(e._access_count for e in aria_tm.entries) if aria_tm.entries else 0

print(f"\n  Memories accessed at least once: {len(accessed)}/{aria_total}")
print(f"  Never accessed: {len(never_accessed)}/{aria_total}")
print(f"  Max access count: {max_acc}")
if accessed:
    print(f"\n  Most-accessed memories:")
    for e in sorted(accessed, key=lambda x: x._access_count, reverse=True)[:3]:
        print(f"    acc={e._access_count} imp={e.importance} | {e.summary[:60]}...")

check("Some memories have been accessed (search/retrieval used)",
      len(accessed) > 0,
      f"{len(accessed)} accessed out of {aria_total}")
check("Accessed memories tend to be higher importance",
      (sum(e.importance for e in accessed) / max(len(accessed), 1)) >=
      (sum(e.importance for e in never_accessed) / max(len(never_accessed), 1))
      if accessed and never_accessed else True)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 7: Learning Progression -- Does Quality Improve Over Time?")
print("=" * 76)

# Split deliberate entries into first half and second half chronologically
deliberate_sorted = sorted(aria_deliberate, key=lambda e: e.timestamp)
half = len(deliberate_sorted) // 2
if half > 0:
    early = deliberate_sorted[:half]
    later = deliberate_sorted[half:]

    early_avg_imp = sum(e.importance for e in early) / len(early)
    later_avg_imp = sum(e.importance for e in later) / len(later)
    early_avg_kw = sum(len(e.keywords) for e in early) / len(early)
    later_avg_kw = sum(len(e.keywords) for e in later) / len(later)
    early_with_ref = sum(1 for e in early if e.reflection) / len(early)
    later_with_ref = sum(1 for e in later if e.reflection) / len(later)

    print(f"\n  Early deliberate entries ({len(early)}):")
    print(f"    Avg importance: {early_avg_imp:.1f}, Avg keywords: {early_avg_kw:.1f}, Has reflection: {early_with_ref:.0%}")
    print(f"  Later deliberate entries ({len(later)}):")
    print(f"    Avg importance: {later_avg_imp:.1f}, Avg keywords: {later_avg_kw:.1f}, Has reflection: {later_with_ref:.0%}")

    check("Both early and later entries exist", len(early) > 0 and len(later) > 0)
    check("Reflections present throughout (consistent learning)",
          early_with_ref >= 0.5 and later_with_ref >= 0.5,
          f"Early: {early_with_ref:.0%}, Later: {later_with_ref:.0%}")
else:
    print("  Not enough deliberate entries to split")
    check("Enough deliberate entries for progression analysis", False, f"Only {len(aria_deliberate)} entries")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 8: Rex Core vs Archival -- Important Skills Stay in Core")
print("=" * 76)

if rex_tm.core:
    print(f"\n  Rex core ({len(rex_tm.core)} entries -- always in context):")
    for e in rex_tm.core:
        print(f"    imp={e.importance} | {e.summary[:65]}...")
        if e.reflection:
            print(f"    -> {e.reflection[:65]}...")

    core_avg = sum(e.importance for e in rex_tm.core) / len(rex_tm.core)
    arch_avg = sum(e.importance for e in rex_tm.archival) / max(len(rex_tm.archival), 1)
    print(f"\n  Core avg importance:    {core_avg:.1f}")
    print(f"  Archival avg importance: {arch_avg:.1f}")

    check("Rex core has high-importance entries (>=7)",
          all(e.importance >= 7 for e in rex_tm.core))
    check("Core avg importance > archival avg importance",
          core_avg > arch_avg,
          f"Core: {core_avg:.1f}, Archival: {arch_avg:.1f}")
else:
    check("Rex core has entries", False, "Core is empty")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 9: Context Block Generation -- Usable by LLM")
print("=" * 76)

aria_ctx = aria_tm.get_context_block()
rex_ctx = rex_tm.get_context_block()

print(f"\n  Aria context block ({len(aria_ctx)} chars):")
for line in aria_ctx.split("\n")[:6]:
    print(f"    {line[:80]}")
if len(aria_ctx.split("\n")) > 6:
    print(f"    ... ({len(aria_ctx.split(chr(10)))} lines total)")

print(f"\n  Rex context block ({len(rex_ctx)} chars):")
for line in rex_ctx.split("\n")[:6]:
    print(f"    {line[:80]}")

check("Aria produces a context block", len(aria_ctx) > 0)
check("Rex produces a context block", len(rex_ctx) > 0)
check("Aria block contains task header", "TASK EXPERIENCES" in aria_ctx)
check("Rex block contains core header", "CORE TASK MEMORY" in rex_ctx)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 10: Tag Parsing -- [TASK_MEMORY] Blocks Parse Correctly")
print("=" * 76)

sample = """I've learned a lot from this simulation.

[TASK_MEMORY]
summary: Ran a society stability sim with 3 factions and observed cascade failure
reflection: Small initial biases compound rapidly when feedback loops are present
keywords: simulation, factions, cascade, feedback
importance: 8
[/TASK_MEMORY]

Let me continue working on this."""

clean, entries = parse_task_memory_tags(sample)

print(f"\n  Sample input: (with embedded [TASK_MEMORY] block)")
print(f"  Cleaned output: '{clean[:60]}...'")
print(f"  Parsed entries: {len(entries)}")
if entries:
    e = entries[0]
    print(f"    summary: {e.get('summary', '')[:60]}")
    print(f"    reflection: {e.get('reflection', '')[:60]}")
    print(f"    keywords: {e.get('keywords', [])}")
    print(f"    importance: {e.get('importance', '?')}")

check("Tag parsed successfully", len(entries) == 1)
check("Summary extracted", "society stability" in entries[0].get("summary", "").lower() if entries else False)
check("Reflection extracted", "feedback loops" in entries[0].get("reflection", "").lower() if entries else False)
check("Keywords extracted", "simulation" in entries[0].get("keywords", []) if entries else False)
check("Tags removed from display text", "[TASK_MEMORY]" not in clean)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 11: Topic Clusters -- What Did They Learn About?")
print("=" * 76)

# Extract topic clusters from keywords
all_keywords = {}
for e in aria_tm.entries:
    for kw in e.keywords:
        kw_lower = kw.lower().strip()
        if len(kw_lower) > 3:
            all_keywords[kw_lower] = all_keywords.get(kw_lower, 0) + 1

top_topics = sorted(all_keywords.items(), key=lambda x: -x[1])[:15]
print(f"\n  Aria's top learning topics (by keyword frequency):")
for kw, count in top_topics:
    print(f"    {kw}: {count} mentions")

check("Keywords form meaningful clusters (>1 occurrence)",
      sum(1 for _, c in all_keywords.items() if c > 1) >= 3,
      f"{sum(1 for _, c in all_keywords.items() if c > 1)} keywords appear multiple times")

# Calculate topic diversity
unique_topics = len(all_keywords)
print(f"\n  Unique keywords: {unique_topics}")
print(f"  Topics appearing 2+ times: {sum(1 for _, c in all_keywords.items() if c > 1)}")

check("Topic diversity is reasonable (>10 unique)",
      unique_topics >= 10,
      f"Found {unique_topics} unique keywords")


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  THE LEARNING VERDICT")
print("=" * 76)

# Determine learning signals
has_reflections = sum(1 for e in aria_deliberate if e.reflection) > 0
has_search_hits = len(aria_sim) > 0
has_access_memory = len(accessed) > 0
has_topic_clusters = sum(1 for _, c in all_keywords.items() if c > 1) >= 3
rex_has_core = len(rex_tm.core) > 0

signals = [has_reflections, has_search_hits, has_access_memory, has_topic_clusters, rex_has_core]
verdict = sum(signals)

print(f"""
  Learning Signals:
    Deliberate reflections written:  {"YES" if has_reflections else "NO"}
    Search retrieval works:          {"YES" if has_search_hits else "NO"}
    Memories being re-accessed:      {"YES" if has_access_memory else "NO"}
    Topic clusters forming:          {"YES" if has_topic_clusters else "NO"}
    Rex core skills identified:      {"YES" if rex_has_core else "NO"}

  Verdict: {verdict}/5 learning signals active
""")

if verdict >= 4:
    print("  CONCLUSION: The AIs ARE learning from tasks.")
    print("  They write reflections, build topic expertise,")
    print("  and retrieve past experiences when relevant.")
elif verdict >= 2:
    print("  CONCLUSION: Partial learning -- some signals present")
    print("  but the system could be more actively used.")
else:
    print("  CONCLUSION: Minimal learning observed.")
    print("  The task memory system needs more engagement.")


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  RESULTS SUMMARY")
print("=" * 76)
print(f"  Total:  {pass_count + fail_count}")
print(f"  Passed: {pass_count}")
print(f"  Failed: {fail_count}")
if failures:
    print(f"\n  Failed:")
    for label, detail in failures:
        print(f"    - {label}")
        if detail:
            print(f"      {detail}")
print()

sys.exit(1 if fail_count else 0)
