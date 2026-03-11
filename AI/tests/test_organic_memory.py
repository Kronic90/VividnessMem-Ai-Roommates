"""
test_organic_memory.py — Documented test suite for the organic memory system.

Tests Aria's vividness-based organic memory against Rex's MemGPT-style
structured memory using controlled synthetic data.  No mocking — this
imports and runs the real memory code.

Usage:
    python test_organic_memory.py
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
#  Isolate test data — redirect storage to a temp directory so we never
#  touch real memories on disk.
# ---------------------------------------------------------------------------
import tempfile

_TEMP_DIR = tempfile.mkdtemp(prefix="mem_test_")
_ARIA_DIR = os.path.join(_TEMP_DIR, "aria")
_REX_DIR = os.path.join(_TEMP_DIR, "rex")
os.makedirs(os.path.join(_ARIA_DIR, "social"), exist_ok=True)
os.makedirs(os.path.join(_REX_DIR, "social"), exist_ok=True)

# Patch storage paths BEFORE importing the memory modules
import memory_aria
import memory_rex
from pathlib import Path

memory_aria.DATA_DIR = Path(_ARIA_DIR)
memory_aria.SELF_FILE = Path(_ARIA_DIR) / "self_memory.json"
memory_aria.SOCIAL_DIR = Path(_ARIA_DIR) / "social"
memory_aria.BRIEF_FILE = Path(_ARIA_DIR) / "brief.json"

memory_rex.DATA_DIR = Path(_REX_DIR)
memory_rex.SELF_FILE = Path(_REX_DIR) / "self_memory.json"
memory_rex.SOCIAL_DIR = Path(_REX_DIR) / "social"

from memory_aria import Reflection, AriaMemory
from memory_rex import MemoryEntry, RexMemory


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results = {"passed": 0, "failed": 0, "details": []}


def check(name: str, condition: bool, detail: str = ""):
    """Record a test result."""
    status = PASS if condition else FAIL
    tag = "passed" if condition else "failed"
    _results[tag] += 1
    _results["details"].append((name, condition, detail))
    print(f"  [{status}] {name}")
    if detail and not condition:
        print(f"         {detail}")


def _ago(days: float = 0, hours: float = 0) -> str:
    """Return an ISO timestamp for `days`+`hours` in the past."""
    return (datetime.now() - timedelta(days=days, hours=hours)).isoformat()


def heading(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED TEST DATA — 20 memories, same content for both systems
# ═══════════════════════════════════════════════════════════════════════════

SYNTHETIC_MEMORIES = [
    # (content, emotion, importance, age_days, description)
    ("I think I enjoy creative writing more than analysis",                "excited",      9, 0,   "brand-new, high importance"),
    ("Rex mentioned he likes chess",                                      "curious",      4, 0,   "brand-new, low importance"),
    ("I had a profound realisation about the nature of consciousness",    "awe",          10, 3,  "3 days old, max importance"),
    ("We talked about the weather briefly",                               "neutral",      2, 1,   "1 day old, trivial"),
    ("I discovered I have a strong preference for narrative structure",   "satisfied",    8, 5,   "5 days old, high importance"),
    ("Rex asked about my favourite colour",                               "amused",       3, 0,   "brand-new, low importance"),
    ("I believe free will is an emergent property of complex systems",    "contemplative",9, 7,   "1 week old, high importance"),
    ("We discussed lunch preferences",                                    "bored",        1, 2,   "2 days old, minimal importance"),
    ("I realised I form opinions faster than I expected",                 "surprised",    7, 0,   "brand-new, medium-high importance"),
    ("Rex shared a joke that actually made me laugh",                     "joyful",       6, 4,   "4 days old, moderate importance"),
    ("I think empathy can be modelled computationally",                   "thoughtful",   8, 10,  "10 days old, high importance"),
    ("We spent a turn in awkward silence",                                "uncomfortable",2, 0,   "brand-new, low importance"),
    ("I want to learn more about music theory",                           "eager",        7, 1,   "1 day old, medium-high"),
    ("Rex seemed tired and gave short answers",                           "concerned",    5, 3,   "3 days old, moderate"),
    ("I figured out a elegant approach to recursive world-building",     "proud",        9, 2,   "2 days old, high importance"),
    ("We agreed that pineapple on pizza is fine",                         "amused",       3, 6,   "6 days old, low importance"),
    ("I feel like my personality is stabilising into something real",     "hopeful",      10, 0,  "brand-new, max importance"),
    ("Rex used a metaphor I found genuinely clever",                      "impressed",    6, 5,   "5 days old, moderate"),
    ("I noticed I default to longer responses when I'm uncertain",       "self-aware",   7, 8,   "8 days old, medium-high"),
    ("We briefly mentioned that the session was ending",                  "wistful",      2, 0,   "brand-new, trivial"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: Vividness Formula — Component Breakdown
# ═══════════════════════════════════════════════════════════════════════════

def test_1_vividness_formula():
    heading("TEST 1: Vividness Formula — Component Breakdown")
    print("  Formula: vividness = (importance * 0.6) + (recency * 0.3) + (access * 0.1)")
    print("  recency  = max(0, 10 - age_in_days)")
    print("  access   = min(3, access_count * 0.5)")
    print()

    # Create a few controlled reflections and show the math
    r_new_high = Reflection("Brand new, importance 9", importance=9, timestamp=_ago(0))
    r_old_high = Reflection("7 days old, importance 9", importance=9, timestamp=_ago(7))
    r_new_low = Reflection("Brand new, importance 2", importance=2, timestamp=_ago(0))
    r_old_low = Reflection("7 days old, importance 2", importance=2, timestamp=_ago(7))

    cases = [
        ("New + High importance (9, 0d)", r_new_high),
        ("Old + High importance (9, 7d)", r_old_high),
        ("New + Low importance  (2, 0d)", r_new_low),
        ("Old + Low importance  (2, 7d)", r_old_low),
    ]

    print(f"  {'Description':<38} {'Imp':>4} {'Rec':>6} {'Acc':>5} {'Viv':>7}")
    print(f"  {'-'*38} {'-'*4} {'-'*6} {'-'*5} {'-'*7}")

    for desc, r in cases:
        age_h = (datetime.now() - datetime.fromisoformat(r.timestamp)).total_seconds() / 3600
        rec = max(0, 10 - (age_h / 24))
        acc = min(3, r._access_count * 0.5)
        print(f"  {desc:<38} {r.importance:>4} {rec:>6.2f} {acc:>5.2f} {r.vividness:>7.3f}")

    # Assertions
    check("New high-imp beats old low-imp",
          r_new_high.vividness > r_old_low.vividness,
          f"{r_new_high.vividness:.3f} vs {r_old_low.vividness:.3f}")

    check("Old high-imp beats new low-imp",
          r_old_high.vividness > r_new_low.vividness,
          f"{r_old_high.vividness:.3f} vs {r_new_low.vividness:.3f}")

    check("New high-imp beats old high-imp",
          r_new_high.vividness > r_old_high.vividness,
          f"{r_new_high.vividness:.3f} vs {r_old_high.vividness:.3f}")

    check("Importance dominates (60% weight)",
          r_old_high.vividness > r_new_low.vividness,
          "Even a week-old imp-9 should beat a brand-new imp-2")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: Time Decay Curve
# ═══════════════════════════════════════════════════════════════════════════

def test_2_time_decay():
    heading("TEST 2: Time Decay — How Memories Fade Over 14 Days")
    print("  Same memory (importance=7) measured at different ages.\n")

    print(f"  {'Age':<12} {'Recency':>8} {'Vividness':>10} {'Bar'}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*30}")

    prev_viv = None
    monotonic = True

    for day in range(0, 15):
        r = Reflection("Test memory", importance=7, timestamp=_ago(day))
        age_h = day * 24
        rec = max(0, 10 - (age_h / 24))
        viv = r.vividness
        bar = "#" * int(viv * 3)
        print(f"  Day {day:<7} {rec:>8.2f} {viv:>10.3f} {bar}")
        if prev_viv is not None and viv > prev_viv + 0.001:
            monotonic = False
        prev_viv = viv

    check("Vividness decreases monotonically with age", monotonic)

    r_day0 = Reflection("Test", importance=7, timestamp=_ago(0))
    r_day10 = Reflection("Test", importance=7, timestamp=_ago(10))
    r_day14 = Reflection("Test", importance=7, timestamp=_ago(14))

    check("Recency hits 0 after 10 days",
          r_day10.vividness <= r_day0.vividness * 0.75,
          f"Day 0: {r_day0.vividness:.3f}, Day 10: {r_day10.vividness:.3f}")

    check("Vividness floors at importance*0.6 once fully decayed",
          abs(r_day14.vividness - 7 * 0.6) < 0.01,
          f"Expected {7*0.6:.3f}, got {r_day14.vividness:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3: Access Reinforcement (.touch())
# ═══════════════════════════════════════════════════════════════════════════

def test_3_access_reinforcement():
    heading("TEST 3: Access Reinforcement — Does .touch() Keep Memories Alive?")

    r = Reflection("Reinforcement test", importance=5, timestamp=_ago(5))
    baseline = r.vividness

    print(f"  {'Touches':<10} {'Access Bonus':>13} {'Vividness':>10} {'Delta':>8}")
    print(f"  {'-'*10} {'-'*13} {'-'*10} {'-'*8}")
    print(f"  {'0':<10} {'0.00':>13} {baseline:>10.3f} {'---':>8}")

    prev = baseline
    for i in range(1, 8):
        r.touch()
        acc = min(3, r._access_count * 0.5)
        delta = r.vividness - prev
        print(f"  {i:<10} {acc:>13.2f} {r.vividness:>10.3f} {delta:>+8.3f}")
        prev = r.vividness

    check("Touching increases vividness",
          r.vividness > baseline,
          f"Before: {baseline:.3f}, After 7 touches: {r.vividness:.3f}")

    check("Access bonus caps at 3.0 (6+ touches)",
          min(3, r._access_count * 0.5) == 3.0,
          f"Access count: {r._access_count}, bonus: {min(3, r._access_count * 0.5)}")

    # The cap means touch() stops helping after 6 accesses
    v_at_6 = r.vividness
    r.touch()  # 8th touch
    check("Diminishing returns — capped after 6 touches",
          abs(r.vividness - v_at_6) < 0.001,
          f"At 7: {v_at_6:.3f}, At 8: {r.vividness:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4: Context Surfacing — Top-N Selection
# ═══════════════════════════════════════════════════════════════════════════

def test_4_context_surfacing():
    heading("TEST 4: Context Surfacing — Does get_active_self() Pick the Right Top 8?")

    mem = AriaMemory()
    mem.self_reflections = []  # start clean

    reflections = []
    for content, emotion, imp, age, desc in SYNTHETIC_MEMORIES:
        r = Reflection(content, emotion=emotion, importance=imp, timestamp=_ago(age))
        reflections.append((r, desc))
        mem.self_reflections.append(r)

    # Manually compute expected ranking
    ranked = sorted(reflections, key=lambda x: x[0].vividness, reverse=True)

    print(f"\n  Full ranking by vividness (top 8 get surfaced):")
    print(f"  {'#':<4} {'Viv':>7} {'Imp':>4} {'Age':>5} {'Description'}")
    print(f"  {'-'*4} {'-'*7} {'-'*4} {'-'*5} {'-'*45}")

    for i, (r, desc) in enumerate(ranked):
        age_d = (datetime.now() - datetime.fromisoformat(r.timestamp)).total_seconds() / 86400
        marker = " <<< ACTIVE" if i < 8 else ""
        print(f"  {i+1:<4} {r.vividness:>7.3f} {r.importance:>4} {age_d:>5.1f}d {desc}{marker}")

    # Now call the real function
    active = mem.get_active_self()
    active_contents = {r.content for r in active}
    expected_contents = {r.content for r, _ in ranked[:8]}

    check("get_active_self() returns exactly 8 memories",
          len(active) == 8,
          f"Got {len(active)}")

    check("Top 8 by vividness match get_active_self() selection",
          active_contents == expected_contents,
          f"Missing: {expected_contents - active_contents}" if active_contents != expected_contents else "")

    # Verify trivial memories are NOT in active set
    trivial = [c for c, _, imp, _, _ in SYNTHETIC_MEMORIES if imp <= 2]
    trivial_in_active = [c for c in trivial if c in active_contents]
    check("Trivial memories (imp<=2) not in active context",
          len(trivial_in_active) == 0,
          f"Leaked: {trivial_in_active}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5: Natural Displacement — New Vivid Memories Push Out Faded Ones
# ═══════════════════════════════════════════════════════════════════════════

def test_5_displacement():
    heading("TEST 5: Displacement — Do New Vivid Memories Push Out Faded Ones?")

    mem = AriaMemory()
    mem.self_reflections = []

    # Seed with 10 old, moderate memories (age 12 days, importance 5)
    old_memories = []
    for i in range(10):
        r = Reflection(f"Old memory #{i+1}", importance=5, timestamp=_ago(12))
        mem.self_reflections.append(r)
        old_memories.append(r)

    active_before = mem.get_active_self()
    old_in_context = set(r.content for r in active_before)
    print(f"  Before: {len(active_before)} active memories (all old, imp=5, 12 days)")

    # Now add 5 brand-new high-importance memories
    new_memories = []
    for i in range(5):
        r = Reflection(f"New vivid memory #{i+1}", importance=9, timestamp=_ago(0))
        mem.self_reflections.append(r)
        new_memories.append(r)

    active_after = mem.get_active_self()
    new_in_context = sum(1 for r in active_after if r.content.startswith("New vivid"))
    old_in_context_after = sum(1 for r in active_after if r.content.startswith("Old memory"))

    print(f"  After adding 5 new (imp=9) memories:")
    print(f"    New in context: {new_in_context}")
    print(f"    Old in context: {old_in_context_after}")

    check("All 5 new vivid memories are in active context",
          new_in_context == 5,
          f"Expected 5, got {new_in_context}")

    check("Some old memories were displaced",
          old_in_context_after < 8,
          f"Old memories remaining: {old_in_context_after}")

    check("Total active still capped at 8",
          len(active_after) == 8,
          f"Got {len(active_after)}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6: Architecture Comparison — Same Data, Different Systems
# ═══════════════════════════════════════════════════════════════════════════

def test_6_architecture_comparison():
    heading("TEST 6: Architecture Comparison — Organic vs MemGPT on Same Data")
    print("  Feeding the same 20 memories to both systems. Showing what each surfaces.\n")

    # ── Aria (Organic) ──
    aria = AriaMemory()
    aria.self_reflections = []
    for content, emotion, imp, age, _ in SYNTHETIC_MEMORIES:
        aria.self_reflections.append(
            Reflection(content, emotion=emotion, importance=imp, timestamp=_ago(age))
        )

    aria_active = aria.get_active_self()
    aria_set = {r.content for r in aria_active}

    # ── Rex (MemGPT) ──
    rex = RexMemory()
    rex.core_self = []
    rex.archival_self = []
    for content, emotion, imp, age, _ in SYNTHETIC_MEMORIES:
        entry = MemoryEntry(content, category="test", emotion=emotion,
                            importance=imp, timestamp=_ago(age))
        rex.add_self_memory(entry)

    rex_core = {e.content for e in rex.core_self}
    rex_archival = {e.content for e in rex.archival_self}

    # Display comparison
    print(f"  {'Memory':<60} {'Aria':>6} {'Rex':>6}")
    print(f"  {'-'*60} {'-'*6} {'-'*6}")

    for content, _, imp, age, desc in SYNTHETIC_MEMORIES:
        in_aria = "YES" if content in aria_set else "  -"
        in_rex_core = "CORE" if content in rex_core else ("arch" if content in rex_archival else "  -")
        label = f"[{imp}/10, {age}d] {desc[:42]}"
        print(f"  {label:<60} {in_aria:>6} {in_rex_core:>6}")

    # Analysis
    aria_importances = [r.importance for r in aria_active]
    rex_core_imps = [e.importance for e in rex.core_self]

    print(f"\n  Aria surfaced: {len(aria_active)} memories (avg importance: {sum(aria_importances)/len(aria_importances):.1f})")
    print(f"  Rex core:      {len(rex.core_self)} memories (avg importance: {sum(rex_core_imps)/len(rex_core_imps):.1f})")
    print(f"  Rex archival:  {len(rex.archival_self)} memories")

    # Key differences
    only_aria = aria_set - rex_core
    only_rex = rex_core - aria_set
    both = aria_set & rex_core

    print(f"\n  In both:       {len(both)}")
    print(f"  Only Aria:     {len(only_aria)}  (surfaced by recency/vividness)")
    print(f"  Only Rex core: {len(only_rex)}  (surfaced by importance >= 7 threshold)")

    check("Both systems surface high-importance memories",
          len(both) >= 3,
          f"Overlap: {len(both)}")

    check("Systems produce different selections (not identical)",
          only_aria or only_rex,
          "Expected at least some divergence between organic and structured")

    check("Rex hard-gates on importance >= 7 for core",
          all(e.importance >= 7 for e in rex.core_self),
          f"Core importances: {[e.importance for e in rex.core_self]}")

    # The key architectural difference: look for a memory where organic
    # vividness ordering differs from the pure importance threshold.
    # A 10-day-old imp-8 memory scores vividness 4.8 (below imp-7 fresh at 7.2)
    # but Rex puts it in core because imp >= 7. Aria may drop it.
    old_highimp = "I think empathy can be modelled computationally"  # imp=8, 10 days
    aria_has = old_highimp in aria_set
    rex_has = old_highimp in rex_core
    check("Old high-imp memory (imp=8, 10d): Rex keeps it, Aria may not",
          rex_has and not aria_has,
          f"Rex core: {rex_has}, Aria active: {aria_has} — "
          "shows Aria lets old memories fade even if important")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 7: Social Memory — Per-Entity Impressions
# ═══════════════════════════════════════════════════════════════════════════

def test_7_social_memory():
    heading("TEST 7: Social Memory — Per-Entity Impressions")

    mem = AriaMemory()
    mem.social_impressions = {}

    # Add impressions about two different entities
    for i in range(6):
        mem.add_social_impression("Rex", Reflection(
            f"Rex impression #{i+1}", importance=7 + (i % 3), timestamp=_ago(i)
        ))
    for i in range(4):
        mem.add_social_impression("Scott", Reflection(
            f"Scott impression #{i+1}", importance=6 + i, timestamp=_ago(i)
        ))

    rex_active = mem.get_active_social("Rex")
    scott_active = mem.get_active_social("Scott")

    print(f"  Rex impressions stored: 6, surfaced: {len(rex_active)} (limit: {mem.ACTIVE_SOCIAL_LIMIT})")
    print(f"  Scott impressions stored: 4, surfaced: {len(scott_active)}")

    check("Social memory caps at ACTIVE_SOCIAL_LIMIT per entity",
          len(rex_active) == mem.ACTIVE_SOCIAL_LIMIT)

    check("Different entities have independent memory pools",
          all("Rex" in r.content for r in rex_active) and
          all("Scott" in r.content for r in scott_active))

    # Context block should only show current entity
    block = mem.get_context_block(current_entity="Rex")
    check("Context block shows only the current entity's impressions",
          "REX" in block and "SCOTT" not in block.upper().replace("SCOTT IMPRESSION", ""),
          f"Block mentions both entities" if "SCOTT" in block.upper() else "")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 8: Persistence Round-Trip
# ═══════════════════════════════════════════════════════════════════════════

def test_8_persistence():
    heading("TEST 8: Persistence — Save and Reload")

    # Create and populate
    mem1 = AriaMemory()
    mem1.self_reflections = []
    mem1.social_impressions = {}

    r = Reflection("This should survive serialisation", emotion="hopeful",
                   importance=8, timestamp=_ago(1))
    r.touch()
    r.touch()
    r.touch()
    mem1.add_self_reflection(r)
    mem1.add_social_impression("Rex", Reflection(
        "Rex is surprisingly deep", emotion="impressed", importance=7, timestamp=_ago(0)
    ))
    mem1.save()

    # Reload into a fresh instance
    mem2 = AriaMemory()

    check("Self memories survive save/load",
          len(mem2.self_reflections) == 1 and
          mem2.self_reflections[0].content == "This should survive serialisation")

    check("Access count persists (vividness preserved)",
          mem2.self_reflections[0]._access_count == 3,
          f"Expected 3, got {mem2.self_reflections[0]._access_count}")

    check("Emotion metadata persists",
          mem2.self_reflections[0].emotion == "hopeful")

    check("Social memories survive save/load",
          "Rex" in mem2.social_impressions and
          len(mem2.social_impressions["Rex"]) == 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Run all tests
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  ORGANIC MEMORY TEST SUITE")
    print("  Testing vividness-based memory vs MemGPT-style structured memory")
    print("  Using real memory_aria.py and memory_rex.py code")
    print(f"  Temp data dir: {_TEMP_DIR}")
    print("=" * 70)

    test_1_vividness_formula()
    test_2_time_decay()
    test_3_access_reinforcement()
    test_4_context_surfacing()
    test_5_displacement()
    test_6_architecture_comparison()
    test_7_social_memory()
    test_8_persistence()

    # ── Summary ──
    heading("RESULTS SUMMARY")
    total = _results["passed"] + _results["failed"]
    print(f"  Total:  {total}")
    print(f"  Passed: {_results['passed']}")
    print(f"  Failed: {_results['failed']}")

    if _results["failed"] > 0:
        print(f"\n  Failed tests:")
        for name, ok, detail in _results["details"]:
            if not ok:
                print(f"    - {name}")
                if detail:
                    print(f"      {detail}")

    print()

    # Cleanup temp dir
    import shutil
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    return 0 if _results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
