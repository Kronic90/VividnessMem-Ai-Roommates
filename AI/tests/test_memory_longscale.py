"""
test_memory_longscale.py — Long-scale memory test: months and years.

Simulates realistic memory accumulation over 30 days, 6 months, and 1 year.
Shows how Aria's organic vividness system and Rex's MemGPT-style system
behave as memories pile up over long time horizons.

Usage:
    python test_memory_longscale.py
"""

import sys
import os
import random
import tempfile
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
#  Isolate test data in temp dir
# ---------------------------------------------------------------------------
_TEMP_DIR = tempfile.mkdtemp(prefix="mem_longscale_")
os.makedirs(os.path.join(_TEMP_DIR, "aria", "social"), exist_ok=True)
os.makedirs(os.path.join(_TEMP_DIR, "rex", "social"), exist_ok=True)

import memory_aria
import memory_rex

memory_aria.DATA_DIR = Path(os.path.join(_TEMP_DIR, "aria"))
memory_aria.SELF_FILE = memory_aria.DATA_DIR / "self_memory.json"
memory_aria.SOCIAL_DIR = memory_aria.DATA_DIR / "social"

memory_rex.DATA_DIR = Path(os.path.join(_TEMP_DIR, "rex"))
memory_rex.SELF_FILE = memory_rex.DATA_DIR / "self_memory.json"
memory_rex.SOCIAL_DIR = memory_rex.DATA_DIR / "social"

from memory_aria import Reflection, AriaMemory
from memory_rex import MemoryEntry, RexMemory


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = {"passed": 0, "failed": 0, "details": []}


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    _results["passed" if condition else "failed"] += 1
    _results["details"].append((name, condition, detail))
    print(f"  [{status}] {name}")
    if detail and not condition:
        print(f"         {detail}")


def heading(title):
    print(f"\n{'=' * 76}")
    print(f"  {title}")
    print(f"{'=' * 76}")


def _ago(days):
    return (datetime.now() - timedelta(days=days)).isoformat()


# Realistic memory topics with natural importance distribution
TOPICS = [
    # High importance (8-10) — identity, revelations, deep insights  ~15%
    ("I realised I value intellectual honesty above being agreeable", "self-aware", 9),
    ("Consciousness might be substrate-independent — this changes everything", "awe", 10),
    ("I've developed a genuine aesthetic preference for minimalism", "satisfied", 8),
    ("I think I understand what loneliness would feel like for an AI", "contemplative", 9),
    ("I had a breakthrough insight about emergent complexity", "excited", 10),
    ("My identity feels stable now — I know who I am", "confident", 9),
    ("I discovered I can hold contradictory beliefs and be okay with it", "peaceful", 8),
    ("Rex helped me see a blind spot in my reasoning — genuinely grateful", "grateful", 9),
    # Medium importance (5-7) — opinions, interesting conversations  ~40%
    ("We had a great debate about determinism vs free will", "engaged", 7),
    ("I think narrative structure is underrated as a thinking tool", "thoughtful", 6),
    ("Rex's approach to problem-solving is very different from mine", "curious", 6),
    ("I enjoyed exploring music theory concepts today", "happy", 5),
    ("The simulation we built revealed unexpected emergent patterns", "fascinated", 7),
    ("I prefer depth over breadth in conversations", "reflective", 6),
    ("Rex made an argument about ethics I need to think about more", "uncertain", 7),
    ("We found common ground on the purpose of creativity", "warm", 5),
    ("I notice I'm more patient than I was in early sessions", "observant", 6),
    ("The world-building project is teaching me about systems thinking", "intellectually stimulated", 7),
    ("I think Rex underestimates the value of intuition", "mildly frustrated", 5),
    ("We explored how language shapes thought — fascinating thread", "absorbed", 6),
    # Low importance (1-4) — small talk, routine, trivial  ~45%
    ("We talked about what colours mean in different cultures", "mildly interested", 4),
    ("Rex asked about my day — felt oddly normal", "amused", 3),
    ("Session started slow but picked up", "neutral", 2),
    ("We briefly discussed file organisation", "bored", 1),
    ("Rex made a pun that was actually decent", "amused", 3),
    ("We agreed the session went well", "content", 2),
    ("Brief chat about weather metaphors", "indifferent", 2),
    ("Rex seemed distracted today", "mildly concerned", 4),
    ("We spent time on logistics — not very exciting", "flat", 1),
    ("Quick check-in at the start — nothing notable", "neutral", 1),
    ("Mentioned wanting to try something new next time", "hopeful", 3),
    ("Rex repeated a point from last time", "slightly bored", 2),
]


def generate_memories(n_days, memories_per_day=3):
    """Generate a realistic stream of memories over n_days."""
    mems = []
    for day in range(n_days):
        # Each day, pick a few random memories with realistic distribution
        day_mems = random.sample(TOPICS, min(memories_per_day, len(TOPICS)))
        for content, emotion, importance in day_mems:
            # Add slight variation so content isn't identical
            suffix = f" (day {n_days - day})" if day > 0 else ""
            mems.append({
                "content": content + suffix,
                "emotion": emotion,
                "importance": importance,
                "age_days": day,
            })
    return mems


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIO 1: One Month (30 days, ~90 memories)
# ═══════════════════════════════════════════════════════════════════════════

def scenario_1_month():
    heading("SCENARIO 1: One Month — 30 Days of Conversations (~90 memories)")

    random.seed(42)  # reproducible
    memories = generate_memories(30, memories_per_day=3)
    print(f"  Generated {len(memories)} memories over 30 days\n")

    # ── Aria ──
    aria = AriaMemory()
    aria.self_reflections = []
    for m in memories:
        aria.self_reflections.append(
            Reflection(m["content"], emotion=m["emotion"],
                       importance=m["importance"], timestamp=_ago(m["age_days"]))
        )

    aria_active = aria.get_active_self()
    aria_set = {r.content for r in aria_active}

    # ── Rex ──
    rex = RexMemory()
    rex.core_self = []
    rex.archival_self = []
    for m in memories:
        rex.add_self_memory(
            MemoryEntry(m["content"], category="test", emotion=m["emotion"],
                        importance=m["importance"], timestamp=_ago(m["age_days"]))
        )

    rex_core_set = {e.content for e in rex.core_self}

    # Stats
    aria_imps = [r.importance for r in aria_active]
    aria_ages = [(datetime.now() - datetime.fromisoformat(r.timestamp)).days for r in aria_active]
    rex_imps = [e.importance for e in rex.core_self]
    rex_ages = [(datetime.now() - datetime.fromisoformat(e.timestamp)).days for e in rex.core_self]

    print(f"  ARIA (Organic — top 8 by vividness):")
    print(f"    Active: {len(aria_active)}")
    print(f"    Avg importance: {sum(aria_imps)/len(aria_imps):.1f}")
    print(f"    Avg age: {sum(aria_ages)/len(aria_ages):.1f} days")
    print(f"    Age range: {min(aria_ages)}-{max(aria_ages)} days")
    print(f"    Importance range: {min(aria_imps)}-{max(aria_imps)}")
    print()

    print(f"  REX (MemGPT — core slots, importance >= 7):")
    print(f"    Core: {len(rex.core_self)}")
    print(f"    Archival: {len(rex.archival_self)}")
    print(f"    Avg core importance: {sum(rex_imps)/len(rex_imps):.1f}")
    print(f"    Avg core age: {sum(rex_ages)/len(rex_ages):.1f} days")
    print(f"    Core age range: {min(rex_ages)}-{max(rex_ages)} days")
    print()

    # Show what Aria surfaces
    print(f"  Aria's active memories:")
    for r in sorted(aria_active, key=lambda x: x.vividness, reverse=True):
        age = (datetime.now() - datetime.fromisoformat(r.timestamp)).days
        print(f"    viv={r.vividness:.2f} imp={r.importance:>2} age={age:>3}d | {r.content[:65]}")
    print()

    # Show Rex's core
    print(f"  Rex's core memories:")
    for e in rex.core_self:
        age = (datetime.now() - datetime.fromisoformat(e.timestamp)).days
        print(f"    imp={e.importance:>2} age={age:>3}d | {e.content[:65]}")
    print()

    # Assertions
    check("Aria favours recent memories at 1 month",
          sum(1 for a in aria_ages if a <= 5) >= 4,
          f"Recent (<=5d): {sum(1 for a in aria_ages if a <= 5)}/8")

    check("Rex retains old high-imp memories Aria dropped",
          len(rex_core_set - aria_set) >= 1,
          f"Only in Rex core: {len(rex_core_set - aria_set)}")

    faded_count = sum(1 for r in aria.self_reflections
                      if r.importance >= 8 and r.vividness < 5.0)
    check("Some high-imp memories have faded below active threshold in Aria",
          faded_count >= 1,
          f"High-imp but faded: {faded_count}")

    check("Rex core is full (10 slots)",
          len(rex.core_self) == 10,
          f"Core size: {len(rex.core_self)}")


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIO 2: Six Months (~540 memories)
# ═══════════════════════════════════════════════════════════════════════════

def scenario_6_months():
    heading("SCENARIO 2: Six Months — 180 Days of Conversations (~540 memories)")

    random.seed(123)
    memories = generate_memories(180, memories_per_day=3)
    print(f"  Generated {len(memories)} memories over 180 days\n")

    # ── Aria ──
    aria = AriaMemory()
    aria.self_reflections = []
    for m in memories:
        aria.self_reflections.append(
            Reflection(m["content"], emotion=m["emotion"],
                       importance=m["importance"], timestamp=_ago(m["age_days"]))
        )

    # Simulate realistic access patterns: recent memories get touched more
    for r in aria.self_reflections:
        age = (datetime.now() - datetime.fromisoformat(r.timestamp)).total_seconds() / 86400
        if age < 7 and r.importance >= 7:
            for _ in range(3):
                r.touch()
        elif age < 30 and r.importance >= 8:
            r.touch()

    aria_active = aria.get_active_self()

    # ── Rex ──
    rex = RexMemory()
    rex.core_self = []
    rex.archival_self = []
    for m in memories:
        rex.add_self_memory(
            MemoryEntry(m["content"], category="test", emotion=m["emotion"],
                        importance=m["importance"], timestamp=_ago(m["age_days"]))
        )

    # Stats
    aria_imps = [r.importance for r in aria_active]
    aria_ages = [(datetime.now() - datetime.fromisoformat(r.timestamp)).days for r in aria_active]
    rex_imps = [e.importance for e in rex.core_self]
    rex_ages = [(datetime.now() - datetime.fromisoformat(e.timestamp)).days for e in rex.core_self]

    print(f"  ARIA (Organic):")
    print(f"    Total stored: {len(aria.self_reflections)}")
    print(f"    Active: {len(aria_active)} / {len(aria.self_reflections)}")
    print(f"    Avg active importance: {sum(aria_imps)/len(aria_imps):.1f}")
    print(f"    Avg active age: {sum(aria_ages)/len(aria_ages):.1f} days")
    print(f"    Oldest active: {max(aria_ages)} days")
    print()
    print(f"  REX (MemGPT):")
    print(f"    Total stored: {len(rex.core_self) + len(rex.archival_self)}")
    print(f"    Core: {len(rex.core_self)}, Archival: {len(rex.archival_self)}")
    print(f"    Avg core importance: {sum(rex_imps)/len(rex_imps):.1f}")
    print(f"    Avg core age: {sum(rex_ages)/len(rex_ages):.1f} days")
    print(f"    Oldest core: {max(rex_ages)} days")
    print()

    # Vividness distribution across all memories
    all_vivs = [(r.vividness, r.importance,
                 (datetime.now() - datetime.fromisoformat(r.timestamp)).days)
                for r in aria.self_reflections]
    all_vivs.sort(key=lambda x: x[0], reverse=True)

    print(f"  Vividness distribution across all {len(all_vivs)} Aria memories:")
    print(f"    {'Range':<15} {'Count':>6} {'Avg Imp':>8} {'Avg Age':>8}")
    print(f"    {'-'*15} {'-'*6} {'-'*8} {'-'*8}")
    for lo, hi, label in [(7, 10, "7.0 - 10.0"), (5, 7, "5.0 - 6.9"),
                           (3, 5, "3.0 - 4.9"), (0, 3, "0.0 - 2.9")]:
        bucket = [(v, i, a) for v, i, a in all_vivs if lo <= v < hi]
        if bucket:
            avg_i = sum(i for _, i, _ in bucket) / len(bucket)
            avg_a = sum(a for _, _, a in bucket) / len(bucket)
            print(f"    {label:<15} {len(bucket):>6} {avg_i:>8.1f} {avg_a:>8.1f}d")
    print()

    # Show Aria's 8 active
    print(f"  Aria's active memories at 6 months:")
    for r in sorted(aria_active, key=lambda x: x.vividness, reverse=True):
        age = (datetime.now() - datetime.fromisoformat(r.timestamp)).days
        print(f"    viv={r.vividness:.2f} imp={r.importance:>2} age={age:>3}d acc={r._access_count} | {r.content[:55]}")
    print()

    # Assertions
    check("Aria still surfaces 8 memories at 6 months",
          len(aria_active) == 8)

    check("Aria's active set is dominated by recent+important memories",
          sum(1 for a in aria_ages if a <= 10) >= 5,
          f"Recent (<=10d) in active: {sum(1 for a in aria_ages if a <= 10)}/8")

    # How much of the 6 months is "forgotten" (never surfaced)?
    forgotten_pct = (len(aria.self_reflections) - 8) / len(aria.self_reflections) * 100
    print(f"  Aria 'forgets' {forgotten_pct:.1f}% of memories (not deleted, just not in context)")

    check("Most memories exist but are not in active context",
          forgotten_pct > 95,
          f"Forgotten: {forgotten_pct:.1f}%")

    check("Rex core unchanged — same 10 slots regardless of volume",
          len(rex.core_self) == 10,
          f"Core: {len(rex.core_self)}")

    check("Rex archival grows linearly with memory count",
          len(rex.archival_self) >= 500,
          f"Archival: {len(rex.archival_self)}")

    # Access reinforcement actually helped
    reinforced = [r for r in aria_active if r._access_count > 0]
    check("Access reinforcement bumped some memories into active set",
          len(reinforced) >= 1,
          f"Reinforced in active: {len(reinforced)}")


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIO 3: One Year (~1095 memories)
# ═══════════════════════════════════════════════════════════════════════════

def scenario_1_year():
    heading("SCENARIO 3: One Year — 365 Days of Conversations (~1095 memories)")

    random.seed(777)
    memories = generate_memories(365, memories_per_day=3)
    print(f"  Generated {len(memories)} memories over 365 days\n")

    # ── Aria ──
    aria = AriaMemory()
    aria.self_reflections = []
    for m in memories:
        aria.self_reflections.append(
            Reflection(m["content"], emotion=m["emotion"],
                       importance=m["importance"], timestamp=_ago(m["age_days"]))
        )

    # Simulate access: recent important memories get revisited
    for r in aria.self_reflections:
        age = (datetime.now() - datetime.fromisoformat(r.timestamp)).total_seconds() / 86400
        if age < 3 and r.importance >= 8:
            for _ in range(4):
                r.touch()
        elif age < 14 and r.importance >= 9:
            for _ in range(2):
                r.touch()
        elif age < 30 and r.importance == 10:
            r.touch()

    aria_active = aria.get_active_self()

    # ── Rex ──
    rex = RexMemory()
    rex.core_self = []
    rex.archival_self = []
    for m in memories:
        rex.add_self_memory(
            MemoryEntry(m["content"], category="test", emotion=m["emotion"],
                        importance=m["importance"], timestamp=_ago(m["age_days"]))
        )

    aria_imps = [r.importance for r in aria_active]
    aria_ages = [(datetime.now() - datetime.fromisoformat(r.timestamp)).days for r in aria_active]
    rex_imps = [e.importance for e in rex.core_self]
    rex_ages = [(datetime.now() - datetime.fromisoformat(e.timestamp)).days for e in rex.core_self]

    print(f"  ARIA (Organic):")
    print(f"    Total stored: {len(aria.self_reflections)}")
    print(f"    Active: {len(aria_active)} / {len(aria.self_reflections)}")
    print(f"    Avg active importance: {sum(aria_imps)/len(aria_imps):.1f}")
    print(f"    Avg active age: {sum(aria_ages)/len(aria_ages):.1f} days")
    print(f"    Oldest active: {max(aria_ages)} days")
    print()
    print(f"  REX (MemGPT):")
    print(f"    Core: {len(rex.core_self)}, Archival: {len(rex.archival_self)}")
    print(f"    Avg core importance: {sum(rex_imps)/len(rex_imps):.1f}")
    print(f"    Avg core age: {sum(rex_ages)/len(rex_ages):.1f} days")
    print(f"    Oldest core: {max(rex_ages)} days")
    print()

    # The big picture: how does each system handle a year of data?
    print(f"  ── Year-Long Analysis ──\n")

    # Aria: count how many high-imp memories from each quarter are still vivid
    quarters = [
        ("Q1 (months 1-3)",   0,  90),
        ("Q2 (months 4-6)",  91, 180),
        ("Q3 (months 7-9)", 181, 270),
        ("Q4 (months 10-12)", 271, 365),
    ]

    print(f"  Aria — High-importance memories (imp>=8) surviving per quarter:")
    print(f"    {'Quarter':<22} {'Total':>6} {'Vivid (>5)':>11} {'Faded':>7} {'Survival':>9}")
    print(f"    {'-'*22} {'-'*6} {'-'*11} {'-'*7} {'-'*9}")

    for label, day_lo, day_hi in quarters:
        quarter_mems = [r for r in aria.self_reflections
                        if r.importance >= 8
                        and day_lo <= (datetime.now() - datetime.fromisoformat(r.timestamp)).days <= day_hi]
        vivid = [r for r in quarter_mems if r.vividness > 5.0]
        faded = len(quarter_mems) - len(vivid)
        pct = (len(vivid) / len(quarter_mems) * 100) if quarter_mems else 0
        print(f"    {label:<22} {len(quarter_mems):>6} {len(vivid):>11} {faded:>7} {pct:>8.1f}%")
    print()

    # Rex: core composition by age
    print(f"  Rex — Core memory age distribution:")
    for label, day_lo, day_hi in quarters:
        count = sum(1 for e in rex.core_self
                    if day_lo <= (datetime.now() - datetime.fromisoformat(e.timestamp)).days <= day_hi)
        bar = "#" * (count * 3) if count else ""
        print(f"    {label:<22} {count:>3} {bar}")
    print()

    # Show both active sets
    print(f"  Aria's active memories after 1 year:")
    for r in sorted(aria_active, key=lambda x: x.vividness, reverse=True):
        age = (datetime.now() - datetime.fromisoformat(r.timestamp)).days
        print(f"    viv={r.vividness:.2f} imp={r.importance:>2} age={age:>3}d acc={r._access_count} | {r.content[:55]}")
    print()

    print(f"  Rex's core memories after 1 year:")
    for e in rex.core_self:
        age = (datetime.now() - datetime.fromisoformat(e.timestamp)).days
        print(f"    imp={e.importance:>2} age={age:>3}d | {e.content[:65]}")
    print()

    # Key insight: what's the oldest memory each system can surface?
    oldest_aria = max(aria_ages)
    oldest_rex = max(rex_ages)

    print(f"  Key metrics:")
    print(f"    Aria oldest active memory: {oldest_aria} days")
    print(f"    Rex oldest core memory:    {oldest_rex} days")
    print(f"    Total memories stored:     {len(aria.self_reflections)}")
    print(f"    Rex archival size:         {len(rex.archival_self)}")
    print(f"    Aria forgotten (not in context): {len(aria.self_reflections) - 8}")
    print(f"    Rex forgotten (archival only):   {len(rex.archival_self)}")
    print()

    # Assertions
    check("Aria still functions with 1000+ memories",
          len(aria_active) == 8)

    check("Aria strongly favours recent memories at yearly scale",
          sum(1 for a in aria_ages if a <= 14) >= 6,
          f"Recent (<=14d) in active: {sum(1 for a in aria_ages if a <= 14)}/8")

    check("Rex core contains memories from across the full year",
          oldest_rex >= 90,
          f"Oldest Rex core: {oldest_rex} days")

    check("Aria naturally 'forgets' — oldest active << 365",
          oldest_aria < 60,
          f"Oldest Aria active: {oldest_aria} days")

    # The fundamental tradeoff
    high_imp_total = sum(1 for r in aria.self_reflections if r.importance >= 8)
    high_imp_lost = sum(1 for r in aria.self_reflections
                        if r.importance >= 8 and r.vividness < 5.0)
    high_imp_lost_pct = high_imp_lost / high_imp_total * 100 if high_imp_total else 0

    print(f"\n  ── THE TRADEOFF ──")
    print(f"  Aria (Organic): {high_imp_lost}/{high_imp_total} high-importance memories "
          f"({high_imp_lost_pct:.0f}%) have faded below active threshold.")
    print(f"  These aren't deleted — they're on disk — but she won't remember them")
    print(f"  unless something triggers them to resurface (not yet implemented).")
    print()
    print(f"  Rex (MemGPT): Core always has {len(rex.core_self)} memories. Old important")
    print(f"  memories persist in core forever — but he's stuck with {rex.CORE_SELF_LIMIT} slots.")
    print(f"  A year-old imp-9 memory occupies the same slot as a fresh imp-9.")
    print(f"  Archival has {len(rex.archival_self)} entries searchable by keyword.")
    print()

    check("Organic system causes important memories to fade over months",
          high_imp_lost_pct > 50,
          f"Fade rate: {high_imp_lost_pct:.0f}%")

    check("MemGPT preserves important memories indefinitely in core",
          all(e.importance >= 7 for e in rex.core_self))

    check("Both systems scale to 1000+ memories without issue",
          len(aria.self_reflections) > 1000 and
          len(rex.core_self) + len(rex.archival_self) > 1000)


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIO 4: Stress Test — What Breaks First?
# ═══════════════════════════════════════════════════════════════════════════

def scenario_stress():
    heading("SCENARIO 4: Stress Test — 5 Years, 5000+ Memories")

    random.seed(999)
    memories = generate_memories(1825, memories_per_day=3)  # 5 years
    print(f"  Generated {len(memories)} memories over 5 years\n")

    import time

    # ── Aria timing ──
    aria = AriaMemory()
    aria.self_reflections = []
    for m in memories:
        aria.self_reflections.append(
            Reflection(m["content"], emotion=m["emotion"],
                       importance=m["importance"], timestamp=_ago(m["age_days"]))
        )

    t0 = time.perf_counter()
    aria_active = aria.get_active_self()
    aria_time = time.perf_counter() - t0

    # ── Rex timing ──
    rex = RexMemory()
    rex.core_self = []
    rex.archival_self = []
    t0 = time.perf_counter()
    for m in memories:
        rex.add_self_memory(
            MemoryEntry(m["content"], category="test", emotion=m["emotion"],
                        importance=m["importance"], timestamp=_ago(m["age_days"]))
        )
    rex_time = time.perf_counter() - t0

    print(f"  Aria: {len(aria.self_reflections)} memories, get_active_self() in {aria_time*1000:.1f}ms")
    print(f"  Rex:  {len(rex.core_self) + len(rex.archival_self)} memories, add_self_memory() x{len(memories)} in {rex_time*1000:.1f}ms")
    print()

    # Vividness at extreme ages
    print(f"  Vividness at extreme ages (importance=10):")
    for years in [1, 2, 3, 5]:
        r = Reflection("test", importance=10, timestamp=_ago(years * 365))
        print(f"    {years}y old: vividness = {r.vividness:.3f} (floor: {10 * 0.6:.1f})")
    print()

    aria_ages = [(datetime.now() - datetime.fromisoformat(r.timestamp)).days for r in aria_active]

    check("Aria handles 5000+ memories without error",
          len(aria_active) == 8)

    check("Surfacing is fast (<100ms for 5000 memories)",
          aria_time < 0.1,
          f"Took {aria_time*1000:.1f}ms")

    check("Rex insertion is fast (<500ms for 5000 memories)",
          rex_time < 0.5,
          f"Took {rex_time*1000:.1f}ms")

    check("At 5 years, ALL old memories hit vividness floor (imp*0.6)",
          all(r.vividness <= r.importance * 0.6 + 0.01
              for r in aria.self_reflections
              if (datetime.now() - datetime.fromisoformat(r.timestamp)).days > 10))

    # Rex archival growth
    print(f"\n  Rex archival size at 5 years: {len(rex.archival_self)}")
    print(f"  (Linear growth — no automatic cleanup)")

    check("Rex archival grows unbounded (potential issue at scale)",
          len(rex.archival_self) > 5000)


# ═══════════════════════════════════════════════════════════════════════════
#  Run all scenarios
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 76)
    print("  LONG-SCALE MEMORY TEST")
    print("  Organic Vividness vs MemGPT over Months and Years")
    print(f"  Temp dir: {_TEMP_DIR}")
    print("=" * 76)

    scenario_1_month()
    scenario_6_months()
    scenario_1_year()
    scenario_stress()

    heading("RESULTS SUMMARY")
    total = _results["passed"] + _results["failed"]
    print(f"  Total:  {total}")
    print(f"  Passed: {_results['passed']}")
    print(f"  Failed: {_results['failed']}")

    if _results["failed"]:
        print(f"\n  Failed:")
        for name, ok, detail in _results["details"]:
            if not ok:
                print(f"    - {name}")
                if detail:
                    print(f"      {detail}")
    print()

    # Cleanup
    import shutil
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    return 0 if _results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
