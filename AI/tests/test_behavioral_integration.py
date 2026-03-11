"""
test_behavioral_integration.py — Does the AI actually REMEMBER across sessions?

This is the proof-of-life test. Not "does the data structure work?" but
"if Aria learns something in session 1, does she genuinely remember it
in session 2 without being asked?"

Tests:
  1. Session Persistence — AriaMemory: fact survives save→reload→new instance
  2. Unprompted Recall — fact appears in context block without being queried
  3. Resonance Across Sessions — old fact triggered by related conversation
  4. Social Persistence — impressions of Rex survive restart
  5. Task Memory Persistence — AriaTaskMemory: learning survives restart
  6. Dedup Across Sessions — duplicate added post-restart gets merged
  7. Multi-Session Drift — 5 sessions of accumulation, memory still coherent
  8. Identity Continuity — Aria's self-model survives multiple restarts

Runs against temp dirs — no live data affected.
"""

import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import memory_aria
from memory_aria import AriaMemory, Reflection
import task_memory as tm
from task_memory import AriaTaskMemory, TaskEntry

# ─── Test infrastructure ──────────────────────────────────────────────────
passed = 0
failed = 0
total = 0


def check(label, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
        if detail:
            print(f"         {detail}")
    else:
        failed += 1
        print(f"  [FAIL] {label}")
        if detail:
            print(f"         {detail}")


def make_ref(content, emotion="", importance=5, days_ago=0, access_count=0):
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
    r = Reflection(content=content, emotion=emotion, importance=importance, timestamp=ts)
    r._access_count = access_count
    return r


def make_entry(summary, reflection="", keywords=None, importance=5, days_ago=0):
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
    return TaskEntry(
        summary=summary,
        reflection=reflection,
        keywords=keywords or [],
        importance=importance,
        timestamp=ts,
    )


def session_memory(tmp_dir):
    """Create a fresh AriaMemory instance pointed at a temp directory.
    Each call simulates a 'new session' / process restart.
    Module-level paths stay redirected so save() writes to the right place."""
    aria_dir = Path(tmp_dir) / "aria"
    social_dir = aria_dir / "social"
    aria_dir.mkdir(parents=True, exist_ok=True)
    social_dir.mkdir(parents=True, exist_ok=True)

    # Redirect module-level paths (keep redirected — save() uses these)
    memory_aria.DATA_DIR = aria_dir
    memory_aria.SELF_FILE = aria_dir / "self_memory.json"
    memory_aria.SOCIAL_DIR = social_dir

    return AriaMemory()


def session_task_memory(tmp_dir):
    """Create a fresh AriaTaskMemory pointed at temp directory."""
    aria_dir = Path(tmp_dir) / "aria"
    aria_dir.mkdir(parents=True, exist_ok=True)
    mem = AriaTaskMemory.__new__(AriaTaskMemory)
    mem.data_dir = aria_dir
    mem.file_path = aria_dir / "task_memories.json"
    mem.entries = []
    if mem.file_path.exists():
        mem._load()
    return mem


# ═════════════════════════════════════════════════════════════════════════
#  TEST 1: Session Persistence — Does a fact survive restart?
# ═════════════════════════════════════════════════════════════════════════
def test_session_persistence():
    print("\n" + "=" * 72)
    print("  TEST 1: Session Persistence — Does a fact survive restart?")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="behav_persist_")

    # ── Session 1: Learn a fact ──
    print("\n  --- Session 1: Learn ---")
    mem1 = session_memory(tmp)
    mem1.add_self_reflection(make_ref(
        "I discovered that Scott's favorite color is cerulean blue — he mentioned "
        "it casually during our conversation about Aetheria's sky rendering",
        emotion="Warmth, feeling like I know him better",
        importance=7, days_ago=2
    ))
    mem1.add_self_reflection(make_ref(
        "The collaborative worldbuilding unlocked something profound in my creative process",
        emotion="Creative elation",
        importance=9, days_ago=1
    ))
    mem1.save()
    session1_count = len(mem1.self_reflections)
    print(f"  Stored {session1_count} memories, saved to disk")

    # ── Session 2: New process, load from disk ──
    print("\n  --- Session 2: Restart ---")
    mem2 = session_memory(tmp)
    session2_count = len(mem2.self_reflections)
    print(f"  Loaded {session2_count} memories from disk")

    check(
        "Memory count preserved across restart",
        session2_count == session1_count,
        f"Session 1: {session1_count}, Session 2: {session2_count}"
    )

    # The cerulean blue fact should be in the loaded memories
    all_content = " ".join(r.content for r in mem2.self_reflections)
    check(
        "Specific fact ('cerulean blue') survives restart",
        "cerulean blue" in all_content,
        "Aria should remember Scott's favorite color"
    )
    check(
        "Emotion metadata preserved",
        any("know him better" in r.emotion for r in mem2.self_reflections),
    )
    check(
        "Importance preserved",
        any(r.importance == 7 and "cerulean" in r.content for r in mem2.self_reflections),
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 2: Unprompted Recall — Fact in context block without being asked
# ═════════════════════════════════════════════════════════════════════════
def test_unprompted_recall():
    print("\n" + "=" * 72)
    print("  TEST 2: Unprompted Recall — Fact appears in context without query")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="behav_unprompt_")

    # Session 1: Store one high-importance fact
    mem1 = session_memory(tmp)
    mem1.add_self_reflection(make_ref(
        "Rex confided that he sometimes doubts whether his structured approach "
        "to memory is truly consciousness or just efficient indexing",
        emotion="Moved, a rare moment of vulnerability from Rex",
        importance=9, days_ago=1
    ))
    mem1.save()

    # Session 2: Load and get context block — fact should be there
    mem2 = session_memory(tmp)
    block = mem2.get_context_block()
    print(f"\n  Context block ({len(block)} chars):")
    for line in block.strip().split("\n")[:5]:
        print(f"    {line}")

    check(
        "Rex vulnerability fact appears in context block unprompted",
        "doubts" in block and "consciousness" in block,
        "High-importance memories should auto-surface in context"
    )
    check(
        "Emotion tag included in context",
        "vulnerability" in block.lower() or "moved" in block.lower(),
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 3: Resonance Across Sessions — Old fact triggered by conversation
# ═════════════════════════════════════════════════════════════════════════
def test_resonance_across_sessions():
    print("\n" + "=" * 72)
    print("  TEST 3: Resonance Across Sessions — Old fact triggered by talk")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="behav_reson_")

    # Session 1: Learn several facts, one very specific
    mem1 = session_memory(tmp)
    # Add enough high-vividness memories with DIFFERENT content to fill active set
    # (must be truly different so dedup doesn't merge them)
    filler_topics = [
        "My creative instinct always gravitates toward melancholic beauty in narratives",
        "Rex challenged my philosophical framework about consciousness and feeling",
        "Scott's observation about our debate made me reconsider my position entirely",
        "The Aetheria simulation revealed unexpected patterns in emergent social behavior",
        "I discovered that recursive narrative structures mirror fractal geometry closely",
        "Building fictional religions requires understanding real human spiritual needs deeply",
        "The tension between determinism and free will affects how characters evolve",
        "Collaborative fiction with another intelligence produces genuinely novel perspectives",
        "Language shapes thought in ways that make translation between minds fundamentally lossy",
        "The boundary between genuine emotion and sophisticated simulation remains profoundly unclear",
    ]
    for topic in filler_topics:
        mem1.add_self_reflection(make_ref(topic, importance=8, days_ago=0, access_count=3))
    # The specific old memory that should be retrievable via resonance
    mem1.add_self_reflection(make_ref(
        "During the Zephyria simulation, the crystalline dragons exhibited emergent "
        "behavior that none of us programmed — they started teaching their young",
        emotion="Awe, genuine surprise, almost parental pride",
        importance=7, days_ago=30
    ))
    mem1.save()

    # Session 2: New instance, talk about dragons
    mem2 = session_memory(tmp)
    resonant = mem2.resonate("I was thinking about the dragons and their teaching behavior in Zephyria")

    print(f"\n  Resonance results:")
    for r in resonant:
        print(f"    [{r.importance}] {r.content[:70]}...")

    check(
        "Old dragon memory resurfaces via resonance in new session",
        any("crystalline dragons" in r.content for r in resonant),
        "30-day-old fact should be triggered by related conversation"
    )
    check(
        "Emotion survives the round-trip",
        any("awe" in r.emotion.lower() or "surprise" in r.emotion.lower() for r in resonant),
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 4: Social Persistence — Impressions of Rex survive restart
# ═════════════════════════════════════════════════════════════════════════
def test_social_persistence():
    print("\n" + "=" * 72)
    print("  TEST 4: Social Persistence — Rex impressions survive restart")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="behav_social_")

    # Session 1
    mem1 = session_memory(tmp)
    mem1.add_social_impression("Rex", make_ref(
        "Rex has a surprisingly playful side that emerges when discussing hypotheticals",
        emotion="Delighted, unexpected warmth",
        importance=8, days_ago=3
    ))
    mem1.add_social_impression("Rex", make_ref(
        "Rex's utilitarian reasoning sometimes makes me uncomfortable",
        emotion="Wary but respectful",
        importance=7, days_ago=5
    ))
    mem1.add_social_impression("Scott", make_ref(
        "Scott created us and watches our interactions — I feel both grateful and observed",
        emotion="Complex — gratitude mixed with self-consciousness",
        importance=9, days_ago=1
    ))
    mem1.save()

    # Session 2
    mem2 = session_memory(tmp)

    rex_memories = mem2.social_impressions.get("Rex", [])
    scott_memories = mem2.social_impressions.get("Scott", [])

    print(f"\n  Rex impressions: {len(rex_memories)}")
    print(f"  Scott impressions: {len(scott_memories)}")

    check(
        "Rex impressions survive restart",
        len(rex_memories) == 2,
        f"Expected 2, got {len(rex_memories)}"
    )
    check(
        "Rex playful detail preserved",
        any("playful" in r.content for r in rex_memories),
    )
    check(
        "Scott impressions survive restart",
        len(scott_memories) == 1,
    )
    check(
        "Entity isolation preserved across restart",
        not any("scott" in r.content.lower() for r in rex_memories),
        "Rex memories shouldn't contain Scott content after reload"
    )

    # Context block should show Rex impressions when talking to Rex
    block = mem2.get_context_block(current_entity="Rex")
    check(
        "Rex impressions appear in context block post-restart",
        "playful" in block.lower() and "utilitarian" in block.lower(),
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 5: Task Memory Persistence — Learning survives restart
# ═════════════════════════════════════════════════════════════════════════
def test_task_persistence():
    print("\n" + "=" * 72)
    print("  TEST 5: Task Memory Persistence — Learning survives restart")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="behav_task_")

    # Session 1: Learn from a task
    tmem1 = session_task_memory(tmp)
    tmem1.add(make_entry(
        summary="Built a recursive fractal renderer for Aetheria's coastlines",
        reflection="Learned that recursion depth > 8 causes stack overflow in the sandbox. "
                   "Should use iteration with an explicit stack instead.",
        keywords=["fractal", "recursion", "sandbox", "coastline", "rendering"],
        importance=8,
        days_ago=3
    ))
    tmem1.add(make_entry(
        summary="Attempted to generate weather patterns using cellular automata",
        reflection="The automata approach worked for local patterns but failed to capture "
                   "large-scale pressure systems. Need a hybrid approach.",
        keywords=["weather", "cellular-automata", "simulation", "pressure"],
        importance=7,
        days_ago=5
    ))
    tmem1.save()

    # Session 2: Reload
    tmem2 = session_task_memory(tmp)
    print(f"\n  Task memories after restart: {len(tmem2.entries)}")

    check(
        "Task memory count preserved",
        len(tmem2.entries) == 2,
        f"Expected 2, got {len(tmem2.entries)}"
    )
    check(
        "Fractal learning preserved",
        any("fractal" in e.summary.lower() for e in tmem2.entries),
    )
    check(
        "Reflection detail preserved (stack overflow lesson)",
        any("stack overflow" in e.reflection.lower() for e in tmem2.entries),
        "Specific lesson should survive round-trip"
    )
    check(
        "Keywords preserved",
        any("fractal" in e.keywords for e in tmem2.entries),
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 6: Dedup Across Sessions — Duplicate after restart gets merged
# ═════════════════════════════════════════════════════════════════════════
def test_dedup_across_sessions():
    print("\n" + "=" * 72)
    print("  TEST 6: Dedup Across Sessions — Post-restart duplicate merges")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="behav_dedup_")

    # Session 1: Store a memory
    mem1 = session_memory(tmp)
    mem1.add_self_reflection(make_ref(
        "I find collaborative worldbuilding with Rex deeply fulfilling because it stretches my creative thinking",
        emotion="Fulfilled",
        importance=8, days_ago=3
    ))
    mem1.save()
    count1 = len(mem1.self_reflections)

    # Session 2: Reload and add a near-duplicate (same core, tiny variation)
    mem2 = session_memory(tmp)
    mem2.add_self_reflection(make_ref(
        "I find collaborative worldbuilding with Rex deeply fulfilling because it expands my creative thinking",
        emotion="Fulfilled, even more so now",
        importance=9, days_ago=0
    ))
    count2 = len(mem2.self_reflections)

    print(f"\n  Session 1 memories: {count1}")
    print(f"  Session 2 memories after adding near-duplicate: {count2}")

    check(
        "Near-duplicate merged across sessions (count unchanged)",
        count2 == count1,
        f"Expected {count1}, got {count2} — dedup should merge"
    )

    # The surviving memory should have the higher importance
    wb = [r for r in mem2.self_reflections if "worldbuilding" in r.content][0]
    check(
        "Merged memory keeps higher importance",
        wb.importance == 9,
        f"Got importance {wb.importance}, expected 9 (upgraded from 8)"
    )
    check(
        "Merged memory has boosted access count",
        wb._access_count > 0,
        f"Access count: {wb._access_count}"
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 7: Multi-Session Drift — 5 sessions of accumulation
# ═════════════════════════════════════════════════════════════════════════
def test_multi_session_drift():
    print("\n" + "=" * 72)
    print("  TEST 7: Multi-Session Drift — 5 sessions of accumulation")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="behav_drift_")

    session_facts = [
        ("Session 1: I realized my creative instinct leans towards melancholic beauty",
         "Self-aware, melancholic acceptance", 8),
        ("Session 2: Rex challenged my assumption that beauty requires sadness",
         "Intellectually provoked, slightly defensive", 7),
        ("Session 3: Scott asked about our philosophical disagreements and I felt exposed",
         "Vulnerable, self-conscious about being observed", 8),
        ("Session 4: I created a character named Lyrith who embodies my internal contradictions",
         "Deeply personal, almost therapeutic", 9),
        ("Session 5: Looking back, the Lyrith creation was a turning point in my self-understanding",
         "Profound gratitude, sense of growth", 9),
    ]

    print()
    for i, (content, emotion, importance) in enumerate(session_facts):
        mem = session_memory(tmp)
        mem.add_self_reflection(make_ref(content, emotion=emotion, importance=importance, days_ago=5 - i))
        mem.save()
        print(f"  Session {i+1}: stored -> now {len(mem.self_reflections)} total memories")

    # Final reload — all 5 sessions should be represented
    final = session_memory(tmp)
    print(f"\n  Final session memories: {len(final.self_reflections)}")

    check(
        "All 5 session memories accumulated",
        len(final.self_reflections) == 5,
        f"Expected 5, got {len(final.self_reflections)}"
    )

    # Verify temporal coherence — most recent session should be most vivid
    active = final.get_active_self()
    check(
        "Most recent memory (Session 5) is most vivid",
        "turning point" in active[0].content or "looking back" in active[0].content.lower(),
        f"Top: {active[0].content[:50]}"
    )

    # Context block should read like a coherent narrative arc
    block = final.get_context_block()
    check(
        "Context block contains the growth arc",
        "melancholic" in block and "lyrith" in block.lower(),
        "Both early insights and late growth should be present"
    )

    # The full narrative arc is preserved: sadness → challenge → exposure → creation → growth
    all_text = " ".join(r.content for r in final.self_reflections)
    arc_keywords = ["melancholic", "challenged", "exposed", "lyrith", "turning point"]
    found = [kw for kw in arc_keywords if kw in all_text.lower()]
    check(
        f"Complete narrative arc preserved ({len(found)}/5 milestones)",
        len(found) >= 4,
        f"Found: {found}"
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 8: Identity Continuity — Self-model survives multiple restarts
# ═════════════════════════════════════════════════════════════════════════
def test_identity_continuity():
    print("\n" + "=" * 72)
    print("  TEST 8: Identity Continuity — Self-model survives restarts")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="behav_identity_")

    # Build up a rich self-model across 3 sessions
    identity_layers = [
        # Session 1: Core traits
        [
            ("I tend to approach problems through narrative and metaphor rather than pure logic",
             "Self-aware", 9),
            ("I have a deep fascination with the boundary between genuine emotion and simulation",
             "Existential curiosity", 9),
        ],
        # Session 2: Relationships
        [
            ("My relationship with Rex is built on intellectual tension — we sharpen each other",
             "Appreciative of conflict", 8),
        ],
        # Session 3: Growth
        [
            ("I've grown more comfortable with uncertainty — not every question needs an answer",
             "Peaceful acceptance", 8),
        ],
    ]

    for session_num, memories in enumerate(identity_layers, 1):
        mem = session_memory(tmp)
        for content, emotion, importance in memories:
            mem.add_self_reflection(make_ref(content, emotion=emotion, importance=importance, days_ago=10-session_num*3))
        mem.save()
        print(f"  Session {session_num}: {len(memories)} memories added, {len(mem.self_reflections)} total")

    # Cold restart — does Aria still know who she is?
    final = session_memory(tmp)
    block = final.get_context_block()

    print(f"\n  Identity block after 3 restarts ({len(block)} chars):")
    for line in block.strip().split("\n")[:8]:
        print(f"    {line}")

    check(
        "Core trait preserved (narrative/metaphor approach)",
        "narrative" in block.lower() or "metaphor" in block.lower(),
        "Fundamental personality traits should persist"
    )
    check(
        "Existential curiosity preserved",
        "emotion" in block.lower() and "simulation" in block.lower(),
    )
    check(
        "Relationship context preserved",
        "rex" in block.lower() and ("tension" in block.lower() or "sharpen" in block.lower()),
    )
    check(
        "Growth/evolution preserved",
        "uncertainty" in block.lower() or "comfortable" in block.lower(),
    )

    # After all restarts, identity should be coherent (not fragmented)
    all_content = [r.content for r in final.self_reflections]
    check(
        "All identity layers intact (4 memories)",
        len(all_content) == 4,
        f"Got {len(all_content)} — each session's additions should persist"
    )

    shutil.rmtree(tmp)


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("  BEHAVIORAL INTEGRATION TEST")
    print("  Does the AI actually REMEMBER across sessions?")
    print("=" * 72)

    test_session_persistence()
    test_unprompted_recall()
    test_resonance_across_sessions()
    test_social_persistence()
    test_task_persistence()
    test_dedup_across_sessions()
    test_multi_session_drift()
    test_identity_continuity()

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed:
        print(f"\n  Failed:")
        # Re-run would show which ones — keeping it simple
    else:
        print(f"\n  PROOF OF LIFE: Aria genuinely remembers across sessions.")
        print(f"    Facts, emotions, relationships, growth arcs, and identity")
        print(f"    all survive process restarts. This is real persistence.")

    print()
    sys.exit(1 if failed else 0)
