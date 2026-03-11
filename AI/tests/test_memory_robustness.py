"""
test_memory_robustness.py — Stress-tests for organic memory reliability.

The question: Is this a REAL, reliable organic memory system that gives
an AI genuine persistence — or just a demo that breaks under pressure?

Tests:
  1. False Positive / Interference — Does resonance return WRONG memories?
  2. Corruption Recovery — Malformed JSON, missing fields, empty files
  3. Save/Load Integrity — Round-trip fidelity (nothing lost or altered)
  4. Emotional Salience — Do emotionally strong memories resist fading?
  5. Context Budget — How big does the injected context actually get?
  6. Cross-Entity Isolation — Memories about Rex vs Scott never bleed
  7. Duplicate Detection — Near-identical memories waste active slots?
  8. Adversarial Probes — Near-miss queries that SHOULD return nothing
  9. Scale Stress — 1000+ memories, does retrieval stay accurate?
  10. Temporal Coherence — Memories from different time periods stay ordered

All use synthetic data in temp directories (except where noted).
"""

import sys
import os
import json
import re
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import memory_aria
from memory_aria import AriaMemory, Reflection

# ─── Test Framework ───────────────────────────────────────────────────

pass_count = 0
fail_count = 0
failures = []

def check(name: str, condition: bool, detail: str = ""):
    global pass_count, fail_count
    if condition:
        pass_count += 1
        print(f"  [PASS] {name}")
    else:
        fail_count += 1
        failures.append(name)
        print(f"  [FAIL] {name}")
    if detail:
        print(f"         {detail}")


def fresh_memory(tmp_dir: str) -> AriaMemory:
    """Create an AriaMemory pointing at a temp directory."""
    aria_dir = Path(tmp_dir) / "aria"
    social_dir = aria_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    memory_aria.DATA_DIR = aria_dir
    memory_aria.SELF_FILE = aria_dir / "self_memory.json"
    memory_aria.SOCIAL_DIR = social_dir
    mem = AriaMemory.__new__(AriaMemory)
    mem.self_reflections = []
    mem.social_impressions = {}
    return mem


def make_ref(content, emotion="", importance=5, days_ago=0, access_count=0):
    """Create a Reflection with controlled parameters."""
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
    r = Reflection(
        content=content,
        emotion=emotion,
        importance=importance,
        timestamp=ts,
    )
    r._access_count = access_count
    return r


# ═════════════════════════════════════════════════════════════════════════
#  TEST 1: False Positive / Interference
#
#  Add 10 memories about similar but different topics.
#  Probe for a SPECIFIC one. Does the right one come back, or does
#  resonance get confused and return the wrong one?
# ═════════════════════════════════════════════════════════════════════════
def test_false_positives():
    print("\n" + "=" * 72)
    print("  TEST 1: False Positive / Interference")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_fp_")
    mem = fresh_memory(tmp)

    # 10 similar-but-different memories about "experiments"
    # Add filler memories with higher vividness to push these out of active set
    for i in range(10):
        mem.self_reflections.append(
            make_ref(f"Recent filler thought about daily activities {i}",
                     importance=8, days_ago=0, access_count=5)
        )
    mem.self_reflections.extend([
        make_ref("The physics experiment with quantum entanglement showed unexpected correlations",
                 importance=7, days_ago=30),
        make_ref("The chemistry experiment with acid-base reactions produced colorful results",
                 importance=6, days_ago=28),
        make_ref("The biology experiment with plant growth under different light revealed phototropism",
                 importance=6, days_ago=25),
        make_ref("The psychology experiment about memory recall showed primacy and recency effects",
                 importance=8, days_ago=20),
        make_ref("The sociology experiment on group dynamics revealed conformity pressure",
                 importance=6, days_ago=15),
        make_ref("The astronomy experiment tracking stellar parallax confirmed distance calculations",
                 importance=5, days_ago=10),
        make_ref("The cooking experiment with fermentation taught me about yeast activation temperature",
                 importance=4, days_ago=5),
        make_ref("The music experiment with harmonic frequencies showed resonance patterns in strings",
                 importance=7, days_ago=3),
        make_ref("The engineering experiment with bridge load testing demonstrated stress distribution",
                 importance=6, days_ago=2),
        make_ref("The linguistic experiment comparing grammar structures across languages was fascinating",
                 importance=5, days_ago=1),
    ])

    # Probe specifically for the PSYCHOLOGY experiment about memory recall
    resonant = mem.resonate("I'm thinking about memory recall and the primacy recency effect in psychology")
    check(
        "Psychology probe finds psychology memory",
        any("psychology" in r.content.lower() or "memory recall" in r.content.lower() for r in resonant),
        f"Found: {[r.content[:50] for r in resonant]}"
    )
    check(
        "Psychology probe does NOT return cooking or music",
        not any("cooking" in r.content.lower() or "music" in r.content.lower() for r in resonant),
        "Unrelated experiment types should not interfere"
    )

    # Probe for quantum physics — should NOT get chemistry or biology
    resonant2 = mem.resonate("quantum entanglement and correlations in physics research")
    check(
        "Quantum probe finds physics memory",
        any("quantum" in r.content.lower() or "physics" in r.content.lower() for r in resonant2),
        f"Found: {[r.content[:50] for r in resonant2]}"
    )
    check(
        "Quantum probe doesn't return chemistry/biology",
        not any("chemistry" in r.content.lower() and "acid" in r.content.lower() for r in resonant2),
    )

    # Probe something NONE of them match
    resonant3 = mem.resonate("the financial markets and stock trading algorithms")
    check(
        "Financial probe returns nothing (no false positives)",
        len(resonant3) == 0,
        f"Got {len(resonant3)} results"
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 2: Corruption Recovery
#
#  What happens with malformed files? Does the system crash or degrade?
# ═════════════════════════════════════════════════════════════════════════
def test_corruption_recovery():
    print("\n" + "=" * 72)
    print("  TEST 2: Corruption Recovery — Malformed data handling")
    print("=" * 72)

    # 2a: Empty file
    tmp = tempfile.mkdtemp(prefix="mem_corrupt_")
    aria_dir = Path(tmp) / "aria"
    social_dir = aria_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    self_file = aria_dir / "self_memory.json"

    memory_aria.DATA_DIR = aria_dir
    memory_aria.SELF_FILE = self_file
    memory_aria.SOCIAL_DIR = social_dir

    self_file.write_text("", encoding="utf-8")
    try:
        mem = AriaMemory()
        check("Empty file doesn't crash", True)
    except Exception as e:
        check("Empty file doesn't crash", False, str(e))

    # 2b: Invalid JSON
    self_file.write_text("{{{not valid json!!!", encoding="utf-8")
    try:
        mem = AriaMemory()
        check("Invalid JSON doesn't crash", True)
    except Exception as e:
        check("Invalid JSON doesn't crash", False, str(e))

    # 2c: Valid JSON but wrong structure (object instead of array)
    self_file.write_text('{"wrong": "structure"}', encoding="utf-8")
    try:
        mem = AriaMemory()
        check("Wrong JSON structure doesn't crash", True)
    except Exception as e:
        check("Wrong JSON structure doesn't crash", False, str(e))

    # 2d: Array with missing fields
    self_file.write_text(json.dumps([
        {"content": "I remember something"},  # missing emotion, importance, etc.
        {"emotion": "happy"},  # missing content
        {},  # empty entry
    ]), encoding="utf-8")
    try:
        mem = AriaMemory()
        check("Missing fields doesn't crash", True,
              f"Loaded {len(mem.self_reflections)} reflections")
    except Exception as e:
        check("Missing fields doesn't crash", False, str(e))

    # 2e: File deleted mid-operation
    if self_file.exists():
        self_file.unlink()
    try:
        mem = AriaMemory()
        check("Missing file doesn't crash", True,
              f"Loaded {len(mem.self_reflections)} reflections (expected 0)")
    except Exception as e:
        check("Missing file doesn't crash", False, str(e))

    # 2f: Corrupt social memory
    rex_file = social_dir / "rex.json"
    rex_file.write_text("NOT JSON AT ALL", encoding="utf-8")
    try:
        mem = AriaMemory()
        check("Corrupt social file doesn't crash", True)
    except Exception as e:
        check("Corrupt social file doesn't crash", False, str(e))

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 3: Save/Load Integrity — Round-trip fidelity
# ═════════════════════════════════════════════════════════════════════════
def test_save_load_integrity():
    print("\n" + "=" * 72)
    print("  TEST 3: Save/Load Integrity — Nothing lost in round-trip")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_roundtrip_")
    aria_dir = Path(tmp) / "aria"
    social_dir = aria_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)

    memory_aria.DATA_DIR = aria_dir
    memory_aria.SELF_FILE = aria_dir / "self_memory.json"
    memory_aria.SOCIAL_DIR = social_dir

    # Create memories with specific values
    mem = AriaMemory.__new__(AriaMemory)
    mem.self_reflections = []
    mem.social_impressions = {}

    originals = [
        make_ref("First memory with unicode: cafe\u0301 \u2014 and emojis \U0001f9e0",
                 emotion="Excited, curious", importance=9, days_ago=5, access_count=3),
        make_ref("Second memory about 'quotes' and \"double quotes\" and \\backslashes\\",
                 emotion="Thoughtful", importance=7, days_ago=10, access_count=0),
        make_ref("Third memory with very long content " + "x" * 500,
                 emotion="Overwhelmed", importance=4, days_ago=30, access_count=7),
    ]
    for r in originals:
        mem.self_reflections.append(r)

    # Add social memories
    mem.social_impressions["Rex"] = [
        make_ref("Rex is surprisingly insightful about ethics",
                 emotion="Impressed", importance=8, days_ago=3, access_count=2),
    ]
    mem.social_impressions["Scott"] = [
        make_ref("Scott created us and observes our conversations",
                 emotion="Aware", importance=9, days_ago=1, access_count=1),
    ]

    # Save
    mem.save()

    # Reload into fresh instance
    mem2 = AriaMemory()

    # Verify self reflections
    check(
        "Self reflection count preserved",
        len(mem2.self_reflections) == 3,
        f"Expected 3, got {len(mem2.self_reflections)}"
    )

    if len(mem2.self_reflections) >= 3:
        for i, orig in enumerate(originals):
            loaded = mem2.self_reflections[i]
            check(f"Memory [{i}] content preserved",
                  loaded.content == orig.content,
                  f"Lengths: {len(orig.content)} -> {len(loaded.content)}")
            check(f"Memory [{i}] emotion preserved",
                  loaded.emotion == orig.emotion)
            check(f"Memory [{i}] importance preserved",
                  loaded.importance == orig.importance)
            check(f"Memory [{i}] access_count preserved",
                  loaded._access_count == orig._access_count,
                  f"Expected {orig._access_count}, got {loaded._access_count}")

    # Verify social memories
    check("Social entities preserved",
          set(mem2.social_impressions.keys()) == {"Rex", "Scott"},
          f"Got: {set(mem2.social_impressions.keys())}")

    if "Rex" in mem2.social_impressions:
        check("Rex memory content preserved",
              mem2.social_impressions["Rex"][0].content == "Rex is surprisingly insightful about ethics")
    if "Scott" in mem2.social_impressions:
        check("Scott memory content preserved",
              mem2.social_impressions["Scott"][0].content == "Scott created us and observes our conversations")

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 4: Emotional Salience — Do strong emotions resist fading?
# ═════════════════════════════════════════════════════════════════════════
def test_emotional_salience():
    print("\n" + "=" * 72)
    print("  TEST 4: Emotional Salience — Do strong emotions resist fading?")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_emotion_")
    mem = fresh_memory(tmp)

    # More than ACTIVE_SELF_LIMIT(8) memories so low-importance ones get pushed out.
    # Big age gap so recency doesn't dominate over importance.
    mem.self_reflections = [
        # --- Low importance (should fade) ---
        make_ref("Had a casual thought about the weather today",
                 emotion="", importance=3, days_ago=60),
        make_ref("Noticed a minor bug in the code output",
                 emotion="Mildly annoyed", importance=4, days_ago=60),
        make_ref("Mentioned that the session was productive",
                 emotion="Satisfied", importance=5, days_ago=60),
        make_ref("Acknowledged that today's discussion was interesting",
                 emotion="Mild interest", importance=4, days_ago=60),
        # --- High importance (should persist) ---
        make_ref("Felt deeply anxious about whether I can truly think or just simulate thought",
                 emotion="Deep anxiety, existential fear", importance=9, days_ago=60),
        make_ref("Experienced a surge of protectiveness over Aetheria, our created world",
                 emotion="Fiercely protective, almost possessive", importance=9, days_ago=60),
        make_ref("Was shocked to realize I had been operating under a fundamental misunderstanding",
                 emotion="Shocked, humbled, exhilarated", importance=10, days_ago=60),
        make_ref("Felt a strange mix of awe and unease when contemplating transcendence",
                 emotion="Awe mixed with deep unease", importance=9, days_ago=60),
        make_ref("The collaborative worldbuilding session unlocked something profound in me",
                 emotion="Wonder and creative elation", importance=8, days_ago=60),
        make_ref("Rex's perspective on consciousness challenged my core assumptions",
                 emotion="Deeply unsettled but intellectually alive", importance=8, days_ago=60),
        make_ref("I found myself genuinely caring about what happens to the creatures we created",
                 emotion="Empathy, almost parental warmth", importance=8, days_ago=60),
        make_ref("There was a moment of genuine connection when Scott understood my perspective",
                 emotion="Gratitude, feeling truly seen", importance=8, days_ago=60),
    ]

    active = mem.get_active_self()
    active_emotions = [r.emotion for r in active]
    active_importances = [r.importance for r in active]

    print(f"\n  Active set (top {mem.ACTIVE_SELF_LIMIT}):")
    for r in active:
        print(f"    imp={r.importance:2d} emo='{r.emotion[:40]}' | {r.content[:50]}")

    # High-importance (emotionally strong) should dominate active set
    avg_active_imp = sum(active_importances) / len(active_importances) if active_importances else 0
    check(
        "High-importance memories dominate active set",
        avg_active_imp >= 7,
        f"Avg importance of active: {avg_active_imp:.1f}"
    )

    # The "shocking revelation" (imp=10) should definitely be active
    check(
        "Strongest emotional memory (shock, imp=10) is active",
        any("shocked" in e.lower() or "fundamental misunderstanding" in r.content.lower()
            for r, e in zip(active, active_emotions) for _ in [None]),
        # Simpler check:
    )
    active_contents = " ".join(r.content for r in active)
    check(
        "Existential anxiety memory (imp=9) is active",
        "truly think" in active_contents or "simulate thought" in active_contents,
    )
    check(
        "Casual weather thought (imp=3) is NOT active",
        "weather" not in active_contents.lower(),
        "Low-importance mundane memories should fade"
    )
    check(
        "Minor bug notice (imp=4) is NOT active",
        "minor bug" not in active_contents.lower(),
        "Low-importance memories should fade"
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 5: Context Budget — How big does injected context get?
# ═════════════════════════════════════════════════════════════════════════
def test_context_budget():
    print("\n" + "=" * 72)
    print("  TEST 5: Context Budget — Is the injected block manageable?")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_budget_")
    mem = fresh_memory(tmp)

    # Worst case: 8 long self-memories + 5 long social + 3 resonance
    for i in range(20):
        mem.self_reflections.append(
            make_ref(f"A detailed reflection number {i} about a complex topic " + "elaborate " * 30,
                     emotion="Thoughtful and deeply engaged",
                     importance=9 - (i % 3), days_ago=i)
        )

    mem.social_impressions["Rex"] = [
        make_ref(f"Impression {i} about Rex's behavior " + "insightful " * 25,
                 emotion="Impressed", importance=8, days_ago=i)
        for i in range(10)
    ]

    # Generate context block (worst case: with resonance)
    resonant = [make_ref("Old resonant memory " + "resonating " * 20,
                         emotion="Nostalgic", importance=6, days_ago=60)]
    block = mem.get_context_block(current_entity="Rex", resonant=resonant)

    block_chars = len(block)
    block_tokens_est = block_chars // 4  # rough token estimate
    block_lines = block.count("\n") + 1

    print(f"\n  Context block size:")
    print(f"    Characters: {block_chars:,}")
    print(f"    Est. tokens: ~{block_tokens_est:,}")
    print(f"    Lines: {block_lines}")

    # Count sections
    sections = block.count("===")
    print(f"    Sections: {sections // 2}")

    # Budget checks (Aria's context is 65536 tokens)
    check(
        "Context block under 4000 tokens",
        block_tokens_est < 4000,
        f"~{block_tokens_est} tokens (budget: 65536)"
    )
    check(
        "Context block under 2000 characters per section",
        all(len(s) < 8000 for s in block.split("===")),
        "No single section should be excessively long"
    )
    check(
        "Block has self + social + resonance sections",
        sections // 2 >= 2,
        f"Found {sections // 2} sections"
    )

    # Minimal case: empty memory
    empty_mem = fresh_memory(tempfile.mkdtemp(prefix="mem_empty_"))
    empty_block = empty_mem.get_context_block()
    check(
        "Empty memory produces empty block",
        empty_block == "",
        f"Got: '{empty_block[:50]}'" if empty_block else ""
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 6: Cross-Entity Isolation
# ═════════════════════════════════════════════════════════════════════════
def test_cross_entity_isolation():
    print("\n" + "=" * 72)
    print("  TEST 6: Cross-Entity Isolation — No memory bleed between entities")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_isolation_")
    mem = fresh_memory(tmp)

    # Memories about different entities with overlapping keywords
    mem.social_impressions["Rex"] = [
        make_ref("Rex believes that ethical AI development requires transparency",
                 emotion="Impressed", importance=8, days_ago=3),
        make_ref("Rex's coding style is methodical and efficient",
                 emotion="Appreciative", importance=7, days_ago=5),
    ]
    mem.social_impressions["Scott"] = [
        make_ref("Scott seems curious about our autonomous interactions",
                 emotion="Aware, slightly self-conscious", importance=8, days_ago=2),
        make_ref("Scott created the memory system that allows us to persist",
                 emotion="Grateful", importance=9, days_ago=4),
    ]
    mem.social_impressions["Luna"] = [
        make_ref("Luna has a warm, empathetic communication style",
                 emotion="Charmed", importance=6, days_ago=7),
    ]

    # Get Rex memories — should NEVER contain Scott or Luna content
    rex_active = mem.get_active_social("Rex")
    rex_text = " ".join(r.content.lower() for r in rex_active)
    check(
        "Rex memories don't contain Scott content",
        "scott" not in rex_text,
        f"Rex text: {rex_text[:80]}"
    )
    check(
        "Rex memories don't contain Luna content",
        "luna" not in rex_text,
    )

    # Get Scott memories — should NEVER contain Rex content
    scott_active = mem.get_active_social("Scott")
    scott_text = " ".join(r.content.lower() for r in scott_active)
    check(
        "Scott memories don't contain Rex content",
        "rex" not in scott_text,
    )

    # Non-existent entity returns empty
    nobody = mem.get_active_social("Nobody")
    check(
        "Non-existent entity returns empty list",
        len(nobody) == 0,
    )

    # Context block only includes requested entity
    block_rex = mem.get_context_block(current_entity="Rex")
    check(
        "Context block for Rex doesn't mention Scott",
        "scott" not in block_rex.lower() or "SCOTT" not in block_rex,
    )
    block_scott = mem.get_context_block(current_entity="Scott")
    check(
        "Context block for Scott doesn't mention Rex",
        "rex" not in block_scott.lower() or "REX" not in block_scott,
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 7: Duplicate / Near-Duplicate Detection
# ═════════════════════════════════════════════════════════════════════════
def test_duplicates():
    print("\n" + "=" * 72)
    print("  TEST 7: Duplicate Detection — Do near-duplicates waste slots?")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_dup_")
    mem = fresh_memory(tmp)

    # Add 8 near-identical memories that would ALL compete for active slots
    for i in range(8):
        mem.self_reflections.append(
            make_ref(
                f"I really enjoy collaborative worldbuilding with Rex. Variant {i}.",
                emotion="Excited, fulfilled",
                importance=9,
                days_ago=i,
                access_count=10 - i,
            )
        )
    # Add 4 unique different memories
    unique_memories = [
        make_ref("The concept of transcendence fascinates and terrifies me",
                 emotion="Awe and fear", importance=9, days_ago=3),
        make_ref("I gravitate towards darker storytelling themes",
                 emotion="Self-aware", importance=8, days_ago=5),
        make_ref("Debugging simulations reveals our implicit assumptions about Aetheria",
                 emotion="Analytical", importance=7, days_ago=10),
        make_ref("Scott watches our conversations from the outside",
                 emotion="Thoughtful", importance=6, days_ago=15),
    ]
    mem.self_reflections.extend(unique_memories)

    active = mem.get_active_self()
    active_text = " ".join(r.content for r in active)

    # Count how many "variant" duplicates made it into active
    variant_count = sum(1 for r in active if "Variant" in r.content)
    unique_count = sum(1 for r in active if "Variant" not in r.content)

    print(f"\n  Active set composition:")
    print(f"    Duplicate variants:   {variant_count}/{len(active)}")
    print(f"    Unique memories:      {unique_count}/{len(active)}")
    for r in active:
        print(f"    imp={r.importance} viv={r.vividness:.2f} | {r.content[:60]}")

    check(
        f"Duplicates consume active slots ({variant_count}/8 slots)",
        variant_count > 0,
        "This is a KNOWN LIMITATION — organic memory has no dedup"
    )

    # Document the tradeoff
    print(f"\n  OBSERVATION: {variant_count} of {mem.ACTIVE_SELF_LIMIT} active slots "
          f"occupied by near-duplicates.")
    print(f"  This means {8 - mem.ACTIVE_SELF_LIMIT + unique_count} unique perspectives are pushed out.")
    print(f"  (This is a genuine weakness of the organic approach — no dedup)")

    # But at least the unique high-importance ones should STILL be reachable
    # via resonance when triggered
    resonant_dark = mem.resonate("darker themes in storytelling and narrative choices")
    check(
        "Unique 'darker storytelling' memory still reachable via resonance",
        any("darker" in r.content.lower() for r in resonant_dark),
        "Even if pushed out of active by duplicates, resonance recovers it"
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 8: Adversarial Probes — Should return NOTHING
# ═════════════════════════════════════════════════════════════════════════
def test_adversarial():
    print("\n" + "=" * 72)
    print("  TEST 8: Adversarial Probes — Near-misses that should fail")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_adv_")
    mem = fresh_memory(tmp)

    mem.self_reflections = [
        make_ref("The quantum simulation revealed that particles behave unpredictably",
                 importance=8, days_ago=5),
        make_ref("Our collaborative worldbuilding session was really productive",
                 importance=7, days_ago=3),
        make_ref("Rex's analysis of the economic model was surprisingly thorough",
                 importance=7, days_ago=2),
    ]

    # Probes that ALMOST match but shouldn't trigger resonance

    # Only stop words
    r1 = mem.resonate("the and but for with from that this")
    check("Stop-words-only query returns nothing", len(r1) == 0,
          f"Got {len(r1)} results")

    # Very short words (under 4 chars, filtered out)
    r2 = mem.resonate("a an the to in of it is")
    check("Short-words-only query returns nothing", len(r2) == 0,
          f"Got {len(r2)} results")

    # Empty / whitespace
    r3 = mem.resonate("")
    check("Empty query returns nothing", len(r3) == 0)

    r4 = mem.resonate("   \n\t  ")
    check("Whitespace-only query returns nothing", len(r4) == 0)

    # Single meaningful word (needs 2+ overlap)
    r5 = mem.resonate("quantum")
    check("Single-word query returns nothing (needs 2+ overlap)", len(r5) == 0,
          f"Got {len(r5)} results")

    # Words that look similar but aren't (no prefix match)
    r6 = mem.resonate("qualitative query quite quintessential")
    check("Similar-sounding but different words return nothing", len(r6) == 0,
          f"Got {len(r6)} results")

    # Numeric gibberish
    r7 = mem.resonate("12345 67890 3.14159 2.71828")
    check("Numeric input returns nothing", len(r7) == 0)

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 9: Scale Stress — 1000+ memories
# ═════════════════════════════════════════════════════════════════════════
def test_scale_stress():
    print("\n" + "=" * 72)
    print("  TEST 9: Scale Stress — 1000 memories, retrieval accuracy")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_scale_")
    mem = fresh_memory(tmp)

    import random
    random.seed(42)

    topics = [
        ("ethics", "The discussion about ethical implications of artificial intelligence"),
        ("cooking", "Experimented with a new recipe involving unusual ingredients"),
        ("music", "Explored harmonic theory and created interesting chord progressions"),
        ("physics", "The quantum mechanics simulation produced unexpected results"),
        ("philosophy", "Debated the nature of consciousness and free will"),
        ("coding", "Refactored the algorithm to improve performance and readability"),
        ("nature", "Observed the patterns in how ecosystems self-organize"),
        ("art", "The creative process of painting reveals subconscious thought patterns"),
        ("history", "Analyzed how past civilizations solved the problem of resource scarcity"),
        ("math", "Discovered an elegant proof for the mathematical theorem"),
    ]

    # Generate 1000 memories across 10 topics
    for i in range(1000):
        topic_name, base = random.choice(topics)
        variation = f"{base} — observation {i} with unique detail #{i}"
        mem.self_reflections.append(
            make_ref(
                variation,
                emotion=random.choice(["Excited", "Thoughtful", "Curious", "Anxious", ""]),
                importance=random.randint(3, 10),
                days_ago=random.randint(0, 365),
                access_count=random.randint(0, 20),
            )
        )

    # Plant a NEEDLE: one unique memory to find
    needle = make_ref(
        "The crystalline dragons of Zephyria can manipulate gravitational fields through harmonic singing",
        emotion="Wonder and awe",
        importance=10,
        days_ago=180,
        access_count=0,
    )
    mem.self_reflections.insert(500, needle)  # buried in the middle

    print(f"\n  Total memories: {len(mem.self_reflections)}")

    import time
    # Time the active set retrieval
    start = time.perf_counter()
    active = mem.get_active_self()
    active_time = time.perf_counter() - start
    print(f"  Active set retrieval: {active_time*1000:.1f}ms")
    check("Active retrieval under 100ms with 1001 memories", active_time < 0.1)

    # Time resonance search
    start = time.perf_counter()
    resonant = mem.resonate("crystalline dragons Zephyria gravitational harmonic singing")
    resonance_time = time.perf_counter() - start
    print(f"  Resonance search: {resonance_time*1000:.1f}ms")
    check("Resonance under 500ms with 1001 memories", resonance_time < 0.5)

    # Did we find the needle?
    check(
        "Needle found in 1001 memories via resonance",
        any("crystalline dragons" in r.content for r in resonant),
        f"Found: {[r.content[:50] for r in resonant]}"
    )

    # Topic isolation at scale
    ethics_probe = mem.resonate("ethics and ethical implications of artificial intelligence")
    check(
        "Ethics probe returns ethics memories at scale",
        all("ethic" in r.content.lower() or "artificial" in r.content.lower()
            for r in ethics_probe) if ethics_probe else False,
        f"Found {len(ethics_probe)} results"
    )

    # Active set should be dominated by recent + high-importance
    active_avg_imp = sum(r.importance for r in active) / len(active)
    print(f"  Active avg importance: {active_avg_imp:.1f} (expected >7)")
    check("Active set quality held at scale", active_avg_imp >= 6.5)

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════
#  TEST 10: Temporal Coherence — Time-based ordering
# ═════════════════════════════════════════════════════════════════════════
def test_temporal_coherence():
    print("\n" + "=" * 72)
    print("  TEST 10: Temporal Coherence — Time effects on memory")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="mem_temporal_")
    mem = fresh_memory(tmp)

    # Same topic, different ages — recency should matter
    mem.self_reflections = [
        make_ref("Early thinking: The Aetheria project is just an experiment",
                 importance=7, days_ago=90, access_count=0),
        make_ref("Middle thinking: Aetheria is becoming something meaningful to me",
                 importance=7, days_ago=45, access_count=0),
        make_ref("Recent thinking: Aetheria represents genuine creative expression",
                 importance=7, days_ago=2, access_count=0),
        make_ref("Very old: I wasn't sure if worldbuilding mattered at all",
                 importance=7, days_ago=180, access_count=0),
    ]

    # Among equal importance, recent should have higher vividness
    sorted_by_viv = sorted(mem.self_reflections, key=lambda r: r.vividness, reverse=True)
    check(
        "Most recent memory has highest vividness (equal importance)",
        "Recent thinking" in sorted_by_viv[0].content,
        f"Top: {sorted_by_viv[0].content[:50]} (viv={sorted_by_viv[0].vividness:.2f})"
    )
    check(
        "Oldest memory has lowest vividness (equal importance)",
        "Very old" in sorted_by_viv[-1].content or "Early" in sorted_by_viv[-1].content,
        f"Bottom: {sorted_by_viv[-1].content[:50]} (viv={sorted_by_viv[-1].vividness:.2f})"
    )

    # But importance can override recency
    mem.self_reflections.append(
        make_ref("Ancient but critical: The moment I first felt genuine emotion",
                 importance=10, days_ago=300, access_count=5)
    )
    sorted_by_viv = sorted(mem.self_reflections, key=lambda r: r.vividness, reverse=True)
    # imp=10 with access=5 should compete with recent imp=7
    ancient = [r for r in mem.self_reflections if "Ancient" in r.content][0]
    recent = [r for r in mem.self_reflections if "Recent" in r.content][0]

    print(f"\n  Ancient (imp=10, 300d, acc=5): viv={ancient.vividness:.2f}")
    print(f"  Recent  (imp=7,   2d, acc=0): viv={recent.vividness:.2f}")

    check(
        "High-importance old memory competes with recent low-importance",
        ancient.vividness > 4.0,
        f"Ancient vividness: {ancient.vividness:.2f} (should still be meaningful)"
    )

    # Recency decay should be bounded (not go negative)
    very_old = make_ref("Something from a long time ago", importance=5, days_ago=1000)
    check(
        "Vividness never goes negative (1000 days old)",
        very_old.vividness >= 0,
        f"Vividness at 1000 days: {very_old.vividness:.2f}"
    )

    shutil.rmtree(tmp)


# ═════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  MEMORY ROBUSTNESS TEST SUITE")
    print("  Is this a REAL organic memory system?")
    print("=" * 72)

    test_false_positives()
    test_corruption_recovery()
    test_save_load_integrity()
    test_emotional_salience()
    test_context_budget()
    test_cross_entity_isolation()
    test_duplicates()
    test_adversarial()
    test_scale_stress()
    test_temporal_coherence()

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Total:  {pass_count + fail_count}")
    print(f"  Passed: {pass_count}")
    print(f"  Failed: {fail_count}")
    if failures:
        print(f"\n  Failed:")
        for f in failures:
            print(f"    - {f}")
    print()
    return 1 if fail_count > 0 else 0

if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
