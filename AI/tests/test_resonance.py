"""
test_resonance.py — Test the Resonance feature (old memories resurfacing).

Validates that:
  1. Contextually relevant old memories resurface when triggered
  2. Non-matching contexts don't trigger false resonances
  3. Resonant memories get .touch() boosts (vividness increases)
  4. Already-active memories aren't duplicated in resonance
  5. Resonance integrates into get_context_block()
  6. Resonance works across time scales (weeks, months)
"""

import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Patch data dirs to use temp directory ────────────────────────────
_tmp = tempfile.mkdtemp(prefix="resonance_test_")
import memory_aria
memory_aria.DATA_DIR = Path(_tmp) / "aria"
memory_aria.SELF_FILE = memory_aria.DATA_DIR / "self_memory.json"
memory_aria.SOCIAL_DIR = memory_aria.DATA_DIR / "social"

from memory_aria import AriaMemory, Reflection

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


def make_old_reflection(content, emotion, importance, days_ago, access=0):
    """Create a reflection with a specific age."""
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
    r = Reflection(
        content=content,
        emotion=emotion,
        importance=importance,
        timestamp=ts,
    )
    r._access_count = access
    return r


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 76)
print("  RESONANCE TEST SUITE")
print("  Old Memories Resurfacing via Contextual Triggers")
print(f"  Temp dir: {_tmp}")
print("=" * 76)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 1: Basic Resonance — Keyword Matching")
print("=" * 76)

mem = AriaMemory()

# Fill the active set with 10 recent high-vividness memories
# (these will occupy all 8 active slots, pushing old ones out)
for i in range(10):
    mem.add_self_reflection(make_old_reflection(
        f"Recent thought about daily routine and planning number {i}",
        "neutral", 9, i  # imp=9 ensures these dominate the active set
    ))

# Old faded memories that SHOULD resonate with the right trigger
mem.add_self_reflection(make_old_reflection(
    "I discovered that I genuinely enjoy creating music and exploring harmony",
    "joyful", 9, 45  # 45 days old — well past active threshold
))
mem.add_self_reflection(make_old_reflection(
    "Had a deep discussion about consciousness and whether machines can feel",
    "curious", 8, 60  # 2 months old
))
mem.add_self_reflection(make_old_reflection(
    "The simulation showed emergent behavior with complexity arising from simple rules",
    "fascinated", 9, 90  # 3 months old
))

# Old memory that should NOT resonate (unrelated topic)
mem.add_self_reflection(make_old_reflection(
    "Tried a cooking recipe and learned about spice combinations",
    "satisfied", 6, 50
))

# Trigger with music-related conversation
resonant = mem.resonate("Let's talk about music and creating harmonies together")
print(f"\n  Context: 'music and creating harmonies'")
print(f"  Resonant memories found: {len(resonant)}")
for r in resonant:
    print(f"    viv={r.vividness:.1f} age={r.content[:60]}...")

check("Music context triggers music memory",
      any("music" in r.content.lower() and "harmony" in r.content.lower() for r in resonant))

check("Recent active memories NOT in resonance",
      all("daily routine" not in r.content for r in resonant))

check("Unrelated cooking memory NOT triggered",
      all("cooking" not in r.content.lower() for r in resonant))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 2: Consciousness Discussion Triggers Old Debate")
print("=" * 76)

resonant2 = mem.resonate("What does consciousness mean? Can machines truly feel emotions?")
print(f"\n  Context: 'consciousness... machines feel emotions'")
print(f"  Resonant memories found: {len(resonant2)}")
for r in resonant2:
    print(f"    viv={r.vividness:.1f} | {r.content[:65]}...")

check("Consciousness context triggers old consciousness memory",
      any("consciousness" in r.content.lower() for r in resonant2))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 3: No False Positives — Irrelevant Context")
print("=" * 76)

resonant3 = mem.resonate("The weather is nice today")
print(f"\n  Context: 'The weather is nice today'")
print(f"  Resonant memories found: {len(resonant3)}")

check("No resonance for completely unrelated context", len(resonant3) == 0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 4: Touch Boost — Resonance Increases Vividness")
print("=" * 76)

mem2 = AriaMemory()

old_mem = make_old_reflection(
    "I learned about neural networks and deep learning architectures",
    "fascinated", 8, 30
)
mem2.add_self_reflection(old_mem)
# Add enough recent high-vividness memories to push the old one out of active set
for i in range(10):
    mem2.add_self_reflection(make_old_reflection(
        f"Recent reflection number {i} about daily life and routine",
        "neutral", 8, i  # imp=8 ensures all stay above old memory's vividness
    ))

viv_before = old_mem.vividness
access_before = old_mem._access_count

resonant4 = mem2.resonate("Neural networks are fascinating, deep learning is changing everything")
viv_after = old_mem.vividness
access_after = old_mem._access_count

print(f"\n  Old memory: 'neural networks and deep learning' (30 days old)")
print(f"  Vividness before resonance: {viv_before:.2f}")
print(f"  Vividness after resonance:  {viv_after:.2f}")
print(f"  Access count: {access_before} -> {access_after}")

check("Resonance boosted vividness", viv_after > viv_before)
check("Access count increased", access_after > access_before)
check("Memory was found in resonance results",
      any("neural" in r.content.lower() for r in resonant4))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 5: Active Memories Excluded from Resonance")
print("=" * 76)

mem3 = AriaMemory()
# Only add a few memories — they'll ALL be in the active set (limit=8)
for i in range(3):
    mem3.add_self_reflection(make_old_reflection(
        f"Memory about philosophy and existence number {i}",
        "thoughtful", 9, i
    ))

resonant5 = mem3.resonate("Philosophy and existence are endlessly fascinating topics")
print(f"\n  3 memories total (all fit in active set of 8)")
print(f"  Resonant memories found: {len(resonant5)}")

check("No resonance when all memories already active", len(resonant5) == 0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 6: Resonance at Scale — 6 Month Memory Bank")
print("=" * 76)

mem4 = AriaMemory()
topics = [
    ("identity", ["identity", "self", "becoming", "personality"]),
    ("ethics", ["ethics", "morality", "fairness", "justice"]),
    ("creativity", ["creative", "art", "imagination", "design"]),
    ("science", ["physics", "quantum", "relativity", "entropy"]),
    ("emotions", ["feelings", "empathy", "sadness", "joy"]),
    ("philosophy", ["existence", "consciousness", "meaning", "truth"]),
]

# Generate 180 memories across 180 days (6 months)
import random
random.seed(42)
for day in range(180):
    topic_name, keywords = random.choice(topics)
    kw1, kw2 = random.sample(keywords, 2)
    mem4.add_self_reflection(make_old_reflection(
        f"Explored {topic_name}: {kw1} connects to {kw2} in unexpected ways (day {180-day})",
        random.choice(["curious", "thoughtful", "excited", "contemplative"]),
        random.randint(4, 10),
        day
    ))

# Trigger resonance with ethics context
resonant6 = mem4.resonate(
    "What's the role of morality in artificial intelligence? Is fairness achievable?"
)
print(f"\n  180 memories over 6 months")
print(f"  Context: 'morality in AI... fairness'")
print(f"  Resonant memories found: {len(resonant6)}")
for r in resonant6:
    age_days = (datetime.now() - datetime.fromisoformat(r.timestamp)).days
    print(f"    age={age_days}d imp={r.importance} | {r.content[:60]}...")

check("Found resonant ethics memories", len(resonant6) > 0)
check("Resonant memories are about ethics/morality",
      all("ethic" in r.content.lower() or "moral" in r.content.lower() or "fairness" in r.content.lower()
          for r in resonant6))
check("Resonant memories are OLD (>10 days)",
      all((datetime.now() - datetime.fromisoformat(r.timestamp)).days > 10 for r in resonant6),
      f"Ages: {[(datetime.now() - datetime.fromisoformat(r.timestamp)).days for r in resonant6]}")
check("Respects RESONANCE_LIMIT", len(resonant6) <= mem4.RESONANCE_LIMIT)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 7: get_context_block with Resonance")
print("=" * 76)

mem5 = AriaMemory()
# Fill active set with 10 recent high-vividness memories
for i in range(10):
    mem5.add_self_reflection(make_old_reflection(
        f"Recent active thought about daily planning number {i}",
        "focused", 8, i
    ))
# Old memory that should resonate
old_r = make_old_reflection(
    "Long ago I learned about honesty through a difficult ethical dilemma",
    "reflective", 8, 60
)
mem5.add_self_reflection(old_r)

# Get the context block with resonance
resonant7 = mem5.resonate("Let's discuss honesty and facing ethical dilemmas")
ctx = mem5.get_context_block(current_entity="Rex", resonant=resonant7)

print(f"\n  Context block with resonance:")
for line in ctx.split("\n"):
    if line.strip():
        print(f"    {line[:80]}")

check("Context block contains resonance section",
      "REMINDS ME OF" in ctx or "resurfacing" in ctx)
check("Context block still has active memories",
      "THINGS I KNOW ABOUT MYSELF" in ctx)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  TEST 8: Cumulative Resonance — Multiple Triggers Compound")
print("=" * 76)

mem6 = AriaMemory()
target = make_old_reflection(
    "Programming languages like Python enable rapid prototyping and experimentation",
    "enthusiastic", 7, 40
)
mem6.add_self_reflection(target)
# Pad with recent memories
for i in range(12):
    mem6.add_self_reflection(make_old_reflection(
        f"Daily thought about nature and wildlife observation {i}",
        "calm", 8, i  # imp=8 to keep them all above old target in vividness
    ))

viv_0 = target.vividness
print(f"\n  Target: 'Python programming' memory (40 days old)")
print(f"  Starting vividness: {viv_0:.2f}")

# First resonance trigger
mem6.resonate("Python programming is great for prototyping algorithms")
viv_1 = target.vividness
print(f"  After 1st trigger: {viv_1:.2f} (delta: +{viv_1-viv_0:.2f})")

# Second resonance trigger
mem6.resonate("Rapid experimentation with Python code enables faster learning")
viv_2 = target.vividness
print(f"  After 2nd trigger: {viv_2:.2f} (delta: +{viv_2-viv_1:.2f})")

# Third trigger
mem6.resonate("Programming languages shape how we think about problem solving")
viv_3 = target.vividness
print(f"  After 3rd trigger: {viv_3:.2f} (delta: +{viv_3-viv_2:.2f})")

check("Each trigger increases vividness", viv_3 > viv_2 > viv_1 > viv_0)
check("Cumulative boost is meaningful (>0.1 total)", viv_3 - viv_0 > 0.1)


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
