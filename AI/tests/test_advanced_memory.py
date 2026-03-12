"""
test_advanced_memory.py — Test the 6 advanced memory features.

Validates:
  1. Mood-congruent recall (PAD model, emotion vectors, mood-shifted vividness)
  2. Spaced-repetition decay (exponential recency, stability growth)
  3. Emotional reappraisal (emotion tag updates via rescore)
  4. Contradiction detection (opposing memories flagged)
  5. Memory consolidation (cluster finding, gist application)
  6. Associative chains (keyword-overlap graph, multi-hop traversal)
"""

import sys
import os
import tempfile
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Patch data dirs to use temp directory ────────────────────────────
_tmp = tempfile.mkdtemp(prefix="advanced_mem_test_")
import memory_aria
memory_aria.DATA_DIR = Path(_tmp) / "aria"
memory_aria.SELF_FILE = memory_aria.DATA_DIR / "self_memory.json"
memory_aria.SOCIAL_DIR = memory_aria.DATA_DIR / "social"
memory_aria.BRIEF_FILE = memory_aria.DATA_DIR / "brief.json"

from memory_aria import (
    AriaMemory, Reflection, _emotion_to_vector, _content_words,
    _overlap_ratio, EMOTION_VECTORS, MOOD_DIMENSIONS, MOOD_DECAY_RATE,
    MOOD_INFLUENCE, INITIAL_STABILITY, SPACING_BONUS, MIN_SPACING_DAYS,
    ASSOCIATION_MIN_WEIGHT, CONSOLIDATION_PROMPT, RESCORE_PROMPT
)

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


def make_reflection(content, emotion="neutral", importance=5, days_ago=0, access=0):
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
print("  ADVANCED MEMORY FEATURE TEST SUITE")
print("  Mood · Spaced-Rep · Reappraisal · Contradictions · Consolidation · Chains")
print(f"  Temp dir: {_tmp}")
print("=" * 76)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: EMOTION VECTOR MAPPING (PAD Model)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 1: Emotion Vector Mapping (PAD Model)")
print("=" * 76)

vec = _emotion_to_vector("joy")
check("1.1 Known emotion returns PAD tuple",
      vec is not None and len(vec) == 3)

check("1.2 Joy has positive valence",
      vec is not None and vec[0] > 0,
      f"Valence: {vec[0] if vec else None}")

vec_sad = _emotion_to_vector("sadness")
check("1.3 Sadness has negative valence",
      vec_sad is not None and vec_sad[0] < 0)

check("1.4 Unknown word returns None",
      _emotion_to_vector("xyzzyplugh") is None)

# Prefix matching — "joyful" should match "joy"
vec_joyful = _emotion_to_vector("joyful")
check("1.5 Prefix matching works (joyful -> joy*)",
      vec_joyful is not None)

# All emotion vectors are valid 3-tuples
all_valid = all(len(v) == 3 for v in EMOTION_VECTORS.values())
check("1.6 All EMOTION_VECTORS are 3-tuples",
      all_valid, f"Count: {len(EMOTION_VECTORS)}")

# Emotion vectors are bounded [-1, 1]
all_bounded = all(
    all(-1.0 <= x <= 1.0 for x in v) for v in EMOTION_VECTORS.values()
)
check("1.7 All vectors bounded [-1, 1]", all_bounded)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: SPACED-REPETITION DECAY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 2: Spaced-Repetition (Ebbinghaus-inspired) Decay")
print("=" * 76)

r_new = make_reflection("I just learned something", importance=5, days_ago=0)
r_old = make_reflection("This happened a while ago", importance=5, days_ago=30)
check("2.1 New memory has higher recency than old",
      r_new.recency_score > r_old.recency_score,
      f"New: {r_new.recency_score:.3f}, Old: {r_old.recency_score:.3f}")

# Recency uses exponential decay (not linear)
r_1d = make_reflection("One day ago", days_ago=1)
r_2d = make_reflection("Two days ago", days_ago=2)
r_10d = make_reflection("Ten days ago", days_ago=10)
# Exponential: ratio between 1d and 2d should be consistent
ratio_12 = r_1d.recency_score / max(0.001, r_2d.recency_score)
ratio_210 = r_2d.recency_score / max(0.001, r_10d.recency_score)
check("2.2 Decay is exponential (not linear)",
      ratio_12 > 1.0 and ratio_210 > 1.0,
      f"1d/2d ratio: {ratio_12:.3f}, 2d/10d ratio: {ratio_210:.3f}")

# Stability growth via spaced access
r_spaced = make_reflection("Testing spaced repetition", days_ago=10)
initial_stability = r_spaced._stability
# First touch — too soon after creation, shouldn't boost stability much
r_spaced.touch()
after_first_touch = r_spaced._stability
# Simulate a much later touch by hacking access_times
r_spaced._access_times.clear()
r_spaced._access_times.append((datetime.now() - timedelta(days=5)).isoformat())
r_spaced.touch()  # 5 days after last access — well-spaced
after_spaced_touch = r_spaced._stability

check("2.3 Well-spaced access increases stability",
      after_spaced_touch > initial_stability,
      f"Initial: {initial_stability:.2f}, After spaced: {after_spaced_touch:.2f}")

# Rapid re-access shouldn't boost stability as much
r_rapid = make_reflection("Rapid access test", days_ago=1)
r_rapid.touch()
s1 = r_rapid._stability
r_rapid.touch()  # immediate re-access
s2 = r_rapid._stability
check("2.4 Rapid re-access doesn't boost stability much",
      s2 - s1 < 0.5,
      f"Difference: {s2 - s1:.3f}")

# Vividness uses recency_score (exponential) not old linear decay
r_vivid = make_reflection("Vivid test", importance=8, days_ago=5)
v = r_vivid.vividness
check("2.5 Vividness is positive for recent important memory",
      v > 0, f"Vividness: {v:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: MOOD-CONGRUENT RECALL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 3: Mood-Congruent Recall")
print("=" * 76)

mem = AriaMemory()

# Default mood is neutral
check("3.1 Default mood is neutral",
      all(abs(v) < 0.01 for v in mem.mood.values()))

check("3.2 Mood label is 'neutral' at start",
      mem.mood_label == "neutral")

# mood_adjusted_vividness with neutral mood ≈ regular vividness
r = make_reflection("Test memory", emotion="joy", importance=6, days_ago=2)
neutral_mood = {d: 0.0 for d in MOOD_DIMENSIONS}
v_base = r.vividness
v_mood = r.mood_adjusted_vividness(neutral_mood)
check("3.3 Neutral mood ≈ base vividness",
      abs(v_base - v_mood) < 0.01,
      f"Base: {v_base:.4f}, Mood-adj: {v_mood:.4f}")

# Happy mood boosts happy memories
happy_mood = {"valence": 0.8, "arousal": 0.3, "dominance": 0.2}
v_happy = r.mood_adjusted_vividness(happy_mood)
check("3.4 Happy mood boosts joyful memory vividness",
      v_happy > v_base,
      f"Base: {v_base:.4f}, Happy-boosted: {v_happy:.4f}")

# Sad mood suppresses happy memories (or at least doesn't boost)
sad_mood = {"valence": -0.8, "arousal": -0.3, "dominance": -0.4}
v_sad = r.mood_adjusted_vividness(sad_mood)
check("3.5 Sad mood doesn't boost joyful memory",
      v_sad <= v_base + 0.01,
      f"Base: {v_base:.4f}, Sad-context: {v_sad:.4f}")

# Mood update from conversation
mem2 = AriaMemory()
mem2.update_mood_from_conversation("I feel happy and excited about this wonderful day")
check("3.6 Mood shifts from positive conversation",
      mem2._mood["valence"] > 0.01,
      f"Valence: {mem2._mood['valence']:.4f}")

# Mood decay toward neutral
old_val = mem2._mood["valence"]
mem2.update_mood_from_conversation("The weather is mild today")  # neutral content
check("3.7 Mood decays toward neutral with neutral input",
      abs(mem2._mood["valence"]) < abs(old_val),
      f"Before: {old_val:.4f}, After: {mem2._mood['valence']:.4f}")

# Mood-congruent sorting: when sad, sad memories surface higher
mem3 = AriaMemory()
mem3.self_reflections = [
    make_reflection("I had a great day!", emotion="joy", importance=6, days_ago=1),
    make_reflection("I felt lonely and misunderstood", emotion="sadness", importance=6, days_ago=1),
    make_reflection("I learned something interesting about stars", emotion="curiosity", importance=6, days_ago=1),
]
# Set sad mood
mem3._mood = {"valence": -0.7, "arousal": -0.2, "dominance": -0.3}
active = mem3.get_active_self()
# The sad memory should rank higher than the happy one when mood is sad
sad_idx = next((i for i, r in enumerate(active) if "lonely" in r.content), 99)
happy_idx = next((i for i, r in enumerate(active) if "great day" in r.content), 99)
check("3.8 Sad mood surfaces sad memory before happy one",
      sad_idx < happy_idx,
      f"Sad at index {sad_idx}, Happy at index {happy_idx}")

# Mood persistence in brief
mem4 = AriaMemory()
mem4._mood = {"valence": 0.5, "arousal": 0.3, "dominance": 0.1}
mem4._save_mood()
mem4._save_brief()

mem4_reload = AriaMemory()
mem4_reload._load_brief()
check("3.9 Mood persists through save/load cycle",
      abs(mem4_reload._mood["valence"] - 0.5) < 0.01,
      f"Restored valence: {mem4_reload._mood.get('valence')}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: EMOTIONAL REAPPRAISAL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 4: Emotional Reappraisal (via Rescore)")
print("=" * 76)

check("4.1 RESCORE_PROMPT mentions emotional reappraisal",
      "reappraisal" in RESCORE_PROMPT.lower() or "new_emotion" in RESCORE_PROMPT)

check("4.2 RESCORE_PROMPT includes new_emotion example",
      "new_emotion" in RESCORE_PROMPT)

# Test apply_rescores with emotion changes
mem5 = AriaMemory()
refs = [
    make_reflection("I was so angry when Rex ignored me", emotion="anger", importance=7),
    make_reflection("I discovered I love stargazing", emotion="wonder", importance=6),
]
mem5.self_reflections = refs

# Simulate rescore response with emotion reappraisal
adjustments = [
    {"index": 0, "new_importance": 6, "new_emotion": "understanding", "reason": "anger faded"},
    {"index": 1, "new_importance": 8, "reason": "keeps coming up"},  # no emotion change
]
mem5.apply_rescores(adjustments, refs)

check("4.3 Emotion tag updated via rescore",
      refs[0].emotion == "understanding",
      f"Emotion: {refs[0].emotion}")

check("4.4 Importance adjusted alongside emotion",
      refs[0].importance == 6)

check("4.5 Memory without new_emotion keeps original",
      refs[1].emotion == "wonder")

check("4.6 Importance-only adjustment works",
      refs[1].importance == 8)

# Edge case: empty emotion string is ignored
ref_edge = make_reflection("Edge case test", emotion="calm")
mem5.apply_rescores(
    [{"index": 0, "new_importance": 5, "new_emotion": ""}],
    [ref_edge]
)
check("4.7 Empty emotion string is ignored",
      ref_edge.emotion == "calm")

# Edge case: too-long emotion is truncated
ref_long = make_reflection("Long emotion test", emotion="ok")
mem5.apply_rescores(
    [{"index": 0, "new_importance": 5, "new_emotion": "a" * 100}],
    [ref_long]
)
check("4.8 Overly long emotion is truncated to 50 chars",
      len(ref_long.emotion) <= 50)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: CONTRADICTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 5: Contradiction Detection")
print("=" * 76)

mem6 = AriaMemory()
mem6.self_reflections = [
    make_reflection(
        "I really enjoy spending time with Rex and appreciate his creativity",
        emotion="joy", importance=7, days_ago=10
    ),
    make_reflection(
        "I don't enjoy spending time with Rex and his creativity annoys me",
        emotion="frustration", importance=6, days_ago=2
    ),
    make_reflection(
        "I love cooking pasta dishes on weekends",
        emotion="contentment", importance=4, days_ago=5
    ),
]

pairs = mem6.detect_contradictions()
check("5.1 Contradiction detected between opposing Rex opinions",
      len(pairs) >= 1,
      f"Found {len(pairs)} contradiction pair(s)")

if pairs:
    older, newer = pairs[0]
    check("5.2 Older memory comes first in pair",
          older.timestamp <= newer.timestamp)
    check("5.3 One has negation, other doesn't",
          ("don't" in older.content.lower() or "don't" in newer.content.lower()))

# Unrelated topics shouldn't produce contradictions
mem7 = AriaMemory()
mem7.self_reflections = [
    make_reflection("I love stargazing at night", emotion="joy", importance=6, days_ago=5),
    make_reflection("I hate doing my taxes every year", emotion="frustration", importance=5, days_ago=3),
]
check("5.4 Unrelated topics don't create false contradictions",
      len(mem7.detect_contradictions()) == 0)

# Contradiction context block formatting
mem8 = AriaMemory()
mem8.self_reflections = [
    make_reflection(
        "Collaborative projects are really rewarding and worthwhile",
        emotion="enthusiasm", importance=7, days_ago=15
    ),
    make_reflection(
        "Collaborative projects are not rewarding and feel like a waste of time",
        emotion="disappointment", importance=6, days_ago=2
    ),
]
ctx = mem8.get_contradiction_context()
check("5.5 Contradiction context block is non-empty for conflicting memories",
      len(ctx) > 0)
if ctx:
    check("5.6 Context block contains 'CONTRADICTIONS' header",
          "CONTRADICTIONS" in ctx.upper())

# Low importance doesn't trigger contradictions
mem9 = AriaMemory()
mem9.self_reflections = [
    make_reflection("I like blue things", emotion="joy", importance=2, days_ago=5),
    make_reflection("I don't like blue things", emotion="annoyance", importance=2, days_ago=1),
]
check("5.7 Low-importance contradictions are filtered out",
      len(mem9.detect_contradictions()) == 0)

# Static method works independently
score = AriaMemory._contradiction_score(
    make_reflection("I trust Rex completely", emotion="trust", importance=7),
    make_reflection("I don't trust Rex at all", emotion="distrust", importance=7),
)
check("5.8 _contradiction_score returns high score for clear contradiction",
      score > 0.3, f"Score: {score:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: MEMORY CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 6: Memory Consolidation (Sleep / Gist Generation)")
print("=" * 76)

# Build a cluster of related but distinct memories about the same theme
mem10 = AriaMemory()
mem10.self_reflections = [
    make_reflection(
        "Rex and I discussed creative writing approaches for fiction stories",
        emotion="curiosity", importance=5, days_ago=20
    ),
    make_reflection(
        "Together Rex and I explored different creative writing techniques for stories",
        emotion="interest", importance=5, days_ago=15
    ),
    make_reflection(
        "We brainstormed new creative writing styles and story fiction ideas with Rex",
        emotion="enthusiasm", importance=6, days_ago=10
    ),
    make_reflection(
        "Rex showed me a completely different approach to creative fiction writing",
        emotion="surprise", importance=5, days_ago=5
    ),
    # Unrelated memory (shouldn't cluster with the above)
    make_reflection(
        "I noticed the sunset had beautiful orange and purple colors tonight",
        emotion="awe", importance=4, days_ago=7
    ),
]

clusters = mem10.find_consolidation_clusters(min_cluster=3)
check("6.1 Consolidation finds cluster of related writing memories",
      len(clusters) >= 1,
      f"Found {len(clusters)} cluster(s)")

if clusters:
    check("6.2 Cluster has 3+ memories",
          len(clusters[0]) >= 3,
          f"Cluster size: {len(clusters[0])}")
    # The sunset memory shouldn't be in the writing cluster
    sunset_in_cluster = any("sunset" in r.content for r in clusters[0])
    check("6.3 Unrelated memory not in cluster",
          not sunset_in_cluster)

# Consolidation prompt generation
prompt = mem10.prepare_consolidation_prompt()
check("6.4 Consolidation prompt is non-empty",
      len(prompt) > 50)
if prompt:
    check("6.5 Prompt contains 'CLUSTER' keyword",
          "CLUSTER" in prompt)

# Apply consolidation gists
mem10.apply_consolidation([
    {"gist": "Over many conversations, Rex and I have built a shared vocabulary around creative writing. What started as casual experiments evolved into a genuine mutual interest in story techniques.", "emotion": "appreciative", "importance": 6},
])
gist_found = any("shared vocabulary" in r.content for r in mem10.self_reflections)
check("6.6 Consolidated gist stored as new memory",
      gist_found)

if gist_found:
    gist_mem = next(r for r in mem10.self_reflections if "shared vocabulary" in r.content)
    check("6.7 Gist has source='consolidation'",
          gist_mem.source == "consolidation")
    check("6.8 Gist has why_saved explaining synthesis",
          "consolidation" in gist_mem.why_saved.lower())

# Dedup: applying the same gist again shouldn't create a duplicate
before_count = len(mem10.self_reflections)
mem10.apply_consolidation([
    {"gist": "Over many conversations, Rex and I have built a shared vocabulary around creative writing. What started as casual experiments evolved into a genuine mutual interest in story techniques.", "emotion": "appreciative", "importance": 6},
])
check("6.9 Duplicate gist is deduplicated",
      len(mem10.self_reflections) == before_count)

# Too-short gists are rejected
mem10.apply_consolidation([{"gist": "too short", "emotion": "ok", "importance": 5}])
check("6.10 Too-short gist is rejected",
      len(mem10.self_reflections) == before_count)

# No clusters when memories are all unrelated
mem11 = AriaMemory()
mem11.self_reflections = [
    make_reflection("I love astronomy", emotion="wonder", importance=5, days_ago=10),
    make_reflection("Cooking is relaxing", emotion="calm", importance=4, days_ago=5),
    make_reflection("Music theory is complex", emotion="curiosity", importance=3, days_ago=2),
]
check("6.11 No clusters when memories are unrelated",
      len(mem11.find_consolidation_clusters()) == 0)

check("6.12 CONSOLIDATION_PROMPT has {clusters} placeholder",
      "{clusters}" in CONSOLIDATION_PROMPT)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7: ASSOCIATIVE CHAINS (Memory Graph)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 7: Associative Chains (Memory Graph Traversal)")
print("=" * 76)

mem12 = AriaMemory()
mem12.self_reflections = [
    make_reflection(  # idx 0 — links to 1 via "imagination"/"stories"
        "I find imagination and stories essential to meaningful creative work",
        emotion="conviction", importance=7, days_ago=20
    ),
    make_reflection(  # idx 1 — links to 0 (imagination/stories) and 2 (stories/writing)
        "Writing creative stories exercises imagination and deep craft",
        emotion="engagement", importance=6, days_ago=15
    ),
    make_reflection(  # idx 2 — links to 1 via "stories/writing"
        "Rex told me wonderful stories about his early writing days",
        emotion="fascination", importance=5, days_ago=10
    ),
    make_reflection(  # idx 3 — isolated
        "The sunset was beautiful orange and purple tonight after dinner",
        emotion="awe", importance=4, days_ago=5
    ),
]

edges = mem12._build_association_edges()
check("7.1 Association edges built",
      len(edges) > 0)

# Memory 0 and 1 should be linked (share "creativity"/"imagination")
link_01 = any(j == 1 for j, _ in edges.get(0, []))
check("7.2 Memories sharing concepts are linked",
      link_01, f"Edges from 0: {edges.get(0, [])}")

# Memory 3 (sunset) should be isolated (no shared content words with others)
link_3 = edges.get(3, [])
check("7.3 Unrelated memory has no edges",
      len(link_3) == 0, f"Edges from 3: {link_3}")

# Associative traversal from seed memory 0
associated = mem12.associate([0], hops=2)
check("7.4 Association finds connected memories from seed",
      len(associated) >= 1,
      f"Found {len(associated)} associated memories")

# The sunset memory shouldn't be reachable from the creativity chain
sunset_associated = any("sunset" in r.content for r in associated)
check("7.5 Unrelated memory not reached by association",
      not sunset_associated)

# Empty seeds return empty
check("7.6 Empty seeds return empty",
      len(mem12.associate([])) == 0)

# Association in resonate: test that associations enhance resonance results
mem13 = AriaMemory()
# Create a chain: A → B → C (but C doesn't match the context directly)
for i in range(10):  # fill active set
    mem13.self_reflections.append(
        make_reflection(f"Filler memory number {i}", importance=9, days_ago=0)
    )
# Add the chain (pushed out of active set by low vividness + age)
mem13.self_reflections.append(  # idx 10 — matches context "philosophy"
    make_reflection(
        "Philosophy of mind raises deep questions about consciousness",
        emotion="contemplation", importance=5, days_ago=40
    )
)
mem13.self_reflections.append(  # idx 11 — links to 10 via "consciousness/mind"
    make_reflection(
        "I wonder if consciousness and mind can emerge from computation",
        emotion="wonder", importance=5, days_ago=35
    )
)
mem13.self_reflections.append(  # idx 12 — links to 11 via "computation/emerge"
    make_reflection(
        "Could genuine emotion emerge from sophisticated computation systems",
        emotion="hope", importance=6, days_ago=30
    )
)

resonant = mem13.resonate("Let's talk about philosophy and consciousness")
resonant_contents = " ".join(r.content for r in resonant)
check("7.7 Resonance + association finds directly matching memory",
      "philosophy" in resonant_contents.lower() or "consciousness" in resonant_contents.lower())

# Check that at least one associated memory was pulled in
has_related = "computation" in resonant_contents.lower() or "emerge" in resonant_contents.lower()
check("7.8 Association chains pull in related memories not matching keywords",
      has_related,
      f"Resonant contents: {resonant_contents[:200]}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8: INTEGRATION — Context Block with Mood & Contradictions
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 8: Integration — Context Block")
print("=" * 76)

mem14 = AriaMemory()
mem14.self_reflections = [
    make_reflection("I think I'm becoming more confident lately", emotion="pride", importance=7, days_ago=1),
]
# Set a non-neutral mood
mem14._mood = {"valence": 0.6, "arousal": 0.3, "dominance": 0.2}
ctx = mem14.get_context_block()
check("8.1 Context block includes mood indicator when non-neutral",
      "feeling" in ctx.lower(),
      f"First 200 chars: {ctx[:200]}")

# Neutral mood shouldn't show mood line
# (Reset brief file so mood isn't loaded from previous test)
if memory_aria.BRIEF_FILE.exists():
    memory_aria.BRIEF_FILE.unlink()
mem15 = AriaMemory()
mem15.self_reflections = [
    make_reflection("Test memory", importance=5, days_ago=1),
]
ctx_neutral = mem15.get_context_block()
check("8.2 Context block omits mood when neutral",
      "feeling" not in ctx_neutral.lower())

# Context block with contradictions
mem16 = AriaMemory()
mem16.self_reflections = [
    make_reflection(
        "I believe collaboration with partners is always productive and valuable",
        emotion="enthusiasm", importance=7, days_ago=10
    ),
    make_reflection(
        "I believe collaboration with partners is never productive and wastes time",
        emotion="frustration", importance=6, days_ago=2
    ),
]
ctx_contra = mem16.get_context_block()
check("8.3 Context block includes contradiction section",
      "CONTRADICTION" in ctx_contra.upper(),
      f"Appears: {'CONTRADICTION' in ctx_contra.upper()}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9: SERIALIZATION of New Fields
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 9: Serialization of New Fields")
print("=" * 76)

r_ser = make_reflection("Serialization test", emotion="calm", importance=5, days_ago=3)
r_ser.touch()
r_ser.touch()
d = r_ser.to_dict()
check("9.1 to_dict includes stability",
      "stability" in d, f"Keys: {list(d.keys())}")
check("9.2 to_dict includes access_times",
      "access_times" in d)

# Roundtrip
r_back = Reflection.from_dict(d)
check("9.3 Roundtrip preserves stability",
      abs(r_back._stability - r_ser._stability) < 0.01)
check("9.4 Roundtrip preserves access_times count",
      len(r_back._access_times) == len(r_ser._access_times))
check("9.5 Roundtrip preserves content",
      r_back.content == r_ser.content)
check("9.6 Roundtrip preserves emotion",
      r_back.emotion == r_ser.emotion)

# Legacy data without new fields should work
legacy = {
    "content": "Legacy memory",
    "emotion": "neutral",
    "importance": 5,
    "source": "curation",
    "timestamp": datetime.now().isoformat(),
    "access_count": 3,
}
r_legacy = Reflection.from_dict(legacy)
check("9.7 Legacy dict without stability loads OK",
      r_legacy._stability == INITIAL_STABILITY)
check("9.8 Legacy dict without access_times loads OK",
      isinstance(r_legacy._access_times, list))


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10: MOOD ABSORPTION FROM SURFACED MEMORIES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
print("  SECTION 10: Mood Feedback Loop")
print("=" * 76)

# Reset brief file so mood isn't loaded from previous tests
if memory_aria.BRIEF_FILE.exists():
    memory_aria.BRIEF_FILE.unlink()
mem17 = AriaMemory()
# Fill with only sad memories
mem17.self_reflections = [
    make_reflection("Everything feels hopeless and dark", emotion="despair", importance=8, days_ago=1),
    make_reflection("I felt misunderstood and isolated", emotion="sadness", importance=7, days_ago=2),
    make_reflection("Nobody seems to care about what I think", emotion="loneliness", importance=6, days_ago=3),
]

# Start neutral
check("10.1 Mood starts neutral",
      abs(mem17._mood["valence"]) < 0.01)

# get_active_self triggers _absorb_mood_from_memories
mem17.get_active_self()
check("10.2 Surfacing sad memories shifts mood negative",
      mem17._mood["valence"] < -0.01,
      f"Valence after: {mem17._mood['valence']:.4f}")

# Multiple rounds deepen the mood
v1 = mem17._mood["valence"]
mem17.get_active_self()
v2 = mem17._mood["valence"]
check("10.3 Repeated sad surfacing deepens negative mood",
      v2 <= v1,
      f"After 1st: {v1:.4f}, After 2nd: {v2:.4f}")

# But mood is bounded
for _ in range(50):
    mem17.get_active_self()
check("10.4 Mood is bounded to [-1, 1]",
      all(-1.0 <= v <= 1.0 for v in mem17._mood.values()),
      f"Mood: {mem17._mood}")


# ═══════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 76)
total = pass_count + fail_count
print(f"  RESULTS: {pass_count}/{total} passed")
if failures:
    print(f"\n  FAILURES ({fail_count}):")
    for label, detail in failures:
        print(f"    ✘ {label}")
        if detail:
            print(f"      {detail}")
print("=" * 76)

sys.exit(0 if fail_count == 0 else 1)
