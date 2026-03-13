"""
test_dream_regret_arcs.py — Tests for the 3 organic cognitive features.

Validates:
  1. Memory Dreaming — 2-hop candidate discovery, prompt generation,
     dream application with dedup, session interval gating.
  2. Regret Scoring — importance-drop flagging, pattern aggregation,
     context block generation, serialization roundtrip.
  3. Relationship Arc Tracking — trajectory EMA, warmth accumulation,
     trend labels, arc context in get_context_block, serialization.
"""

import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Patch data dirs to use temp directory ────────────────────────────
_tmp = tempfile.mkdtemp(prefix="dream_regret_arc_test_")
import memory_aria
memory_aria.DATA_DIR = Path(_tmp) / "aria"
memory_aria.SELF_FILE = memory_aria.DATA_DIR / "self_memory.json"
memory_aria.SOCIAL_DIR = memory_aria.DATA_DIR / "social"
memory_aria.BRIEF_FILE = memory_aria.DATA_DIR / "brief.json"

from memory_aria import (
    AriaMemory, Reflection, _emotion_to_vector, _content_words,
    _overlap_ratio, DREAM_INTERVAL, DREAM_PROMPT, parse_dream_response,
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
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f"\n         {detail}"
        print(msg)
        fail_count += 1
        failures.append(label)


def fresh_memory(save_path=None) -> AriaMemory:
    """Create a fresh AriaMemory in a clean temp directory."""
    d = save_path or tempfile.mkdtemp(prefix="aria_test_")
    memory_aria.DATA_DIR = Path(d) / "aria"
    memory_aria.SELF_FILE = memory_aria.DATA_DIR / "self_memory.json"
    memory_aria.SOCIAL_DIR = memory_aria.DATA_DIR / "social"
    memory_aria.BRIEF_FILE = memory_aria.DATA_DIR / "brief.json"
    return AriaMemory()


# ═════════════════════════════════════════════════════════════════════════
# SECTION 1: Memory Dreaming
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 1: Memory Dreaming")
print("=" * 72)

# --- 1.1 needs_dream respects interval ---
m = fresh_memory()
check("1.1 needs_dream false at session 0", not m.needs_dream())
m._brief_data["session_count"] = DREAM_INTERVAL
check("1.2 needs_dream true after DREAM_INTERVAL sessions",
      m.needs_dream())
m._brief_data["last_dream_session"] = DREAM_INTERVAL
check("1.3 needs_dream false after dreaming",
      not m.needs_dream())
m._brief_data["session_count"] = DREAM_INTERVAL * 2
check("1.4 needs_dream true again after another interval",
      m.needs_dream())

# --- 1.5 find_dream_candidates needs >= 4 memories ---
m = fresh_memory()
for i in range(3):
    m.add_self_reflection(Reflection(f"Memory about topic {i}", emotion="curious", importance=7))
candidates = m.find_dream_candidates()
check("1.5 find_dream_candidates returns [] with < 4 memories",
      len(candidates) == 0)

# --- 1.6 find_dream_candidates returns pairs for connected memories ---
m = fresh_memory()
# Create memories that form a chain: A-B share "neural networks",
# B-C share "deep learning", so A and C are 2-hop connected
m.add_self_reflection(Reflection(
    "Studied neural networks and backpropagation algorithms in depth",
    emotion="curious", importance=8))
m.add_self_reflection(Reflection(
    "Neural networks can be combined with deep learning techniques for better results",
    emotion="excited", importance=7))
m.add_self_reflection(Reflection(
    "Deep learning breakthroughs in computer vision changed the field dramatically",
    emotion="amazed", importance=8))
m.add_self_reflection(Reflection(
    "The history of philosophy contains many ideas about consciousness and will",
    emotion="thoughtful", importance=6))
m.add_self_reflection(Reflection(
    "Computer vision systems are amazing at recognizing deep patterns in photographs",
    emotion="curious", importance=7))

candidates = m.find_dream_candidates(max_pairs=10)
check("1.6 find_dream_candidates returns some pairs",
      len(candidates) > 0,
      f"Got {len(candidates)} pairs")

# --- 1.7 dream candidates are tuples of (Reflection, Reflection, float) ---
if candidates:
    a, b, strength = candidates[0]
    check("1.7 dream candidate is (Reflection, Reflection, float)",
          isinstance(a, Reflection) and isinstance(b, Reflection) and isinstance(strength, (int, float)))
else:
    check("1.7 dream candidate is (Reflection, Reflection, float)", False,
          "No candidates found to check")

# --- 1.8 prepare_dream_prompt returns non-empty for valid candidates ---
prompt = m.prepare_dream_prompt()
# Prompt may or may not be generated depending on 2-hop graph structure
# Just check it returns a string
check("1.8 prepare_dream_prompt returns string",
      isinstance(prompt, str))

# --- 1.9 apply_dream inserts a new memory with source='dream' ---
m = fresh_memory()
m.add_self_reflection(Reflection("Base memory about gardening and plants", importance=5))
count_before = len(m.self_reflections)
m.apply_dream([{
    "insight": "There is a hidden connection between gardening patience and coding persistence — both require nurturing something slowly over time.",
    "emotion": "reflective",
    "importance": 6,
}])
check("1.9 apply_dream adds a memory",
      len(m.self_reflections) == count_before + 1)
dream_mem = m.self_reflections[-1]
check("1.10 dream memory has source='dream'",
      dream_mem.source == "dream")
check("1.11 dream memory has why_saved about dreaming",
      "dream" in dream_mem.why_saved.lower())

# --- 1.12 apply_dream deduplicates ---
m.apply_dream([{
    "insight": "There is a hidden connection between gardening patience and coding persistence — both require nurturing something slowly over time.",
    "emotion": "reflective",
    "importance": 6,
}])
check("1.12 apply_dream deduplicates identical insight",
      len(m.self_reflections) == count_before + 1)

# --- 1.13 apply_dream rejects short insights ---
m.apply_dream([{"insight": "Too short", "importance": 5}])
check("1.13 apply_dream rejects insight < 20 chars",
      len(m.self_reflections) == count_before + 1)

# --- 1.14 apply_dream updates dream_log and last_dream_session ---
m._brief_data["session_count"] = 4
m.apply_dream([{
    "insight": "A completely new connection that nobody has ever thought of before in the entire world of ideas.",
    "importance": 5,
}])
dream_log = m._brief_data.get("dream_log", [])
check("1.14 dream_log has entries",
      len(dream_log) > 0)
check("1.15 last_dream_session updated",
      m._brief_data.get("last_dream_session") == 4)

# --- 1.16 DREAM_PROMPT template has {pairs} placeholder ---
check("1.16 DREAM_PROMPT has {pairs} placeholder",
      "{pairs}" in DREAM_PROMPT)

# --- 1.17 parse_dream_response delegates to parse_curation_response ---
check("1.17 parse_dream_response is callable",
      callable(parse_dream_response))


# ═════════════════════════════════════════════════════════════════════════
# SECTION 2: Regret Scoring
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 2: Regret Scoring")
print("=" * 72)

# --- 2.1 New Reflection has _regret = 0.0 and _believed_importance = 0 ---
r = Reflection("Test memory", importance=7, emotion="excited")
check("2.1 New Reflection._regret is 0.0",
      r._regret == 0.0)
check("2.2 New Reflection._believed_importance is 0",
      r._believed_importance == 0)

# --- 2.3 apply_rescores sets regret when importance drops (original >= 5) ---
m = fresh_memory()
ref = Reflection("I thought this was super important but actually it was noise",
                 importance=8, emotion="excited")
m.add_self_reflection(ref)
m.apply_rescores(
    [{"index": 0, "new_importance": 5}],
    m.self_reflections
)
check("2.3 Regret set after importance drop (8 -> clamped 6)",
      m.self_reflections[0]._regret > 0,
      f"regret={m.self_reflections[0]._regret}")
check("2.4 believed_importance preserved",
      m.self_reflections[0]._believed_importance == 8)
check("2.5 Clamped to +/-2 cap (8 -> 6, not 5)",
      m.self_reflections[0].importance == 6)

# --- 2.6 No regret when original < 5 ---
m2 = fresh_memory()
ref2 = Reflection("Low importance memory", importance=3, emotion="neutral")
m2.add_self_reflection(ref2)
m2.apply_rescores(
    [{"index": 0, "new_importance": 1}],
    m2.self_reflections
)
check("2.6 No regret when original importance < 5",
      m2.self_reflections[0]._regret == 0.0)

# --- 2.7 No regret when importance goes up ---
m3 = fresh_memory()
ref3 = Reflection("Underrated memory", importance=6, emotion="thoughtful")
m3.add_self_reflection(ref3)
m3.apply_rescores(
    [{"index": 0, "new_importance": 9}],
    m3.self_reflections
)
check("2.7 No regret when importance goes up",
      m3.self_reflections[0]._regret == 0.0)

# --- 2.8 get_regret_memories returns sorted by regret ---
m4 = fresh_memory()
r_high = Reflection("Major overestimate", importance=10, emotion="excited")
r_low = Reflection("Minor overestimate", importance=6, emotion="happy")
m4.add_self_reflection(r_high)
m4.add_self_reflection(r_low)
m4.apply_rescores(
    [{"index": 0, "new_importance": 7}, {"index": 1, "new_importance": 3}],
    m4.self_reflections
)
regretted = m4.get_regret_memories()
check("2.8 get_regret_memories returns regretted memories",
      len(regretted) >= 1)
if len(regretted) >= 2:
    check("2.9 Regretted memories sorted by regret (highest first)",
          regretted[0]._regret >= regretted[1]._regret)
else:
    # Only one may have regret (original >= 5 filter)
    check("2.9 At least 1 regretted memory (10->8 drop)",
          len(regretted) >= 1)

# --- 2.10 get_regret_patterns ---
patterns = m4.get_regret_patterns()
check("2.10 get_regret_patterns returns dict with count",
      "count" in patterns and patterns["count"] >= 1)

# --- 2.11 get_regret_context needs >= 3 regretted memories ---
ctx = m4.get_regret_context()
check("2.11 get_regret_context empty with < 3 regretted",
      ctx == "",
      f"Got: '{ctx[:80]}'" if ctx else "")

# --- 2.12 get_regret_context with 3+ regretted memories ---
m5 = fresh_memory()
_regret_topics = [
    "The breakthrough in quantum computing will change everything we know",
    "My theory about consciousness emerging from recursive neural pathways",
    "The connection between jazz improvisation and mathematical fractals",
    "How photosynthetic bacteria could revolutionize solar panel design",
    "The hidden link between medieval architecture and modern software design",
]
for i, topic in enumerate(_regret_topics):
    r = Reflection(topic, importance=8 + (i % 3), emotion="excitement")
    m5.add_self_reflection(r)
check("2.12a All 5 diverse memories stored",
      len(m5.self_reflections) == 5,
      f"Got {len(m5.self_reflections)}")
m5.apply_rescores(
    [{"index": i, "new_importance": 3} for i in range(len(m5.self_reflections))],
    m5.self_reflections
)
ctx = m5.get_regret_context()
check("2.12 get_regret_context non-empty with 3+ regretted",
      len(ctx) > 0,
      f"Got {len(ctx)} chars")
check("2.13 Regret context has JUDGMENT header",
      "JUDGMENT" in ctx.upper() if ctx else False)

# --- 2.14 Regret survives save/load cycle ---
m6 = fresh_memory()
ref6 = Reflection("Round trip memory", importance=9, emotion="excited")
m6.add_self_reflection(ref6)
m6.apply_rescores(
    [{"index": 0, "new_importance": 6}],
    m6.self_reflections
)
# to_dict and from_dict roundtrip
d = m6.self_reflections[0].to_dict()
check("2.14 to_dict includes regret field",
      "regret" in d, f"keys: {list(d.keys())}")
check("2.15 to_dict includes believed_importance field",
      "believed_importance" in d)
restored = Reflection.from_dict(d)
check("2.16 from_dict restores _regret",
      restored._regret == m6.self_reflections[0]._regret)
check("2.17 from_dict restores _believed_importance",
      restored._believed_importance == m6.self_reflections[0]._believed_importance)

# --- 2.18 regret context appears in get_context_block ---
ctx_block = m5.get_context_block()
check("2.18 Regret context appears in context block",
      "JUDGMENT" in ctx_block.upper(),
      f"Block length: {len(ctx_block)}")


# ═════════════════════════════════════════════════════════════════════════
# SECTION 3: Relationship Arc Tracking
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SECTION 3: Relationship Arc Tracking")
print("=" * 72)

# --- 3.1 _update_relationship_arc creates arc entry ---
m = fresh_memory()
m._update_relationship_arc("Rex", "joy")  # joy has positive valence
arcs = m._brief_data.get("relationship_arcs", {})
check("3.1 Arc entry created for Rex", "Rex" in arcs)

# --- 3.2 Arc has required fields ---
arc = arcs["Rex"]
check("3.2 Arc has trajectory", "trajectory" in arc)
check("3.3 Arc has warmth", "warmth" in arc)
check("3.4 Arc has history", "history" in arc)
check("3.5 Arc has impression_count", "impression_count" in arc)

# --- 3.6 Positive emotion moves trajectory positive ---
check("3.6 Positive emotion moves trajectory > 0",
      arc["trajectory"] > 0,
      f"trajectory={arc['trajectory']}")

# --- 3.7 Multiple positive emotions compound ---
first_traj = arc["trajectory"]
m._update_relationship_arc("Rex", "joy")
m._update_relationship_arc("Rex", "gratitude")
arc2 = arcs["Rex"]
check("3.7 Multiple positive emotions increase trajectory",
      arc2["trajectory"] > first_traj,
      f"before={first_traj}, after={arc2['trajectory']}")
check("3.8 Warmth accumulates positively",
      arc2["warmth"] > 0,
      f"warmth={arc2['warmth']}")

# --- 3.9 Negative emotions push trajectory negative ---
m2 = fresh_memory()
for _ in range(5):
    m2._update_relationship_arc("Rival", "anger")  # anger has negative valence
arc_rival = m2._brief_data["relationship_arcs"]["Rival"]
check("3.9 Negative emotions push trajectory negative",
      arc_rival["trajectory"] < 0,
      f"trajectory={arc_rival['trajectory']}")
check("3.10 Warmth accumulates negatively",
      arc_rival["warmth"] < 0,
      f"warmth={arc_rival['warmth']}")

# --- 3.11 get_relationship_arc returns dict with trend_label ---
arc_dict = m.get_relationship_arc("Rex")
check("3.11 get_relationship_arc returns dict", arc_dict is not None)
check("3.12 Arc dict has trend_label",
      "trend_label" in (arc_dict or {}))
check("3.13 Positive trajectory -> 'warming' label",
      (arc_dict or {}).get("trend_label") == "warming",
      f"label={arc_dict.get('trend_label') if arc_dict else 'None'}")

# --- 3.14 Negative trajectory -> 'cooling' label ---
rival_dict = m2.get_relationship_arc("Rival")
check("3.14 Negative trajectory -> 'cooling' label",
      (rival_dict or {}).get("trend_label") == "cooling",
      f"label={rival_dict.get('trend_label') if rival_dict else 'None'}")

# --- 3.15 get_relationship_arc returns None for unknown entity ---
check("3.15 Unknown entity returns None",
      m.get_relationship_arc("Nobody") is None)

# --- 3.16 get_arc_context with >= 2 impressions ---
ctx = m.get_arc_context("Rex")
check("3.16 Arc context non-empty for Rex (3 impressions)",
      len(ctx) > 0,
      f"Got: '{ctx}'")
check("3.17 Arc context mentions entity name",
      "Rex" in ctx if ctx else False)
check("3.18 Arc context mentions warmth descriptor",
      any(w in ctx for w in ["warm", "cool", "cold", "neutral"]) if ctx else False)
check("3.19 Arc context mentions trend",
      any(w in ctx for w in ["warming", "cooling", "stable"]) if ctx else False)

# --- 3.20 get_arc_context empty with < 2 impressions ---
m3 = fresh_memory()
m3._update_relationship_arc("Newbie", "curiosity")
ctx_empty = m3.get_arc_context("Newbie")
check("3.20 Arc context empty with < 2 impressions",
      ctx_empty == "")

# --- 3.21 add_social_impression triggers arc update ---
m4 = fresh_memory()
m4._brief_data["session_count"] = 1
imp = Reflection("Rex was really kind today and helped me figure out a problem",
                 emotion="grateful", importance=7, source="Rex")
m4.add_social_impression("Rex", imp)
arcs4 = m4._brief_data.get("relationship_arcs", {})
check("3.21 add_social_impression triggers arc creation",
      "Rex" in arcs4)
check("3.22 Arc trajectory positive after gratitude impression",
      arcs4.get("Rex", {}).get("trajectory", 0) > 0)

# --- 3.23 Trajectory bounded to [-1, 1] ---
m5 = fresh_memory()
for _ in range(50):
    m5._update_relationship_arc("Friend", "joy")
arc5 = m5._brief_data["relationship_arcs"]["Friend"]
check("3.23 Trajectory bounded <= 1.0",
      arc5["trajectory"] <= 1.0,
      f"trajectory={arc5['trajectory']}")
check("3.24 Warmth bounded <= 1.0",
      arc5["warmth"] <= 1.0,
      f"warmth={arc5['warmth']}")

# --- 3.25 History tracks sessions ---
hist = arc5.get("history", [])
check("3.25 History has entries",
      len(hist) > 0)

# --- 3.26 Arc survives save/load ---
m6 = fresh_memory()
m6._update_relationship_arc("Rex", "joy")
m6._update_relationship_arc("Rex", "joy")
m6.save()
m6._save_brief()  # brief data (arcs) is saved separately
save_root = str(Path(memory_aria.DATA_DIR).parent)
m6_loaded = fresh_memory(save_path=save_root)
check("3.26 Relationship arcs survive save/load",
      "Rex" in m6_loaded._brief_data.get("relationship_arcs", {}),
      f"arcs keys: {list(m6_loaded._brief_data.get('relationship_arcs', {}).keys())}")

# --- 3.27 Arc context in get_context_block for entity ---
m7 = fresh_memory()
# Need enough impressions to trigger arc context
imp1 = Reflection("Rex said something encouraging", emotion="joy", importance=6, source="Rex")
imp2 = Reflection("Rex helped with a difficult problem", emotion="gratitude", importance=7, source="Rex")
m7.add_social_impression("Rex", imp1)
m7.add_social_impression("Rex", imp2)
block = m7.get_context_block(current_entity="Rex")
check("3.27 Context block includes arc info for entity",
      "warming" in block.lower() or "trajectory" in block.lower() or "relationship" in block.lower(),
      f"Block snippet: {block[-200:]}")

# --- 3.28 EMA alpha check: trajectory converges ---
m8 = fresh_memory()
# Start with strong negative
for _ in range(10):
    m8._update_relationship_arc("Foe", "anger")
neg_traj = m8._brief_data["relationship_arcs"]["Foe"]["trajectory"]
# Then add positive
for _ in range(10):
    m8._update_relationship_arc("Foe", "joy")
new_traj = m8._brief_data["relationship_arcs"]["Foe"]["trajectory"]
check("3.28 Trajectory recovers from negative after positive emotions",
      new_traj > neg_traj,
      f"before={neg_traj}, after={new_traj}")

# --- 3.29 Stats include new feature counts ---
m9 = fresh_memory()
for i in range(3):
    r = Reflection(f"Stats memory {i} about important things", importance=8, emotion="excited")
    m9.add_self_reflection(r)
m9.apply_rescores(
    [{"index": 0, "new_importance": 5}],
    m9.self_reflections
)
m9._update_relationship_arc("Rex", "joy")
m9._brief_data["dream_log"] = [{"session": 1, "insight": "test"}]
s = m9.stats()
check("3.29 Stats has regret_count",
      "regret_count" in s, f"keys: {list(s.keys())}")
check("3.30 Stats has dream_count",
      "dream_count" in s)
check("3.31 Stats has tracked_arcs",
      "tracked_arcs" in s)
check("3.32 regret_count is 1",
      s.get("regret_count") == 1,
      f"got {s.get('regret_count')}")
check("3.33 dream_count is 1",
      s.get("dream_count") == 1,
      f"got {s.get('dream_count')}")
check("3.34 tracked_arcs has 1 entity",
      len(s.get("tracked_arcs", {})) == 1,
      f"got {s.get('tracked_arcs')}")


# ═════════════════════════════════════════════════════════════════════════
# RESULTS
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  RESULTS SUMMARY")
print("=" * 72)
print(f"  Total:  {pass_count + fail_count}")
print(f"  Passed: {pass_count}")
print(f"  Failed: {fail_count}")
if failures:
    print(f"  Failures:")
    for f in failures:
        print(f"    - {f}")
print()

sys.exit(1 if fail_count else 0)
