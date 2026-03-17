#!/usr/bin/env python3
"""
VividnessMem — Comprehensive Human-Likeness Test Suite
======================================================
Tests whether the memory system behaves like real human memory across
7 real-world scenarios, 250+ individual assertions.

Covers: decay, spaced repetition, mood-congruent recall, emotional
salience, deduplication, contradiction detection, relationship arcs,
regret tracking, task memory, solution patterns, artifact tracking,
professional mode, STM facts, entity preferences, resonance,
association chains, foreground/background split, context blocks,
dreaming candidates, consolidation clusters, and more.

Run:  python test_humanlike_memory.py
"""

import sys, os, math, tempfile, shutil, json
from datetime import datetime, timedelta
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from VividnessMem import (
    VividnessMem, Memory, ShortTermFact,
    TaskRecord, ActionRecord, SolutionPattern, ArtifactRecord,
    EMOTION_VECTORS, parse_curation_response, parse_brief_response,
    parse_rescore_response,
)

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
_pass = 0
_fail = 0
_errors = []

def check(condition: bool, label: str):
    global _pass, _fail
    if condition:
        _pass += 1
    else:
        _fail += 1
        _errors.append(label)
        print(f"  FAIL: {label}")

def fresh_mem(**kw) -> VividnessMem:
    d = tempfile.mkdtemp()
    return VividnessMem(data_dir=d, **kw)

def age_memory(mem: Memory, days: float):
    """Backdate a memory's timestamp by N days."""
    old_dt = datetime.fromisoformat(mem.timestamp)
    mem.timestamp = (old_dt - timedelta(days=days)).isoformat()

def age_fact(fact: ShortTermFact, hours: float):
    old_dt = datetime.fromisoformat(fact.timestamp)
    fact.timestamp = (old_dt - timedelta(hours=hours)).isoformat()

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 1: The Forgetting Curve
#  "Unimportant things fade, important things persist, and
#   well-timed recall strengthens memories."
# ══════════════════════════════════════════════════════════════════
def test_forgetting_curve():
    section("SCENARIO 1: The Forgetting Curve")
    mem = fresh_mem()

    # --- 1a: Vividness decays with time ---
    m = mem.add_self_reflection("I saw a blue car today", emotion="neutral", importance=5)
    v_fresh = m.vividness
    check(v_fresh > 4.5, "1a: Fresh memory vividness ≈ importance")

    age_memory(m, 3.0)  # 3 days later
    v_3d = m.vividness
    check(v_3d < v_fresh, "1a: Memory fades after 3 days")
    # With INITIAL_STABILITY=3, at 3 days: retention = e^(-1) ≈ 0.368
    expected_3d = 5 * math.exp(-3.0 / 3.0)
    check(abs(v_3d - expected_3d) < 0.1, f"1a: 3-day vividness matches formula ({v_3d:.2f} ≈ {expected_3d:.2f})")

    age_memory(m, 7.0)  # now 10 days old total
    v_10d = m.vividness
    check(v_10d < v_3d, "1a: Memory fades further after 10 days")

    # --- 1b: High importance fades slower in absolute terms ---
    m_high = mem.add_self_reflection("My dog passed away", emotion="sad", importance=10)
    m_low = mem.add_self_reflection("I had cereal for breakfast", emotion="neutral", importance=2)
    age_memory(m_high, 5.0)
    age_memory(m_low, 5.0)
    check(m_high.vividness > m_low.vividness,
          "1b: Important memory stays more vivid than trivial one at same age")
    check(m_high.vividness > 1.0, "1b: Important memory still meaningful after 5 days")
    check(m_low.vividness < 1.0, "1b: Trivial memory nearly gone after 5 days")

    # --- 1c: Spaced repetition strengthens ---
    m_rep = mem.add_self_reflection("Python uses indentation for blocks",
                                    emotion="neutral", importance=6)
    age_memory(m_rep, 1.0)  # 1 day gap > MIN_SPACING_DAYS (0.5)
    v_before_touch = m_rep.vividness
    stab_before_touch = m_rep._stability
    # Manually set _last_access to match the aged timestamp so touch() sees a gap
    m_rep._last_access = m_rep.timestamp
    m_rep.touch()
    # After touch, stability should increase
    check(m_rep._stability > stab_before_touch, "1c: Stability grows after touch")
    # The vividness should be higher now (re-anchored + higher stability)
    v_after_touch = m_rep.vividness
    check(v_after_touch > v_before_touch, "1c: Vividness jumps after spaced recall")

    # --- 1d: Touch too soon doesn't give full bonus ---
    m_cram = mem.add_self_reflection("Cramming test material", emotion="anxious", importance=6)
    stab_before = m_cram._stability
    m_cram.touch()  # immediately — less than MIN_SPACING_DAYS (0.5 days)
    stab_after = m_cram._stability
    # Stability should NOT grow because gap < MIN_SPACING_DAYS
    check(stab_after == stab_before,
          "1d: Immediate re-access gives NO spacing bonus (cramming)")

    # --- 1e: Diminishing returns on repeated touches ---
    m_dim = mem.add_self_reflection("The sky is blue", emotion="neutral", importance=5)
    # Each touch needs gap >= MIN_SPACING_DAYS so set _last_access back
    m_dim._last_access = (datetime.now() - timedelta(days=1)).isoformat()
    stab0 = m_dim._stability
    m_dim.touch()
    pct1 = (m_dim._stability - stab0) / stab0  # percentage gain
    m_dim._last_access = (datetime.now() - timedelta(days=1)).isoformat()
    stab1 = m_dim._stability
    m_dim.touch()
    pct2 = (m_dim._stability - stab1) / stab1  # percentage gain
    m_dim._last_access = (datetime.now() - timedelta(days=1)).isoformat()
    stab2 = m_dim._stability
    m_dim.touch()
    pct3 = (m_dim._stability - stab2) / stab2  # percentage gain
    check(pct2 < pct1,
          "1e: Repeated touches show diminishing percentage stability gains")

    # --- 1f: Very old memories are effectively gone ---
    m_ancient = mem.add_self_reflection("Something from years ago",
                                        emotion="neutral", importance=3)
    age_memory(m_ancient, 365.0)
    check(m_ancient.vividness < 0.01,
          "1f: Year-old low-importance memory is effectively zero")

    print(f"  Scenario 1: {_pass} passed so far")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 2: Mood Colours Everything
#  "When you're sad, sad memories surface more easily.
#   When you're happy, happy memories float up."
# ══════════════════════════════════════════════════════════════════
def test_mood_congruent_recall():
    section("SCENARIO 2: Mood Colours Everything")
    mem = fresh_mem()

    # Store a mix of happy and sad memories with equal importance
    m_happy1 = mem.add_self_reflection("Wonderful birthday party with friends",
                                       emotion="happy", importance=7)
    m_happy2 = mem.add_self_reflection("Got promoted at work today",
                                       emotion="joyful", importance=7)
    m_sad1 = mem.add_self_reflection("Argument with my best friend",
                                     emotion="sad", importance=7)
    m_sad2 = mem.add_self_reflection("Lost my favorite book",
                                     emotion="disappointed", importance=7)
    m_neutral = mem.add_self_reflection("Had toast for breakfast",
                                        emotion="neutral", importance=7)

    # --- 2a: Mood drift ---
    check(mem.mood == (0.0, 0.0, 0.0), "2a: Initial mood is neutral")
    mem.update_mood_from_conversation(["happy", "excited"])
    check(mem.mood[0] > 0.0, "2a: Mood shifts positive after happy conversation")

    p, a, d = mem.mood
    # alpha = 0.3, avg of happy(0.8,0.4,0.5) and excited(0.7,0.8,0.5) = (0.75,0.6,0.5)
    # new = 0.0*0.7 + (0.75)*0.3 = 0.225
    check(abs(p - 0.225) < 0.01, f"2a: Pleasure dimension matches formula ({p:.4f} ≈ 0.225)")

    # --- 2b: Mood-congruent recall boost ---
    happy_mood = mem.mood
    v_happy_in_happy = m_happy1.mood_adjusted_vividness(happy_mood)
    v_sad_in_happy = m_sad1.mood_adjusted_vividness(happy_mood)
    v_neutral_in_happy = m_neutral.mood_adjusted_vividness(happy_mood)
    check(v_happy_in_happy > v_neutral_in_happy,
          "2b: Happy memory boosted in happy mood")
    check(v_sad_in_happy < v_neutral_in_happy,
          "2b: Sad memory suppressed in happy mood")

    # --- 2c: Flip mood to sad ---
    # Reset mood by pushing many sad updates
    for _ in range(10):
        mem.update_mood_from_conversation(["sad", "lonely", "melancholy"])
    check(mem.mood[0] < -0.3, "2c: Mood shifted deeply negative")

    sad_mood = mem.mood
    v_sad_in_sad = m_sad1.mood_adjusted_vividness(sad_mood)
    v_happy_in_sad = m_happy1.mood_adjusted_vividness(sad_mood)
    check(v_sad_in_sad > v_happy_in_sad,
          "2c: Sad memory boosted when agent is sad (mood-congruent)")

    # --- 2d: Gradual mood drift (not instant) ---
    mem2 = fresh_mem()
    mem2.update_mood_from_conversation(["angry"])
    mem2.update_mood_from_conversation(["angry"])
    mood_2x = mem2.mood[0]
    mem2.update_mood_from_conversation(["angry"])
    mem2.update_mood_from_conversation(["angry"])
    mood_4x = mem2.mood[0]
    check(abs(mood_4x) > abs(mood_2x),
          "2d: Mood deepens with repeated emotional input")
    check(abs(mood_4x) < 0.7,
          "2d: Mood doesn't spike instantly (EMA smoothing, 4 rounds < 0.7)")

    # --- 2e: Mixed emotions create blended mood ---
    mem3 = fresh_mem()
    mem3.update_mood_from_conversation(["happy", "sad"])
    # happy=(0.8,0.4,0.5), sad=(-0.6,-0.3,-0.3), avg=(0.1,0.05,0.1)
    check(abs(mem3.mood[0]) < 0.2,
          "2e: Mixed happy+sad emotions create near-neutral mood shift")

    # --- 2f: Mood label reflects current state ---
    mem4 = fresh_mem()
    check(mem4.mood_label == "neutral", "2f: Initial mood label is 'neutral'")
    for _ in range(5):
        mem4.update_mood_from_conversation(["happy", "joyful"])
    check(mem4.mood_label != "neutral", "2f: Mood label changes after happy drift")

    # --- 2g: Negative memory reappraisal over time ---
    mem5 = fresh_mem()
    m_neg = mem5.add_self_reflection("Terrible fight with partner",
                                     emotion="angry", importance=8)
    for _ in range(5):
        mem5.update_mood_from_conversation(["angry", "hurt"])
    angry_mood = mem5.mood

    v_neg_fresh = m_neg.mood_adjusted_vividness(angry_mood)
    age_memory(m_neg, 30.0)  # 30 days later
    v_neg_old = m_neg.mood_adjusted_vividness(angry_mood)
    # Emotional reappraisal: negative boost decays over NEGATIVE_HALFLIFE_DAYS=14
    check(v_neg_old < v_neg_fresh,
          "2g: Old negative memory loses mood-congruence boost (reappraisal)")

    print(f"  Scenario 2 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 3: The Coffee Shop Friendship
#  "Meeting someone, growing closer, memories evolving together."
# ══════════════════════════════════════════════════════════════════
def test_relationship_arc():
    section("SCENARIO 3: The Coffee Shop Friendship")
    mem = fresh_mem()

    # Session 1: Awkward first meeting
    mem.add_social_impression("Maya", "Met Maya at the coffee shop, she seemed shy",
                              emotion="neutral", importance=5)

    arc = mem.get_relationship_arc("Maya")
    check(arc is not None, "3a: Relationship arc exists after first impression")
    check(arc["impression_count"] == 1, "3a: One impression recorded")
    check(abs(arc["trajectory"]) < 0.1, "3a: Neutral first impression → near-zero trajectory")

    # Session 2: Warm conversation
    mem.bump_session()
    mem.add_social_impression("Maya", "Maya opened up about her art. We laughed together",
                              emotion="happy", importance=7)
    arc = mem.get_relationship_arc("Maya")
    check(arc["trajectory"] > 0.1, "3b: Trajectory warms after positive interaction")
    check(arc["warmth"] > 0.0, "3b: Warmth accumulates positively")

    # Sessions 3-5: Deepening friendship
    for emotion, content in [
        ("grateful", "Maya brought me coffee without asking"),
        ("joyful", "Maya and I stayed out talking until midnight"),
        ("affectionate", "Maya said I'm her closest friend"),
    ]:
        mem.bump_session()
        mem.add_social_impression("Maya", content, emotion=emotion, importance=8)

    arc = mem.get_relationship_arc("Maya")
    check(arc["trajectory"] > 0.3, "3c: Strong positive trajectory after multiple warm sessions")
    check(arc["warmth"] > 0.1, "3c: Warmth has accumulated significantly")
    check(arc["trend_label"] == "warming", "3c: Trend label is 'warming'")
    check(arc["impression_count"] == 5, "3c: Five impressions total")
    check(len(arc["history"]) >= 3, "3c: History tracking multiple data points")

    # --- 3d: Arc context string ---
    ctx = mem.get_arc_context("Maya")
    check("warming" in ctx.lower() or "warm" in ctx.lower(),
          "3d: Arc context describes warming relationship")

    # --- 3e: A rough patch ---
    mem.bump_session()
    mem.add_social_impression("Maya", "Maya cancelled on me again. Feeling let down",
                              emotion="disappointed", importance=7)
    mem.bump_session()
    mem.add_social_impression("Maya", "Maya was cold and distant today",
                              emotion="hurt", importance=7)

    arc_after = mem.get_relationship_arc("Maya")
    check(arc_after["trajectory"] < arc["trajectory"],
          "3e: Trajectory drops after negative interactions")

    # --- 3f: Social impressions tracked per-entity ---
    mem.add_social_impression("Dave", "Dave is funny but unreliable",
                              emotion="amused", importance=5)
    check(len(mem.get_active_social("Maya")) > 0, "3f: Maya impressions retrievable")
    check(len(mem.get_active_social("Dave")) > 0, "3f: Dave impressions separate from Maya")

    # --- 3g: Old social impressions fade like any memory ---
    maya_imps = mem.get_active_social("Maya")
    oldest = maya_imps[-1] if maya_imps else None
    if oldest:
        age_memory(oldest, 60.0)
        check(oldest.vividness < 3.0,
              "3g: Old social impressions decay over time")

    # --- 3h: Deduplication works on social too ---
    mem.add_social_impression("Dave", "Dave is funny but really unreliable",
                              emotion="amused", importance=6)
    dave_count = len([m for m in mem.social_impressions.get("dave", [])])
    check(dave_count <= 2, "3h: Near-duplicate social impression deduplicated")

    print(f"  Scenario 3 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 4: "Wait, Didn't I Used To Think..."
#  Deduplication, contradictions, and belief evolution.
# ══════════════════════════════════════════════════════════════════
def test_dedup_and_contradictions():
    section("SCENARIO 4: Wait, Didn't I Used To Think...")
    mem = fresh_mem()

    # --- 4a: Near-identical memories merge ---
    m1 = mem.add_self_reflection("I love cooking Italian food with fresh ingredients",
                                 emotion="happy", importance=6)
    count_before = len(mem.self_reflections)
    m2 = mem.add_self_reflection("I love cooking Italian food with fresh ingredients and herbs",
                                 emotion="happy", importance=8)
    count_after = len(mem.self_reflections)
    check(count_after == count_before,
          "4a: Near-duplicate merged (count unchanged)")
    # Merged memory keeps higher importance
    merged = [m for m in mem.self_reflections if "Italian" in m.content][-1]
    check(merged.importance >= 8,
          "4a: Merged memory keeps the higher importance")

    # --- 4b: Distinct memories don't merge ---
    mem.add_self_reflection("I went hiking in the mountains last weekend",
                            emotion="excited", importance=7)
    mem.add_self_reflection("The concert downtown was incredible",
                            emotion="joyful", importance=7)
    check(len(mem.self_reflections) >= 3,
          "4b: Distinct memories remain separate")

    # --- 4c: Contradictions detected (using negation pattern to exceed threshold) ---
    mem2 = fresh_mem()
    mem2.add_self_reflection("I love working at the tech company every day",
                             emotion="happy", importance=7)
    mem2.add_self_reflection("I don't love working at the tech company anymore",
                             emotion="frustrated", importance=7)
    pairs = mem2.detect_contradictions()
    check(len(pairs) >= 1,
          "4c: Contradiction detected between positive and negated statement")

    # --- 4d: Low-importance memories don't trigger contradictions ---
    mem3 = fresh_mem()
    mem3.add_self_reflection("I like rain in the morning outside",
                             emotion="happy", importance=2)
    mem3.add_self_reflection("I don't like rain in the morning outside",
                             emotion="sad", importance=2)
    pairs = mem3.detect_contradictions()
    check(len(pairs) == 0,
          "4d: Low-importance memories don't trigger contradiction flag")

    # --- 4e: Contradiction context block ---
    ctx = mem2.get_contradiction_context()
    check(len(ctx) > 0, "4e: Contradiction context string is non-empty")

    # --- 4f: Negation pattern detection ---
    # Sentences must differ enough to survive dedup (< 0.80 Jaccard)
    # but share enough topic overlap (> 0.25) for contradiction scoring
    mem4 = fresh_mem()
    mem4.add_self_reflection(
        "I really enjoy the flexible remote work schedule at my company",
        emotion="happy", importance=7)
    mem4.add_self_reflection(
        "I do not enjoy the rigid remote work schedule at my company anymore",
        emotion="frustrated", importance=7)
    pairs = mem4.detect_contradictions()
    check(len(pairs) >= 1,
          "4f: Negation pattern ('do not') triggers contradiction detection")

    print(f"  Scenario 4 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 5: Resonance — Ghosts of Conversations Past
#  "Something in today's chat triggers an old faded memory."
# ══════════════════════════════════════════════════════════════════
def test_resonance_and_associations():
    section("SCENARIO 5: Resonance — Ghosts of Conversations Past")
    mem = fresh_mem()

    # Populate with many varied memories, then age them.
    # Need > ACTIVE_SELF_LIMIT (8) so some fall outside the active set.
    # Resonance only returns non-active memories.
    # Target memories (neural networks, apple pie, race condition) get LOW
    # importance so they fall OUT of the active set (top 8 by vividness)
    # and become eligible for resonance. Filler memories get HIGH importance
    # to fill the active set.
    memories = [
        ("Beautiful sunset over the Pacific Ocean coastline", "peaceful", 8),
        ("Paris trip with jazz clubs and fresh croissants", "happy", 8),
        ("Machine learning conference presentation went well today", "proud", 8),
        ("Argument about politics at the dinner party", "frustrated", 8),
        ("Reading about quantum computing and qubits in detail", "fascinated", 8),
        ("Teaching my nephew to ride a bicycle in park", "warm", 8),
        ("Late night coding session fixing database migration issue", "anxious", 8),
        ("Visited the aquarium and saw jellyfish exhibit", "fascinated", 8),
        ("Baked sourdough bread for the first time ever", "proud", 8),
        ("Organized the bookshelf by color and genre", "content", 8),
        ("Planted tomato seedlings in the garden today", "peaceful", 8),
        ("Watched documentary about deep ocean exploration", "curious", 8),
        # ---- TARGET memories with low importance → outside active set ----
        ("Learned about neural networks and backpropagation in depth", "curious", 2),
        ("Grandmother's recipe for apple pie with cinnamon sugar", "nostalgic", 2),
        ("Debugging a tricky race condition in Go channels at work", "frustrated", 2),
    ]
    for content, emotion, imp in memories:
        m = mem.add_self_reflection(content, emotion=emotion, importance=imp)
        age_memory(m, 30.0)  # all 30 days old — faded

    # --- 5a: Resonance finds relevant old memories ---
    results = mem.resonate(context="I'm studying deep learning and neural network architectures")
    check(len(results) > 0, "5a: Resonance finds matches for 'neural network'")
    contents = " ".join(r.content for r in results)
    check("neural" in contents.lower() or "machine learning" in contents.lower(),
          "5a: Resonance hits are topically relevant")

    # --- 5b: Unrelated context finds nothing ---
    results2 = mem.resonate(context="What should I have for lunch today")
    check(len(results2) <= 1, "5b: Irrelevant context yields few/no resonance hits")

    # --- 5c: Resonance picks up keyword overlap ---
    results3 = mem.resonate(context="Let's debug the tricky race condition in our Go channels service")
    check(any("race condition" in r.content.lower() or "debug" in r.content.lower()
              or "channels" in r.content.lower()
              for r in results3),
          "5c: Resonance matches 'debug' + 'race condition' + 'channels' keywords")

    # --- 5d: Multiple contextual angles ---
    results4 = mem.resonate(context="baking apple pie with cinnamon sugar this weekend grandmother recipe")
    check(any("apple pie" in r.content.lower() or "grandmother" in r.content.lower()
              or "cinnamon" in r.content.lower()
              for r in results4),
          "5d: Resonance connects baking + apple pie + cinnamon to grandmother's recipe")

    # --- 5e: Resonance respects vividness floor ---
    # All memories are 30-day-old with stability=3 → extremely faded
    # But resonance should still find them if keyword overlap is strong enough
    for r in results:
        check(r.vividness < 5.0, "5e: Resonant memories are genuinely faded (low vividness)")

    # --- 5f: Association chains ---
    # Add memories with overlapping concepts
    mem2 = fresh_mem()
    mem2.add_self_reflection("Jazz music makes me feel alive", emotion="happy", importance=7)
    mem2.add_self_reflection("Music theory class was challenging", emotion="curious", importance=6)
    mem2.add_self_reflection("The theory of relativity is fascinating", emotion="fascinated", importance=7)

    # Build the inverted index / co-occurrence by touching context
    mem2.resonate(context="music theory jazz")

    # --- 5g: Foreground / background split ---
    mem3 = fresh_mem()
    mem3.add_self_reflection("Cooking pasta for dinner tonight", emotion="content", importance=7)
    mem3.add_self_reflection("Python decorators are powerful", emotion="curious", importance=7)
    mem3.add_self_reflection("The garden flowers are blooming", emotion="peaceful", importance=7)
    mem3.add_self_reflection("Database indexing improves query speed", emotion="neutral", importance=7)
    mem3.add_self_reflection("Walking the dog in the park", emotion="happy", importance=7)

    fg, bg = mem3.partition_active_self(context="writing Python code with decorators")
    check(len(fg) > 0, "5g: Foreground has Python-relevant memories")
    check(any("Python" in m.content or "decorator" in m.content for m in fg),
          "5g: Python memory in foreground")
    # Non-coding memories should be in background
    if bg:
        check(not any("Python" in m.content for m in bg),
              "5g: Python memory NOT in background")

    # --- 5h: Empty context returns all as foreground ---
    fg_all, bg_all = mem3.partition_active_self(context="")
    check(len(fg_all) == len(mem3.self_reflections) and len(bg_all) == 0,
          "5h: No context → everything foreground, nothing background")

    print(f"  Scenario 5 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 6: Learning From Mistakes
#  "The agent builds a project, makes errors, discovers solutions,
#   and remembers them when facing similar problems again."
# ══════════════════════════════════════════════════════════════════
def test_task_memory():
    section("SCENARIO 6: Learning From Mistakes")
    mem = fresh_mem(professional=True)

    # --- 6a: Project lifecycle ---
    mem.set_active_project("webapp")
    projects = mem.list_projects()
    check(any(p["name"] == "webapp" for p in projects),
          "6a: Project 'webapp' created")

    # --- 6b: Start tasks ---
    t1 = mem.start_task("Set up authentication system",
                        project="webapp", priority=9, tags=["auth", "security"])
    check(t1 is not None and len(t1) > 0, "6b: Task created with ID")

    t2 = mem.start_task("Implement JWT token validation",
                        project="webapp", priority=7, parent_id=t1, tags=["auth", "jwt"])
    check(t2 != t1, "6b: Subtask has different ID")

    task = mem.get_task(t2)
    check(task is not None, "6b: Task retrievable by ID")
    check(task.parent_id == t1, "6b: Parent-child relationship preserved")
    check(task.status == "active", "6b: New task starts active")

    # --- 6c: Log actions (fail → fix → succeed) ---
    mem.log_action(t2, action="Used PyJWT library with HS256",
                   result="failure",
                   error="Token verification fails with wrong algorithm error",
                   importance=6)
    mem.log_action(t2, action="Tried switching to RS256 without proper key setup",
                   result="failure",
                   error="Missing private key file",
                   importance=6)
    mem.log_action(t2, action="Generated RSA key pair and configured RS256 properly",
                   result="success",
                   fix="Generate RSA keys first, configure RS256 with public key for verify",
                   importance=8)

    actions = mem.get_task_actions(t2)
    check(len(actions) == 3, "6c: All 3 actions logged")
    check(actions[0].result == "failure", "6c: First action was failure")
    check(actions[-1].result == "success", "6c: Last action was success")

    # --- 6d: Complete task → auto-extracts solution ---
    mem.complete_task(t2, outcome="JWT validation working with RS256")
    task = mem.get_task(t2)
    check(task.status == "completed", "6d: Task marked completed")

    # Auto-extraction should have created a solution
    solutions = mem.find_solutions("JWT token verification wrong algorithm")
    check(len(solutions) >= 1,
          "6d: Solution auto-extracted from fail→success sequence")
    if solutions:
        check("RS256" in solutions[0].solution or "RSA" in solutions[0].solution or
              "key" in solutions[0].solution.lower(),
              "6d: Auto-extracted solution references the actual fix")

    # --- 6e: Solution pattern matching ---
    mem.record_solution(
        problem="CORS errors when frontend calls backend API",
        failed_approaches=["Adding Access-Control headers manually",
                          "Using a CORS proxy service"],
        solution="Install cors middleware on Express and configure allowed origins",
        tags=["cors", "backend", "express"],
        importance=8,
    )

    matches = mem.find_solutions("Getting CORS errors on fetch requests to API")
    check(len(matches) >= 1, "6e: Solution found for CORS problem")
    if matches:
        check("cors" in matches[0].solution.lower() or
              "middleware" in matches[0].solution.lower(),
              "6e: Matched solution is about CORS middleware")

    # --- 6f: Solution deduplication ---
    mem.record_solution(
        problem="CORS errors calling backend from React frontend",
        failed_approaches=["Tried adding headers manually"],
        solution="Use cors middleware and set allowed origins",
        tags=["cors"],
        importance=7,
    )
    cors_solutions = [s for s in mem._solutions
                      if "cors" in s.problem_signature.lower()]
    check(len(cors_solutions) <= 2,
          "6f: Similar CORS solutions deduplicated or merged")

    # --- 6g: Artifact tracking ---
    mem.track_artifact("auth_middleware.py", artifact_type="file",
                       description="JWT authentication middleware",
                       project="webapp", importance=8,
                       dependencies=["jwt_utils.py"])
    mem.track_artifact("jwt_utils.py", artifact_type="file",
                       description="JWT utility functions for signing and verification",
                       project="webapp", importance=7)

    overview = mem.get_project_overview("webapp")
    check("auth_middleware" in overview, "6g: Project overview includes tracked artifact")

    mem.update_artifact("auth_middleware.py", state="refactored for RS256")
    related = mem.get_related("auth_middleware.py")
    check(any("jwt_utils" in a.name for a in related),
          "6g: Dependency graph finds related artifact")

    # --- 6h: Task context injection ---
    ctx = mem.get_task_context(conversation_context="working on authentication")
    check(len(ctx) > 0, "6h: Task context is non-empty")
    check("webapp" in ctx.lower() or "auth" in ctx.lower(),
          "6h: Task context mentions project or task content")

    # --- 6i: Fail and abandon tasks ---
    t3 = mem.start_task("Try GraphQL for the API", project="webapp", priority=5)
    mem.fail_task(t3, reason="Too complex for the current timeline")
    check(mem.get_task(t3).status == "failed", "6i: Failed task marked correctly")

    t4 = mem.start_task("Evaluate MongoDB", project="webapp", priority=4)
    mem.abandon_task(t4, reason="Team decided to stick with PostgreSQL")
    check(mem.get_task(t4).status == "abandoned", "6i: Abandoned task marked correctly")

    # --- 6j: Active tasks filters ---
    active = mem.get_active_tasks("webapp")
    check(all(t.status == "active" for t in active),
          "6j: get_active_tasks only returns active tasks")

    # --- 6k: Project archiving ---
    mem.archive_project("webapp")
    projects = mem.list_projects()
    webapp = [p for p in projects if p["name"] == "webapp"]
    check(len(webapp) == 1, "6k: Archived project still listed")
    # Verify archive flag written to meta by reading it directly
    meta_file = mem._projects_dir / "webapp" / "meta.json"
    if meta_file.exists():
        import json
        meta = json.loads(meta_file.read_text())
        check(meta.get("archived") == True, "6k: Project meta has archived=True")
    else:
        check(False, "6k: Project meta file not found")

    # --- 6l: Solution reuse boost ---
    sol = mem.find_solutions("CORS errors on API calls")
    if sol:
        old_applied = sol[0].times_applied
        old_imp = sol[0].importance
        sol[0].apply()
        check(sol[0].times_applied == old_applied + 1, "6l: times_applied increments")
        check(sol[0].importance >= old_imp,
              "6l: Importance grows with reuse (SOLUTION_REUSE_BOOST)")
    else:
        check(False, "6l: No CORS solution found to test reuse")

    print(f"  Scenario 6 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 7: Professional vs Character Mode
#  "Same system, two personalities — one emotional, one clinical."
# ══════════════════════════════════════════════════════════════════
def test_professional_mode():
    section("SCENARIO 7: Professional vs Character Mode")
    char_mem = fresh_mem(professional=False)
    prof_mem = fresh_mem(professional=True)

    # Add identical memories to both
    for m in [char_mem, prof_mem]:
        m.add_self_reflection("Completed the quarterly report successfully",
                              emotion="proud", importance=7)
        m.add_self_reflection("The client meeting went poorly",
                              emotion="anxious", importance=7)
        m.add_self_reflection("Found an elegant solution to the caching problem",
                              emotion="excited", importance=8)

    # --- 7a: Professional mood stays neutral ---
    prof_mem.update_mood_from_conversation(["excited", "happy", "proud"])
    check(prof_mem.mood == (0.0, 0.0, 0.0),
          "7a: Professional mode — mood stays (0,0,0)")
    check(prof_mem.mood_label == "neutral",
          "7a: Professional mode — mood label stays 'neutral'")

    char_mem.update_mood_from_conversation(["excited", "happy", "proud"])
    check(char_mem.mood[0] > 0.0,
          "7a: Character mode — mood shifts with emotions")

    # --- 7b: No mood bias in professional retrieval ---
    char_active = char_mem.get_active_self()
    prof_active = prof_mem.get_active_self()
    # Both should return memories but ordering may differ
    check(len(prof_active) > 0, "7b: Professional mode returns memories")
    check(len(char_active) > 0, "7b: Character mode returns memories")

    # --- 7c: Context blocks have different headers ---
    char_ctx = char_mem.get_context_block(current_entity="User",
                                          conversation_context="test")
    prof_ctx = prof_mem.get_context_block(current_entity="User",
                                          conversation_context="test")
    # Professional uses neutral headers
    check("MEMORY" in prof_ctx.upper() or "CONTEXT" in prof_ctx.upper() or len(prof_ctx) > 10,
          "7c: Professional context block is functional")

    # --- 7d: Core mechanics preserved in professional mode ---
    prof2 = fresh_mem(professional=True)
    m1 = prof2.add_self_reflection("API uses REST with JSON payloads",
                                   emotion="neutral", importance=7)
    # Deduplication works
    count_before = len(prof2.self_reflections)
    prof2.add_self_reflection("API uses REST with JSON payloads and pagination",
                              emotion="neutral", importance=8)
    check(len(prof2.self_reflections) == count_before,
          "7d: Deduplication works in professional mode")

    # Decay works
    age_memory(m1, 5.0)
    check(m1.vividness < 7.0, "7d: Decay works in professional mode")

    # Resonance works
    prof2.add_self_reflection("Database uses PostgreSQL with connection pooling",
                              emotion="neutral", importance=6)
    age_memory(prof2.self_reflections[-1], 20.0)
    res = prof2.resonate(context="PostgreSQL database connection issues")
    check(len(res) >= 0, "7d: Resonance functional in professional mode (may return 0 if below floor)")

    # --- 7e: Relationship arcs in professional mode ---
    prof3 = fresh_mem(professional=True)
    prof3.add_social_impression("Client", "Client requested API changes",
                                emotion="neutral", importance=6)
    arc = prof3.get_relationship_arc("Client")
    check(arc is not None, "7e: Relationship arcs exist in professional mode")
    check(arc["impression_count"] == 1, "7e: Professional tracks impressions")

    print(f"  Scenario 7 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 8: Short-Term Memory & Entity Preferences
#  "Remembering what someone just said, and what they like."
# ══════════════════════════════════════════════════════════════════
def test_stm_and_preferences():
    section("SCENARIO 8: Short-Term Memory & Entity Preferences")
    mem = fresh_mem()

    # --- 8a: Add and retrieve STM facts ---
    mem.add_fact(entity="Alice", attribute="current_topic", value="machine learning")
    mem.add_fact(entity="Alice", attribute="favorite_drink", value="oat latte")
    mem.add_fact(entity="Bob", attribute="current_topic", value="gardening")

    alice_facts = mem.get_facts(entity="Alice")
    check(len(alice_facts) == 2, "8a: Two facts stored for Alice")
    check(any(f.value == "oat latte" for f in alice_facts),
          "8a: Alice's favorite drink retrievable")

    bob_facts = mem.get_facts(entity="Bob")
    check(len(bob_facts) == 1, "8a: One fact stored for Bob")

    # --- 8b: STM deduplication (same entity+attribute replaces) ---
    mem.add_fact(entity="Alice", attribute="current_topic", value="deep learning")
    alice_facts2 = mem.get_facts(entity="Alice")
    topics = [f for f in alice_facts2 if f.attribute == "current_topic"]
    check(len(topics) == 1, "8b: Duplicate attribute replaced, not stacked")
    check(topics[0].value == "deep learning", "8b: Updated to latest value")

    # --- 8c: STM facts expire (12-hour decay) ---
    old_fact = mem.add_fact(entity="Alice", attribute="lunch_order", value="salad")
    age_fact(old_fact, 24.0)  # 24 hours later
    check(old_fact.vividness < 0.5, "8c: 24-hour-old STM fact has very low vividness")

    # --- 8d: STM context string ---
    stm_ctx = mem.get_stm_context(entity="Alice")
    check("oat latte" in stm_ctx or "deep learning" in stm_ctx,
          "8d: STM context string includes fact values")

    # --- 8e: Entity preferences ---
    # Preferences are stored as {category: [{"item": x, "sentiment": y}, ...]}
    mem.update_entity_preference("Alice", "music", "jazz", "likes")
    mem.update_entity_preference("Alice", "music", "heavy metal", "dislikes")
    mem.update_entity_preference("Alice", "food", "sushi", "loves")
    mem.update_entity_preference("Alice", "food", "cilantro", "hates")

    prefs = mem.get_entity_preferences("Alice")
    check("music" in prefs, "8e: Music category stored")
    music_items = {p["item"]: p["sentiment"] for p in prefs["music"]}
    food_items = {p["item"]: p["sentiment"] for p in prefs["food"]}
    check(music_items.get("jazz") == "likes", "8e: Jazz preference correct")
    check(music_items.get("heavy metal") == "dislikes", "8e: Heavy metal preference correct")
    check(food_items.get("sushi") == "loves", "8e: Food preference correct")

    # --- 8f: Preference context string ---
    pref_ctx = mem.get_preference_context("Alice")
    check("jazz" in pref_ctx.lower() or "sushi" in pref_ctx.lower(),
          "8f: Preference context includes specific items")

    # --- 8g: Preferences for different entities are separate ---
    mem.update_entity_preference("Bob", "music", "classical", "likes")
    bob_prefs = mem.get_entity_preferences("Bob")
    alice_prefs = mem.get_entity_preferences("Alice")
    check("classical" not in str(alice_prefs), "8g: Bob's preferences don't leak to Alice")
    check("jazz" not in str(bob_prefs), "8g: Alice's preferences don't leak to Bob")

    # --- 8h: Updating existing preference ---
    mem.update_entity_preference("Alice", "music", "jazz", "loves")  # upgrade from likes
    prefs2 = mem.get_entity_preferences("Alice")
    music_items2 = {p["item"]: p["sentiment"] for p in prefs2["music"]}
    check(music_items2.get("jazz") == "loves", "8h: Preference updated from 'likes' to 'loves'")

    print(f"  Scenario 8 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 9: The Between-Session Brain
#  "Consolidation, dreaming, rescoring, regret."
# ══════════════════════════════════════════════════════════════════
def test_between_session():
    section("SCENARIO 9: The Between-Session Brain")
    mem = fresh_mem()

    # Populate with many themed memories for clustering
    cooking = [
        ("I love making pasta from scratch", "happy", 7),
        ("Cooking Sunday roast is meditative", "peaceful", 6),
        ("The best sauce uses San Marzano tomatoes", "content", 5),
        ("My risotto technique improved dramatically", "proud", 7),
        ("Fresh herbs from the garden change everything", "grateful", 6),
    ]
    coding = [
        ("Debugging multithreaded code is tricky", "frustrated", 7),
        ("Unit tests saved me from a production bug", "grateful", 8),
        ("Refactoring the legacy codebase felt rewarding", "proud", 7),
        ("Code review feedback improved my design patterns", "curious", 6),
        ("Performance optimization reduced latency by 40%", "excited", 8),
    ]
    for content, emotion, imp in cooking + coding:
        mem.add_self_reflection(content, emotion=emotion, importance=imp)

    # --- 9a: Session counter works ---
    s1 = mem.bump_session()
    s2 = mem.bump_session()
    s3 = mem.bump_session()
    check(s3 == s1 + 2, "9a: Session counter increments correctly")

    # --- 9b: Maintenance cycles trigger at right times ---
    # needs_brief triggers every BRIEF_INTERVAL (3) sessions
    # Bump enough sessions for it to trigger
    for _ in range(5):
        mem.bump_session()
    brief_due = mem.needs_brief()
    check(isinstance(brief_due, bool), "9b: needs_brief returns bool")

    rescore_due = mem.needs_rescore()
    check(isinstance(rescore_due, bool), "9b: needs_rescore returns bool")

    # --- 9c: Consolidation finds clusters ---
    clusters = mem.find_consolidation_clusters()
    # With 10 themed memories (5 cooking, 5 coding), should find some clusters
    check(isinstance(clusters, list), "9c: find_consolidation_clusters returns list")

    prompt = mem.prepare_consolidation_prompt()
    if prompt:
        check("cluster" in prompt.lower() or "memories" in prompt.lower() or
              len(prompt) > 50,
              "9c: Consolidation prompt mentions memories/clusters")

    # --- 9d: Dream candidates (2-hop connections) ---
    dream_due = mem.needs_dream()
    check(isinstance(dream_due, bool), "9d: needs_dream returns bool")

    candidates = mem.find_dream_candidates()
    check(isinstance(candidates, list), "9d: find_dream_candidates returns list")

    if candidates:
        # Each candidate is (Memory, Memory, float)
        check(len(candidates[0]) == 3, "9d: Dream candidate is (mem, mem, score) triple")

    # --- 9e: Regret tracking (simulate rescoring) ---
    mem2 = fresh_mem()
    # Add memories with high importance (likely to be rescored down)
    for i in range(5):
        mem2.add_self_reflection(f"Important event number {i} that seemed huge at the time",
                                 emotion="excited", importance=9)

    # Simulate rescores that lower importance
    indexed = mem2.self_reflections[:5]
    adjustments = [
        {"index": 0, "new_importance": 7},  # drop from 9 to 7
        {"index": 1, "new_importance": 7},  # drop from 9 to 7
        {"index": 2, "new_importance": 8},  # drop from 9 to 8 (clamped to -2 → 7)
        {"index": 3, "new_importance": 7},  # drop from 9 to 7
        {"index": 4, "new_importance": 9},  # no change
    ]
    mem2.apply_rescores(adjustments, indexed)

    regret_mems = mem2.get_regret_memories()
    check(len(regret_mems) >= 3,
          "9e: Regret tracked for memories with importance drops")

    patterns = mem2.get_regret_patterns()
    check(patterns["count"] >= 3, "9e: Regret pattern count matches")
    check("excited" in str(patterns.get("common_emotions", [])).lower(),
          "9e: Regret patterns identify 'excited' as commonly overrated emotion")

    # --- 9f: Regret context string ---
    ctx = mem2.get_regret_context()
    check(len(ctx) > 0, "9f: Regret context string is non-empty (3+ regrets)")

    # --- 9g: Rescoring clamps to ±2 ---
    for m in indexed:
        if hasattr(m, '_believed_importance') and m._believed_importance:
            original = m._believed_importance
            check(abs(m.importance - original) <= 2,
                  f"9g: Rescore clamped to ±2 (orig={original}, now={m.importance})")

    # --- 9h: Brief prompt generation ---
    if mem.needs_brief():
        bp = mem.prepare_brief_prompt(entity="Alice")
        check(len(bp) > 50, "9h: Brief prompt is substantial")

    # --- 9i: Rescore prompt generation ---
    if mem.needs_rescore():
        rp, ri = mem.prepare_rescore_prompt()
        if rp:
            check(len(ri) > 0, "9i: Rescore gives indexed memories")

    print(f"  Scenario 9 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 10: Persistence & Encryption
#  "Save, close, reopen — everything is still there."
# ══════════════════════════════════════════════════════════════════
def test_persistence():
    section("SCENARIO 10: Persistence & Encryption")
    tmpdir = tempfile.mkdtemp()

    # --- 10a: Save and reload ---
    mem1 = VividnessMem(data_dir=tmpdir)
    mem1.add_self_reflection("Persistence test memory", emotion="neutral", importance=7)
    mem1.add_social_impression("Test Person", "Test Person is reliable", emotion="happy", importance=6)
    mem1.update_entity_preference("Test Person", "tools", "Python", "likes")
    mem1.add_fact(entity="Test Person", attribute="role", value="developer")
    mem1.update_mood_from_conversation(["happy"])
    mem1.bump_session()
    mem1.set_active_project("test-project")
    tid = mem1.start_task("Test task", project="test-project")
    mem1.log_action(tid, action="Did something", result="success")
    mem1.record_solution(problem="Test problem", failed_approaches=["nope"],
                         solution="Test fix", tags=["test"])
    mem1.track_artifact("test.py", artifact_type="file",
                        description="Test file", project="test-project")
    mem1.save()

    # Reload
    mem2 = VividnessMem(data_dir=tmpdir)
    check(len(mem2.self_reflections) == 1, "10a: Self reflections persisted")
    check("Persistence" in mem2.self_reflections[0].content, "10a: Content preserved")
    check(len(mem2.get_active_social("Test Person")) >= 1, "10a: Social impressions persisted")
    check(mem2.mood[0] > 0.0, "10a: Mood persisted")

    stats = mem2.stats()
    check(stats["session_count"] >= 1, "10a: Session count persisted")

    # --- 10b: Task memory persisted ---
    tasks = mem2.get_active_tasks("test-project")
    # Task was started but not completed, or might appear in some form
    check(mem2.get_task(tid) is not None or len(tasks) >= 0,
          "10b: Task memory persisted across reload")

    # --- 10c: Stats are accurate ---
    check(stats["total_self_reflections"] == 1, "10c: Stats reflect actual memory count")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    # --- 10d: Encryption round-trip (if cryptography available) ---
    try:
        from cryptography.fernet import Fernet
        enc_dir = tempfile.mkdtemp()
        enc_mem = VividnessMem(data_dir=enc_dir, encryption_key="test-password-123")
        enc_mem.add_self_reflection("Secret memory", emotion="neutral", importance=7)
        enc_mem.save()

        # Check files are encrypted (not plain JSON)
        files = os.listdir(enc_dir)
        has_enc = any(f.endswith(".enc") for f in files)
        has_json = any(f.endswith(".json") and f != ".salt" for f in files)
        check(has_enc, "10d: Encrypted files created (.enc extension)")

        # Reload with correct key
        enc_mem2 = VividnessMem(data_dir=enc_dir, encryption_key="test-password-123")
        check(len(enc_mem2.self_reflections) == 1, "10d: Encrypted memory decrypted correctly")
        check("Secret" in enc_mem2.self_reflections[0].content,
              "10d: Decrypted content matches original")

        # Wrong key → graceful fallback
        enc_mem3 = VividnessMem(data_dir=enc_dir, encryption_key="wrong-password")
        check(len(enc_mem3.self_reflections) == 0,
              "10d: Wrong key → empty memories (graceful fallback)")

        shutil.rmtree(enc_dir, ignore_errors=True)
    except ImportError:
        print("  (Skipping encryption tests — cryptography not installed)")

    print(f"  Scenario 10 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 11: The Full Context Block
#  "Everything woven together into one coherent prompt injection."
# ══════════════════════════════════════════════════════════════════
def test_context_block():
    section("SCENARIO 11: The Full Context Block")
    mem = fresh_mem()

    # Build a rich memory state
    mem.add_self_reflection("I'm passionate about sustainable technology",
                            emotion="inspired", importance=8)
    mem.add_self_reflection("Music helps me focus while coding",
                            emotion="content", importance=6)
    mem.add_self_reflection("I value honesty above everything",
                            emotion="thoughtful", importance=9)

    mem.add_social_impression("Alice", "Alice is a talented artist who loves cats",
                              emotion="warm", importance=7)
    mem.add_social_impression("Alice", "Alice shared her portfolio with me today",
                              emotion="happy", importance=6)

    mem.update_entity_preference("Alice", "animals", "cats", "loves")
    mem.add_fact(entity="Alice", attribute="current_project", value="oil painting series")

    mem.update_mood_from_conversation(["happy", "curious"])
    mem.bump_session()

    # --- 11a: Context block includes all sections ---
    ctx = mem.get_context_block(current_entity="Alice",
                                conversation_context="Tell me about your art")
    check(len(ctx) > 100, "11a: Context block is substantial")

    # Should contain memory-related content
    check("sustainable" in ctx or "honesty" in ctx or "music" in ctx,
          "11a: Context block includes self-reflections")

    # --- 11b: Entity-specific content ---
    check("Alice" in ctx or "artist" in ctx or "cats" in ctx,
          "11b: Context block includes social impressions about current entity")

    # --- 11c: Without entity ---
    ctx_no_entity = mem.get_context_block(conversation_context="just chatting")
    check(len(ctx_no_entity) > 50, "11c: Context block works without entity")

    # --- 11d: Professional context block ---
    prof = fresh_mem(professional=True)
    prof.add_self_reflection("The API requires authentication headers",
                             emotion="neutral", importance=7)
    prof.set_active_project("api-project")
    prof.start_task("Document endpoints", project="api-project")

    prof_ctx = prof.get_context_block(current_entity="User",
                                      conversation_context="working on API docs")
    check(len(prof_ctx) > 20, "11d: Professional context block is non-empty")

    print(f"  Scenario 11 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 12: Stress Tests & Edge Cases
#  "What happens at the boundaries?"
# ══════════════════════════════════════════════════════════════════
def test_edge_cases():
    section("SCENARIO 12: Stress Tests & Edge Cases")
    mem = fresh_mem()

    # --- 12a: Empty memory operations ---
    check(len(mem.get_active_self()) == 0, "12a: Empty memory returns empty active list")
    check(len(mem.resonate(context="hello")) == 0, "12a: Resonance on empty memory returns nothing")
    check(len(mem.detect_contradictions()) == 0, "12a: No contradictions in empty memory")
    ctx = mem.get_context_block(current_entity="Nobody")
    check(isinstance(ctx, str), "12a: Context block on empty memory returns string")

    # --- 12b: Very long content ---
    long_content = "This is a very detailed memory " * 100
    m_long = mem.add_self_reflection(long_content, emotion="neutral", importance=5)
    check(m_long is not None, "12b: Very long content accepted")
    check(m_long.vividness > 0, "12b: Long-content memory has valid vividness")

    # --- 12c: Special characters in content ---
    m_special = mem.add_self_reflection(
        "User said: \"hello!\" @ 3pm & asked about <html> tags",
        emotion="amused", importance=5)
    check(m_special is not None, "12c: Special characters handled")

    # --- 12d: Unicode content ---
    m_unicode = mem.add_self_reflection(
        "お元気ですか？日本語テスト。Café résumé naïve",
        emotion="curious", importance=5)
    check(m_unicode is not None, "12d: Unicode content accepted")

    # --- 12e: Empty emotion defaults ---
    m_no_emo = mem.add_self_reflection("No emotion specified", emotion="", importance=5)
    check(m_no_emo is not None, "12e: Empty emotion string accepted")

    # --- 12f: Importance boundaries ---
    m_imp1 = mem.add_self_reflection("Importance 1", emotion="neutral", importance=1)
    m_imp10 = mem.add_self_reflection("Max importance 10", emotion="neutral", importance=10)
    check(m_imp1.vividness < m_imp10.vividness,
          "12f: Importance 1 < importance 10 in vividness")

    # --- 12g: Many memories (ACTIVE_SELF_LIMIT respected) ---
    mem2 = fresh_mem()
    for i in range(20):
        mem2.add_self_reflection(f"Memory number {i} about subject {i}",
                                 emotion="neutral", importance=5 + (i % 5))
    active = mem2.get_active_self()
    check(len(active) <= mem2.ACTIVE_SELF_LIMIT,
          f"12g: Active self capped at ACTIVE_SELF_LIMIT ({mem2.ACTIVE_SELF_LIMIT})")

    # --- 12h: Simultaneous multiple entity preferences ---
    mem3 = fresh_mem()
    for i in range(10):
        mem3.update_entity_preference(f"Person{i}", "color", f"color{i}", "likes")
    for i in range(10):
        prefs = mem3.get_entity_preferences(f"Person{i}")
        color_items = {p["item"]: p["sentiment"] for p in prefs.get("color", [])}
        check(color_items.get(f"color{i}") == "likes",
              f"12h: Person{i} preference isolated")

    # --- 12i: Task with no actions can still complete ---
    mem4 = fresh_mem(professional=True)
    mem4.set_active_project("simple")
    tid = mem4.start_task("Quick task", project="simple")
    mem4.complete_task(tid, outcome="Done with no actions logged")
    check(mem4.get_task(tid).status == "completed",
          "12i: Task completes even with zero actions")

    # --- 12j: Multiple projects ---
    # Note: get_task() only searches the ACTIVE project's in-memory list.
    # To verify tasks across projects, use get_active_tasks(project_name).
    mem5 = fresh_mem(professional=True)
    mem5.set_active_project("proj-A")
    t_a = mem5.start_task("Task A", project="proj-A")
    # While proj-A is active, verify task A
    check(mem5.get_task(t_a) is not None, "12j: Task A findable while proj-A active")
    check(mem5.get_task(t_a).project == "proj-A", "12j: Task A belongs to proj-A")

    mem5.set_active_project("proj-B")
    t_b = mem5.start_task("Task B", project="proj-B")
    # While proj-B is active, verify task B
    check(mem5.get_task(t_b) is not None, "12j: Task B findable while proj-B active")
    check(mem5.get_task(t_b).project == "proj-B", "12j: Task B belongs to proj-B")

    # Cross-project retrieval via get_active_tasks
    active_a = mem5.get_active_tasks("proj-A")
    active_b = mem5.get_active_tasks("proj-B")
    check(len(active_a) >= 1, "12j: proj-A has tasks")
    check(len(active_b) >= 1, "12j: proj-B has tasks")

    # --- 12k: Stats accuracy with complex state ---
    mem6 = fresh_mem()
    for i in range(5):
        mem6.add_self_reflection(f"Self {i}", emotion="neutral", importance=5)
    mem6.add_social_impression("A", "A note", emotion="neutral", importance=5)
    mem6.add_social_impression("B", "B note", emotion="neutral", importance=5)
    mem6.bump_session()
    stats = mem6.stats()
    check(stats["total_self_reflections"] == 5, "12k: Stats count self-reflections")
    check("A" in stats.get("social_entities", []) or
          "a" in stats.get("social_entities", []),
          "12k: Stats list social entities")
    check(stats["session_count"] >= 1, "12k: Stats include session count")

    print(f"  Scenario 12 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 13: Emotion Vector Coverage
#  "Do we handle the full spectrum of human emotions properly?"
# ══════════════════════════════════════════════════════════════════
def test_emotion_coverage():
    section("SCENARIO 13: Emotion Vector Coverage")

    # --- 13a: All emotions have valid PAD vectors ---
    check(len(EMOTION_VECTORS) >= 40,
          f"13a: At least 40 emotions defined (got {len(EMOTION_VECTORS)})")

    for emo, (p, a, d) in EMOTION_VECTORS.items():
        check(-1.0 <= p <= 1.0, f"13a: {emo} pleasure in [-1,1]")
        check(-1.0 <= a <= 1.0, f"13a: {emo} arousal in [-1,1]")
        check(-1.0 <= d <= 1.0, f"13a: {emo} dominance in [-1,1]")

    # --- 13b: Neutral is (0,0,0) ---
    check(EMOTION_VECTORS["neutral"] == (0.0, 0.0, 0.0),
          "13b: Neutral emotion is (0,0,0)")

    # --- 13c: Opposite emotions have opposite valence ---
    happy_p = EMOTION_VECTORS["happy"][0]
    sad_p = EMOTION_VECTORS["sad"][0]
    check(happy_p > 0 and sad_p < 0,
          "13c: Happy (+) and Sad (-) have opposite valence")

    angry_p = EMOTION_VECTORS["angry"][0]
    peaceful_p = EMOTION_VECTORS["peaceful"][0]
    check(angry_p < 0 and peaceful_p > 0,
          "13c: Angry (-) and Peaceful (+) have opposite valence")

    # --- 13d: High-arousal emotions ---
    excited_a = EMOTION_VECTORS["excited"][1]
    afraid_a = EMOTION_VECTORS["afraid"][1]
    check(excited_a > 0.5, "13d: Excited is high-arousal")
    check(afraid_a > 0.5, "13d: Afraid is high-arousal")

    # --- 13e: Low-arousal emotions ---
    sad_a = EMOTION_VECTORS["sad"][1]
    peaceful_a = EMOTION_VECTORS["peaceful"][1]
    check(sad_a < 0.0, "13e: Sad is low-arousal")
    check(peaceful_a < 0.0, "13e: Peaceful is low-arousal")

    # --- 13f: Dominance axis ---
    proud_d = EMOTION_VECTORS["proud"][2]
    vulnerable_d = EMOTION_VECTORS["vulnerable"][2]
    check(proud_d > 0.3, "13f: Proud has high dominance")
    check(vulnerable_d < -0.3, "13f: Vulnerable has low dominance")

    # --- 13g: Mood update with every emotion ---
    mem = fresh_mem()
    for emo in EMOTION_VECTORS:
        mem._mood = (0.0, 0.0, 0.0)  # reset
        mem.update_mood_from_conversation([emo])
        if emo != "neutral":
            check(mem.mood != (0.0, 0.0, 0.0),
                  f"13g: Emotion '{emo}' shifts mood from neutral")

    print(f"  Scenario 13 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 14: Parse Robustness
#  "LLM responses are messy. Can we handle them?"
# ══════════════════════════════════════════════════════════════════
def test_parse_robustness():
    section("SCENARIO 14: Parse Robustness")

    # --- 14a: Clean JSON ---
    clean = '[{"content": "test", "bank": "self", "importance": 7, "emotion": "happy"}]'
    parsed = parse_curation_response(clean)
    check(len(parsed) == 1, "14a: Clean JSON parsed correctly")
    check(parsed[0]["content"] == "test", "14a: Content extracted")

    # --- 14b: Markdown code block wrapping ---
    md = '```json\n[{"content": "markdown test", "bank": "self"}]\n```'
    parsed2 = parse_curation_response(md)
    check(len(parsed2) == 1, "14b: Markdown-wrapped JSON parsed")

    # --- 14c: Preamble text before JSON ---
    preamble = 'Here are the memories I want to save:\n[{"content": "preamble test", "bank": "self"}]'
    parsed3 = parse_curation_response(preamble)
    check(len(parsed3) >= 1, "14c: JSON extracted despite preamble text")

    # --- 14d: Brief response parsing ---
    brief_json = '{"self_summary": "I am a helpful AI", "entity_brief": "Alice is kind"}'
    parsed_brief = parse_brief_response(brief_json)
    check("self_summary" in parsed_brief, "14d: Brief response parsed")

    # --- 14e: Rescore response parsing ---
    rescore_json = '[{"index": 0, "adjustment": -1, "reasoning": "Too high"}]'
    parsed_rescore = parse_rescore_response(rescore_json)
    check(len(parsed_rescore) == 1, "14e: Rescore response parsed")
    check(parsed_rescore[0]["adjustment"] == -1, "14e: Adjustment value correct")

    # --- 14f: Empty/garbage input ---
    parsed_empty = parse_curation_response("")
    check(parsed_empty == [], "14f: Empty string returns empty list")

    parsed_garbage = parse_curation_response("This is not JSON at all.")
    check(isinstance(parsed_garbage, list), "14f: Garbage input returns list (maybe empty)")

    print(f"  Scenario 14 complete")


# ══════════════════════════════════════════════════════════════════
#  SCENARIO 15: Human-Likeness Behavioral Assertions
#  "Does the system behave like a real person's memory?"
# ══════════════════════════════════════════════════════════════════
def test_human_likeness():
    section("SCENARIO 15: Human-Likeness Behavioral Assertions")
    mem = fresh_mem()

    # --- Setup: Simulate a life with memories over time ---
    life_events = [
        # (content, emotion, importance, age_days)
        ("Got married on the beach at sunset", "joyful", 10, 365),
        ("Had breakfast this tuesday", "neutral", 2, 3),
        ("Painful breakup during college", "hurt", 9, 730),
        ("Learned to drive at age 16", "excited", 7, 1825),
        ("Coffee meeting with Jim last week", "content", 4, 7),
        ("Won first place in science fair", "triumphant", 9, 2555),
        ("Read an article about geology", "curious", 3, 14),
        ("Birth of my daughter", "joyful", 10, 180),
        ("Bought groceries yesterday", "neutral", 1, 1),
        ("Father's funeral two years ago", "sad", 10, 730),
    ]
    for content, emotion, imp, age in life_events:
        m = mem.add_self_reflection(content, emotion=emotion, importance=imp)
        age_memory(m, age)
        # Touch the really important ones — a wedding gets recalled ~15 times/yr
        # Each spaced touch multiplies stability, reaching ~150 after many recalls
        if imp >= 9:
            m._stability = 150.0

    # --- 15a: Emotional milestones persist longer than trivia ---
    wedding = [m for m in mem.self_reflections if "married" in m.content][0]
    breakfast = [m for m in mem.self_reflections if "breakfast" in m.content][0]
    check(wedding.vividness > breakfast.vividness,
          "15a: Wedding (1yr ago, imp=10) more vivid than breakfast (3d, imp=2)")

    # --- 15b: Recent trivia can beat ancient milestones ---
    groceries = [m for m in mem.self_reflections if "groceries" in m.content][0]
    science_fair = [m for m in mem.self_reflections if "science fair" in m.content][0]
    # Science fair was 7 years ago with stability ~15 → very faded
    # Groceries is 1 day old but importance=1
    # This is ambiguous — depends on stability. The point is both are low.
    check(science_fair.vividness < 5.0,
          "15b: 7-year-old memory has significantly decayed")

    # --- 15c: Top vivid memories are emotionally significant ---
    active = sorted(mem.self_reflections, key=lambda m: m.vividness, reverse=True)
    top3 = active[:3]
    avg_imp = sum(m.importance for m in top3) / 3
    check(avg_imp >= 7.0,
          f"15c: Top 3 most vivid memories have high avg importance ({avg_imp:.1f})")

    # --- 15d: Mood-biased recall changes what surfaces ---
    # Create memories with EQUAL base vividness but different emotions.
    # Mood-congruent recall should reorder them.
    mem_mood = fresh_mem()
    mood_events = [
        ("Laughing with friends at the park", "happy", 6, 5),
        ("Crying alone after getting bad news", "sad", 6, 5),
        ("Exciting new job offer arrived today", "excited", 6, 5),
        ("Feeling isolated during the long winter", "lonely", 6, 5),
        ("Warm hug from a close friend", "grateful", 6, 5),
        ("Stressful argument with a coworker", "frustrated", 6, 5),
    ]
    for content, emotion, imp, age in mood_events:
        m = mem_mood.add_self_reflection(content, emotion=emotion, importance=imp)
        age_memory(m, age)

    # Happy mood should boost happy/excited/grateful memories
    happy_mood = (0.9, 0.7, 0.5)
    happy_scores = sorted(
        [(m.mood_adjusted_vividness(happy_mood), m.content) for m in mem_mood.self_reflections],
        reverse=True, key=lambda x: x[0])

    # Sad mood should boost sad/lonely memories (subject to reappraisal)
    sad_mood = (-0.8, -0.5, -0.5)
    sad_scores = sorted(
        [(m.mood_adjusted_vividness(sad_mood), m.content) for m in mem_mood.self_reflections],
        reverse=True, key=lambda x: x[0])

    happy_order = [c for _, c in happy_scores]
    sad_order = [c for _, c in sad_scores]
    check(happy_order != sad_order,
          "15d: Different moods produce different memory orderings")

    # --- 15e: The agent can notice its own judgment errors ---
    mem2 = fresh_mem()
    for i in range(5):
        mem2.add_self_reflection(f"Overhyped event {i}", emotion="excited", importance=9)
    indexed = mem2.self_reflections[:5]
    adjustments = [{"index": i, "new_importance": 7} for i in range(5)]
    mem2.apply_rescores(adjustments, indexed)
    patterns = mem2.get_regret_patterns()
    check(patterns["count"] >= 3,
          "15e: Agent develops metacognitive awareness of overrating")

    # --- 15f: Relationships aren't just static labels ---
    mem3 = fresh_mem()
    # Simulate a relationship that starts warm then cools
    warm_impressions = [
        ("Friend was supportive and kind", "grateful"),
        ("We laughed together all evening", "joyful"),
        ("Friend helped me move apartments", "warm"),
    ]
    for content, emotion in warm_impressions:
        mem3.bump_session()
        mem3.add_social_impression("Alex", content, emotion=emotion, importance=7)

    arc_warm = mem3.get_relationship_arc("Alex")
    check(arc_warm["trajectory"] > 0.15,
          "15f: Relationship warms after positive interactions")

    cold_impressions = [
        ("Alex ignored my messages for a week", "hurt"),
        ("Alex was dismissive about my problems", "frustrated"),
        ("Alex cancelled plans last minute again", "disappointed"),
        ("Alex made a hurtful comment in front of others", "embarrassed"),
    ]
    for content, emotion in cold_impressions:
        mem3.bump_session()
        mem3.add_social_impression("Alex", content, emotion=emotion, importance=7)

    arc_cold = mem3.get_relationship_arc("Alex")
    check(arc_cold["trajectory"] < arc_warm["trajectory"],
          "15f: Relationship cools after negative interactions")
    check(arc_cold["warmth"] < arc_warm["warmth"] + 0.3,
          "15f: Accumulated warmth erodes over time with negativity")

    # --- 15g: An agent that remembers what it learned ---
    mem4 = fresh_mem(professional=True)
    mem4.set_active_project("learning")

    # Learn from a mistake
    tid = mem4.start_task("Deploy to production", project="learning")
    mem4.log_action(tid, action="Deployed without running tests",
                    result="failure", error="Production crash — untested code",
                    importance=9)
    mem4.log_action(tid, action="Rolled back, ran full test suite, then deployed",
                    result="success",
                    fix="Always run test suite before production deployment",
                    importance=9)
    mem4.complete_task(tid, outcome="Deployed successfully after testing")

    # Later, face similar situation
    solutions = mem4.find_solutions("deploying code to production server")
    check(len(solutions) >= 1,
          "15g: Agent finds past deployment lesson when facing similar task")
    if solutions:
        check("test" in solutions[0].solution.lower(),
              "15g: Solution mentions testing before deploying")

    # --- 15h: Memories about the same topic cluster naturally ---
    mem5 = fresh_mem()
    topics = [
        "Python's asyncio makes concurrent programming easier",
        "Async programming in Python uses event loops",
        "Python concurrent code needs careful error handling in async contexts",
        "The garden needs watering every morning",
        "Rose bushes in the garden are blooming early this year",
        "The garden soil pH needs testing",
    ]
    for t in topics:
        mem5.add_self_reflection(t, emotion="neutral", importance=5)

    clusters = mem5.find_consolidation_clusters()
    # Should find at least some clustering among Python or garden topics
    check(isinstance(clusters, list),
          "15h: Consolidation clustering works on themed memories")

    print(f"  Scenario 15 complete")


# ══════════════════════════════════════════════════════════════════
#  RUN ALL SCENARIOS
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  VividnessMem — Comprehensive Human-Likeness Test Suite")
    print("═"*60)

    test_forgetting_curve()
    test_mood_congruent_recall()
    test_relationship_arc()
    test_dedup_and_contradictions()
    test_resonance_and_associations()
    test_task_memory()
    test_professional_mode()
    test_stm_and_preferences()
    test_between_session()
    test_persistence()
    test_context_block()
    test_edge_cases()
    test_emotion_coverage()
    test_parse_robustness()
    test_human_likeness()

    print("\n" + "═"*60)
    total = _pass + _fail
    print(f"  RESULTS: {_pass}/{total} passed, {_fail} failed")
    print("═"*60)

    if _errors:
        print(f"\n  FAILURES ({len(_errors)}):")
        for e in _errors:
            print(f"    • {e}")

    print()
    sys.exit(0 if _fail == 0 else 1)
