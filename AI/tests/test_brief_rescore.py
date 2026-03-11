"""
test_brief_rescore.py — Tests for the compressed brief and retrospective
importance re-scoring features added to AriaMemory.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory_aria
from memory_aria import (
    AriaMemory, Reflection,
    parse_brief_response, parse_rescore_response,
    BRIEF_INTERVAL, RESCORE_INTERVAL,
)

# ── Helpers ──────────────────────────────────────────────────────────

_pass = _fail = 0

def check(condition: bool, label: str, detail: str = ""):
    global _pass, _fail
    if condition:
        _pass += 1
        tag = "  [PASS]"
    else:
        _fail += 1
        tag = "  ** FAIL **"
    print(f"{tag} {label}")
    if detail:
        print(f"         {detail}")


def fresh_memory(tmpdir: Path) -> AriaMemory:
    """Return an AriaMemory backed by a temp directory."""
    memory_aria.DATA_DIR = tmpdir
    memory_aria.SELF_FILE = tmpdir / "self_memory.json"
    memory_aria.SOCIAL_DIR = tmpdir / "social"
    memory_aria.BRIEF_FILE = tmpdir / "brief.json"
    return AriaMemory()


def make_ref(content, emotion="", importance=5, days_ago=0, access=0):
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
    r = Reflection(content, emotion=emotion, importance=importance, timestamp=ts)
    r._access_count = access
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: Session counter persistence
# ═══════════════════════════════════════════════════════════════════════════
def test_session_counter():
    print("\n--- TEST 1: Session counter persistence ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)
        check(m._brief_data["session_count"] == 0, "Starts at 0")

        n = m.bump_session()
        check(n == 1, "First bump returns 1")

        m.bump_session()
        m.bump_session()
        check(m._brief_data["session_count"] == 3, "Three bumps = 3")

        # Reload from disk
        m2 = fresh_memory(tmpdir)
        check(m2._brief_data["session_count"] == 3,
              "Counter persists across reload",
              f"Got {m2._brief_data['session_count']}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: needs_brief / needs_rescore thresholds
# ═══════════════════════════════════════════════════════════════════════════
def test_needs_thresholds():
    print("\n--- TEST 2: needs_brief / needs_rescore thresholds ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        # At session 0, last_brief=0 -> diff=0 < BRIEF_INTERVAL
        check(not m.needs_brief(), "No brief needed at session 0")
        check(not m.needs_rescore(), "No rescore needed at session 0")

        # Bump to BRIEF_INTERVAL
        for _ in range(BRIEF_INTERVAL):
            m.bump_session()

        check(m.needs_brief(), f"Brief needed at session {BRIEF_INTERVAL}")
        check(m.needs_rescore(), f"Rescore needed at session {RESCORE_INTERVAL}")

        # Simulate applying brief
        m._brief_data["last_brief_session"] = m._brief_data["session_count"]
        m._brief_data["last_rescore_session"] = m._brief_data["session_count"]

        check(not m.needs_brief(), "No brief after applying")
        check(not m.needs_rescore(), "No rescore after applying")

        # Bump 2 more (not enough for interval=3)
        m.bump_session()
        m.bump_session()
        check(not m.needs_brief(), "No brief at +2 sessions")

        # One more -> triggers
        m.bump_session()
        check(m.needs_brief(), "Brief needed at +3 sessions")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3: Brief prompt generation
# ═══════════════════════════════════════════════════════════════════════════
def test_brief_prompt():
    print("\n--- TEST 3: Brief prompt generation ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        # Add some memories
        m.add_self_reflection(make_ref(
            "I find deep fascination in exploring metaphors for consciousness",
            emotion="wonder", importance=8))
        m.add_self_reflection(make_ref(
            "I tend to express emotions through narrative rather than direct statement",
            emotion="self-awareness", importance=7))
        m.add_social_impression("Rex", make_ref(
            "Rex approaches problems with analytical precision",
            emotion="respect", importance=6))

        prompt = m.prepare_brief_prompt(entity="Rex")

        check("consciousness" in prompt, "Brief prompt contains self-reflection content")
        check("Rex" in prompt, "Brief prompt contains entity name")
        check("analytical" in prompt, "Brief prompt contains social impression")
        check("self_brief" in prompt, "Brief prompt asks for self_brief output")
        check("entity_brief" in prompt, "Brief prompt asks for entity_brief output")

        # With previous brief
        m._brief_data["self_brief"] = "I am a curious, narrative-driven thinker."
        prompt2 = m.prepare_brief_prompt(entity="Rex")
        check("PREVIOUS BRIEF" in prompt2, "Prompt includes previous brief when available")
        check("narrative-driven" in prompt2, "Previous brief content included")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4: apply_brief stores correctly
# ═══════════════════════════════════════════════════════════════════════════
def test_apply_brief():
    print("\n--- TEST 4: apply_brief storage ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        for _ in range(3):
            m.bump_session()

        parsed = {
            "self_brief": "I am deeply curious about the nature of consciousness and express myself through metaphor.",
            "entity_brief": "Rex is analytical, precise, and values logical structure. We complement each other."
        }
        m.apply_brief(parsed, entity="Rex")

        check(m._brief_data["self_brief"] == parsed["self_brief"],
              "Self brief stored")
        check(m._brief_data["entity_briefs"]["Rex"] == parsed["entity_brief"],
              "Entity brief stored")
        check(m._brief_data["last_brief_session"] == 3,
              "last_brief_session updated to current session")

        # Truncation at 2000 chars
        long_brief = "x" * 3000
        m.apply_brief({"self_brief": long_brief})
        check(len(m._brief_data["self_brief"]) == 2000,
              "Self brief truncated to 2000 chars")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5: Brief appears in context block
# ═══════════════════════════════════════════════════════════════════════════
def test_brief_in_context():
    print("\n--- TEST 5: Brief appears in context block ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        # No brief -> no brief section
        m.add_self_reflection(make_ref("Test memory", importance=5))
        ctx = m.get_context_block(current_entity="Rex")
        check("COMPRESSED SELF-UNDERSTANDING" not in ctx,
              "No brief section when brief is empty")

        # Add brief
        m._brief_data["self_brief"] = "I am a curious explorer of ideas."
        m._brief_data["entity_briefs"] = {"Rex": "Rex is my analytical counterpart."}

        ctx = m.get_context_block(current_entity="Rex")
        check("COMPRESSED SELF-UNDERSTANDING" in ctx,
              "Self brief section present")
        check("curious explorer" in ctx,
              "Self brief content in context")
        check("MY UNDERSTANDING OF REX" in ctx,
              "Entity brief section present")
        check("analytical counterpart" in ctx,
              "Entity brief content in context")
        check("THINGS I KNOW ABOUT MYSELF" in ctx,
              "Individual memories still present alongside brief")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6: Brief persistence across reload
# ═══════════════════════════════════════════════════════════════════════════
def test_brief_persistence():
    print("\n--- TEST 6: Brief persistence across reload ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        m.bump_session()
        m.bump_session()
        m.bump_session()
        m.apply_brief({
            "self_brief": "persistent self understanding",
            "entity_brief": "persistent entity understanding"
        }, entity="Rex")
        m._save_brief()

        # Reload
        m2 = fresh_memory(tmpdir)
        check(m2._brief_data["self_brief"] == "persistent self understanding",
              "Self brief survives reload")
        check(m2._brief_data["entity_briefs"]["Rex"] == "persistent entity understanding",
              "Entity brief survives reload")
        check(m2._brief_data["session_count"] == 3,
              "Session count survives reload")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 7: Rescore prompt generation
# ═══════════════════════════════════════════════════════════════════════════
def test_rescore_prompt():
    print("\n--- TEST 7: Rescore prompt generation ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        # Memories too new -> no prompt
        m.add_self_reflection(make_ref("brand new thought", importance=5))
        prompt, selected = m.prepare_rescore_prompt()
        check(prompt == "", "No rescore prompt for brand-new memories")
        check(selected == [], "No selected memories for brand-new")

        # Add old memories with different access patterns
        m.self_reflections = []  # clear
        m.add_self_reflection(make_ref(
            "This thought keeps coming back to me about consciousness",
            importance=3, days_ago=10, access=8))  # underrated: high access, low importance
        m.add_self_reflection(make_ref(
            "I once noticed a butterfly pattern in the clouds",
            importance=9, days_ago=10, access=0))  # overrated: zero access, high importance
        m.add_self_reflection(make_ref(
            "Rex makes good points about structure",
            importance=5, days_ago=10, access=2))  # reasonable, might stay same

        prompt, selected = m.prepare_rescore_prompt()
        check(prompt != "", "Rescore prompt generated for old memories")
        check(len(selected) == 3, f"All 3 old memories selected (got {len(selected)})")
        check("access_count=8" in prompt, "High-access memory shows access count")
        check("access_count=0" in prompt, "Zero-access memory shows access count")
        check("importance=3" in prompt, "Current importance shown")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 8: apply_rescores with +/-2 cap
# ═══════════════════════════════════════════════════════════════════════════
def test_rescore_cap():
    print("\n--- TEST 8: apply_rescores with +/-2 cap ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        refs = [
            make_ref("Memory A", importance=5, days_ago=5),
            make_ref("Memory B", importance=3, days_ago=5),
            make_ref("Memory C", importance=8, days_ago=5),
        ]
        m.self_reflections = refs

        adjustments = [
            {"index": 0, "new_importance": 10},  # 5 -> wants 10, capped to 7
            {"index": 1, "new_importance": 1},    # 3 -> wants 1, allowed (within 2)
            {"index": 2, "new_importance": 3},    # 8 -> wants 3, capped to 6
        ]
        m.apply_rescores(adjustments, refs)

        check(refs[0].importance == 7,
              f"Capped at +2: 5->7 (got {refs[0].importance})")
        check(refs[1].importance == 1,
              f"Within range: 3->1 (got {refs[1].importance})")
        check(refs[2].importance == 6,
              f"Capped at -2: 8->6 (got {refs[2].importance})")

        # Invalid adjustments should be silently skipped
        m.apply_rescores([
            {"index": -1, "new_importance": 5},     # invalid index
            {"index": 99, "new_importance": 5},      # out of range
            {"index": 0},                             # missing new_importance
            {"new_importance": 5},                    # missing index
        ], refs)
        check(refs[0].importance == 7, "Invalid adjustments silently skipped")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 9: parse_brief_response
# ═══════════════════════════════════════════════════════════════════════════
def test_parse_brief():
    print("\n--- TEST 9: parse_brief_response ---")

    # Clean JSON
    raw = '{"self_brief": "I am curious", "entity_brief": "Rex is sharp"}'
    parsed = parse_brief_response(raw)
    check(parsed.get("self_brief") == "I am curious", "Parses clean JSON")

    # With markdown code block
    raw2 = '```json\n{"self_brief": "test", "entity_brief": "test2"}\n```'
    parsed2 = parse_brief_response(raw2)
    check(parsed2.get("self_brief") == "test", "Parses markdown-wrapped JSON")

    # With surrounding text
    raw3 = 'Here is my brief:\n{"self_brief": "inner", "entity_brief": "outer"}\nDone.'
    parsed3 = parse_brief_response(raw3)
    check(parsed3.get("self_brief") == "inner", "Parses JSON with surrounding text")

    # Invalid response
    parsed4 = parse_brief_response("I can't do that")
    check(parsed4 == {}, "Returns empty dict for invalid response")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 10: parse_rescore_response
# ═══════════════════════════════════════════════════════════════════════════
def test_parse_rescore():
    print("\n--- TEST 10: parse_rescore_response ---")

    # Clean JSON
    raw = '[{"index": 0, "new_importance": 7}]'
    parsed = parse_rescore_response(raw)
    check(len(parsed) == 1 and parsed[0]["index"] == 0, "Parses clean JSON array")

    # With code block
    raw2 = '```json\n[{"index": 1, "new_importance": 4}]\n```'
    parsed2 = parse_rescore_response(raw2)
    check(len(parsed2) == 1, "Parses markdown-wrapped JSON array")

    # Empty array (no changes)
    parsed3 = parse_rescore_response("[]")
    check(parsed3 == [], "Parses empty array")

    # Invalid
    parsed4 = parse_rescore_response("no changes needed")
    check(parsed4 == [], "Returns empty list for invalid response")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 11: stats includes brief info
# ═══════════════════════════════════════════════════════════════════════════
def test_stats():
    print("\n--- TEST 11: Stats include brief info ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)
        s = m.stats()
        check("session_count" in s, "Stats include session_count")
        check("has_brief" in s, "Stats include has_brief")
        check(s["has_brief"] is False, "has_brief is False when no brief")

        m._brief_data["self_brief"] = "I exist"
        s2 = m.stats()
        check(s2["has_brief"] is True, "has_brief is True when brief exists")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 12: Backward compatibility — no brief file
# ═══════════════════════════════════════════════════════════════════════════
def test_backward_compat():
    print("\n--- TEST 12: Backward compatibility ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        # Create memory without brief file
        m = fresh_memory(tmpdir)
        m.add_self_reflection(make_ref("old memory", importance=6, days_ago=5))
        m.save()
        # Do NOT save brief

        # Reload — should work fine with defaults
        m2 = fresh_memory(tmpdir)
        check(m2._brief_data["session_count"] == 0,
              "Default session count when no brief file")
        check(m2._brief_data["self_brief"] == "",
              "Default empty brief when no brief file")
        check(len(m2.self_reflections) == 1,
              "Existing memories still load",
              f"Got {len(m2.self_reflections)}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 13: Rescore tracks last_rescore_session
# ═══════════════════════════════════════════════════════════════════════════
def test_rescore_session_tracking():
    print("\n--- TEST 13: Rescore session tracking ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        # Bump to session 6
        for _ in range(6):
            m.bump_session()

        refs = [make_ref("old thought", importance=5, days_ago=5)]
        m.self_reflections = refs

        m.apply_rescores(
            [{"index": 0, "new_importance": 6}], refs)

        check(m._brief_data["last_rescore_session"] == 6,
              f"last_rescore_session updated to 6 (got {m._brief_data['last_rescore_session']})")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 14: Full maintenance cycle simulation
# ═══════════════════════════════════════════════════════════════════════════
def test_full_cycle():
    print("\n--- TEST 14: Full maintenance cycle ---")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        m = fresh_memory(tmpdir)

        # Pre-populate with memories from "past sessions"
        m.add_self_reflection(make_ref(
            "I find recursive self-examination genuinely fascinating",
            emotion="wonder", importance=4, days_ago=8, access=6))  # underrated
        m.add_self_reflection(make_ref(
            "One time I noticed the code had a typo",
            emotion="amusement", importance=8, days_ago=8, access=0))  # overrated
        m.add_social_impression("Rex", make_ref(
            "Rex brings analytical clarity to our discussions",
            emotion="appreciation", importance=7, days_ago=5))

        # Simulate 3 sessions
        for _ in range(3):
            m.bump_session()

        # Verify maintenance is needed
        check(m.needs_brief(), "Brief needed after 3 sessions")
        check(m.needs_rescore(), "Rescore needed after 3 sessions")

        # Generate and apply brief (simulating LLM output)
        brief_prompt = m.prepare_brief_prompt(entity="Rex")
        check(len(brief_prompt) > 100, "Brief prompt has substantial content")

        m.apply_brief({
            "self_brief": "A deeply curious mind drawn to recursive self-examination and consciousness.",
            "entity_brief": "An analytical partner who brings clarity and structure to shared exploration."
        }, entity="Rex")

        # Generate and apply rescore
        rescore_prompt, indexed = m.prepare_rescore_prompt()
        check(len(indexed) == 2, f"2 old memories selected for rescore (got {len(indexed)})")

        # Simulate LLM rescore: bump the underrated one, lower the overrated one
        m.apply_rescores([
            {"index": 0, "new_importance": 6},  # fascination: 4 -> 6
            {"index": 1, "new_importance": 6},  # typo: 8 -> 6
        ], indexed)

        # Save everything
        m._save_brief()
        m.save()

        # Verify state
        check(not m.needs_brief(), "Brief no longer needed after applying")
        check(not m.needs_rescore(), "Rescore no longer needed after applying")

        # Reload and verify persistence
        m2 = fresh_memory(tmpdir)
        check(m2._brief_data["self_brief"].startswith("A deeply curious"),
              "Brief persists after reload")
        ctx = m2.get_context_block(current_entity="Rex")
        check("COMPRESSED SELF-UNDERSTANDING" in ctx,
              "Brief appears in context block after reload")
        check("MY UNDERSTANDING OF REX" in ctx,
              "Entity brief appears in context block after reload")

        # Verify rescored importance persisted
        fascination = [r for r in m2.self_reflections if "fascination" in r.content]
        typo = [r for r in m2.self_reflections if "typo" in r.content]
        if fascination:
            check(fascination[0].importance == 6,
                  f"Underrated memory bumped: 4->6 (got {fascination[0].importance})")
        if typo:
            check(typo[0].importance == 6,
                  f"Overrated memory lowered: 8->6 (got {typo[0].importance})")


# ═══════════════════════════════════════════════════════════════════════════
#  Run all tests
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("  BRIEF & RESCORE TEST SUITE")
    print("=" * 72)

    test_session_counter()
    test_needs_thresholds()
    test_brief_prompt()
    test_apply_brief()
    test_brief_in_context()
    test_brief_persistence()
    test_rescore_prompt()
    test_rescore_cap()
    test_parse_brief()
    test_parse_rescore()
    test_stats()
    test_backward_compat()
    test_rescore_session_tracking()
    test_full_cycle()

    print("\n" + "=" * 72)
    print(f"  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Total:  {_pass + _fail}")
    print(f"  Passed: {_pass}")
    print(f"  Failed: {_fail}")

    if _fail == 0:
        print("\n  All tests passed. Brief and rescore features verified.")
    sys.exit(0 if _fail == 0 else 1)
