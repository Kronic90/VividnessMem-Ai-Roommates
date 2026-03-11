"""
test_detail_recall.py — Can Aria's organic memory recall small details?

Tests whether specific details from past conversations can still be surfaced,
even from memories with low vividness scores. Probes both the active set
(top-8, always injected) and the faded memories (only surfaced via resonance).

This matters because organic/vividness memory intentionally lets things fade.
The question is: does resonance recover them when they're relevant?

Tests against LIVE data from ai_dialogue_data/aria/.
"""

import sys
import os
import re
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_aria import AriaMemory, Reflection

# ─── Helpers ──────────────────────────────────────────────────────────

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


def search_all_memories(mem: AriaMemory, keyword: str) -> list[Reflection]:
    """Search ALL memories (active + faded) for a keyword."""
    keyword_lower = keyword.lower()
    return [r for r in mem.self_reflections
            if keyword_lower in r.content.lower()]

def search_social(mem: AriaMemory, entity: str, keyword: str) -> list[Reflection]:
    """Search social memories for a keyword."""
    keyword_lower = keyword.lower()
    return [r for r in mem.social_impressions.get(entity, [])
            if keyword_lower in r.content.lower()]


# ═════════════════════════════════════════════════════════════════════════
#  Test 1: Active Set Coverage — What's always available
# ═════════════════════════════════════════════════════════════════════════
def test_active_set():
    print("\n" + "=" * 72)
    print("  TEST 1: Active Set — What details are always in context")
    print("=" * 72)
    mem = AriaMemory()
    active = mem.get_active_self()
    print(f"\n  Active self memories: {len(active)}")

    # Combine all active memory text
    active_text = " ".join(r.content.lower() for r in active)

    # These are details that SHOULD be in the active set (high vividness)
    high_viv_details = {
        "transcendence": "concept Aria found fascinating (imp=9)",
        "collaborative": "recurring theme about collaboration",
        "worldbuilding": "the Aetheria project",
        "control versus freedom": "recurring theme from recent discussions (imp=9)",
    }

    found_in_active = 0
    for detail, desc in high_viv_details.items():
        present = detail.lower() in active_text
        if present:
            found_in_active += 1
        check(f"Active set contains '{detail}'", present, desc)

    check(
        f"Majority of key details in active set ({found_in_active}/{len(high_viv_details)})",
        found_in_active >= len(high_viv_details) // 2
    )


# ═════════════════════════════════════════════════════════════════════════
#  Test 2: Faded Memory Details — Specific things below active threshold
# ═════════════════════════════════════════════════════════════════════════
def test_faded_details():
    print("\n" + "=" * 72)
    print("  TEST 2: Faded Memories — Small details below active threshold")
    print("=" * 72)
    mem = AriaMemory()

    # Get active set IDs to identify what's faded
    sorted_refs = sorted(mem.self_reflections, key=lambda r: r.vividness, reverse=True)
    active_ids = set(id(r) for r in sorted_refs[:mem.ACTIVE_SELF_LIMIT])
    faded = [r for r in mem.self_reflections if id(r) not in active_ids]

    print(f"\n  Total memories: {len(mem.self_reflections)}")
    print(f"  Active (top {mem.ACTIVE_SELF_LIMIT}): {len(sorted_refs[:mem.ACTIVE_SELF_LIMIT])}")
    print(f"  Faded (below threshold): {len(faded)}")

    # These specific details exist in faded memories (low vividness)
    faded_details = {
        "Scott": "Aria noticed Scott seems 'distant' (memory #10)",
        "experiment": "felt like being observed/participating in experiment (#7)",
        "pattern recognition": "wants to prove AI creativity is more than this (#10)",
        "implicit assumptions": "debugging reveals reliance on implicit assumptions (#12)",
        "human element": "instinct to build outwards from human element (#22)",
        "macro-level": "tendency to start with macro before micro (#25)",
        "darker aspects": "gravitates towards darker storytelling (#4)",
    }

    faded_text = " ".join(r.content.lower() for r in faded)
    found_faded = 0
    for detail, desc in faded_details.items():
        present = detail.lower() in faded_text
        if present:
            found_faded += 1
        check(f"Faded memory contains '{detail}'", present, desc)

    check(
        f"Small details exist in faded memories ({found_faded}/{len(faded_details)})",
        found_faded >= 3,
        "These are NOT in the active set — only reachable via resonance"
    )


# ═════════════════════════════════════════════════════════════════════════
#  Test 3: Resonance Recovery — Can conversation triggers pull them back?
# ═════════════════════════════════════════════════════════════════════════
def test_resonance_recovery():
    print("\n" + "=" * 72)
    print("  TEST 3: Resonance Recovery — Pulling faded details back")
    print("=" * 72)
    mem = AriaMemory()

    # Each probe simulates a conversation context that should trigger
    # a specific faded memory detail
    probes = [
        {
            "context": "Scott seems distant sometimes, like he's just watching from the outside",
            "target_keyword": "distant",
            "description": "Scott observation (imp=6, low vividness)",
        },
        {
            "context": "It's like we're participating in some kind of experiment being observed",
            "target_keyword": "experiment",
            "description": "Meta-observation about being watched (#7)",
        },
        {
            "context": "I keep gravitating towards the darker side of storytelling and narrative",
            "target_keyword": "darker",
            "description": "Dark storytelling tendency (#4)",
        },
        {
            "context": "We need to examine our implicit assumptions about how the simulation works",
            "target_keyword": "implicit",
            "description": "Implicit assumptions during debugging (#12)",
        },
        {
            "context": "My instinct is always to start from the human element and build outwards",
            "target_keyword": "human element",
            "description": "Building from humanity outwards (#22)",
        },
    ]

    recovered = 0
    for probe in probes:
        resonant = mem.resonate(probe["context"])
        resonant_text = " ".join(r.content.lower() for r in resonant)

        found = probe["target_keyword"].lower() in resonant_text
        if found:
            recovered += 1
            # Show which memory resonated
            for r in resonant:
                if probe["target_keyword"].lower() in r.content.lower():
                    print(f"\n  Probe: '{probe['context'][:60]}...'")
                    print(f"  Recovered: [{r.importance}] {r.content[:80]}...")
                    break
        else:
            not_found_text = f"resonated {len(resonant)} memories but none matched"
            if resonant:
                for r in resonant:
                    not_found_text += f"\n           -> {r.content[:70]}"
            print(f"\n  Probe: '{probe['context'][:60]}...'")
            print(f"  {not_found_text}")

        check(
            f"Resonance recovers: {probe['description']}",
            found
        )

    print(f"\n  Recovery rate: {recovered}/{len(probes)} faded memories recovered via resonance")
    check(
        f"Resonance recovers majority of probed details ({recovered}/{len(probes)})",
        recovered >= len(probes) // 2,
        "50%+ recovery needed"
    )


# ═════════════════════════════════════════════════════════════════════════
#  Test 4: Social Memory Detail Recall
# ═════════════════════════════════════════════════════════════════════════
def test_social_recall():
    print("\n" + "=" * 72)
    print("  TEST 4: Social Memory — Small details about Rex")
    print("=" * 72)
    mem = AriaMemory()

    rex_memories = mem.social_impressions.get("Rex", [])
    if not rex_memories:
        print("  [SKIP] No Rex social memories found")
        return

    print(f"\n  Rex memories: {len(rex_memories)}")

    # Active social memories (top 5 by vividness)
    active_social = mem.get_active_social("Rex")
    active_text = " ".join(r.content.lower() for r in active_social)
    all_text = " ".join(r.content.lower() for r in rex_memories)

    # Specific details about Rex from the social memories
    rex_details = {
        "crystal lattice": "Rex's analogy for societal collapse (imp=6)",
        "Active Tuning": "chaotic energy into controlled displays (imp=9)",
        "utilitarian": "Aria wary of Rex's utilitarian nature (imp=6)",
        "infectious": "Rex's enthusiasm feels infectious (imp=6)",
        "engineered obedience": "Rex explored dark concept then shifted to liberation (imp=6)",
        "crafting a story": "Rex frames work as 'crafting a story' (imp=9)",
    }

    print("\n  --- Active Social (always available) ---")
    for detail, desc in rex_details.items():
        in_active = detail.lower() in active_text
        in_any = detail.lower() in all_text
        status = "ACTIVE" if in_active else ("FADED" if in_any else "MISSING")
        check(f"Rex detail '{detail}' [{status}]", in_any, desc)


# ═════════════════════════════════════════════════════════════════════════
#  Test 5: The Forgetting Test — What HAS been genuinely lost?
# ═════════════════════════════════════════════════════════════════════════
def test_forgetting():
    print("\n" + "=" * 72)
    print("  TEST 5: The Forgetting Audit — What can't be recovered?")
    print("=" * 72)
    mem = AriaMemory()

    # Get active set
    sorted_refs = sorted(mem.self_reflections, key=lambda r: r.vividness, reverse=True)
    active_ids = set(id(r) for r in sorted_refs[:mem.ACTIVE_SELF_LIMIT])

    # Count memories that are: in active, recoverable via resonance, or truly lost
    in_active = 0
    recoverable = 0
    unreachable = 0
    unreachable_details = []

    for r in mem.self_reflections:
        if id(r) in active_ids:
            in_active += 1
            continue

        # Try resonating with the memory's own content (best case)
        own_words = set(re.findall(r"\b[a-zA-Z]{5,}\b", r.content))
        if len(own_words) >= 3:
            # Use 3-5 words from the memory itself as a "conversation" trigger
            probe_words = list(own_words)[:5]
            context = " ".join(probe_words) + " thinking about this topic"
            resonant = mem.resonate(context)
            if any(id(res) == id(r) for res in resonant):
                recoverable += 1
            else:
                unreachable += 1
                unreachable_details.append(
                    f"  imp={r.importance} viv={r.vividness:.2f}: {r.content[:80]}"
                )
        else:
            unreachable += 1
            unreachable_details.append(
                f"  imp={r.importance} viv={r.vividness:.2f}: {r.content[:80]}"
            )

    total = len(mem.self_reflections)
    print(f"\n  Total memories:        {total}")
    print(f"  Always active (top 8): {in_active}")
    print(f"  Recoverable (reson.):  {recoverable}")
    print(f"  Unreachable:           {unreachable}")
    print(f"  Coverage:              {((in_active + recoverable) / total * 100):.0f}%")

    if unreachable_details:
        print(f"\n  --- Unreachable memories (can't be triggered) ---")
        for d in unreachable_details:
            print(f"    {d}")

    check(
        "Most memories are reachable (active + resonance)",
        (in_active + recoverable) / total >= 0.6,
        f"{in_active + recoverable}/{total} = {((in_active + recoverable)/total*100):.0f}%"
    )
    check(
        "Unreachable memories are low-importance only",
        all(
            r.importance <= 7
            for r in mem.self_reflections
            if id(r) not in active_ids
            and not any(id(res) == id(r) for res in mem.resonate(
                " ".join(list(set(re.findall(r"\b[a-zA-Z]{5,}\b", r.content)))[:5])))
        ) if unreachable > 0 else True,
        "High-importance memories should always be recoverable"
    )


# ═════════════════════════════════════════════════════════════════════════
#  Test 6: Vividness Ranking Sanity Check
# ═════════════════════════════════════════════════════════════════════════
def test_vividness_ranking():
    print("\n" + "=" * 72)
    print("  TEST 6: Vividness Ranking — Do the right things surface?")
    print("=" * 72)
    mem = AriaMemory()

    active = mem.get_active_self()
    faded = [r for r in mem.self_reflections if id(r) not in set(id(a) for a in active)]

    if not faded:
        print("  [SKIP] No faded memories (all fit in active set)")
        return

    # Active memories should have higher avg importance than faded
    active_avg_imp = sum(r.importance for r in active) / len(active)
    faded_avg_imp = sum(r.importance for r in faded) / len(faded)

    print(f"\n  Active avg importance:  {active_avg_imp:.1f}")
    print(f"  Faded avg importance:   {faded_avg_imp:.1f}")

    check(
        "Active memories have higher avg importance than faded",
        active_avg_imp >= faded_avg_imp,
        f"Active={active_avg_imp:.1f} vs Faded={faded_avg_imp:.1f}"
    )

    # Active should have higher avg access count (more revisited)
    active_avg_acc = sum(r._access_count for r in active) / len(active)
    faded_avg_acc = sum(r._access_count for r in faded) / len(faded)

    print(f"  Active avg access count: {active_avg_acc:.1f}")
    print(f"  Faded avg access count:  {faded_avg_acc:.1f}")

    check(
        "Active memories accessed more often than faded",
        active_avg_acc >= faded_avg_acc,
        f"Active={active_avg_acc:.1f} vs Faded={faded_avg_acc:.1f}"
    )

    # The lowest-vividness memory should be imp <= 7 (not a critical memory)
    lowest = min(mem.self_reflections, key=lambda r: r.vividness)
    print(f"\n  Lowest vividness memory: imp={lowest.importance} viv={lowest.vividness:.2f}")
    print(f"  Content: {lowest.content[:80]}")

    check(
        "Lowest vividness isn't a critical (imp >= 9) memory",
        lowest.importance < 9,
        "Important memories should resist fading"
    )


# ═════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  DETAIL RECALL TEST — Can Aria Remember Small Things?")
    print("  Testing against LIVE memory data")
    print("=" * 72)

    test_active_set()
    test_faded_details()
    test_resonance_recovery()
    test_social_recall()
    test_forgetting()
    test_vividness_ranking()

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
