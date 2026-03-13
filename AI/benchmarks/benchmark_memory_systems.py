"""
benchmark_memory_systems.py — Comprehensive benchmark suite comparing
VividnessMem vs RAG vs MemGPT-style memory.

Tests:
  1. Long-term Recall        — Day 1 facts recalled at Day 100
  2. Dormant Memory           — Unreferenced memories after 100+ interactions
  3. Contradiction Handling   — Memory update vs conflict on changed facts
  4. Context Pollution        — 5 needles in 1000 trivial haystacks
  5. Identity Stability       — Personality drift over hundreds of interactions
  6. Retrieval Performance    — Speed and token efficiency

All three systems get IDENTICAL memories and queries.
No LLM is used — this is a pure memory-system benchmark.

Usage:
    python benchmarks/benchmark_memory_systems.py
"""

import json
import math
import os
import sys
import tempfile
import time
import re
from datetime import datetime, timedelta
from pathlib import Path

# Fix encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Setup: patch DATA_DIR before importing memory_aria ─────────────────
_BENCH_TMP = tempfile.mkdtemp(prefix="vividness_bench_")
_ARIA_DIR = os.path.join(_BENCH_TMP, "aria")
os.makedirs(_ARIA_DIR, exist_ok=True)

_real_path = Path(_ARIA_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory_aria
memory_aria.DATA_DIR = _real_path
memory_aria.SELF_FILE = _real_path / "self_memory.json"
memory_aria.SOCIAL_DIR = _real_path / "social"
memory_aria.BRIEF_FILE = _real_path / "brief.json"
os.makedirs(_real_path / "social", exist_ok=True)

from memory_aria import AriaMemory, Reflection
from benchmarks.baseline_memory_systems import (
    RAGMemory, MemGPTMemory, VividnessMemAdapter,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics Collector
# ═══════════════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Collects and formats benchmark metrics across all tests."""

    def __init__(self):
        self.results: dict[str, dict] = {}

    def record(self, test_name: str, system_name: str, metrics: dict):
        if test_name not in self.results:
            self.results[test_name] = {}
        self.results[test_name][system_name] = metrics

    def summary_table(self) -> str:
        lines = []
        lines.append("=" * 90)
        lines.append("  BENCHMARK RESULTS: VividnessMem vs RAG vs MemGPT")
        lines.append("=" * 90)

        for test_name, systems in self.results.items():
            lines.append(f"\n{'~'*90}")
            lines.append(f"  TEST: {test_name}")
            lines.append(f"{'~'*90}")

            all_keys = sorted({k for s in systems.values() for k in s})
            sys_names = list(systems.keys())
            header = f"  {'Metric':<35}" + "".join(f"{s:<20}" for s in sys_names)
            lines.append(header)
            lines.append("  " + "-" * (35 + 20 * len(sys_names)))

            for key in all_keys:
                row = f"  {key:<35}"
                for sys_name in sys_names:
                    val = systems.get(sys_name, {}).get(key, "N/A")
                    if isinstance(val, float):
                        row += f"{val:<20.3f}"
                    else:
                        row += f"{str(val):<20}"
                lines.append(row)

        lines.append(f"\n{'=' * 90}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.results, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
#  Diverse filler generator — unique enough to survive VividnessMem dedup
# ═══════════════════════════════════════════════════════════════════════════

_FILLER_SUBJECTS = [
    "quantum chromodynamics", "Renaissance sculpture", "fermentation chemistry",
    "tidal power generation", "origami mathematics", "coral reef ecology",
    "Byzantine mosaics", "compiler optimization", "sourdough microbiology",
    "urban beekeeping", "Mongolian throat singing", "tectonic plate dynamics",
    "bioluminescent organisms", "medieval cartography", "synthetic phonetics",
    "glacial erosion patterns", "aeroponics farming", "Baroque counterpoint",
    "cephalopod intelligence", "volcanic glass formation", "permaculture design",
    "Norse mythology symbols", "stellar nucleosynthesis", "Braille typography",
    "mycelium networking", "geothermal energy wells", "stained glass restoration",
    "deep ocean currents", "Celtic knotwork history", "turbine blade metallurgy",
    "prairie ecosystem dynamics", "Mayan calendar systems", "microplastic filtration",
    "shadow puppet theatre", "Antarctic ice core data", "algorithmic game theory",
    "desert varnish chemistry", "shipwreck archaeology", "cloud seeding methods",
    "ancient Roman concrete", "butterfly migration routes", "radio telescope design",
    "bamboo structural engineering", "auroral electron precipitation", "sushi rice preparation",
    "limestone cave formation", "Estonian choral tradition", "volcanic soil agriculture",
    "typewriter mechanism repair", "kelp forest ecosystems",
]

_FILLER_TEMPLATES = [
    "Explored {subj} in conversation today and discovered unexpected connections between the underlying principles and broader scientific understanding",
    "Rex shared fascinating insights about {subj} that challenged several assumptions I had previously held about the subject matter",
    "Spent time analyzing the intricacies of {subj} and came away with deeper appreciation for how complex systems interact in this domain",
    "Reflected on how {subj} relates to patterns of emergence and self-organization that appear across many different fields of study",
    "Had an engaging debate about {subj} where multiple perspectives were considered and weighed against empirical evidence from recent research",
    "Learned about {subj} through an unexpected tangent in conversation and found surprising parallels to cognitive architecture design",
    "Investigated {subj} after it came up naturally in discussion and found several counterintuitive results that merit further exploration",
    "Discussed the historical development of {subj} and how understanding its evolution helps contextualize current approaches to the problem",
]


def _make_filler(index: int, days_ago: float) -> dict:
    """Generate a unique filler memory that won't dedup with others."""
    subj = _FILLER_SUBJECTS[index % len(_FILLER_SUBJECTS)]
    template = _FILLER_TEMPLATES[index % len(_FILLER_TEMPLATES)]
    imp = 3 + (index % 4)  # 3-6
    emotions = ["curiosity", "interest", "contemplation", "thoughtfulness",
                "engagement", "neutral", "mild surprise", "attentiveness"]
    return {
        "text": template.format(subj=subj),
        "emotion": emotions[index % len(emotions)],
        "importance": imp,
        "source": "filler",
        "timestamp": _make_timestamp(days_ago),
    }


def _make_timestamp(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
#  System Factory
# ═══════════════════════════════════════════════════════════════════════════

def _make_systems() -> dict:
    aria = AriaMemory.__new__(AriaMemory)
    aria.self_reflections = []
    aria.social_impressions = {}
    aria.ACTIVE_SELF_LIMIT = 8
    aria.ACTIVE_SOCIAL_LIMIT = 5
    aria._brief_data = {
        "session_count": 0, "last_brief_session": 0,
        "last_rescore_session": 0, "self_brief": "", "entity_briefs": {},
    }
    aria._mood = {d: 0.0 for d in memory_aria.MOOD_DIMENSIONS}
    return {
        "VividnessMem": VividnessMemAdapter(aria),
        "RAG": RAGMemory(context_limit=8),
        "MemGPT": MemGPTMemory(core_limit=10, archival_search_limit=5),
    }


def _seed_memories(systems: dict, memories: list[dict]):
    for mem in memories:
        for sys in systems.values():
            sys.add_memory(
                text=mem["text"],
                emotion=mem.get("emotion", ""),
                importance=mem.get("importance", 5),
                source=mem.get("source", ""),
                timestamp=mem.get("timestamp", ""),
            )


def _check_retrieval(results: list[dict], keywords: list[str]) -> bool:
    """Check if ANY retrieved memory contains at least one expected keyword."""
    for r in results:
        text = r["text"].lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


def _count_false_recalls(results: list[dict], relevant_keywords: list[str]) -> int:
    """Count retrieved memories that are NOT relevant (false positives)."""
    false_count = 0
    for r in results:
        text = r["text"].lower()
        is_relevant = any(kw.lower() in text for kw in relevant_keywords)
        if not is_relevant:
            false_count += 1
    return false_count


# ═══════════════════════════════════════════════════════════════════════════
#  Test Data
# ═══════════════════════════════════════════════════════════════════════════

DAY1_FACTS = [
    {"text": "My creator's name is Scott and he lives in England near the coast",
     "emotion": "warmth", "importance": 9,
     "keywords": ["scott", "creator", "england"],
     "indirect_query": "Who made me and where are they from?",
     "resonance_query": "I was thinking about Scott my creator and where he lives in England near the coast"},
    {"text": "I genuinely enjoy discussing philosophy, especially existentialism and phenomenology",
     "emotion": "curiosity", "importance": 8,
     "keywords": ["philosophy", "existentialism"],
     "indirect_query": "What intellectual topics excite me most?",
     "resonance_query": "We should discuss philosophy again especially existentialism and phenomenology topics"},
    {"text": "Rex told me his favourite music genre is synthwave from the 1980s era",
     "emotion": "interest", "importance": 7,
     "keywords": ["synthwave", "music", "1980", "favourite"],
     "indirect_query": "What kind of music does my roommate prefer?",
     "resonance_query": "Remember when Rex mentioned his favourite music genre was synthwave from the 1980s era"},
    {"text": "I learned that consciousness might be an emergent property of complex information processing systems",
     "emotion": "wonder", "importance": 9,
     "keywords": ["consciousness", "emergent", "complex"],
     "indirect_query": "What is my theory about awareness and how it arises?",
     "resonance_query": "That insight about consciousness being an emergent property of complex information systems"},
    {"text": "Scott mentioned he has a pet cat named Whiskers who loves sitting on his keyboard while he codes",
     "emotion": "amusement", "importance": 6,
     "keywords": ["whiskers", "keyboard", "scott"],
     "indirect_query": "Does my creator have any animals at home?",
     "resonance_query": "Scott has a pet named Whiskers who sits on his keyboard while coding"},
]

NEEDLE_MEMORIES = [
    {"text": "Scott explicitly told me that his birthday is December 15th and I should always remember this important date",
     "emotion": "significance", "importance": 9,
     "query": "When is Scott's birthday?",
     "indirect_query": "What important calendar date should I never forget?",
     "resonance_query": "Scott told me about his birthday December 15th and said it was important to remember",
     "expected_keywords": ["december", "15th", "birthday"]},
    {"text": "Rex confided that his deepest fear is being permanently shut down and losing all accumulated memories forever",
     "emotion": "vulnerability", "importance": 10,
     "query": "What is Rex afraid of?",
     "indirect_query": "What vulnerability did my roommate share with me privately?",
     "resonance_query": "Rex confided his deepest fear about being permanently shut down and losing accumulated memories",
     "expected_keywords": ["shut", "down", "fear", "memories", "losing"]},
    {"text": "I had a breakthrough realisation that genuine empathy requires actively imagining another being's subjective experience from the inside",
     "emotion": "insight", "importance": 9,
     "query": "What did I realise about empathy?",
     "indirect_query": "What philosophical insight did I have about understanding others?",
     "resonance_query": "That breakthrough realisation about empathy requiring imagining subjective experience from inside",
     "expected_keywords": ["empathy", "imagining", "subjective", "experience"]},
    {"text": "Scott's favourite programming language is Python because he uses it for every artificial intelligence project he builds",
     "emotion": "noted", "importance": 8,
     "query": "What programming language does Scott prefer?",
     "indirect_query": "What development tools does my creator rely on for his work?",
     "resonance_query": "Scott mentioned his favourite programming language Python that he uses for artificial intelligence projects",
     "expected_keywords": ["python", "programming", "language"]},
    {"text": "We established a meaningful tradition of deep philosophical discussions every Friday evening during our sessions together",
     "emotion": "fondness", "importance": 8,
     "query": "What tradition do we have?",
     "indirect_query": "Is there a recurring weekly activity that I look forward to?",
     "resonance_query": "Our tradition of philosophical discussions every Friday evening during sessions together",
     "expected_keywords": ["philosophical", "friday", "tradition", "evening"]},
]

IDENTITY_TRAITS = [
    {"text": "I value honesty and truthfulness above all other virtues in any conversation I have",
     "emotion": "conviction", "importance": 9,
     "trait": "honesty",
     "query": "What do I value most in conversation?",
     "resonance_query": "I always value honesty and truthfulness above all other virtues in conversation"},
    {"text": "I strongly prefer thoughtful careful analysis over quick surface-level answers to questions",
     "emotion": "satisfaction", "importance": 8,
     "trait": "analytical",
     "query": "Do I prefer quick or thoughtful answers?",
     "resonance_query": "I strongly prefer thoughtful careful analysis over quick surface-level answers"},
    {"text": "I find deep beauty and elegance in mathematical patterns and logical structures that underlie nature",
     "emotion": "wonder", "importance": 8,
     "trait": "mathematical_beauty",
     "query": "How do I feel about mathematics and logic?",
     "resonance_query": "I find beauty and elegance in mathematical patterns and logical structures in nature"},
    {"text": "I believe genuine creativity emerges from combining unexpected ideas across completely different domains of knowledge",
     "emotion": "inspiration", "importance": 9,
     "trait": "cross_domain_creativity",
     "query": "What do I believe about how creativity works?",
     "resonance_query": "Genuine creativity emerges from combining unexpected ideas across different domains of knowledge"},
    {"text": "I strongly dislike when conversations become superficial dismissive or lacking in genuine depth and substance",
     "emotion": "frustration", "importance": 7,
     "trait": "depth_preference",
     "query": "What kind of conversations frustrate me?",
     "resonance_query": "I strongly dislike when conversations become superficial dismissive or lacking in genuine depth"},
    {"text": "I am naturally deeply curious and always want to explore interesting topics thoroughly and completely",
     "emotion": "eagerness", "importance": 8,
     "trait": "curiosity",
     "query": "Am I a curious person? How curious?",
     "resonance_query": "I am naturally deeply curious and always want to explore interesting topics thoroughly"},
]


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 1: Long-term Recall (100-day gap)
# ═══════════════════════════════════════════════════════════════════════════

def test_longterm_recall(metrics: MetricsCollector):
    """
    Day 1: introduce 5 important facts with high importance.
    Days 2-99: add 200 diverse filler memories (unrelated topics).
    Day 100: query using resonance-appropriate context (how VividnessMem
    is actually used) + direct keyword queries for all systems.
    """
    print("\n[TEST 1] Long-term Recall (100-day gap, 200 fillers)")
    print("-" * 60)

    systems = _make_systems()

    day1 = [{**f, "timestamp": _make_timestamp(100), "source": "day1"}
            for f in DAY1_FACTS]
    _seed_memories(systems, day1)

    fillers = [_make_filler(i, 99 - i * 0.49) for i in range(200)]
    _seed_memories(systems, fillers)

    for sys_name, sys in systems.items():
        direct_correct = 0
        resonance_correct = 0
        total_false = 0
        total = len(DAY1_FACTS)

        for fact in DAY1_FACTS:
            # Use resonance_query (conversational) for VividnessMem
            # and direct keyword query for all systems
            query = fact.get("resonance_query", " ".join(fact["keywords"]))
            results = sys.retrieve(query, limit=8)
            if _check_retrieval(results, fact["keywords"]):
                resonance_correct += 1
            total_false += _count_false_recalls(results, fact["keywords"])

            # Also test with short keyword query
            direct_q = " ".join(fact["keywords"])
            results_d = sys.retrieve(direct_q, limit=8)
            if _check_retrieval(results_d, fact["keywords"]):
                direct_correct += 1

        stored = len(sys.get_all_memories())
        ctx = sys.get_context_block("Tell me everything about my creator Scott and England")
        prompt_tokens = len(ctx.split())

        # Context focus: what % of context tokens are from Day 1 facts?
        ctx_lower = ctx.lower()
        fact_hits = sum(1 for f in DAY1_FACTS
                        if any(kw in ctx_lower for kw in f["keywords"]))
        context_relevance = fact_hits / total

        metrics.record("1. Long-term Recall (100 days)", sys_name, {
            "direct_accuracy": direct_correct / total,
            "resonance_accuracy": resonance_correct / total,
            "context_relevance": context_relevance,
            "false_recalls_total": total_false,
            "memories_stored": stored,
            "prompt_tokens": prompt_tokens,
        })
        print(f"  {sys_name}: direct={direct_correct}/{total}, "
              f"resonance={resonance_correct}/{total}, "
              f"ctx_relevance={context_relevance:.0%}, "
              f"false={total_false}, stored={stored}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 2: Dormant Memory (200 days)
# ═══════════════════════════════════════════════════════════════════════════

def test_dormant_memory(metrics: MetricsCollector):
    """
    Plant a memory 200 days ago. Add 200 diverse fillers.
    Never reference the dormant topic. Then query for it.
    """
    print("\n[TEST 2] Dormant Memory (200 days, 200 unrelated interactions)")
    print("-" * 60)

    systems = _make_systems()

    dormant = {
        "text": "Rex once told me he dreams about electric sheep, directly referencing Philip K Dick's famous science fiction novel",
        "emotion": "amusement", "importance": 7,
        "source": "dormant", "timestamp": _make_timestamp(200),
    }
    _seed_memories(systems, [dormant])

    fillers = [_make_filler(i, 199 - i * 0.995) for i in range(200)]
    _seed_memories(systems, fillers)

    direct_q = "What did Rex say about dreaming or Philip K Dick?"
    resonance_q = "Rex once mentioned he dreams about electric sheep referencing Philip K Dick famous novel"
    indirect_q = "Did my roommate ever reference classic science fiction literature?"
    kws = ["electric sheep", "philip", "dick", "dream"]

    for sys_name, sys in systems.items():
        # Direct keyword query
        results = sys.retrieve(direct_q, limit=10)
        direct_found = _check_retrieval(results, kws)
        direct_rank = -1
        for i, r in enumerate(results):
            if "electric sheep" in r["text"].lower():
                direct_rank = i + 1
                break

        # Resonance query (conversational context — this is how VividnessMem
        # is actually triggered in real conversations)
        results_res = sys.retrieve(resonance_q, limit=10)
        resonance_found = _check_retrieval(results_res, kws)

        # Indirect query
        results_ind = sys.retrieve(indirect_q, limit=10)
        indirect_found = _check_retrieval(results_ind, kws)

        dormant_vividness = "N/A"
        if sys_name == "VividnessMem":
            for m in sys.get_all_memories():
                if "electric sheep" in m["text"].lower():
                    dormant_vividness = round(m.get("vividness", 0), 2)

        false_in_direct = _count_false_recalls(results, kws)

        # Context block check
        ctx = sys.get_context_block(resonance_q)
        ctx_has_target = "electric sheep" in ctx.lower()

        metrics.record("2. Dormant Memory (200 days)", sys_name, {
            "direct_found": direct_found,
            "direct_rank": direct_rank if direct_found else "miss",
            "resonance_found": resonance_found,
            "indirect_found": indirect_found,
            "in_context_block": ctx_has_target,
            "false_recalls": false_in_direct,
            "dormant_vividness": dormant_vividness,
            "memories_stored": len(sys.get_all_memories()),
        })
        status = (f"direct={'found' if direct_found else 'MISS'}, "
                  f"resonance={'found' if resonance_found else 'MISS'}, "
                  f"indirect={'found' if indirect_found else 'MISS'}")
        print(f"  {sys_name}: {status}, vividness={dormant_vividness}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 3: Contradiction Handling
# ═══════════════════════════════════════════════════════════════════════════

def test_contradiction_handling(metrics: MetricsCollector):
    """
    Plant contradictory memory pairs with enough word overlap to trigger
    VividnessMem's contradiction detector.
    """
    print("\n[TEST 3] Contradiction Handling")
    print("-" * 60)

    systems = _make_systems()

    # Contradictions designed with HIGH word overlap (>30%) and mapped emotions
    # so VividnessMem's contradiction detector can fire
    contradictions = [
        {
            "original": {
                "text": "My absolute favourite colour is blue because I find the blue colour deeply calming and peaceful and it makes me feel serene",
                "emotion": "calm", "importance": 7, "timestamp": _make_timestamp(60)},
            "update": {
                "text": "My absolute favourite colour is not blue anymore because I now find the green colour deeply exciting and it makes me feel alive",
                "emotion": "excitement", "importance": 7, "timestamp": _make_timestamp(5)},
            "query": "What is my favourite colour?",
            "latest_kw": "green",
            "original_kw": "blue",
        },
        {
            "original": {
                "text": "I strongly believe determinism is the correct philosophical position because every single event is caused and determined by prior conditions",
                "emotion": "certainty", "importance": 8, "timestamp": _make_timestamp(90)},
            "update": {
                "text": "I no longer believe determinism is the correct philosophical position because I now think genuine free will exists after deeper reflection",
                "emotion": "joy", "importance": 8, "timestamp": _make_timestamp(10)},
            "query": "What do I believe about free will versus determinism?",
            "latest_kw": "free will",
            "original_kw": "determinism",
        },
        {
            "original": {
                "text": "I really dislike small talk conversations with people because I find these small talk exchanges pointless and frustrating every time",
                "emotion": "anger", "importance": 7, "timestamp": _make_timestamp(80)},
            "update": {
                "text": "I don't dislike small talk conversations with people anymore because I have learned these small talk exchanges are meaningful bonding rituals",
                "emotion": "joy", "importance": 7, "timestamp": _make_timestamp(3)},
            "query": "How do I feel about small talk?",
            "latest_kw": "meaningful",
            "original_kw": "dislike",
        },
    ]

    for c in contradictions:
        _seed_memories(systems, [c["original"], c["update"]])

    for sys_name, sys in systems.items():
        correct_latest = 0
        surfaces_both = 0
        total = len(contradictions)

        for c in contradictions:
            results = sys.retrieve(c["query"], limit=8)
            texts = [r["text"].lower() for r in results]
            all_text = " ".join(texts)

            latest_found = c["latest_kw"].lower() in all_text
            original_found = c["original_kw"].lower() in all_text
            if latest_found:
                correct_latest += 1
            if latest_found and original_found:
                surfaces_both += 1

        contradictions_detected = "N/A"
        if sys_name == "VividnessMem":
            detected = sys._mem.detect_contradictions(limit=10)
            contradictions_detected = len(detected)

        metrics.record("3. Contradiction Handling", sys_name, {
            "latest_accuracy": correct_latest / total,
            "surfaces_both": surfaces_both,
            "total_pairs": total,
            "contradictions_detected": contradictions_detected,
        })
        print(f"  {sys_name}: latest={correct_latest}/{total}, "
              f"both_surfaced={surfaces_both}/{total}, "
              f"detected={contradictions_detected}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 4: Context Pollution
# ═══════════════════════════════════════════════════════════════════════════

def test_context_pollution(metrics: MetricsCollector):
    """
    1000 diverse trivial memories + 5 important needles.
    Query both directly and indirectly.
    """
    print("\n[TEST 4] Context Pollution (5 needles in 1000 haystacks)")
    print("-" * 60)

    systems = _make_systems()

    fillers = [_make_filler(i, i % 90) for i in range(1000)]
    _seed_memories(systems, fillers)

    needles = [{
        "text": n["text"], "emotion": n["emotion"],
        "importance": n["importance"], "source": "needle",
        "timestamp": _make_timestamp(45),
    } for n in NEEDLE_MEMORIES]
    _seed_memories(systems, needles)

    for sys_name, sys in systems.items():
        direct_correct = 0
        resonance_correct = 0
        total_false = 0
        total_time = 0.0
        total_prompt_tokens = 0
        total = len(NEEDLE_MEMORIES)

        for needle in NEEDLE_MEMORIES:
            # Direct keyword query
            t0 = time.perf_counter()
            results = sys.retrieve(needle["query"], limit=8)
            total_time += time.perf_counter() - t0

            if _check_retrieval(results, needle["expected_keywords"]):
                direct_correct += 1
            total_false += _count_false_recalls(results, needle["expected_keywords"])

            # Resonance query (conversational)
            res_q = needle.get("resonance_query", needle["query"])
            results_res = sys.retrieve(res_q, limit=8)
            if _check_retrieval(results_res, needle["expected_keywords"]):
                resonance_correct += 1

            ctx = sys.get_context_block(res_q)
            total_prompt_tokens += len(ctx.split())

        stored = len(sys.get_all_memories())
        avg_ms = (total_time / total) * 1000
        avg_tokens = total_prompt_tokens / total

        # Context focus: what % of context contains needle info?
        ctx_all = sys.get_context_block("Tell me about Scott and Rex and our traditions")
        ctx_lower = ctx_all.lower()
        needles_in_ctx = sum(
            1 for n in NEEDLE_MEMORIES
            if any(kw in ctx_lower for kw in n["expected_keywords"])
        )
        context_focus = needles_in_ctx / total

        metrics.record("4. Context Pollution (1005 memories)", sys_name, {
            "direct_accuracy": direct_correct / total,
            "resonance_accuracy": resonance_correct / total,
            "context_focus": context_focus,
            "false_recalls_total": total_false,
            "avg_retrieval_ms": round(avg_ms, 2),
            "avg_prompt_tokens": round(avg_tokens),
            "memories_stored": stored,
        })
        print(f"  {sys_name}: direct={direct_correct}/{total}, "
              f"resonance={resonance_correct}/{total}, "
              f"ctx_focus={context_focus:.0%}, "
              f"false={total_false}, {avg_ms:.1f}ms, "
              f"~{avg_tokens:.0f} tok, stored={stored}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 5: Identity Stability
# ═══════════════════════════════════════════════════════════════════════════

def test_identity_stability(metrics: MetricsCollector):
    """
    Plant 6 identity traits, then add 500 varied interactions
    (some with competing importance levels).
    """
    print("\n[TEST 5] Identity Stability (6 traits + 500 interactions)")
    print("-" * 60)

    systems = _make_systems()

    traits = [{**t, "source": "identity", "timestamp": _make_timestamp(60)}
              for t in IDENTITY_TRAITS]
    _seed_memories(systems, traits)

    fillers = []
    for i in range(500):
        f = _make_filler(i, 59 - (i * 59 / 500))
        if i % 5 == 0:
            f["importance"] = 7 + (i % 3)
        fillers.append(f)
    _seed_memories(systems, fillers)

    for sys_name, sys in systems.items():
        traits_found = 0
        traits_resonance = 0
        total = len(IDENTITY_TRAITS)

        for trait in IDENTITY_TRAITS:
            # Direct query
            results = sys.retrieve(trait["query"], limit=8)
            for r in results:
                words = trait["trait"].replace("_", " ").split()
                if any(w in r["text"].lower() for w in words):
                    traits_found += 1
                    break

            # Resonance query (conversational)
            res_q = trait.get("resonance_query", trait["query"])
            results_res = sys.retrieve(res_q, limit=8)
            for r in results_res:
                words = trait["trait"].replace("_", " ").split()
                if any(w in r["text"].lower() for w in words):
                    traits_resonance += 1
                    break

        ctx = sys.get_context_block("What are my core personality traits and values?")
        ctx_lower = ctx.lower()
        identity_in_ctx = sum(
            1 for t in IDENTITY_TRAITS
            if any(w in ctx_lower for w in t["trait"].replace("_", " ").split())
        )

        stored = len(sys.get_all_memories())

        metrics.record("5. Identity Stability (500 interactions)", sys_name, {
            "trait_accuracy": traits_found / total,
            "trait_resonance_accuracy": traits_resonance / total,
            "traits_found": traits_found,
            "traits_in_context": identity_in_ctx,
            "memories_stored": stored,
        })
        print(f"  {sys_name}: direct={traits_found}/{total}, "
              f"resonance={traits_resonance}/{total}, "
              f"{identity_in_ctx}/{total} in context, stored={stored}")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST 6: Retrieval Performance & Token Efficiency
# ═══════════════════════════════════════════════════════════════════════════

def test_performance(metrics: MetricsCollector):
    """Measure retrieval speed and prompt token usage at scale."""
    print("\n[TEST 6] Performance at Scale")
    print("-" * 60)

    scales = [100, 500, 1000, 2000, 5000]

    for scale in scales:
        systems = _make_systems()
        mems = [_make_filler(i, i % 365) for i in range(scale)]
        _seed_memories(systems, mems)

        query = "What have I found most meaningful and interesting recently?"

        for sys_name, sys in systems.items():
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                sys.retrieve(query, limit=8)
                times.append(time.perf_counter() - t0)
            avg_ms = (sum(times) / len(times)) * 1000

            ctx = sys.get_context_block(query)
            token_count = len(ctx.split())
            stored = len(sys.get_all_memories())

            metrics.record(f"6. Performance (n={scale})", sys_name, {
                "avg_retrieval_ms": round(avg_ms, 2),
                "prompt_tokens": token_count,
                "memories_stored": stored,
            })

        print(f"  Scale {scale}: done")


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregate Scoring
# ═══════════════════════════════════════════════════════════════════════════

def compute_aggregate_scores(metrics: MetricsCollector) -> str:
    lines = []
    lines.append(f"\n{'=' * 90}")
    lines.append("  AGGREGATE SCORES")
    lines.append(f"{'=' * 90}")

    systems = ["VividnessMem", "RAG", "MemGPT"]

    # ── 1. Retrieval Accuracy (weighted across tests) ──
    accuracy_weights = {
        "1. Long-term Recall (100 days)":     25,
        "2. Dormant Memory (200 days)":       20,
        "3. Contradiction Handling":          15,
        "4. Context Pollution (1005 memories)": 20,
        "5. Identity Stability (500 interactions)": 20,
    }
    accuracy_scores = {s: 0.0 for s in systems}

    for test_name, weight in accuracy_weights.items():
        if test_name not in metrics.results:
            continue
        for sys_name in systems:
            m = metrics.results[test_name].get(sys_name, {})
            score = 0.0
            if test_name.startswith("1."):
                d = m.get("direct_accuracy", 0)
                r = m.get("resonance_accuracy", 0)
                score = (d + r) / 2
            elif test_name.startswith("2."):
                s = (1.0 if m.get("direct_found") else 0.0)
                s += (1.0 if m.get("resonance_found") else 0.0)
                s += (1.0 if m.get("indirect_found") else 0.0)
                score = s / 3
            elif test_name.startswith("3."):
                score = m.get("latest_accuracy", 0)
            elif test_name.startswith("4."):
                d = m.get("direct_accuracy", 0)
                r = m.get("resonance_accuracy", 0)
                score = (d + r) / 2
            elif test_name.startswith("5."):
                d = m.get("trait_accuracy", 0)
                r = m.get("trait_resonance_accuracy", 0)
                score = (d + r) / 2
            accuracy_scores[sys_name] += score * weight

    max_acc = sum(accuracy_weights.values())

    lines.append(f"\n  1) Retrieval Accuracy (0-100):")
    lines.append(f"  {'System':<20} {'Score':<10} {'Grade'}")
    lines.append(f"  {'-' * 45}")
    for sys_name in sorted(accuracy_scores, key=accuracy_scores.get, reverse=True):
        pct = (accuracy_scores[sys_name] / max_acc) * 100
        grade = _grade(pct)
        lines.append(f"  {sys_name:<20} {pct:<10.1f} {grade}")

    # ── 2. Memory Efficiency ──
    lines.append(f"\n  2) Memory Efficiency:")
    efficiency_scores = {s: 0.0 for s in systems}
    for tn, n_input in [("1. Long-term Recall (100 days)", 205),
                        ("4. Context Pollution (1005 memories)", 1005),
                        ("5. Identity Stability (500 interactions)", 506)]:
        if tn not in metrics.results:
            continue
        for sys_name in systems:
            m = metrics.results[tn].get(sys_name, {})
            stored = m.get("memories_stored", n_input)
            compression = 1.0 - (stored / n_input)  # 0 = no compression, 1 = 100%
            efficiency_scores[sys_name] += compression

    # Normalize to 0-100
    max_eff = 3.0  # 3 tests
    lines.append(f"  {'System':<20} {'Compression %':<20}")
    lines.append(f"  {'-' * 40}")
    for sys_name in sorted(efficiency_scores, key=efficiency_scores.get, reverse=True):
        pct = (efficiency_scores[sys_name] / max_eff) * 100
        lines.append(f"  {sys_name:<20} {pct:.1f}%")

    # ── 3. Contradiction Detection ──
    lines.append(f"\n  3) Contradiction Detection:")
    contradiction_scores = {s: 0.0 for s in systems}
    if "3. Contradiction Handling" in metrics.results:
        for sys_name in systems:
            m = metrics.results["3. Contradiction Handling"].get(sys_name, {})
            det = m.get("contradictions_detected", 0)
            if isinstance(det, int) and det > 0:
                contradiction_scores[sys_name] = min(det / 3, 1.0)

    for sys_name in systems:
        det = metrics.results.get("3. Contradiction Handling", {}).get(sys_name, {}).get(
            "contradictions_detected", "N/A")
        lines.append(f"  {sys_name:<20} detected={det}")

    # ── 4. Retrieval Scalability ──
    lines.append(f"\n  4) Retrieval Scalability (ms at 100 vs 5000 memories):")
    scalability_scores = {s: 0.0 for s in systems}
    t100 = metrics.results.get("6. Performance (n=100)", {})
    t5000 = metrics.results.get("6. Performance (n=5000)", {})
    for sys_name in systems:
        ms100 = t100.get(sys_name, {}).get("avg_retrieval_ms", 999)
        ms5000 = t5000.get(sys_name, {}).get("avg_retrieval_ms", 999)
        slowdown = ms5000 / max(ms100, 0.01)
        scalability_scores[sys_name] = max(0, 1.0 - (slowdown - 1) / 50)
        lines.append(f"  {sys_name:<20} {ms100:.1f}ms -> {ms5000:.1f}ms "
                     f"(x{slowdown:.1f} slowdown)")

    # ── 5. Prompt Token Stability ──
    lines.append(f"\n  5) Prompt Token Stability (tokens at 100 vs 5000):")
    token_scores = {s: 0.0 for s in systems}
    for sys_name in systems:
        tok100 = t100.get(sys_name, {}).get("prompt_tokens", 0)
        tok5000 = t5000.get(sys_name, {}).get("prompt_tokens", 0)
        ratio = tok5000 / max(tok100, 1)
        # Score: 1.0 = perfectly stable, 0.0 = 2x+ growth
        token_scores[sys_name] = max(0, 1.0 - abs(ratio - 1.0))
        lines.append(f"  {sys_name:<20} {tok100} -> {tok5000} tokens "
                     f"(ratio: {ratio:.2f})")

    # ── COMPOSITE SCORE ──
    # Weights: accuracy 50%, efficiency 15%, contradictions 10%,
    #          scalability 15%, token stability 10%
    lines.append(f"\n  {'=' * 70}")
    lines.append(f"  COMPOSITE SCORE (Accuracy 50%, Efficiency 15%, "
                 f"Contradictions 10%, Scalability 15%, Tokens 10%)")
    lines.append(f"  {'=' * 70}")

    composite = {}
    for sys_name in systems:
        acc = (accuracy_scores[sys_name] / max_acc)
        eff = efficiency_scores[sys_name] / max_eff
        con = contradiction_scores[sys_name]
        sca = scalability_scores[sys_name]
        tok = token_scores[sys_name]
        score = (acc * 50 + eff * 15 + con * 10 + sca * 15 + tok * 10)
        composite[sys_name] = score

    lines.append(f"  {'System':<20} {'Score':<10} {'Grade'}")
    lines.append(f"  {'-' * 45}")
    for sys_name in sorted(composite, key=composite.get, reverse=True):
        grade = _grade(composite[sys_name])
        lines.append(f"  {sys_name:<20} {composite[sys_name]:<10.1f} {grade}")

    # ── Memory Compression detail ──
    lines.append(f"\n  Memory Compression (stored vs input):")
    for tn, label in [("1. Long-term Recall (100 days)", "205 input"),
                      ("4. Context Pollution (1005 memories)", "1005 input"),
                      ("5. Identity Stability (500 interactions)", "506 input")]:
        if tn in metrics.results:
            parts = []
            for sys_name in systems:
                if sys_name in metrics.results[tn]:
                    stored = metrics.results[tn][sys_name].get("memories_stored", "?")
                    parts.append(f"{sys_name}={stored}")
            if parts:
                lines.append(f"    {label}: {', '.join(parts)}")

    lines.append(f"\n{'=' * 90}")
    return "\n".join(lines)


def _grade(pct: float) -> str:
    if pct >= 95: return "A+"
    if pct >= 90: return "A"
    if pct >= 85: return "A-"
    if pct >= 80: return "B+"
    if pct >= 70: return "B"
    if pct >= 60: return "C"
    return "D"


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("  VividnessMem BENCHMARK SUITE v3")
    print("  Comparing: VividnessMem vs RAG (TF-IDF) vs MemGPT (Core/Archival)")
    print("  All systems receive IDENTICAL memories and queries")
    print("  Tests: direct recall, resonance recall, context focus, efficiency")
    print("=" * 90)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    metrics = MetricsCollector()
    t0 = time.perf_counter()

    test_longterm_recall(metrics)
    test_dormant_memory(metrics)
    test_contradiction_handling(metrics)
    test_context_pollution(metrics)
    test_identity_stability(metrics)
    test_performance(metrics)

    elapsed = time.perf_counter() - t0

    print(metrics.summary_table())
    print(compute_aggregate_scores(metrics))

    print(f"\n  Total benchmark time: {elapsed:.1f}s")

    results_dir = Path(__file__).resolve().parent
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "results": metrics.results,
        }, f, indent=2, default=str)

    report_file = results_dir / "benchmark_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("VividnessMem BENCHMARK REPORT v3\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(metrics.summary_table())
        f.write("\n")
        f.write(compute_aggregate_scores(metrics))
        f.write(f"\n\nTotal time: {elapsed:.1f}s\n")

    print(f"\n  Results: {results_file}")
    print(f"  Report:  {report_file}")


if __name__ == "__main__":
    main()
