"""
baseline_memory_systems.py — Fair baseline implementations for benchmarking.

Provides two baseline memory systems with the SAME interface so benchmarks
can swap them in and out fairly:

1. RAGMemory       — TF-IDF vector retrieval (represents RAG-style systems)
2. MemGPTMemory    — Core/archival split with keyword search (represents MemGPT)

Both expose the same API:
    .add_memory(text, emotion, importance, source, timestamp)
    .retrieve(query, limit) -> list[dict]
    .get_context_block(query="") -> str
    .get_all_memories() -> list[dict]
    .stats() -> dict
"""

import math
import re
from collections import Counter
from datetime import datetime
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
#  Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

_STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "was",
    "are", "not", "but", "you", "your", "have", "has", "had",
    "been", "will", "would", "could", "should", "can", "did",
    "does", "just", "about", "they", "them", "what", "when",
    "turn", "session", "conversation", "here", "there", "also",
    "like", "know", "think", "said", "really", "going", "right",
    "something", "things", "thing", "well", "yeah", "okay",
}


def _tokenize(text: str) -> list[str]:
    """Extract meaningful words (3+ chars, lowered, no stop words)."""
    return [
        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        if w not in _STOP_WORDS
    ]


class MemoryRecord:
    """Universal memory record used by all baselines."""

    def __init__(self, text: str, emotion: str = "", importance: int = 5,
                 source: str = "", timestamp: str = ""):
        self.text = text
        self.emotion = emotion
        self.importance = importance
        self.source = source
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "importance": self.importance,
            "source": self.source,
            "timestamp": self.timestamp,
        }

    @property
    def age_days(self) -> float:
        return (datetime.now() - datetime.fromisoformat(self.timestamp)).total_seconds() / 86400


# ═══════════════════════════════════════════════════════════════════════════
#  RAG Memory (TF-IDF based retrieval)
# ═══════════════════════════════════════════════════════════════════════════

class RAGMemory:
    """
    Retrieval-Augmented Generation baseline.

    Stores all memories as text documents. On retrieval, uses TF-IDF
    cosine similarity to find the most relevant memories for a query.

    This is a faithful representation of how most RAG memory systems work:
    - No decay, no importance weighting at retrieval time
    - Pure semantic/lexical similarity drives retrieval
    - All memories are equally "alive" regardless of age
    """

    def __init__(self, context_limit: int = 8):
        self.memories: list[MemoryRecord] = []
        self.context_limit = context_limit  # max memories in context
        self._idf_cache: dict[str, float] = {}
        self._dirty = True

    def add_memory(self, text: str, emotion: str = "", importance: int = 5,
                   source: str = "", timestamp: str = ""):
        self.memories.append(MemoryRecord(text, emotion, importance, source, timestamp))
        self._dirty = True

    def _rebuild_idf(self):
        """Recompute IDF scores across all documents."""
        if not self._dirty or not self.memories:
            return
        n = len(self.memories)
        doc_freq: Counter = Counter()
        for mem in self.memories:
            unique_words = set(_tokenize(mem.text))
            for w in unique_words:
                doc_freq[w] += 1
        self._idf_cache = {
            w: math.log((n + 1) / (df + 1)) + 1.0
            for w, df in doc_freq.items()
        }
        self._dirty = False

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        """Compute TF-IDF vector for a text string."""
        words = _tokenize(text)
        tf = Counter(words)
        total = len(words) or 1
        return {
            w: (count / total) * self._idf_cache.get(w, 1.0)
            for w, count in tf.items()
        }

    def _cosine_sim(self, v1: dict[str, float], v2: dict[str, float]) -> float:
        """Cosine similarity between two sparse TF-IDF vectors."""
        shared = set(v1) & set(v2)
        if not shared:
            return 0.0
        dot = sum(v1[w] * v2[w] for w in shared)
        n1 = math.sqrt(sum(v * v for v in v1.values()))
        n2 = math.sqrt(sum(v * v for v in v2.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def retrieve(self, query: str, limit: int = 8) -> list[dict]:
        """Retrieve most relevant memories by TF-IDF cosine similarity."""
        self._rebuild_idf()
        if not self.memories:
            return []
        query_vec = self._tfidf_vector(query)
        scored = []
        for mem in self.memories:
            mem_vec = self._tfidf_vector(mem.text)
            sim = self._cosine_sim(query_vec, mem_vec)
            scored.append((sim, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**m.to_dict(), "retrieval_score": s}
            for s, m in scored[:limit]
        ]

    def get_context_block(self, query: str = "") -> str:
        """Build context block — if query given, retrieve by similarity; else latest."""
        if query:
            results = self.retrieve(query, self.context_limit)
        else:
            # No query: return most recent memories (typical RAG fallback)
            sorted_mems = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)
            results = [m.to_dict() for m in sorted_mems[:self.context_limit]]
        lines = ["=== MEMORY CONTEXT (RAG) ==="]
        for r in results:
            lines.append(f"- [{r.get('emotion', '')}] {r['text']}")
        return "\n".join(lines)

    def get_all_memories(self) -> list[dict]:
        return [m.to_dict() for m in self.memories]

    def stats(self) -> dict:
        return {
            "system": "RAG (TF-IDF)",
            "total_memories": len(self.memories),
            "context_limit": self.context_limit,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  MemGPT-style Memory (Core/Archival with keyword search)
# ═══════════════════════════════════════════════════════════════════════════

class MemGPTMemory:
    """
    MemGPT-inspired memory baseline.

    Two-tier architecture:
    - Core memory: small set of important entries always in context
    - Archival memory: everything else, searchable by keyword

    Routing rule: importance >= 7 → core (capped at core_limit)
    If core is full and a new high-importance entry arrives, the
    lowest-importance core entry is demoted to archival.

    This is a faithful representation of MemGPT's approach:
    - Hard threshold for "important" vs "archival"
    - No continuous decay or vividness curve
    - Keyword-based search for archival retrieval
    - No mood, no spaced repetition, no associative chains
    """

    IMPORTANCE_THRESHOLD = 7  # ≥ this goes to core

    def __init__(self, core_limit: int = 10, archival_search_limit: int = 5):
        self.core: list[MemoryRecord] = []
        self.archival: list[MemoryRecord] = []
        self.core_limit = core_limit
        self.archival_search_limit = archival_search_limit

    def add_memory(self, text: str, emotion: str = "", importance: int = 5,
                   source: str = "", timestamp: str = ""):
        record = MemoryRecord(text, emotion, importance, source, timestamp)
        if importance >= self.IMPORTANCE_THRESHOLD and len(self.core) < self.core_limit:
            self.core.append(record)
        elif importance >= self.IMPORTANCE_THRESHOLD and len(self.core) >= self.core_limit:
            # Try to swap out the least important core entry
            self.core.sort(key=lambda e: e.importance)
            if importance > self.core[0].importance:
                demoted = self.core.pop(0)
                self.archival.append(demoted)
                self.core.append(record)
            else:
                self.archival.append(record)
        else:
            self.archival.append(record)

    def search_archival(self, keyword: str, limit: int = 0) -> list[MemoryRecord]:
        """Keyword search in archival memory, sorted by importance."""
        limit = limit or self.archival_search_limit
        kw = keyword.lower()
        hits = [
            m for m in self.archival
            if kw in m.text.lower() or kw in m.emotion.lower()
        ]
        hits.sort(key=lambda m: m.importance, reverse=True)
        return hits[:limit]

    def retrieve(self, query: str, limit: int = 8) -> list[dict]:
        """Retrieve: core always included + keyword search of archival."""
        results = [m.to_dict() for m in self.core]
        # Search archival with each significant query word
        archival_hits = set()
        for word in _tokenize(query):
            for m in self.search_archival(word, limit=3):
                idx = id(m)
                if idx not in archival_hits:
                    archival_hits.add(idx)
                    results.append(m.to_dict())

        # Sort by importance, take top limit
        results.sort(key=lambda r: r["importance"], reverse=True)
        return results[:limit]

    def get_context_block(self, query: str = "") -> str:
        """Build context block — core always present, archival searched if query."""
        lines = ["=== CORE MEMORY ==="]
        for m in self.core:
            lines.append(f"- [{m.emotion}, {m.importance}/10] {m.text}")

        if query:
            archival_results = []
            for word in _tokenize(query):
                for m in self.search_archival(word, limit=3):
                    if m.to_dict() not in archival_results:
                        archival_results.append(m.to_dict())
            if archival_results:
                lines.append("\n=== ARCHIVAL MATCHES ===")
                for r in archival_results[:self.archival_search_limit]:
                    lines.append(f"- [{r['emotion']}, {r['importance']}/10] {r['text']}")

        return "\n".join(lines)

    def get_all_memories(self) -> list[dict]:
        return [m.to_dict() for m in self.core + self.archival]

    def stats(self) -> dict:
        return {
            "system": "MemGPT (Core/Archival)",
            "core_count": len(self.core),
            "archival_count": len(self.archival),
            "total_memories": len(self.core) + len(self.archival),
            "core_limit": self.core_limit,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  VividnessMem Adapter (wraps our actual system for uniform interface)
# ═══════════════════════════════════════════════════════════════════════════

class VividnessMemAdapter:
    """
    Adapter that wraps memory_aria.AriaMemory to expose the same
    interface as the baselines, so benchmarks can swap systems freely.

    IMPORTANT: This adapter models how VividnessMem is actually used in
    the real system. Active memories (top K by vividness) are always
    present in context. Resonance surfaces older memories when the
    conversation text triggers keyword overlap. retrieve() is non-mutating
    so repeated benchmark calls don't contaminate state.
    """

    def __init__(self, aria_memory):
        """Takes an already-constructed AriaMemory instance."""
        self._mem = aria_memory

    def add_memory(self, text: str, emotion: str = "", importance: int = 5,
                   source: str = "", timestamp: str = ""):
        from memory_aria import Reflection
        r = Reflection(
            content=text,
            emotion=emotion,
            importance=importance,
            source=source,
            timestamp=timestamp,
        )
        self._mem.add_self_reflection(r)

    def retrieve(self, query: str, limit: int = 8) -> list[dict]:
        """Retrieve: resonance (query-relevant) + relevance-gated active.

        In the real system, the context block has SEPARATE sections for
        active memories and resonant memories — they don't compete for
        slots.  We model this by giving resonance priority: resonant
        results first (query-specific), then fill remaining limit with
        active memories that share at least one content word with the
        query (relevance gate).  Active memories with zero word overlap
        are excluded from retrieval results — they're still visible in
        the full context block but aren't retrieval hits for unrelated
        queries.

        Unlike get_active_self(), this does NOT touch() memories or
        mutate mood state — so benchmark calls are idempotent.
        """
        import re as _re
        _GATE_STOP = {
            "the", "and", "for", "that", "this", "with", "from", "was",
            "are", "not", "but", "you", "your", "have", "has", "had",
            "been", "will", "would", "could", "should", "can", "did",
            "does", "just", "about", "they", "them", "what", "when",
            "turn", "session", "conversation", "here", "there", "also",
            "like", "know", "think", "said", "really", "going", "right",
            "something", "things", "thing", "well", "yeah", "okay",
        }
        query_words = {
            w for w in _re.findall(r"\b[a-zA-Z]{4,}\b", query.lower())
        } - _GATE_STOP
        query_prefixes = {w[:5] for w in query_words if len(w) >= 5}

        # Resonance: keyword overlap with query (query-specific retrieval)
        resonant = self._mem.resonate(query, limit=limit)

        # Non-mutating active: sort by vividness without touching
        sorted_refs = sorted(
            self._mem.self_reflections,
            key=lambda r: r.vividness,
            reverse=True,
        )
        active = sorted_refs[:self._mem.ACTIVE_SELF_LIMIT]

        # Build results: resonance FIRST (query-relevant), then active
        seen = set()
        results = []

        # 1. Resonance results take priority (these matched the query)
        for r in resonant:
            if r.content not in seen:
                seen.add(r.content)
                results.append({
                    "text": r.content,
                    "emotion": r.emotion,
                    "importance": r.importance,
                    "source": r.source,
                    "timestamp": r.timestamp,
                    "vividness": r.vividness,
                })

        # 2. Fill remaining slots with active, gated by relevance
        for r in active:
            if len(results) >= limit:
                break
            if r.content not in seen:
                mem_words = set(_re.findall(r"\b[a-zA-Z]{4,}\b",
                                            f"{r.content} {r.emotion}".lower()))
                mem_prefixes = {w[:5] for w in mem_words if len(w) >= 5}
                if (query_words & mem_words) or (query_prefixes & mem_prefixes):
                    seen.add(r.content)
                    results.append({
                        "text": r.content,
                        "emotion": r.emotion,
                        "importance": r.importance,
                        "source": r.source,
                        "timestamp": r.timestamp,
                        "vividness": r.vividness,
                    })

        return results

    def get_context_block(self, query: str = "") -> str:
        resonant = self._mem.resonate(query) if query else None
        return self._mem.get_context_block(
            resonant=resonant, conversation_context=query,
        )

    def get_all_memories(self) -> list[dict]:
        return [
            {
                "text": r.content,
                "emotion": r.emotion,
                "importance": r.importance,
                "source": r.source,
                "timestamp": r.timestamp,
                "vividness": r.vividness,
            }
            for r in self._mem.self_reflections
        ]

    def stats(self) -> dict:
        s = self._mem.stats()
        s["system"] = "VividnessMem"
        return s
