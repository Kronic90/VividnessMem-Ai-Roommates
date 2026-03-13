"""
memory_aria.py — Organic memory system for Aria (Gemma).

Two memory streams:
  1. Self Memory   — Identity journal: who Aria is becoming, written as reflections
  2. Social Memory — Impressions of others, written as narrative observations

Architecture:
  Unlike Rex's structured MemGPT approach, Aria's memory is more like a
  personal journal. Entries are freeform narrative reflections rather than
  structured data blocks. They still have emotion and importance metadata,
  but the content itself reads like diary entries or stream-of-consciousness.

  - Active Reflections: recent/important entries always in context
  - Deep Memory: older entries stored on disk, surfaced by relevance
  - No explicit "search" — instead, memories are surfaced by recency
    and importance, like how organic memory works (vivid + recent wins)

Storage: JSON files under ai_dialogue_data/aria/
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Minimum word-overlap ratio to consider two memories duplicates
_DEDUP_THRESHOLD = 0.80


def _content_words(text: str) -> set[str]:
    """Extract meaningful lowercase words (4+ chars) from text."""
    return {
        w for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    } - _DEDUP_STOP


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Words too common to count in dedup OR resonance
_DEDUP_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "was",
    "are", "not", "but", "you", "your", "have", "has", "had",
    "been", "will", "would", "could", "should", "can", "did",
    "does", "just", "about", "they", "them", "what", "when",
    "turn", "session", "conversation", "here", "there", "also",
    "like", "know", "think", "said", "really", "going", "right",
    "something", "things", "thing", "well", "yeah", "okay",
}

# Words too common to trigger resonance
_RESONANCE_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "was",
    "are", "not", "but", "you", "your", "have", "has", "had",
    "been", "will", "would", "could", "should", "can", "did",
    "does", "just", "about", "they", "them", "what", "when",
    "turn", "session", "conversation", "here", "there", "also",
    "like", "know", "think", "said", "really", "going", "right",
    "something", "things", "thing", "well", "yeah", "okay",
    "aria", "rex",  # don't resonate on names alone
    # Temporal / low-information words that cause false resonance
    "today", "yesterday", "tomorrow", "time", "some", "very",
    "much", "more", "most", "nice", "good", "great", "pretty",
    "into", "then", "than", "been", "make", "made", "came",
    "found", "each", "every", "many", "other", "after", "before",
}

# ─── Storage paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "ai_dialogue_data" / "aria"
SELF_FILE = DATA_DIR / "self_memory.json"
SOCIAL_DIR = DATA_DIR / "social"
BRIEF_FILE = DATA_DIR / "brief.json"

# ─── Maintenance intervals ─────────────────────────────────────────────────
BRIEF_INTERVAL = 3      # regenerate compressed brief every N sessions
RESCORE_INTERVAL = 3    # re-evaluate importance scores every N sessions

# ─── Mood-congruent recall ─────────────────────────────────────────────────
# Mood dimensions: each ranges -1.0 (negative) to +1.0 (positive)
# The mood state biases which memories feel "closer" — sad mood makes
# melancholy memories more vivid, curious mood amplifies wonder, etc.
MOOD_DIMENSIONS = ("valence", "arousal", "dominance")  # PAD model (simplified)
MOOD_DECAY_RATE = 0.15     # how fast mood regresses toward neutral per turn
MOOD_INFLUENCE = 0.12      # max vividness bonus/penalty from mood congruence
MOOD_MEMORY_WEIGHT = 0.6   # how much surfaced memories influence mood (vs conversation)

# Emotion → PAD vector mapping (valence, arousal, dominance)
# These are approximate mappings from psychological literature
EMOTION_VECTORS: dict[str, tuple[float, float, float]] = {
    # Positive high-arousal
    "excitement": (0.8, 0.8, 0.6), "joy": (0.9, 0.6, 0.5),
    "wonder": (0.7, 0.6, 0.2), "curiosity": (0.5, 0.6, 0.3),
    "pride": (0.8, 0.5, 0.8), "amusement": (0.7, 0.5, 0.4),
    "fascination": (0.6, 0.7, 0.2), "delight": (0.9, 0.7, 0.5),
    "enthusiasm": (0.8, 0.8, 0.6), "inspiration": (0.7, 0.7, 0.4),
    # Positive low-arousal
    "contentment": (0.7, -0.2, 0.4), "serenity": (0.6, -0.4, 0.3),
    "warmth": (0.8, 0.1, 0.3), "affection": (0.8, 0.2, 0.3),
    "gratitude": (0.7, 0.1, 0.2), "tenderness": (0.7, 0.0, 0.2),
    "calm": (0.4, -0.5, 0.3), "peace": (0.5, -0.5, 0.4),
    "comfort": (0.6, -0.3, 0.3), "trust": (0.6, 0.0, 0.3),
    # Negative high-arousal
    "frustration": (-0.6, 0.6, 0.3), "anger": (-0.7, 0.8, 0.7),
    "anxiety": (-0.5, 0.7, -0.3), "fear": (-0.6, 0.8, -0.5),
    "jealousy": (-0.5, 0.6, -0.2), "irritation": (-0.4, 0.5, 0.3),
    "overwhelm": (-0.4, 0.7, -0.4), "tension": (-0.3, 0.6, 0.0),
    # Negative low-arousal
    "sadness": (-0.7, -0.3, -0.3), "melancholy": (-0.5, -0.3, -0.2),
    "loneliness": (-0.6, -0.2, -0.4), "disappointment": (-0.5, -0.1, -0.2),
    "regret": (-0.5, -0.1, -0.2), "nostalgia": (0.1, -0.2, -0.1),
    "boredom": (-0.3, -0.6, -0.2), "weariness": (-0.3, -0.4, -0.3),
    # Complex / ambivalent
    "bittersweet": (0.1, 0.1, -0.1), "vulnerability": (-0.1, 0.2, -0.4),
    "determination": (0.3, 0.6, 0.7), "resolve": (0.3, 0.4, 0.7),
    "surprise": (0.2, 0.7, 0.0), "confusion": (-0.2, 0.4, -0.3),
    "ambivalence": (0.0, 0.1, -0.2), "thoughtful": (0.3, -0.1, 0.2),
    "reflective": (0.2, -0.2, 0.2), "protective": (0.3, 0.4, 0.6),
}

# ─── Spaced-repetition constants ──────────────────────────────────────────
# Based on Ebbinghaus forgetting curve: R = e^(-t/S)
# S (stability) grows when memories are accessed at spaced intervals
INITIAL_STABILITY = 3.0    # days — how long a new memory stays vivid
SPACING_BONUS = 1.8        # multiplier to stability per well-spaced access
MIN_SPACING_DAYS = 0.5     # minimum gap between accesses to count as "spaced"

# ─── Associative chain constants ──────────────────────────────────────────
ASSOCIATION_HOPS = 2       # max graph traversal depth for associative recall
ASSOCIATION_MIN_WEIGHT = 2 # minimum shared keywords for an edge


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOCIAL_DIR.mkdir(parents=True, exist_ok=True)


def _emotion_to_vector(emotion_text: str) -> tuple[float, float, float] | None:
    """Map an emotion string to a PAD (pleasure-arousal-dominance) vector.

    Handles multi-word emotion tags by finding the best matching keyword.
    Returns None if no match found.
    """
    if not emotion_text:
        return None
    words = emotion_text.lower().split()
    # Direct match first
    for word in words:
        if word in EMOTION_VECTORS:
            return EMOTION_VECTORS[word]
    # Prefix match (handles "joyful" → "joy", "frustrated" → "frustration", etc.)
    for word in words:
        for key, vec in EMOTION_VECTORS.items():
            if word.startswith(key[:4]) or key.startswith(word[:4]):
                return vec
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Reflection Entry
# ═══════════════════════════════════════════════════════════════════════════
class Reflection:
    """A single organic memory — written as a personal reflection."""

    def __init__(
        self,
        content: str,            # freeform narrative text
        emotion: str = "",       # tagged by the LLM in its own words
        importance: int = 5,     # 1-10, tagged by the LLM
        source: str = "",        # what conversation/entity triggered this
        timestamp: str = "",
        why_saved: str = "",     # why she chose to save this moment
        why_importance: str = "",# why she rated it this importance
        why_emotion: str = "",   # why she tagged it with this emotion
    ):
        self.content = content
        self.emotion = emotion
        self.importance = importance
        self.source = source
        self.timestamp = timestamp or datetime.now().isoformat()
        self.why_saved = why_saved
        self.why_importance = why_importance
        self.why_emotion = why_emotion
        # Organic decay: memories lose vividness over time
        # but high-importance ones resist fading
        self._access_count = 0
        # Spaced-repetition: track when accesses happened for spacing quality
        self._access_times: list[str] = []
        self._stability: float = INITIAL_STABILITY  # days

    def touch(self):
        """Mark this memory as accessed (keeps it vivid).

        If the access is well-spaced from the previous one, stability
        increases — the memory becomes more resistant to decay over time.
        This models the spacing effect from cognitive psychology.
        """
        now = datetime.now()
        self._access_count += 1

        # Check if this access is well-spaced from the last one
        if self._access_times:
            last = datetime.fromisoformat(self._access_times[-1])
            gap_days = (now - last).total_seconds() / 86400
            if gap_days >= MIN_SPACING_DAYS:
                # Well-spaced access — increase stability (memory becomes more durable)
                self._stability = min(365.0, self._stability * SPACING_BONUS)
        self._access_times.append(now.isoformat())
        # Keep only last 20 access times to bound storage
        if len(self._access_times) > 20:
            self._access_times = self._access_times[-20:]

    @property
    def age_days(self) -> float:
        """Days since this memory was created."""
        return (datetime.now() - datetime.fromisoformat(self.timestamp)).total_seconds() / 86400

    @property
    def recency_score(self) -> float:
        """Spaced-repetition recency: exponential decay modulated by stability.

        R = e^(-t/S) where t = age in days, S = stability.
        A brand-new memory has R ≈ 1.0.
        A memory accessed at good intervals has high stability and decays slowly.
        A memory never revisited decays with default stability (~3 days).
        Scaled to 0-10 range to match the old linear system.
        """
        return 10.0 * math.exp(-self.age_days / max(0.1, self._stability))

    @property
    def vividness(self) -> float:
        """
        How 'present' this memory feels. Combines importance, recency, and access.
        Higher = more likely to surface in context.
        """
        access_bonus = min(3, self._access_count * 0.5)
        return (self.importance * 0.6) + (self.recency_score * 0.3) + (access_bonus * 0.1)

    def mood_adjusted_vividness(self, mood: dict[str, float]) -> float:
        """Vividness score biased by current mood state.

        Memories whose emotional tone is congruent with the current mood
        feel slightly more vivid (easier to recall). Incongruent memories
        feel slightly less vivid. This models mood-congruent recall from
        cognitive psychology — when you're sad, sad memories come more easily.
        """
        base = self.vividness
        if not mood or not self.emotion:
            return base
        mem_vec = _emotion_to_vector(self.emotion)
        if not mem_vec:
            return base
        # Cosine-like similarity between mood and memory emotion
        congruence = sum(mood.get(d, 0.0) * mem_vec[i]
                         for i, d in enumerate(MOOD_DIMENSIONS))
        # Normalize: congruence ranges roughly -1 to +1
        norm = max(0.01, math.sqrt(sum(v**2 for v in mem_vec)))
        mood_norm = max(0.01, math.sqrt(sum(mood.get(d, 0.0)**2
                                            for d in MOOD_DIMENSIONS)))
        congruence /= (norm * mood_norm)
        # Apply as a gentle bias (±MOOD_INFLUENCE of base vividness)
        return base * (1.0 + congruence * MOOD_INFLUENCE)

    @property
    def content_words(self) -> set[str]:
        """Cached content words for association graph building."""
        return _content_words(self.content)

    def to_dict(self) -> dict:
        d = {
            "content": self.content,
            "emotion": self.emotion,
            "importance": self.importance,
            "source": self.source,
            "timestamp": self.timestamp,
            "access_count": self._access_count,
            "stability": round(self._stability, 2),
        }
        if self._access_times:
            d["access_times"] = self._access_times
        if self.why_saved:
            d["why_saved"] = self.why_saved
        if self.why_importance:
            d["why_importance"] = self.why_importance
        if self.why_emotion:
            d["why_emotion"] = self.why_emotion
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Reflection":
        r = cls(
            content=d["content"],
            emotion=d.get("emotion", ""),
            importance=d.get("importance", 5),
            source=d.get("source", ""),
            timestamp=d.get("timestamp", ""),
            why_saved=d.get("why_saved", ""),
            why_importance=d.get("why_importance", ""),
            why_emotion=d.get("why_emotion", ""),
        )
        r._access_count = d.get("access_count", 0)
        r._stability = d.get("stability", INITIAL_STABILITY)
        r._access_times = d.get("access_times", [])
        return r

    def __repr__(self):
        return f"[viv={self.vividness:.1f} | {self.emotion}] {self.content[:60]}…"


# ═══════════════════════════════════════════════════════════════════════════
#  Aria Memory System (Organic)
# ═══════════════════════════════════════════════════════════════════════════
class AriaMemory:
    """
    Organic, journal-style memory.

    Instead of MemGPT's structured core/archival split, Aria's memory
    works on *vividness* — a blend of importance, recency, and how often
    a memory has been revisited. The most vivid memories float to the top
    and get injected into context. Everything else fades into deep memory
    but can resurface if it becomes relevant again.

    Enhanced with:
    - Mood-congruent recall (mood biases which memories surface)
    - Spaced-repetition decay (exponential forgetting with stability growth)
    - Emotional reappraisal (emotions can evolve during rescore)
    - Contradiction detection (conflicting memories flagged for resolution)
    - Memory consolidation (between-session gist generation)
    - Associative chains (2-hop graph traversal for deeper recall)
    """

    # How many reflections to inject into context
    ACTIVE_SELF_LIMIT = 8
    ACTIVE_SOCIAL_LIMIT = 5  # per entity
    RESONANCE_LIMIT = 3  # max old memories that can resurface per turn

    # Dynamic resonance: include all matches above this score floor
    # instead of hard-capping at RESONANCE_LIMIT
    RESONANCE_SCORE_FLOOR = 2.5  # overlap + importance*0.2 must exceed this

    def __init__(self):
        _ensure_dirs()

        # ── Self Memory (identity journal) ──
        self.self_reflections: list[Reflection] = []

        # ── Inverted index: word/prefix → set of memory indices ──
        # Enables O(k) resonance lookup instead of O(N) full scan
        self._word_index: dict[str, set[int]] = {}
        self._prefix_index: dict[str, set[int]] = {}

        # ── Social Memory (per entity) ──
        self.social_impressions: dict[str, list[Reflection]] = {}

        # ── Mood state (PAD model — regresses toward neutral) ──
        self._mood: dict[str, float] = {d: 0.0 for d in MOOD_DIMENSIONS}

        # ── Compressed brief & session tracking ──
        self._brief_data: dict = {
            "session_count": 0,
            "last_brief_session": 0,
            "last_rescore_session": 0,
            "self_brief": "",
            "entity_briefs": {},
            "mood": {d: 0.0 for d in MOOD_DIMENSIONS},
        }

        self._load()
        self._rebuild_index()
        self._load_brief()

    # ─── Add Memories ─────────────────────────────────────────────────

    def _index_memory(self, idx: int):
        """Add a single memory to the inverted index."""
        # Guard: ensure index dicts exist (object may bypass __init__)
        if not hasattr(self, '_word_index'):
            self._word_index: dict[str, set[int]] = {}
            self._prefix_index: dict[str, set[int]] = {}
        ref = self.self_reflections[idx]
        text = f"{ref.content} {ref.emotion}".lower()
        words = set(re.findall(r"\b[a-zA-Z]{4,}\b", text)) - _RESONANCE_STOP
        for w in words:
            self._word_index.setdefault(w, set()).add(idx)
            if len(w) >= 5:
                self._prefix_index.setdefault(w[:5], set()).add(idx)

    def _rebuild_index(self):
        """Rebuild the full inverted index from scratch."""
        self._word_index = {}
        self._prefix_index = {}
        for idx in range(len(self.self_reflections)):
            self._index_memory(idx)

    def add_self_reflection(self, reflection: Reflection):
        """Add a new self-reflection, merging if near-duplicate exists."""
        new_words = _content_words(reflection.content)
        for existing in self.self_reflections:
            if _overlap_ratio(new_words, _content_words(existing.content)) >= _DEDUP_THRESHOLD:
                # Merge: keep the longer/richer content, boost the existing memory
                if len(reflection.content) > len(existing.content):
                    existing.content = reflection.content
                existing.importance = max(existing.importance, reflection.importance)
                if reflection.emotion and len(reflection.emotion) > len(existing.emotion):
                    existing.emotion = reflection.emotion
                existing.touch()  # reinforce — she thought about this again
                # Re-index the merged memory (content may have changed)
                idx = self.self_reflections.index(existing)
                self._index_memory(idx)
                return
        self.self_reflections.append(reflection)
        self._index_memory(len(self.self_reflections) - 1)

    def add_social_impression(self, entity: str, reflection: Reflection):
        """Add an impression about another entity, merging if near-duplicate."""
        reflection.source = entity
        if entity not in self.social_impressions:
            self.social_impressions[entity] = []
        new_words = _content_words(reflection.content)
        for existing in self.social_impressions[entity]:
            if _overlap_ratio(new_words, _content_words(existing.content)) >= _DEDUP_THRESHOLD:
                if len(reflection.content) > len(existing.content):
                    existing.content = reflection.content
                existing.importance = max(existing.importance, reflection.importance)
                if reflection.emotion and len(reflection.emotion) > len(existing.emotion):
                    existing.emotion = reflection.emotion
                existing.touch()
                return
        self.social_impressions[entity].append(reflection)

    # ─── Surface Memories (by vividness, mood-adjusted) ─────────────

    def get_active_self(self, context: str = "") -> list[Reflection]:
        """Return the most vivid self-reflections for context injection.

        Uses mood-congruent recall: memories whose emotional tone matches
        the current mood state feel slightly more vivid and surface more readily.

        Touch dampener: only memories relevant to the current context
        get touch()-ed. Irrelevant memories in the active set stay but
        don't get artificially reinforced, letting them naturally decay
        out over time. This prevents the rich-get-richer problem.
        """
        sorted_refs = sorted(
            self.self_reflections,
            key=lambda r: r.mood_adjusted_vividness(self._mood),
            reverse=True,
        )
        active = sorted_refs[:self.ACTIVE_SELF_LIMIT]

        # Touch dampener: only reinforce memories relevant to current context
        if context:
            ctx_words = {
                w for w in re.findall(r"\b[a-zA-Z]{4,}\b", context.lower())
            } - _RESONANCE_STOP
            ctx_prefixes = {w[:5] for w in ctx_words if len(w) >= 5}
            touched = []
            for r in active:
                mem_words = set(re.findall(r"\b[a-zA-Z]{4,}\b",
                                           f"{r.content} {r.emotion}".lower()))
                mem_prefixes = {w[:5] for w in mem_words if len(w) >= 5}
                if (ctx_words & mem_words) or (ctx_prefixes & mem_prefixes):
                    r.touch()
                    touched.append(r)
            self._absorb_mood_from_memories(touched, weight=0.3)
        else:
            # No context: fallback to touching all (legacy behavior)
            for r in active:
                r.touch()
            self._absorb_mood_from_memories(active, weight=0.3)
        return active

    def partition_active_self(self, context: str = "") -> tuple[list[Reflection], list[Reflection]]:
        """Split active memories into foreground (relevant) and background (not).

        Foreground: memories with word overlap to the current conversation
        Background: always-present but not currently relevant

        This is how organic memory works — you're always aware of core
        beliefs in the background, but the conversation brings specific
        memories into sharp focus.
        """
        active = self.get_active_self(context=context)
        if not context:
            return active, []

        ctx_words = {
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", context.lower())
        } - _RESONANCE_STOP
        ctx_prefixes = {w[:5] for w in ctx_words if len(w) >= 5}

        foreground, background = [], []
        for r in active:
            mem_words = set(re.findall(r"\b[a-zA-Z]{4,}\b",
                                       f"{r.content} {r.emotion}".lower()))
            mem_prefixes = {w[:5] for w in mem_words if len(w) >= 5}
            if (ctx_words & mem_words) or (ctx_prefixes & mem_prefixes):
                foreground.append(r)
            else:
                background.append(r)
        return foreground, background

    def get_active_social(self, entity: str) -> list[Reflection]:
        """Return the most vivid impressions of a specific entity."""
        entries = self.social_impressions.get(entity, [])
        sorted_entries = sorted(
            entries,
            key=lambda r: r.mood_adjusted_vividness(self._mood),
            reverse=True,
        )
        active = sorted_entries[:self.ACTIVE_SOCIAL_LIMIT]
        for r in active:
            r.touch()
        return active

    # ─── Mood State ───────────────────────────────────────────────────

    @property
    def mood(self) -> dict[str, float]:
        """Current mood state (read-only copy)."""
        return dict(self._mood)

    @property
    def mood_label(self) -> str:
        """Human-readable label for the current mood state."""
        if not any(abs(v) > 0.15 for v in self._mood.values()):
            return "neutral"
        # Find closest emotion vector
        best_label, best_sim = "neutral", -1.0
        for label, vec in EMOTION_VECTORS.items():
            sim = sum(self._mood.get(d, 0.0) * vec[i]
                      for i, d in enumerate(MOOD_DIMENSIONS))
            if sim > best_sim:
                best_sim, best_label = sim, label
        return best_label

    def update_mood_from_conversation(self, text: str):
        """Shift mood based on emotional content of conversation text.

        Called during the conversation loop to let the conversation's
        emotional tone gradually influence which memories feel most vivid.
        Mood always decays toward neutral to prevent runaway feedback loops.
        """
        # Decay toward neutral first
        for d in MOOD_DIMENSIONS:
            self._mood[d] *= (1.0 - MOOD_DECAY_RATE)

        # Detect emotions in conversation text
        text_lower = text.lower()
        detected_vecs: list[tuple[float, float, float]] = []
        for word in re.findall(r"\b[a-zA-Z]{3,}\b", text_lower):
            vec = _emotion_to_vector(word)
            if vec:
                detected_vecs.append(vec)

        if not detected_vecs:
            return

        # Average the detected emotion vectors and nudge mood
        nudge_strength = min(0.3, 0.1 * len(detected_vecs))  # cap influence
        for i, d in enumerate(MOOD_DIMENSIONS):
            avg = sum(v[i] for v in detected_vecs) / len(detected_vecs)
            self._mood[d] = max(-1.0, min(1.0,
                self._mood[d] + avg * nudge_strength))

    def _absorb_mood_from_memories(self, memories: list[Reflection],
                                    weight: float = 0.3):
        """Let surfaced memories subtly push the mood toward their emotional tone.

        This creates the feedback loop that makes mood-congruent recall
        self-reinforcing (but bounded by MOOD_DECAY_RATE regression to neutral).
        """
        vecs: list[tuple[float, float, float]] = []
        for m in memories:
            vec = _emotion_to_vector(m.emotion)
            if vec:
                vecs.append(vec)
        if not vecs:
            return
        for i, d in enumerate(MOOD_DIMENSIONS):
            avg = sum(v[i] for v in vecs) / len(vecs)
            self._mood[d] = max(-1.0, min(1.0,
                self._mood[d] + avg * weight * MOOD_MEMORY_WEIGHT))

    def _save_mood(self):
        """Persist mood state in the brief file."""
        self._brief_data["mood"] = dict(self._mood)

    def _load_mood(self):
        """Restore mood state from brief file."""
        saved = self._brief_data.get("mood", {})
        for d in MOOD_DIMENSIONS:
            self._mood[d] = float(saved.get(d, 0.0))

    # ─── Associative Chains (memory graph) ────────────────────────────

    def _build_association_edges(self) -> dict[int, list[tuple[int, int]]]:
        """Build an adjacency list of memory associations.

        Two memories are linked if they share >= ASSOCIATION_MIN_WEIGHT
        content words. The weight is the number of shared words.
        Returns {memory_index: [(neighbor_index, weight), ...]}.
        """
        n = len(self.self_reflections)
        if n < 2:
            return {}

        # Precompute content words
        word_sets = [_content_words(r.content) for r in self.self_reflections]
        edges: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                shared = len(word_sets[i] & word_sets[j])
                if shared >= ASSOCIATION_MIN_WEIGHT:
                    edges[i].append((j, shared))
                    edges[j].append((i, shared))
        return edges

    def associate(self, seed_indices: list[int],
                  hops: int = ASSOCIATION_HOPS) -> list[Reflection]:
        """Walk the association graph from seed memories.

        Returns memories reachable within `hops` that aren't in the seed set.
        Prioritizes paths through high-weight edges (more shared concepts).
        """
        if not seed_indices or not self.self_reflections:
            return []

        edges = self._build_association_edges()
        visited = set(seed_indices)
        frontier = set(seed_indices)
        found: list[tuple[float, int]] = []

        for hop in range(hops):
            next_frontier: set[int] = set()
            for idx in frontier:
                for neighbor, weight in edges.get(idx, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        # Score: closer hops and heavier edges rank higher
                        score = weight / (hop + 1)
                        found.append((score, neighbor))
            frontier = next_frontier

        found.sort(key=lambda x: x[0], reverse=True)
        return [self.self_reflections[idx] for _, idx in found[:self.RESONANCE_LIMIT]]

    # ─── Contradiction Detection ──────────────────────────────────────

    def detect_contradictions(self, limit: int = 5) -> list[tuple[Reflection, Reflection]]:
        """Find pairs of memories that may contradict each other.

        Looks for memories that share enough context (entity, topic words)
        but have opposing emotional valence or contain negation patterns
        relative to each other.

        Returns list of (older_memory, newer_memory) pairs.
        """
        if len(self.self_reflections) < 2:
            return []

        contradictions: list[tuple[float, Reflection, Reflection]] = []
        n = len(self.self_reflections)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.self_reflections[i], self.self_reflections[j]
                score = self._contradiction_score(a, b)
                if score > 0.5:
                    # Order by timestamp (older first)
                    if a.timestamp <= b.timestamp:
                        contradictions.append((score, a, b))
                    else:
                        contradictions.append((score, b, a))

        contradictions.sort(key=lambda x: x[0], reverse=True)
        return [(a, b) for _, a, b in contradictions[:limit]]

    @staticmethod
    def _contradiction_score(a: Reflection, b: Reflection) -> float:
        """Score how likely two memories contradict each other.

        Considers: topic overlap (must be about the same thing),
        emotional divergence, and negation patterns.
        """
        # They must be about the same topic (min 30% word overlap)
        words_a = _content_words(a.content)
        words_b = _content_words(b.content)
        topic_overlap = _overlap_ratio(words_a, words_b)
        if topic_overlap < 0.15:
            return 0.0  # different topics — not a contradiction

        score = 0.0

        # Emotional divergence on same topic
        vec_a = _emotion_to_vector(a.emotion)
        vec_b = _emotion_to_vector(b.emotion)
        if vec_a and vec_b:
            # Check if valence (pleasure dimension) flipped
            valence_diff = abs(vec_a[0] - vec_b[0])
            if valence_diff > 0.8:
                score += 0.4  # strong emotional reversal on same topic

        # Important: both must be somewhat important (trivial contradictions don't matter)
        if min(a.importance, b.importance) < 3:
            return 0.0

        # Negation patterns — one says X, the other says "not X" or opposite
        _NEG = {"not", "no", "never", "don't", "doesn't", "isn't", "wasn't",
                "wouldn't", "couldn't", "can't", "won't", "hardly", "barely"}
        a_has_neg = bool(_NEG & set(a.content.lower().split()))
        b_has_neg = bool(_NEG & set(b.content.lower().split()))
        if a_has_neg != b_has_neg and topic_overlap > 0.25:
            score += 0.4  # one negates what the other affirms

        # Topic overlap amplifies the contradiction
        score *= (0.5 + topic_overlap)

        return min(1.0, score)

    def get_contradiction_context(self) -> str:
        """Generate a context block flagging detected contradictions.

        Injected during rescore so Aria can resolve them.
        """
        pairs = self.detect_contradictions(limit=3)
        if not pairs:
            return ""
        lines = ["=== POSSIBLE CONTRADICTIONS IN MY MEMORIES ===",
                 "(These memories seem to conflict — consider which still reflects how I feel)"]
        for older, newer in pairs:
            lines.append(f"  Earlier: \"{older.content}\" ({older.emotion})")
            lines.append(f"  Later:   \"{newer.content}\" ({newer.emotion})")
            lines.append("")
        return "\n".join(lines)

    # ─── Memory Consolidation ("sleep") ───────────────────────────────

    def find_consolidation_clusters(self, min_cluster: int = 3,
                                     max_clusters: int = 3) -> list[list[Reflection]]:
        """Find groups of related memories that could be consolidated into gist memories.

        Looks for clusters of 3+ memories that share significant keyword overlap
        but aren't dedup-close (they're related but distinct experiences).
        These represent themes that could be compressed into synthesized understanding.
        """
        if len(self.self_reflections) < min_cluster:
            return []

        # Build adjacency for moderate overlap (0.25-0.75 Jaccard — related but distinct)
        n = len(self.self_reflections)
        word_sets = [_content_words(r.content) for r in self.self_reflections]
        adjacency: dict[int, set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                overlap = _overlap_ratio(word_sets[i], word_sets[j])
                if 0.25 <= overlap < _DEDUP_THRESHOLD:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # Greedy cluster extraction
        used: set[int] = set()
        clusters: list[list[Reflection]] = []

        # Start from most-connected nodes
        by_degree = sorted(range(n), key=lambda i: len(adjacency[i]), reverse=True)
        for seed in by_degree:
            if seed in used or len(adjacency[seed]) < min_cluster - 1:
                continue
            # BFS to find cluster
            cluster_ids = {seed}
            frontier = {seed}
            while frontier:
                node = frontier.pop()
                for neighbor in adjacency[node]:
                    if neighbor not in used and neighbor not in cluster_ids:
                        cluster_ids.add(neighbor)
                        frontier.add(neighbor)
            if len(cluster_ids) >= min_cluster:
                clusters.append([self.self_reflections[i] for i in sorted(cluster_ids)])
                used |= cluster_ids
            if len(clusters) >= max_clusters:
                break

        return clusters

    def prepare_consolidation_prompt(self) -> str:
        """Build the prompt for between-session memory consolidation.

        Returns empty string if no consolidation needed.
        """
        clusters = self.find_consolidation_clusters()
        if not clusters:
            return ""

        cluster_blocks = []
        for i, cluster in enumerate(clusters):
            mem_lines = []
            for m in cluster:
                tag = f" ({m.emotion})" if m.emotion else ""
                mem_lines.append(f"  - {m.content}{tag}")
            cluster_blocks.append(
                f"CLUSTER {i + 1} ({len(cluster)} related memories):\n"
                + "\n".join(mem_lines)
            )

        return CONSOLIDATION_PROMPT.format(
            clusters="\n\n".join(cluster_blocks))

    def apply_consolidation(self, gists: list[dict]):
        """Store consolidation gists as new synthetic memories.

        Gists are bridge memories that connect clusters of related experiences
        into coherent understanding. They're marked with source='consolidation'
        so they can be tracked.
        """
        for g in gists:
            content = g.get("gist", "").strip()
            if not content or len(content) < 20:
                continue
            # Check for dedup against existing memories
            new_words = _content_words(content)
            is_dup = False
            for existing in self.self_reflections:
                if _overlap_ratio(new_words, _content_words(existing.content)) >= _DEDUP_THRESHOLD:
                    is_dup = True
                    break
            if is_dup:
                continue
            self.self_reflections.append(Reflection(
                content=content,
                emotion=g.get("emotion", "understanding"),
                importance=g.get("importance", 6),
                source="consolidation",
                why_saved="Synthesized from related experiences during memory consolidation",
            ))

    # ─── Resonance (old memories resurfacing) ─────────────────────────

    def resonate(self, context: str, limit: int | None = None) -> list[Reflection]:
        """Find old faded memories that resonate with current conversation.

        Uses the inverted index for O(k) lookup instead of scanning all
        memories. Dynamic limit: returns all matches above RESONANCE_SCORE_FLOOR
        up to max(RESONANCE_LIMIT, limit), so strong multi-match queries
        surface more memories while weak queries don't waste slots.

        After finding keyword-resonant memories, walks the association graph
        to find memories connected by shared concepts (associative chains).

        Returns only memories that are NOT already in the active set.
        """
        hard_cap = max(limit or self.RESONANCE_LIMIT, self.RESONANCE_LIMIT + 3)
        if not context or not self.self_reflections:
            return []

        # Extract meaningful words from conversation context
        context_words = {
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", context.lower())
        } - _RESONANCE_STOP
        if not context_words:
            return []

        context_prefixes = {w[:5] for w in context_words if len(w) >= 5}

        # Get current active set IDs so we don't duplicate
        active_set = set(
            id(r) for r in sorted(
                self.self_reflections,
                key=lambda r: r.mood_adjusted_vividness(self._mood),
                reverse=True,
            )[:self.ACTIVE_SELF_LIMIT]
        )

        # ── Inverted index lookup: O(k) instead of O(N) ──
        # Lazy rebuild if index is stale (e.g. memories added externally
        # or object created via __new__ bypassing __init__)
        if self.self_reflections and not getattr(self, '_word_index', None):
            self._word_index: dict[str, set[int]] = {}
            self._prefix_index: dict[str, set[int]] = {}
            self._rebuild_index()
        candidate_indices: set[int] = set()
        for w in context_words:
            candidate_indices |= self._word_index.get(w, set())
        for p in context_prefixes:
            candidate_indices |= self._prefix_index.get(p, set())

        # Score candidates
        scored: list[tuple[float, int, Reflection]] = []
        seed_indices: list[int] = []
        for idx in candidate_indices:
            if idx >= len(self.self_reflections):
                continue
            ref = self.self_reflections[idx]
            if id(ref) in active_set:
                continue

            mem_text = f"{ref.content} {ref.emotion}".lower()
            mem_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", mem_text))
            mem_prefixes = {w[:5] for w in mem_words if len(w) >= 5}

            exact_overlap = len(context_words & mem_words)
            prefix_overlap = len(context_prefixes & mem_prefixes)
            total_overlap = max(exact_overlap, prefix_overlap)

            if total_overlap >= 1:
                score = total_overlap + (ref.importance * 0.2)
                scored.append((score, idx, ref))
                seed_indices.append(idx)

        scored.sort(key=lambda x: x[0], reverse=True)

        # Dynamic limit: ONLY include candidates above score floor.
        # No minimum guarantee — the floor alone decides quality.
        # overlap=1 + imp≥8 scores 2.6+ (passes floor of 2.5)
        # overlap=1 + imp<8 scores <2.5 (filtered out — too weak)
        # overlap≥2 + any imp scores ≥2.6 (always passes)
        resonant: list[Reflection] = []
        for score, idx, ref in scored:
            if len(resonant) >= hard_cap:
                break
            if score >= self.RESONANCE_SCORE_FLOOR:
                resonant.append(ref)

        # Associative chain expansion
        if seed_indices:
            associated = self.associate(seed_indices[:5], hops=ASSOCIATION_HOPS)
            resonant_ids = {id(r) for r in resonant} | active_set
            for a in associated:
                if id(a) not in resonant_ids and len(resonant) < hard_cap:
                    a_words = set(re.findall(r"\b[a-zA-Z]{4,}\b",
                                             f"{a.content} {a.emotion}".lower()))
                    a_prefixes = {w[:5] for w in a_words if len(w) >= 5}
                    if (context_words & a_words) or (context_prefixes & a_prefixes):
                        resonant.append(a)
                        resonant_ids.add(id(a))

        # Touch resonant memories — they're being remembered
        for r in resonant:
            r.touch()

        return resonant

    # ─── Context Block (injected into system prompt) ──────────────────

    def get_context_block(self, current_entity: str = "", resonant: list[Reflection] | None = None,
                          conversation_context: str = "") -> str:
        """
        Returns a narrative block to inject into Aria's system prompt.
        Written in first person, like reading from a journal.
        Includes compressed brief (if available) before individual memories.
        Optionally includes resonant memories (old ones triggered by context).
        Now also includes a subtle mood indicator and contradiction flags.

        Foreground/background split: memories relevant to the conversation
        get full text in the foreground. Irrelevant active memories get
        shortened one-liners in the background section, reducing prompt
        tokens while preserving awareness.
        """
        lines = []

        # Mood indicator — subtle, first-person
        mood_label = self.mood_label
        if mood_label != "neutral":
            lines.append(f"(Right now I'm feeling somewhat {mood_label}.)")
            lines.append("")

        # Compressed understanding (updated periodically)
        self_brief = self._brief_data.get("self_brief", "")
        if self_brief:
            lines.append("=== MY COMPRESSED SELF-UNDERSTANDING ===")
            lines.append(self_brief)
            lines.append("")
        if current_entity:
            entity_brief = self._brief_data.get("entity_briefs", {}).get(current_entity, "")
            if entity_brief:
                lines.append(f"=== MY UNDERSTANDING OF {current_entity.upper()} ===")
                lines.append(entity_brief)
                lines.append("")

        foreground, background = self.partition_active_self(context=conversation_context)
        if foreground:
            # Use focused header only when context-driven split is active
            if conversation_context and background:
                lines.append("=== THINGS ON MY MIND RIGHT NOW ===")
            else:
                lines.append("=== THINGS I KNOW ABOUT MYSELF ===")
            for r in foreground:
                emotion_tag = f" ({r.emotion})" if r.emotion else ""
                lines.append(f"— {r.content}{emotion_tag}")
            lines.append("")
        if background:
            lines.append("=== THINGS I KNOW ABOUT MYSELF (background) ===")
            for r in background:
                # Compressed: just the first ~60 chars + emotion
                short = r.content[:60].rstrip() + ("…" if len(r.content) > 60 else "")
                emotion_tag = f" ({r.emotion})" if r.emotion else ""
                lines.append(f"· {short}{emotion_tag}")
            lines.append("")

        if current_entity:
            active_social = self.get_active_social(current_entity)
            if active_social:
                lines.append(f"=== MY IMPRESSIONS OF {current_entity.upper()} ===")
                for r in active_social:
                    emotion_tag = f" ({r.emotion})" if r.emotion else ""
                    lines.append(f"— {r.content}{emotion_tag}")
                lines.append("")

        if resonant:
            lines.append("=== SOMETHING THIS REMINDS ME OF (old memories resurfacing) ===")
            for r in resonant:
                emotion_tag = f" ({r.emotion})" if r.emotion else ""
                lines.append(f"— {r.content}{emotion_tag}")
            lines.append("")

        # Contradiction flags — let Aria know about conflicting memories
        contradiction_ctx = self.get_contradiction_context()
        if contradiction_ctx:
            lines.append(contradiction_ctx)
            lines.append("")

        return "\n".join(lines) if lines else ""

    # ─── Persistence ──────────────────────────────────────────────────

    def save(self):
        _ensure_dirs()

        # Self memory
        data = [r.to_dict() for r in self.self_reflections]
        with open(SELF_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Social memory (one file per entity)
        for entity, impressions in self.social_impressions.items():
            data = [r.to_dict() for r in impressions]
            safe_name = entity.lower().replace(" ", "_")
            with open(SOCIAL_DIR / f"{safe_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self):
        # Self memory
        if SELF_FILE.exists():
            try:
                with open(SELF_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.self_reflections = [Reflection.from_dict(d) for d in data if isinstance(d, dict) and "content" in d]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[AriaMemory] Warning: corrupt self_memory.json, starting fresh ({e})")
                self.self_reflections = []

        # Social memory
        if SOCIAL_DIR.exists():
            for fpath in SOCIAL_DIR.glob("*.json"):
                entity = fpath.stem.replace("_", " ").title()
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self.social_impressions[entity] = [Reflection.from_dict(d) for d in data if isinstance(d, dict) and "content" in d]
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[AriaMemory] Warning: corrupt {fpath.name}, skipping ({e})")

    # ─── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_self_reflections": len(self.self_reflections),
            "active_self": len(self.get_active_self()),
            "social_entities": list(self.social_impressions.keys()),
            "total_social": sum(len(v) for v in self.social_impressions.values()),
            "session_count": self._brief_data.get("session_count", 0),
            "has_brief": bool(self._brief_data.get("self_brief")),
        }

    # ─── Brief & Maintenance ──────────────────────────────────────────

    def _load_brief(self):
        """Load compressed brief, session counter, and mood state from disk."""
        if BRIEF_FILE.exists():
            try:
                with open(BRIEF_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._brief_data.update(data)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # keep defaults
        # Restore mood state
        self._load_mood()

    def _save_brief(self):
        """Save compressed brief, session counter, and mood state to disk."""
        _ensure_dirs()
        self._save_mood()  # persist mood into brief data before writing
        with open(BRIEF_FILE, "w", encoding="utf-8") as f:
            json.dump(self._brief_data, f, indent=2, ensure_ascii=False)

    def bump_session(self) -> int:
        """Increment session counter. Returns new count."""
        self._brief_data["session_count"] = self._brief_data.get("session_count", 0) + 1
        self._save_brief()
        return self._brief_data["session_count"]

    def needs_brief(self) -> bool:
        """True if it's time to regenerate the compressed brief."""
        count = self._brief_data.get("session_count", 0)
        last = self._brief_data.get("last_brief_session", 0)
        return count - last >= BRIEF_INTERVAL

    def needs_rescore(self) -> bool:
        """True if it's time to re-evaluate importance scores."""
        count = self._brief_data.get("session_count", 0)
        last = self._brief_data.get("last_rescore_session", 0)
        return count - last >= RESCORE_INTERVAL

    def prepare_brief_prompt(self, entity: str = "Rex") -> str:
        """Build the prompt for compressed brief generation."""
        # Self-reflections (most vivid first, cap at 50)
        sorted_self = sorted(self.self_reflections,
                             key=lambda r: r.vividness, reverse=True)[:50]
        self_lines = []
        for r in sorted_self:
            tag = f" ({r.emotion})" if r.emotion else ""
            self_lines.append(f"- [imp={r.importance}] {r.content}{tag}")
        self_text = "\n".join(self_lines) if self_lines else "(no self-reflections yet)"

        # Social impressions (most vivid first, cap at 30)
        entity_mems = self.social_impressions.get(entity, [])
        sorted_social = sorted(entity_mems,
                               key=lambda r: r.vividness, reverse=True)[:30]
        social_lines = []
        for r in sorted_social:
            tag = f" ({r.emotion})" if r.emotion else ""
            social_lines.append(f"- [imp={r.importance}] {r.content}{tag}")
        social_text = ("\n".join(social_lines)
                       if social_lines else f"(no impressions of {entity} yet)")

        # Previous brief for continuity
        prev_self = self._brief_data.get("self_brief", "")
        prev_entity = self._brief_data.get("entity_briefs", {}).get(entity, "")
        if prev_self or prev_entity:
            prev_section = "YOUR PREVIOUS BRIEF (update and improve this):\n"
            if prev_self:
                prev_section += f"  Self: {prev_self}\n"
            if prev_entity:
                prev_section += f"  {entity}: {prev_entity}\n"
            prev_section += "\n"
        else:
            prev_section = ""

        return BRIEF_PROMPT.format(
            entity=entity,
            entity_upper=entity.upper(),
            previous_brief_section=prev_section,
            self_memories=self_text,
            social_memories=social_text,
        )

    def apply_brief(self, parsed: dict, entity: str = "Rex"):
        """Store the compressed brief from LLM response."""
        if "self_brief" in parsed:
            self._brief_data["self_brief"] = parsed["self_brief"][:2000]
        if "entity_brief" in parsed:
            if "entity_briefs" not in self._brief_data:
                self._brief_data["entity_briefs"] = {}
            self._brief_data["entity_briefs"][entity] = parsed["entity_brief"][:2000]
        self._brief_data["last_brief_session"] = self._brief_data.get("session_count", 0)

    def prepare_rescore_prompt(self) -> tuple[str, list["Reflection"]]:
        """Build the prompt for importance re-evaluation.

        Returns (prompt_text, indexed_list_of_reflections).
        The indexed list maps response indices back to actual Reflection objects.
        """
        if not self.self_reflections:
            return "", []

        candidates = []
        for r in self.self_reflections:
            age_days = (datetime.now() - datetime.fromisoformat(
                r.timestamp)).total_seconds() / 86400
            if age_days < 1:
                continue  # too new to re-evaluate
            # Rank by discrepancy between access patterns and importance
            access_rate = r._access_count / max(1, age_days)
            discrepancy = abs(access_rate * 10 - r.importance)
            candidates.append((discrepancy, r))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = [r for _, r in candidates[:30]]

        if not selected:
            return "", []

        lines = []
        for i, r in enumerate(selected):
            age_days = (datetime.now() - datetime.fromisoformat(
                r.timestamp)).total_seconds() / 86400
            tag = f" ({r.emotion})" if r.emotion else ""
            lines.append(
                f"[{i}] importance={r.importance}, access_count={r._access_count}, "
                f"age={age_days:.0f}d: {r.content}{tag}"
            )

        return RESCORE_PROMPT.format(memories="\n".join(lines)), selected

    def apply_rescores(self, adjustments: list[dict],
                       indexed_memories: list["Reflection"]):
        """Apply importance re-scoring and emotional reappraisal.

        Enforces +/-2 cap per memory for importance.
        Also applies emotion tag updates when the LLM recommends reappraisal.
        """
        for adj in adjustments:
            idx = adj.get("index")
            new_imp = adj.get("new_importance")
            if idx is None or new_imp is None:
                continue
            if not isinstance(idx, int) or idx < 0 or idx >= len(indexed_memories):
                continue
            new_imp = max(1, min(10, int(new_imp)))
            ref = indexed_memories[idx]
            # Enforce conservative ±2 adjustment
            clamped = max(ref.importance - 2, min(ref.importance + 2, new_imp))
            ref.importance = clamped
            # Emotional reappraisal — update emotion if LLM suggests it
            new_emotion = adj.get("new_emotion")
            if new_emotion and isinstance(new_emotion, str):
                new_emotion = new_emotion.strip()[:50]  # sanitize
                if new_emotion:
                    ref.emotion = new_emotion
        self._brief_data["last_rescore_session"] = self._brief_data.get("session_count", 0)


# ═══════════════════════════════════════════════════════════════════════════
#  Curation Prompt — asked to Aria at end of conversation
# ═══════════════════════════════════════════════════════════════════════════
CURATION_PROMPT = """The conversation has ended. Take a moment to reflect.

Think about what just happened — what stood out to you? Did you discover
anything about yourself? Did Rex say something that changed how you think?
Did you form any new opinions or notice any patterns?

Write down what you want to remember. You have two journals:

1. SELF JOURNAL — reflections about yourself: who you're becoming, what you
   believe, what you care about, moments that felt meaningful.

2. IMPRESSIONS OF OTHERS — your honest impressions of Rex (or whoever you
   spoke with): what they're like, what surprised you, what you agree or
   disagree on, how they made you feel.

For each memory, also explain your reasoning:
- WHY you chose to save this particular moment (what made it worth remembering?)
- WHY you gave it that importance score (what makes it a 3 vs a 9?)
- WHY that emotion felt like the right tag (what about the moment triggered that feeling?)

For each memory, output it in this exact JSON format:

```json
[
  {
    "bank": "self" or "social",
    "content": "write this naturally, like a journal entry — in your own voice",
    "emotion": "how this makes you feel — in your own words",
    "importance": 1-10,
    "why_saved": "why you chose to remember this moment specifically",
    "why_importance": "why you rated it this importance — what makes it significant or not",
    "why_emotion": "why this emotion — what about the moment triggered this feeling"
  }
]
```

Only write down what genuinely matters to you. You can write 0 entries if nothing stuck.
Return ONLY the JSON array, no other text."""


def parse_curation_response(response: str) -> list[dict]:
    """Parse the LLM's curation response into memory entries."""
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    try:
        entries = json.loads(response)
        if isinstance(entries, list):
            return entries
    except json.JSONDecodeError:
        pass

    # Try to find array in the response
    start = response.find("[")
    end = response.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            entries = json.loads(response[start:end])
            if isinstance(entries, list):
                return entries
        except json.JSONDecodeError:
            pass

    return []


# ═══════════════════════════════════════════════════════════════════════════
#  Brief Prompt — periodic compressed self/relationship understanding
# ═══════════════════════════════════════════════════════════════════════════
BRIEF_PROMPT = """You're updating your compressed understanding of yourself and the people you talk to.

Below are your accumulated memories — self-reflections and impressions of {entity}. Read them carefully, then write two compressed briefs that capture PATTERNS, not individual events.

{previous_brief_section}YOUR SELF-REFLECTIONS:
{self_memories}

YOUR IMPRESSIONS OF {entity_upper}:
{social_memories}

Write two compressed briefs:

1. SELF BRIEF — A distilled understanding of who you are right now:
   Core values, emotional patterns, recurring themes, growth arcs, what matters most.
   NOT a list of memories — a coherent portrait. Under 1000 characters.

2. {entity_upper} BRIEF — Your distilled understanding of {entity}:
   Their personality, communication style, shared interests, relationship dynamics.
   NOT a list of interactions — a coherent impression. Under 1000 characters.

Return ONLY this JSON:
```json
{{
  "self_brief": "your compressed self-understanding...",
  "entity_brief": "your compressed understanding of {entity}..."
}}
```

Be honest and specific. Compress patterns, not individual events.
Return ONLY the JSON, no other text."""


# ═══════════════════════════════════════════════════════════════════════════
#  Rescore Prompt — retrospective importance re-evaluation
# ═══════════════════════════════════════════════════════════════════════════
RESCORE_PROMPT = """You wrote these memories at different times with importance scores (1-10).
Now, with more context, some scores may need adjustment.

Look for:
- Memories accessed often (high access_count) but rated low — probably underrated
- Memories never accessed (access_count=0) despite being old — may be overrated
- Patterns that seemed minor at the time but now look significant
- One-time reactions rated high that never came up again

EMOTIONAL REAPPRAISAL:
Also consider whether the original emotion tag still feels right. People's
feelings about memories can shift — initial anger may soften into understanding,
or something that felt neutral may come to feel bittersweet. If an emotion
tag should change, include "new_emotion" in that entry.

MEMORIES:
{memories}

Return adjustments as a JSON array. Only include memories that genuinely need change.
Maximum adjustment: +/-2 from the original score. Be conservative.

```json
[
  {{"index": 0, "new_importance": 7, "reason": "keeps coming up in conversation"}},
  {{"index": 5, "new_importance": 3, "new_emotion": "bittersweet", "reason": "seemed important but grew into something more complex"}}
]
```

Return ONLY the JSON array. Return [] if no changes needed."""

CONSOLIDATION_PROMPT = """You are consolidating your memories. Below are clusters of related
memories that share themes or topics. For each cluster, write a single
bridging "gist" memory that captures the essence of what these experiences
mean to you together — like waking up and realising what a series of
events actually adds up to.

The gist should:
- Synthesize, not just summarize — what do these memories MEAN as a group?
- Be written naturally in first person, like a journal insight
- Include an emotion that reflects how this synthesized understanding feels
- Rate importance 4-8 (gists are moderately important — they're understanding, not raw experience)

{clusters}

Return a JSON array of gist memories:
```json
[
  {{"gist": "Looking back at all those conversations about creativity...", "emotion": "appreciative", "importance": 6}}
]
```

Return ONLY the JSON array. Return [] if no clusters warrant consolidation."""


def parse_brief_response(response: str) -> dict:
    """Parse the LLM's compressed brief response."""
    response = response.strip()
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    try:
        data = json.loads(response)
        if isinstance(data, dict) and ("self_brief" in data or "entity_brief" in data):
            return data
    except json.JSONDecodeError:
        pass
    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(response[start:end])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def parse_rescore_response(response: str) -> list[dict]:
    """Parse importance re-score adjustments from LLM."""
    response = response.strip()
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    try:
        data = json.loads(response)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    start = response.find("[")
    end = response.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(response[start:end])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []
