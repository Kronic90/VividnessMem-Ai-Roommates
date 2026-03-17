"""
VividnessMem — Organic Memory System for LLM Agents
====================================================

A standalone, plug-and-play memory system that any LLM-based agent can use.
No personas, no frameworks — just import and go.

    from VividnessMem import VividnessMem

Features:
 • Vividness-ranked recall (spaced-repetition decay, importance weighting)
 • Mood-congruent memory (emotional state biases which memories surface)
 • Associative chains (memories linked by shared concepts)
 • Resonance (old faded memories resurfacing when context matches)
 • Foreground / background context split (saves prompt tokens)
 • Structured background compression (preserves emotion + entity + key facets)
 • Soft deduplication with merge-on-conflict
 • Contradiction detection between memories
 • Memory consolidation (clustering related memories into gist insights)
 • Compressed briefs (periodic LLM-generated self-summaries)
 • Importance re-scoring with emotional reappraisal
 • Memory dreaming (between-session cross-memory pattern discovery)
 • Regret scoring (track importance estimation mistakes over time)
 • Relationship arc tracking (per-entity warmth trajectory)
 • Inverted word/prefix index for O(k) resonance lookup
 • Synonym ring for semantic bridging (afraid↔fear, happy↔joy, etc.)
 • Touch dampener (only reinforces context-relevant memories)
 • Full JSON persistence to disk
 • Optional AES encryption at rest (Fernet + PBKDF2)
 • Professional mode (disable emotion biasing for task-oriented agents)
 • Task memory branch (projects, tasks, action logging, solution patterns)
 • Solution patterns (anti-repeat-mistake memory — "I've seen this before")
 • Project knowledge tracking (artifacts: files, characters, APIs, concepts)
 • Project-scoped decay tiers (active → cooling → cold → archived)
 • Adaptive auto-tracking (professional mode, self-tuning frequency)

Author : Kronic90  — https://github.com/Kronic90/VividnessMem-Ai-Roommates
License: MIT
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Cross-platform file locking (stdlib only)
if sys.platform == "win32":
    import msvcrt

    def _lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, max(1, os.fstat(f.fileno()).st_size or 1))

    def _unlock_file(f):
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, max(1, os.fstat(f.fileno()).st_size or 1))
        except OSError:
            pass
else:
    import fcntl

    def _lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# Optional encryption — only needed if you pass encryption_key
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

# ═══════════════════════════════════════════════════════════════════════════
#  Constants & helpers
# ═══════════════════════════════════════════════════════════════════════════

_DEDUP_THRESHOLD = 0.80  # Jaccard overlap to consider two memories duplicates

# ── Synonym ring for semantic bridging ─────────────────────────────────────
# Each group contains words that should match each other during resonance
# and index lookup.  Kept small and human-curated — no GPU needed.
_SYNONYM_GROUPS: list[frozenset[str]] = [
    frozenset({"afraid", "fear", "scared", "frightened", "terrified", "fearful"}),
    frozenset({"happy", "glad", "joyful", "joyous", "cheerful", "delighted", "pleased"}),
    frozenset({"sad", "unhappy", "sorrowful", "melancholy", "depressed", "gloomy"}),
    frozenset({"angry", "furious", "enraged", "irate", "livid", "outraged", "mad"}),
    frozenset({"anxious", "nervous", "worried", "uneasy", "apprehensive", "tense"}),
    frozenset({"love", "adore", "cherish", "affection", "devotion", "fondness"}),
    frozenset({"hate", "loathe", "detest", "despise", "abhor"}),
    frozenset({"tired", "exhausted", "fatigued", "weary", "drained"}),
    frozenset({"confused", "bewildered", "perplexed", "puzzled", "baffled"}),
    frozenset({"lonely", "isolated", "alone", "solitary"}),
    frozenset({"proud", "pride", "accomplished", "triumphant"}),
    frozenset({"ashamed", "shame", "embarrassed", "humiliated", "mortified"}),
    frozenset({"grateful", "thankful", "appreciative", "gratitude"}),
    frozenset({"jealous", "envious", "envy", "jealousy", "covetous"}),
    frozenset({"trust", "faith", "confidence", "reliance"}),
    frozenset({"distrust", "suspicion", "mistrust", "skeptical", "doubtful"}),
    frozenset({"hopeful", "optimistic", "hope", "expectant"}),
    frozenset({"hopeless", "despair", "despairing", "despondent"}),
    frozenset({"calm", "serene", "peaceful", "tranquil", "relaxed"}),
    frozenset({"excited", "thrilled", "exhilarated", "eager", "enthusiastic"}),
    frozenset({"bored", "tedious", "monotonous", "uninterested"}),
    frozenset({"surprised", "astonished", "amazed", "stunned", "shocked"}),
    frozenset({"disappointed", "letdown", "dissatisfied", "dismayed"}),
    frozenset({"curious", "inquisitive", "interested", "intrigued"}),
    frozenset({"guilty", "remorseful", "regretful", "contrite"}),
    frozenset({"beautiful", "gorgeous", "stunning", "lovely", "attractive"}),
    frozenset({"ugly", "hideous", "grotesque", "unsightly"}),
    frozenset({"smart", "intelligent", "clever", "brilliant", "bright"}),
    frozenset({"stupid", "dumb", "foolish", "idiotic", "ignorant"}),
    frozenset({"fast", "quick", "rapid", "swift", "speedy"}),
    frozenset({"slow", "sluggish", "gradual", "leisurely"}),
    frozenset({"big", "large", "huge", "enormous", "massive", "giant"}),
    frozenset({"small", "tiny", "little", "minuscule", "miniature"}),
    frozenset({"important", "crucial", "vital", "essential", "critical", "significant"}),
    frozenset({"friend", "companion", "buddy", "ally", "pal"}),
    frozenset({"enemy", "foe", "adversary", "opponent", "rival"}),
    frozenset({"help", "assist", "support", "aid"}),
    frozenset({"hurt", "harm", "injure", "wound", "damage"}),
    frozenset({"begin", "start", "commence", "initiate"}),
    frozenset({"end", "finish", "conclude", "terminate", "complete"}),
    frozenset({"create", "build", "construct", "produce", "generate", "craft"}),
    frozenset({"destroy", "demolish", "ruin", "wreck", "annihilate", "remove", "purge"}),
    frozenset({"remember", "recall", "recollect", "reminisce"}),
    frozenset({"forget", "overlook", "neglect", "disregard"}),
    # General-domain synonyms (beyond emotions)
    frozenset({"weather", "climate", "temperature", "forecast", "meteorological"}),
    frozenset({"sports", "athletic", "game", "match", "competition", "tournament"}),
    frozenset({"baseball", "pitching", "batting", "pitcher", "batter", "mound"}),
    frozenset({"finance", "stock", "market", "trading", "investment", "portfolio"}),
    frozenset({"crypto", "bitcoin", "cryptocurrency", "blockchain", "ethereum"}),
    frozenset({"social", "twitter", "reddit", "platform", "trending", "viral"}),
    frozenset({"music", "song", "album", "artist", "playlist", "track"}),
    frozenset({"video", "stream", "watch", "youtube", "content", "clip"}),
    frozenset({"health", "medical", "fitness", "exercise", "wellness", "boxing"}),
    frozenset({"food", "recipe", "cooking", "cuisine", "meal", "ingredient"}),
    frozenset({"travel", "flight", "hotel", "booking", "destination", "trip"}),
    frozenset({"news", "article", "headline", "report", "press", "journalism"}),
    frozenset({"category", "type", "kind", "sort", "class", "genre"}),
    frozenset({"ranking", "leaderboard", "standings", "leaders", "stats", "statistics"}),
    frozenset({"animal", "breed", "species", "pet", "domestic", "wildlife"}),
    frozenset({"latest", "recent", "current", "newest", "updated"}),
    frozenset({"popular", "trending", "famous", "viral", "mainstream"}),
    frozenset({"price", "cost", "rate", "value", "worth", "pricing"}),
    frozenset({"location", "place", "area", "region", "zone", "locale"}),
    frozenset({"schedule", "calendar", "timetable", "agenda", "timeline"}),
    frozenset({"user", "account", "profile", "member", "subscriber"}),
    frozenset({"search", "find", "locate", "lookup", "query"}),
    frozenset({"delete", "drop", "erase", "discard"}),
    frozenset({"update", "modify", "change", "alter", "edit", "revise"}),
]

# Build lookup: word → set of synonyms (excluding itself)
_SYNONYM_MAP: dict[str, frozenset[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        # Merge if a word appears in multiple groups (keeps symmetry)
        existing = _SYNONYM_MAP.get(_word, frozenset())
        _SYNONYM_MAP[_word] = (existing | _group) - {_word}


def _expand_synonyms(words: set[str]) -> set[str]:
    """Expand a word set with synonyms from the built-in ring."""
    extra: set[str] = set()
    for w in words:
        syns = _SYNONYM_MAP.get(w)
        if syns:
            extra |= syns
    return words | extra

# Common English stop words stripped before dedup comparison
_DEDUP_STOP = frozenset(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the "
    "and but if or because as until while of at by for with about against "
    "between through during before after above below to from up down in out "
    "on off over under again further then once here there when where why how "
    "all both each few more most other some such no nor not only own same so "
    "than too very s t can will just don should now d ll m o re ve y ain "
    "aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan "
    "shouldn wasn weren won wouldn really think feel like know want "
    "would could also much get got going one thing".split()
)

# Stop words for resonance matching — skip words too common to be meaningful
_RESONANCE_STOP = frozenset(
    "that this with have from what your about been just like "
    "them they when will been know really think feel would could also much "
    "some more than very into does doing make made been thing there their "
    "then these those which where while each other another such even "
    "want wanted wants going goes gone come came "
    "still always never sometimes maybe perhaps "
    "today yesterday tomorrow tonight morning evening night "
    "something anything everything nothing "
    "someone anyone everyone nobody "
    "here there everywhere somewhere ".split()
)

# Short-word preservation — keep 2-3 char tokens that look like meaningful
# acronyms, codes, or values (API, SQL, US, ERA, 3d, etc.)
_SHORT_WORD_RE = re.compile(r"\b[A-Za-z0-9]{2,3}\b")
_SHORT_STOP = frozenset(
    "the a an is are was were be am do did has had may can its "
    "and but not for nor yet who how why let try got too few own "
    "one two all any her him his our you she her its "
    "said says tell told talk".split()
)


def _extract_short_tokens(text: str) -> set[str]:
    """Keep 2-3 char tokens that look like acronyms, codes, or values."""
    return {
        w.lower() for w in _SHORT_WORD_RE.findall(text)
        if w.lower() not in _SHORT_STOP
    }


def _resonance_words(text: str) -> set[str]:
    """Extract words for resonance/indexing — including short tokens.

    Replaces the old `re.findall(r"\b[a-zA-Z]{4,}\b", ...)` pattern.
    Now also captures 2-3 char acronyms, codes, and abbreviations.
    """
    long_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())) - _RESONANCE_STOP
    short_words = _extract_short_tokens(text)
    return long_words | short_words


def _bigrams(text: str) -> set[str]:
    """Extract contiguous 2-word phrases for phrase-level matching.

    Bigrams capture collocations that single words miss:
    'heat resistant', 'regular season', 'stock market', etc.
    """
    words = [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) if len(w) >= 2]
    return {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}


def _trigrams(text: str) -> set[str]:
    """Extract contiguous 3-word phrases for phrase-level matching.

    Trigrams capture longer collocations that bigrams miss:
    'new york city', 'stock market crash', 'machine learning model', etc.
    """
    words = [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()) if len(w) >= 2]
    return {f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words) - 2)}


def _adaptive_floor(top_score: float, base_floor: float) -> float:
    """Dynamic retrieval floor based on the best match score.

    Strong top match -> higher floor (only return good results).
    Weak top match -> lower floor (return best-effort results).
    """
    if top_score <= 0:
        return base_floor * 0.5
    return max(top_score * 0.4, base_floor * 0.5)


def _content_words(text: str) -> set[str]:
    """Extract meaningful (non-stop) words from text for comparison."""
    return set(text.lower().split()) - _DEDUP_STOP


def _overlap_ratio(words_a: set[str], words_b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# ═══════════════════════════════════════════════════════════════════════════
#  Emotion → PAD vector mapping (Pleasure-Arousal-Dominance)
# ═══════════════════════════════════════════════════════════════════════════
EMOTION_VECTORS: dict[str, tuple[float, float, float]] = {
    # positive-calm
    "content":         ( 0.7,  -0.3,  0.3),
    "peaceful":        ( 0.8,  -0.5,  0.4),
    "serene":          ( 0.8,  -0.6,  0.3),
    "grateful":        ( 0.8,   0.1,  0.4),
    "appreciative":    ( 0.7,   0.0,  0.3),
    "hopeful":         ( 0.6,   0.2,  0.3),
    "warm":            ( 0.7,   0.1,  0.3),
    "tender":          ( 0.6,  -0.1,  0.0),
    "affectionate":    ( 0.7,   0.2,  0.2),
    # positive-active
    "happy":           ( 0.8,   0.4,  0.5),
    "joyful":          ( 0.9,   0.6,  0.5),
    "excited":         ( 0.7,   0.8,  0.5),
    "enthusiastic":    ( 0.7,   0.7,  0.5),
    "proud":           ( 0.7,   0.4,  0.7),
    "amused":          ( 0.6,   0.5,  0.4),
    "playful":         ( 0.7,   0.6,  0.4),
    "inspired":        ( 0.7,   0.5,  0.5),
    "curious":         ( 0.5,   0.6,  0.3),
    "fascinated":      ( 0.6,   0.6,  0.3),
    "motivated":       ( 0.6,   0.5,  0.6),
    "triumphant":      ( 0.8,   0.6,  0.8),
    "delighted":       ( 0.9,   0.5,  0.5),
    # neutral / reflective
    "neutral":         ( 0.0,   0.0,  0.0),
    "thoughtful":      ( 0.2,   0.1,  0.3),
    "reflective":      ( 0.2,  -0.1,  0.2),
    "contemplative":   ( 0.2,  -0.2,  0.2),
    "nostalgic":       ( 0.3,  -0.1,  0.1),
    "bittersweet":     ( 0.1,   0.0,  0.0),
    "wistful":         ( 0.1,  -0.2,  0.0),
    "understanding":   ( 0.4,   0.0,  0.4),
    # negative-low arousal
    "sad":             (-0.6,  -0.3, -0.3),
    "lonely":          (-0.7,  -0.4, -0.5),
    "melancholy":      (-0.5,  -0.4, -0.2),
    "disappointed":    (-0.5,  -0.2, -0.3),
    "guilty":          (-0.5,   0.1, -0.6),
    "insecure":        (-0.4,   0.2, -0.5),
    "vulnerable":      (-0.3,   0.1, -0.5),
    # negative-high arousal
    "anxious":         (-0.5,   0.7, -0.4),
    "frustrated":      (-0.5,   0.6, -0.2),
    "angry":           (-0.7,   0.8,  0.2),
    "hurt":            (-0.6,   0.3, -0.5),
    "confused":        (-0.3,   0.4, -0.3),
    "overwhelmed":     (-0.4,   0.7, -0.5),
    "embarrassed":     (-0.5,   0.5, -0.6),
    "jealous":         (-0.6,   0.6, -0.2),
    "afraid":          (-0.7,   0.8, -0.6),
    "resentful":       (-0.6,   0.5, -0.1),
}

# ── Spaced-repetition constants ───────────────────────────────────────────
INITIAL_STABILITY = 3.0   # days before first ~50 % fade
SPACING_BONUS    = 1.8    # multiplier per well-spaced touch
MIN_SPACING_DAYS = 0.5    # touches closer than this don't boost stability
STABILITY_CAP    = 180.0  # hard ceiling — no memory is forever
DIMINISHING_RATE = 0.85   # each successive touch gives less stability boost

# ── Emotional reappraisal constants ───────────────────────────────────────
NEGATIVE_HALFLIFE_DAYS = 14.0  # negative mood-congruence boost halves every N days
MAX_NEGATIVE_BOOST     = 0.10  # cap on mood-congruence boost for negative memories

# ── Association & brief constants ─────────────────────────────────────────
ASSOCIATION_HOPS       = 2
ASSOCIATION_MIN_WEIGHT = 2
BRIEF_INTERVAL         = 3   # regenerate compressed brief every N sessions
RESCORE_INTERVAL       = 3   # re-evaluate importance scores every N sessions
DREAM_INTERVAL         = 2   # run memory dreaming every N sessions

# ── Task memory constants ─────────────────────────────────────────────────
PROJECT_DECAY_ACTIVE_DAYS   = 7     # 0–7 days since last access: normal decay
PROJECT_DECAY_COOLING_DAYS  = 30    # 8–30 days: accelerated decay
PROJECT_DECAY_COLD_DAYS     = 90    # 31–90 days: aggressive decay
PROJECT_DECAY_COOLING_MULT  = 2.5   # decay multiplier in cooling tier
PROJECT_DECAY_COLD_MULT     = 6.0   # decay multiplier in cold tier
SOLUTION_INITIAL_IMPORTANCE = 8     # explicit record_solution importance
SOLUTION_AUTO_IMPORTANCE    = 6     # auto-extracted solution importance
SOLUTION_REUSE_BOOST        = 0.3   # importance bump per reuse (capped at +3)
AUTO_TRACK_QUALITY_INIT     = 0.5   # starting agent_track_quality score
AUTO_TRACK_EMA_ALPHA        = 0.1   # smoothing factor for quality updates


def _emotion_to_vector(emotion: str) -> tuple[float, float, float] | None:
    """Map an emotion label to its PAD vector, or None if unknown."""
    if not emotion:
        return None
    key = emotion.lower().strip()
    if key in EMOTION_VECTORS:
        return EMOTION_VECTORS[key]
    # Fuzzy: check if the emotion starts with a known key
    for k, v in EMOTION_VECTORS.items():
        if key.startswith(k) or k.startswith(key):
            return v
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Memory (single memory unit)
# ═══════════════════════════════════════════════════════════════════════════

class Memory:
    """A single memory with organic vividness decay.

    Memories fade over time following a spaced-repetition curve.
    Retrieving a memory at the right time reinforces it (touch),
    while cramming (too-frequent access) has diminishing returns.
    Each memory carries an emotion tag mapped to a PAD vector for
    mood-congruent recall.
    """

    __slots__ = (
        "content", "emotion", "importance", "timestamp",
        "source", "entity", "_access_count", "_last_access",
        "_stability", "why_saved", "_regret", "_believed_importance",
    )

    def __init__(self, content: str, emotion: str = "neutral",
                 importance: int = 5, source: str = "reflection",
                 entity: str = "", why_saved: str = ""):
        self.content = content
        self.emotion = emotion
        self.importance = max(1, min(10, importance))
        self.timestamp = datetime.now().isoformat()
        self.source = source
        self.entity = entity
        self._access_count: int = 0
        self._last_access: str = self.timestamp
        self._stability: float = INITIAL_STABILITY
        self.why_saved = why_saved
        self._regret: float = 0.0          # 0.0–1.0, how overestimated this was
        self._believed_importance: int = 0  # original importance before correction (0 = never rescored)

    # ── spaced-repetition touch ───────────────────────────────────────
    def touch(self):
        """Record an access. Well-spaced touches increase stability.

        Stability has diminishing returns (each touch adds less) and a
        hard cap so no memory becomes permanently immune to decay.
        """
        now = datetime.now()
        last = datetime.fromisoformat(self._last_access)
        gap_days = (now - last).total_seconds() / 86400

        if gap_days >= MIN_SPACING_DAYS:
            # Diminishing returns: bonus shrinks with repeated access
            effective_bonus = 1.0 + (SPACING_BONUS - 1.0) * (DIMINISHING_RATE ** self._access_count)
            self._stability = min(self._stability * effective_bonus, STABILITY_CAP)
        self._access_count += 1
        self._last_access = now.isoformat()

    # ── vividness (decayed importance) ────────────────────────────────
    @property
    def vividness(self) -> float:
        age_days = (datetime.now()
                    - datetime.fromisoformat(self.timestamp)
                    ).total_seconds() / 86400
        retention = math.exp(-age_days / max(self._stability, 0.1))
        return self.importance * retention

    def mood_adjusted_vividness(self, mood_vector: tuple[float, float, float]) -> float:
        """Vividness boosted when memory emotion matches current mood.

        Grudge-bug fix: negative-valence memories have their congruence
        boost *capped* and *decayed* over time (emotional reappraisal).
        This prevents a negative-mood spiral from locking the agent into
        perpetual anger/sadness.
        """
        base = self.vividness
        mem_vec = _emotion_to_vector(self.emotion)
        if not mem_vec or mood_vector == (0.0, 0.0, 0.0):
            return base
        dot = sum(a * b for a, b in zip(mem_vec, mood_vector))
        clamped = max(-1.0, min(1.0, dot))

        # ── Emotional reappraisal for negative memories ──────────────
        # If both the memory and mood are negative-valence and the dot
        # is positive (congruence boost), apply two dampeners:
        #   1. Cap the boost at MAX_NEGATIVE_BOOST (lower than the
        #      normal ±0.15)
        #   2. Decay the boost with the memory's age so old grudges
        #      lose their mood-amplification power over time.
        mem_valence = mem_vec[0]           # pleasure axis: <0 = negative
        mood_valence = mood_vector[0]
        if mem_valence < 0 and mood_valence < 0 and clamped > 0:
            age_days = (datetime.now()
                        - datetime.fromisoformat(self.timestamp)
                        ).total_seconds() / 86400
            reappraisal = math.exp(-age_days / (NEGATIVE_HALFLIFE_DAYS / math.log(2)))
            clamped = min(clamped, MAX_NEGATIVE_BOOST) * reappraisal

        return base * (1.0 + 0.15 * clamped)

    @property
    def content_words(self) -> set[str]:
        return _content_words(self.content)

    # ── serialization ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = {
            "content": self.content,
            "emotion": self.emotion,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "source": self.source,
            "entity": self.entity,
            "access_count": self._access_count,
            "last_access": self._last_access,
            "stability": self._stability,
            "why_saved": getattr(self, "why_saved", ""),
        }
        regret = getattr(self, "_regret", 0.0)
        if regret > 0:
            d["regret"] = round(regret, 3)
            d["believed_importance"] = getattr(self, "_believed_importance", 0)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        obj = cls.__new__(cls)
        obj.content       = d.get("content", "")
        obj.emotion       = d.get("emotion", "neutral")
        obj.importance    = max(1, min(10, d.get("importance", 5)))
        obj.timestamp     = d.get("timestamp", datetime.now().isoformat())
        obj.source        = d.get("source", "reflection")
        obj.entity        = d.get("entity", "")
        obj._access_count = d.get("access_count", 0)
        obj._last_access  = d.get("last_access", obj.timestamp)
        obj._stability    = d.get("stability", INITIAL_STABILITY)
        obj.why_saved     = d.get("why_saved", "")
        obj._regret              = d.get("regret", 0.0)
        obj._believed_importance = d.get("believed_importance", 0)
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  ShortTermFact — volatile factual memory with aggressive decay
# ═══════════════════════════════════════════════════════════════════════════

class ShortTermFact:
    """A short-lived factual memory with aggressive decay.

    STM captures entity-specific facts (preferences, states, recent events)
    that are too structured for episodic memory but too volatile to persist
    as permanent knowledge.  Facts auto-decay and are pruned when faded.

    Examples:
        ShortTermFact("Scott", "favorite_food", "pizza")
        ShortTermFact("Luna", "current_mood", "excited about the concert")
    """

    __slots__ = (
        "entity", "attribute", "value", "retrieval_tags",
        "timestamp", "_stability",
    )

    def __init__(self, entity: str, attribute: str, value: str,
                 retrieval_tags: list[str] | None = None):
        self.entity = entity
        self.attribute = attribute
        self.value = value
        self.retrieval_tags = retrieval_tags or self._auto_tag(entity, attribute, value)
        self.timestamp = datetime.now().isoformat()
        self._stability = 0.5  # 12 hours until ~50% fade

    @staticmethod
    def _auto_tag(entity: str, attribute: str, value: str) -> list[str]:
        """Generate retrieval tags from template patterns.

        Auto-creates tags matching common question patterns so facts
        surface when users ask about them naturally.
        """
        tags: list[str] = []
        if entity:
            tags.append(entity.lower())
        if attribute:
            tags.append(attribute.lower())
            # Common question patterns
            for word in attribute.lower().replace("_", " ").split():
                if len(word) >= 3 and word not in _RESONANCE_STOP:
                    tags.append(word)
        if value:
            for word in value.lower().split()[:5]:
                if len(word) >= 3 and word not in _RESONANCE_STOP:
                    tags.append(word)
        if entity and attribute:
            tags.append(f"{entity.lower()}_{attribute.lower()}")
        return tags

    @property
    def vividness(self) -> float:
        """How fresh this fact is — decays aggressively."""
        age_days = (datetime.now()
                    - datetime.fromisoformat(self.timestamp)
                    ).total_seconds() / 86400
        return math.exp(-age_days / max(self._stability, 0.01))

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
            "retrieval_tags": self.retrieval_tags,
            "timestamp": self.timestamp,
            "stability": self._stability,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShortTermFact":
        obj = cls.__new__(cls)
        obj.entity = d.get("entity", "")
        obj.attribute = d.get("attribute", "")
        obj.value = d.get("value", "")
        obj.retrieval_tags = d.get("retrieval_tags", [])
        obj.timestamp = d.get("timestamp", datetime.now().isoformat())
        obj._stability = d.get("stability", 0.5)
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  TaskRecord — tracks what the agent is working on
# ═══════════════════════════════════════════════════════════════════════════

class TaskRecord:
    """A single task or subtask within a project.

    Tasks have a lifecycle: active → completed | failed | abandoned.
    Parent_id enables subtask nesting (e.g. "Fix auth" → "Debug token refresh").
    """

    __slots__ = (
        "task_id", "description", "status", "priority",
        "created_at", "completed_at", "outcome", "parent_id",
        "project", "tags",
    )

    def __init__(self, description: str, project: str = "",
                 priority: int = 5, parent_id: str = "",
                 tags: list[str] | None = None):
        self.task_id = f"t_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000:04d}"
        self.description = description
        self.status = "active"
        self.priority = min(max(priority, 1), 10)
        self.created_at = datetime.now().isoformat()
        self.completed_at = ""
        self.outcome = ""
        self.parent_id = parent_id
        self.project = project
        self.tags = tags or []

    def complete(self, outcome: str = ""):
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        self.outcome = outcome

    def fail(self, reason: str = ""):
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()
        self.outcome = reason

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
            "parent_id": self.parent_id,
            "project": self.project,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskRecord":
        obj = cls.__new__(cls)
        obj.task_id = d.get("task_id", "")
        obj.description = d.get("description", "")
        obj.status = d.get("status", "active")
        obj.priority = d.get("priority", 5)
        obj.created_at = d.get("created_at", datetime.now().isoformat())
        obj.completed_at = d.get("completed_at", "")
        obj.outcome = d.get("outcome", "")
        obj.parent_id = d.get("parent_id", "")
        obj.project = d.get("project", "")
        obj.tags = d.get("tags", [])
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  ActionRecord — what the agent tried and what happened
# ═══════════════════════════════════════════════════════════════════════════

class ActionRecord:
    """A single step taken while working on a task.

    Tracks both successes and failures so the agent can recall
    what it already tried.  The error+fix pair is what feeds
    SolutionPattern auto-extraction.
    """

    __slots__ = (
        "task_id", "action", "result", "error", "fix",
        "timestamp", "importance",
    )

    def __init__(self, task_id: str, action: str, result: str = "success",
                 error: str = "", fix: str = "", importance: int = 5):
        self.task_id = task_id
        self.action = action
        self.result = result          # "success" | "failure" | "partial"
        self.error = error
        self.fix = fix
        self.timestamp = datetime.now().isoformat()
        self.importance = min(max(importance, 1), 10)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "action": self.action,
            "result": self.result,
            "error": self.error,
            "fix": self.fix,
            "timestamp": self.timestamp,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionRecord":
        obj = cls.__new__(cls)
        obj.task_id = d.get("task_id", "")
        obj.action = d.get("action", "")
        obj.result = d.get("result", "success")
        obj.error = d.get("error", "")
        obj.fix = d.get("fix", "")
        obj.timestamp = d.get("timestamp", datetime.now().isoformat())
        obj.importance = d.get("importance", 5)
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  SolutionPattern — "I've seen this before" knowledge
# ═══════════════════════════════════════════════════════════════════════════

class SolutionPattern:
    """A reusable problem→solution pattern extracted from action logs.

    These are the anti-repeat-mistake memories.  When an agent encounters
    an error, find_solutions() checks for matching patterns and returns
    what worked (and what didn't) so the agent skips straight to the fix.

    Vividness decays over time but is boosted every time the pattern
    is reused (spaced-repetition), keeping frequently-used solutions vivid.
    """

    __slots__ = (
        "problem_signature", "failed_approaches", "solution",
        "context_tags", "times_applied", "importance", "timestamp",
        "_stability",
    )

    def __init__(self, problem: str, failed_approaches: list[str],
                 solution: str, tags: list[str] | None = None,
                 importance: int = SOLUTION_INITIAL_IMPORTANCE):
        self.problem_signature = problem
        self.failed_approaches = failed_approaches or []
        self.solution = solution
        self.context_tags = tags or []
        self.times_applied = 0
        self.importance = min(max(importance, 1), 10)
        self.timestamp = datetime.now().isoformat()
        self._stability = INITIAL_STABILITY

    def apply(self):
        """Mark this solution as reused — boosts importance + resets decay."""
        self.times_applied += 1
        self.importance = min(10, self.importance + SOLUTION_REUSE_BOOST)
        # Spaced-repetition: reset decay clock
        self.timestamp = datetime.now().isoformat()
        self._stability = min(
            STABILITY_CAP,
            self._stability * SPACING_BONUS)

    @property
    def vividness(self) -> float:
        age_days = (datetime.now()
                    - datetime.fromisoformat(self.timestamp)
                    ).total_seconds() / 86400
        return self.importance * math.exp(-age_days / max(self._stability, 0.01))

    @property
    def search_text(self) -> str:
        """Combined text for matching against problem queries."""
        parts = [self.problem_signature, self.solution] + self.context_tags + self.failed_approaches
        return " ".join(parts).lower()

    def to_dict(self) -> dict:
        return {
            "problem_signature": self.problem_signature,
            "failed_approaches": self.failed_approaches,
            "solution": self.solution,
            "context_tags": self.context_tags,
            "times_applied": self.times_applied,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "stability": self._stability,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SolutionPattern":
        obj = cls.__new__(cls)
        obj.problem_signature = d.get("problem_signature", "")
        obj.failed_approaches = d.get("failed_approaches", [])
        obj.solution = d.get("solution", "")
        obj.context_tags = d.get("context_tags", [])
        obj.times_applied = d.get("times_applied", 0)
        obj.importance = d.get("importance", SOLUTION_INITIAL_IMPORTANCE)
        obj.timestamp = d.get("timestamp", datetime.now().isoformat())
        obj._stability = d.get("stability", INITIAL_STABILITY)
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  ArtifactRecord — key elements of a project the agent should remember
# ═══════════════════════════════════════════════════════════════════════════

class ArtifactRecord:
    """A tracked artifact within a project (file, function, character, API, concept).

    The artifact_type field is freeform to support any domain:
    - Coding: 'file', 'function', 'endpoint', 'module', 'config'
    - Writing: 'character', 'plot_thread', 'chapter', 'setting', 'theme'
    - Research: 'source', 'hypothesis', 'finding', 'method'
    - DevOps: 'service', 'config', 'pipeline', 'environment'
    """

    __slots__ = (
        "name", "artifact_type", "description", "importance",
        "dependencies", "current_state", "last_updated", "auto_tracked",
    )

    def __init__(self, name: str, artifact_type: str = "",
                 description: str = "", importance: int = 5,
                 dependencies: list[str] | None = None,
                 auto_tracked: bool = False):
        self.name = name
        self.artifact_type = artifact_type
        self.description = description
        self.importance = min(max(importance, 1), 10)
        self.dependencies = dependencies or []
        self.current_state = ""
        self.last_updated = datetime.now().isoformat()
        self.auto_tracked = auto_tracked

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "artifact_type": self.artifact_type,
            "description": self.description,
            "importance": self.importance,
            "dependencies": self.dependencies,
            "current_state": self.current_state,
            "last_updated": self.last_updated,
            "auto_tracked": self.auto_tracked,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArtifactRecord":
        obj = cls.__new__(cls)
        obj.name = d.get("name", "")
        obj.artifact_type = d.get("artifact_type", "")
        obj.description = d.get("description", "")
        obj.importance = d.get("importance", 5)
        obj.dependencies = d.get("dependencies", [])
        obj.current_state = d.get("current_state", "")
        obj.last_updated = d.get("last_updated", datetime.now().isoformat())
        obj.auto_tracked = d.get("auto_tracked", False)
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  VividnessMem — the main memory system
# ═══════════════════════════════════════════════════════════════════════════

class VividnessMem:
    """Organic memory store with vividness-ranked recall.

    Parameters
    ----------
    data_dir : str or Path
        Directory where memory JSON files are persisted.
        Created automatically if it doesn't exist.
    encryption_key : str, optional
        Password for AES encryption at rest (requires ``cryptography``).
    professional : bool, default False
        When True, disables emotion/mood biasing for task-oriented agents.
        Memories are ranked by pure vividness (importance × decay) instead
        of mood-congruent vividness.  Mood tracking, emotional reappraisal,
        and relationship arc warmth are all bypassed.  Context blocks use
        neutral, task-oriented headers.  All other features (spaced-repetition
        decay, resonance, STM facts, dreaming, consolidation, deduplication,
        preferences) work identically.
    """

    # Tuning knobs — override on the class or instance as needed
    ACTIVE_SELF_LIMIT      = 8
    ACTIVE_SOCIAL_LIMIT    = 5
    RESONANCE_LIMIT        = 3
    RESONANCE_SCORE_FLOOR  = 2.5

    def __init__(self, data_dir: str | Path = "memory_data",
                 encryption_key: str | None = None,
                 professional: bool = False):
        self.professional = professional
        self.data_dir   = Path(data_dir)
        self.social_dir = self.data_dir / "social"

        # ── Encryption setup ──────────────────────────────────────────
        self._fernet: Fernet | None = None
        self._hmac_key: bytes = b""
        if encryption_key is not None:
            if not _HAS_CRYPTO:
                raise ImportError(
                    "Optional dependency 'cryptography' is required for "
                    "encryption.  Install it:  pip install cryptography"
                )
            self._init_crypto(encryption_key)

        # File paths — encrypted files use .enc extension
        ext = ".enc" if self._fernet else ".json"
        self.self_file  = self.data_dir / f"self_memory{ext}"
        self.brief_file = self.data_dir / f"brief{ext}"

        self.self_reflections: list[Memory] = []
        self.social_impressions: dict[str, list[Memory]] = {}

        # Inverted index for O(k) resonance lookup
        self._word_index:   dict[str, set[int]] = {}
        self._prefix_index: dict[str, set[int]] = {}

        # Short-term facts (aggressive decay, structured)
        self._stm: list[ShortTermFact] = []

        # Co-occurrence graph for learned associations
        self._cooccurrence: dict[str, dict[str, int]] = {}

        # Compressed brief data (self-summary, entity summaries, session counter)
        self._brief_data: dict = {
            "session_count": 0,
            "self_brief": "",
            "entity_briefs": {},
            "entity_preferences": {},   # {entity: {category: [{item, sentiment, timestamp}]}}
            "relationship_arcs": {},   # {entity: {trajectory, history, ...}}
            "dream_log": [],            # list of dream connection records
        }
        # Mood state (PAD vector) — frozen at neutral in professional mode
        self._mood: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Reverse-lookup: hashed filename → real entity name (for encrypted load)
        self._entity_hash_map: dict[str, str] = {}

        # ── Task memory branch ────────────────────────────────────────
        self._projects_dir = self.data_dir / "projects"
        self._solutions_file = self.data_dir / "solutions.json"
        self._active_project: str = ""          # currently selected project name
        self._solutions: list[SolutionPattern] = []  # global solution patterns
        # Per-project data (loaded lazily for active project only)
        self._project_tasks: list[TaskRecord] = []
        self._project_actions: list[ActionRecord] = []
        self._project_artifacts: list[ArtifactRecord] = []
        self._project_meta: dict = {}           # {last_accessed, created, access_count, agent_track_quality}

        self._ensure_dirs()
        self._load()
        self._load_brief()
        self._load_solutions()
        self._rebuild_index()

    # ── Encryption helpers ─────────────────────────────────────────────

    def _init_crypto(self, password: str):
        """Derive a Fernet key from the user's password using PBKDF2.

        A random salt is generated on first use and stored in the data
        directory.  Subsequent loads read the same salt so the same
        password always produces the same key.
        """
        salt_path = Path(self.data_dir) / ".salt"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if salt_path.exists():
            salt = salt_path.read_bytes()
        else:
            salt = os.urandom(16)
            salt_path.write_bytes(salt)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,          # OWASP 2023 recommendation
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
        self._fernet = Fernet(key)
        # Separate HMAC key for filename hashing (derived from same password
        # but with a different salt prefix so it's independent)
        self._hmac_key = hashlib.sha256(b"filename-hmac:" + key).digest()

    @property
    def encrypted(self) -> bool:
        """True when at-rest encryption is active."""
        return self._fernet is not None

    def _write_json(self, path: Path, obj):
        """Write a Python object as JSON — encrypted if a key is set.

        Uses atomic write (write to temp + rename) with file locking
        to prevent corruption from concurrent access.
        """
        raw = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
        if self._fernet:
            raw = self._fernet.encrypt(raw)
        # Atomic write: write to temp file then rename
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                _lock_file(f)
                try:
                    f.write(raw)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    _unlock_file(f)
            # Atomic rename (overwrites on Windows via os.replace)
            os.replace(tmp, str(path))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read_json(self, path: Path):
        """Read JSON from disk — decrypting if a key is set.

        Uses file locking to prevent reading partial writes.
        """
        with open(path, "rb") as f:
            _lock_file(f)
            try:
                raw = f.read()
            finally:
                _unlock_file(f)
        if self._fernet:
            raw = self._fernet.decrypt(raw)
        return json.loads(raw)

    def _entity_filename(self, entity: str) -> str:
        """Return a safe filename for an entity.

        Plain-text mode: ``alice.json``
        Encrypted mode : HMAC-SHA256 hash + ``.enc`` — hides who the
        agent has interacted with.
        """
        if self._fernet:
            digest = hmac.new(self._hmac_key,
                              entity.lower().encode("utf-8"),
                              hashlib.sha256).hexdigest()[:16]
            return f"{digest}.enc"
        return entity.lower().replace(" ", "_") + ".json"

    # ── directory setup ───────────────────────────────────────────────
    def _ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.social_dir.mkdir(parents=True, exist_ok=True)
        self._projects_dir.mkdir(parents=True, exist_ok=True)

    # ── inverted index ────────────────────────────────────────────────
    def _index_memory(self, idx: int, mem: Memory):
        """Add a single memory to the inverted word/prefix indices.

        Also indexes synonyms so 'afraid' matches memories about 'fear'.
        Also indexes bigrams and trigrams for phrase-level matching."""
        if not hasattr(self, '_word_index'):
            self._word_index = {}
            self._prefix_index = {}
        text = f"{mem.content} {mem.emotion}"
        words = _resonance_words(text)
        # Expand with synonyms so index bridges semantic gaps
        expanded = _expand_synonyms(words)
        for w in expanded:
            self._word_index.setdefault(w, set()).add(idx)
            if len(w) >= 5:
                self._prefix_index.setdefault(w[:5], set()).add(idx)
        # Bigram index for phrase-level matching
        for bg in _bigrams(text):
            self._word_index.setdefault(bg, set()).add(idx)
        # Trigram index for longer phrase matching
        for tg in _trigrams(text):
            self._word_index.setdefault(tg, set()).add(idx)
        # Update co-occurrence graph
        self._update_cooccurrence(expanded)

    def _rebuild_index(self):
        """Rebuild the full inverted index from scratch."""
        self._word_index.clear()
        self._prefix_index.clear()
        self._cooccurrence.clear()
        for i, mem in enumerate(self.self_reflections):
            self._index_memory(i, mem)

    # ── Co-occurrence graph (learned associations) ────────────────────

    def _update_cooccurrence(self, words: set[str]):
        """Track which content words appear together in memories."""
        if not hasattr(self, '_cooccurrence'):
            self._cooccurrence = {}
        word_list = sorted(words)[:30]  # cap to avoid O(n^2) explosion
        for i, w1 in enumerate(word_list):
            for w2 in word_list[i + 1:]:
                self._cooccurrence.setdefault(w1, {})
                self._cooccurrence[w1][w2] = self._cooccurrence[w1].get(w2, 0) + 1
                self._cooccurrence.setdefault(w2, {})
                self._cooccurrence[w2][w1] = self._cooccurrence[w2].get(w1, 0) + 1

    def _expand_via_cooccurrence(self, words: set[str], top_k: int = 10) -> set[str]:
        """Find terms most associated with query words through co-occurrence.

        This is learned query expansion: if 'pizza' and 'favorite' always
        appear in the same memories, querying 'pizza' also finds memories
        mentioning 'favorite' even without that exact word in the query.
        """
        if not hasattr(self, '_cooccurrence'):
            return set()
        scores: dict[str, float] = {}
        for w in words:
            neighbors = self._cooccurrence.get(w, {})
            for neighbor, weight in neighbors.items():
                if neighbor not in words and weight >= 2:
                    scores[neighbor] = scores.get(neighbor, 0) + weight
        if not scores:
            return set()
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {term for term, _ in sorted_terms[:top_k]}

    # ── Add memories ──────────────────────────────────────────────────

    def add_self_reflection(self, content: str, emotion: str = "neutral",
                            importance: int = 5, source: str = "reflection",
                            why_saved: str = "") -> Memory:
        """Store a self-reflection memory. Deduplicates by content overlap.

        If a near-duplicate exists, the new version replaces it (keeping
        the higher importance and merging access history).
        """
        new_words = _content_words(content)
        for i, existing in enumerate(self.self_reflections):
            if _overlap_ratio(new_words, existing.content_words) >= _DEDUP_THRESHOLD:
                # Merge: keep newer text, max importance, preserve history
                merged = Memory(
                    content=content,
                    emotion=emotion,
                    importance=max(importance, existing.importance),
                    source=source,
                    why_saved=why_saved or existing.why_saved,
                )
                merged._access_count = existing._access_count
                merged._stability    = existing._stability
                merged._last_access  = existing._last_access
                merged.timestamp     = existing.timestamp  # keep original creation time
                self.self_reflections[i] = merged
                self._rebuild_index()
                return merged

        mem = Memory(content=content, emotion=emotion,
                     importance=importance, source=source,
                     why_saved=why_saved)
        self.self_reflections.append(mem)
        self._index_memory(len(self.self_reflections) - 1, mem)
        return mem

    def add_social_impression(self, entity: str, content: str,
                              emotion: str = "neutral",
                              importance: int = 5,
                              why_saved: str = "") -> Memory:
        """Store a social impression of an entity. Deduplicates per entity."""
        if entity not in self.social_impressions:
            self.social_impressions[entity] = []
        impressions = self.social_impressions[entity]

        new_words = _content_words(content)
        for i, existing in enumerate(impressions):
            if _overlap_ratio(new_words, existing.content_words) >= _DEDUP_THRESHOLD:
                merged = Memory(
                    content=content,
                    emotion=emotion,
                    importance=max(importance, existing.importance),
                    entity=entity,
                    why_saved=why_saved or existing.why_saved,
                )
                merged._access_count = existing._access_count
                merged._stability    = existing._stability
                merged._last_access  = existing._last_access
                merged.timestamp     = existing.timestamp
                impressions[i] = merged
                self._update_relationship_arc(entity, emotion)
                return merged

        mem = Memory(content=content, emotion=emotion,
                     importance=importance, source="social",
                     entity=entity, why_saved=why_saved)
        impressions.append(mem)
        self._update_relationship_arc(entity, emotion)
        return mem

    # ── Retrieve active memories ──────────────────────────────────────

    def get_active_self(self, context: str = "") -> list[Memory]:
        """Return the most vivid self-memories, mood-weighted.

        Context-overcrowding fix: when a conversation context is provided,
        memories are scored by *both* vividness and topical relevance so
        that highly-vivid-but-irrelevant memories don't crowd out on-topic
        ones.  Without context, pure vividness ranking is kept.

        Touch dampener: memories only get reinforced (touch) if they share
        at least one context word with the current conversation.
        """
        if context:
            ctx_words = _resonance_words(context)
            ctx_words_expanded = _expand_synonyms(ctx_words)

            def _relevance_weighted(mem: Memory) -> float:
                base = (mem.vividness if self.professional
                        else mem.mood_adjusted_vividness(self._mood))
                if not ctx_words_expanded:
                    return base
                mem_words = _resonance_words(f"{mem.content} {mem.emotion}")
                overlap = len(ctx_words_expanded & mem_words)
                # relevance multiplier: 1.0 (no match) to 2.0 (strong match)
                relevance = 1.0 + min(overlap / max(len(ctx_words_expanded), 1), 1.0)
                return base * relevance

            ranked = sorted(self.self_reflections,
                            key=_relevance_weighted, reverse=True)
        else:
            ranked = sorted(
                self.self_reflections,
                key=lambda r: (r.vividness if self.professional
                               else r.mood_adjusted_vividness(self._mood)),
                reverse=True,
            )

        active = ranked[:self.ACTIVE_SELF_LIMIT]

        # Touch dampener — only reinforce memories relevant to context
        if context:
            ctx_words = _resonance_words(context)
            for mem in active:
                mem_words = _resonance_words(f"{mem.content} {mem.emotion}")
                if ctx_words & mem_words:
                    mem.touch()
        else:
            for mem in active:
                mem.touch()

        return active

    def partition_active_self(self, context: str = "") -> tuple[list[Memory], list[Memory]]:
        """Split active memories into foreground (relevant) and background.

        Foreground: memories that share keyword overlap with the current
        conversation context — these get full text in the prompt.
        Background: remaining active memories — shown as compressed
        one-liners to save tokens while preserving awareness.

        Context-overcrowding fix: background list is trimmed so that
        irrelevant memories don't consume the context window.

        If no context is provided, everything goes to foreground.
        """
        active = self.get_active_self(context=context)
        if not context:
            return active, []

        ctx_words = _resonance_words(context)
        ctx_words_expanded = _expand_synonyms(ctx_words) if ctx_words else set()

        if not ctx_words:
            return active, []

        foreground, background = [], []
        for mem in active:
            mem_words = _resonance_words(f"{mem.content} {mem.emotion}")
            if ctx_words_expanded & mem_words:
                foreground.append(mem)
            else:
                background.append(mem)

        # If nothing matched, put top 3 in foreground anyway
        if not foreground and active:
            return active[:3], active[3:]

        # Trim background to at most half the foreground count to prevent
        # irrelevant memories from overwhelming the context window.
        max_bg = max(len(foreground) // 2, 1)
        background = background[:max_bg]

        return foreground, background

    def get_active_social(self, entity: str) -> list[Memory]:
        """Return the most vivid social impressions for a given entity."""
        impressions = self.social_impressions.get(entity, [])
        ranked = sorted(
            impressions,
            key=lambda r: (r.vividness if self.professional
                           else r.mood_adjusted_vividness(self._mood)),
            reverse=True,
        )
        return ranked[:self.ACTIVE_SOCIAL_LIMIT]

    # ── Mood system ───────────────────────────────────────────────────

    @property
    def mood(self) -> tuple[float, float, float]:
        return self._mood

    @property
    def mood_label(self) -> str:
        """Human-readable mood label from current PAD vector."""
        if self._mood == (0.0, 0.0, 0.0):
            return "neutral"
        p, a, d = self._mood
        best_label, best_dot = "neutral", -999.0
        for label, vec in EMOTION_VECTORS.items():
            dot = p * vec[0] + a * vec[1] + d * vec[2]
            if dot > best_dot:
                best_dot = dot
                best_label = label
        return best_label

    def update_mood_from_conversation(self, emotions: list[str]):
        """Shift mood toward emotions expressed in the latest exchange.

        Uses exponential moving average so mood drifts gradually.
        No-op in professional mode (mood stays neutral).
        """
        if self.professional or not emotions:
            return
        vectors = [_emotion_to_vector(e) for e in emotions]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return
        avg = tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(3))
        alpha = 0.3  # blending factor
        self._mood = tuple(
            round(self._mood[i] * (1 - alpha) + avg[i] * alpha, 4)
            for i in range(3)
        )

    def _absorb_mood_from_memories(self):
        """Initialize mood from the most vivid memories on load."""
        top = sorted(self.self_reflections,
                     key=lambda r: r.vividness, reverse=True)[:5]
        if not top:
            return
        vectors = [_emotion_to_vector(r.emotion) for r in top]
        vectors = [v for v in vectors if v is not None]
        if vectors:
            self._mood = tuple(
                round(sum(v[i] for v in vectors) / len(vectors), 4)
                for i in range(3)
            )

    def _save_mood(self):
        self._brief_data["mood"] = list(self._mood)

    def _load_mood(self):
        if self.professional:
            self._mood = (0.0, 0.0, 0.0)
            return
        raw = self._brief_data.get("mood")
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            self._mood = tuple(float(x) for x in raw)
        else:
            self._absorb_mood_from_memories()

    # ── Short-term memory (STM) ───────────────────────────────────────

    def add_fact(self, entity: str, attribute: str, value: str,
                 retrieval_tags: list[str] | None = None) -> ShortTermFact:
        """Store a short-term fact about an entity.

        Facts decay quickly but are auto-included in retrieval when
        the entity is mentioned.  Deduplicates by entity+attribute.

        Examples:
            mem.add_fact("Scott", "favorite_food", "pizza")
            mem.add_fact("Luna", "current_mood", "excited")
        """
        if not hasattr(self, '_stm'):
            self._stm = []
        # Deduplicate: update existing fact for same entity+attribute
        for i, existing in enumerate(self._stm):
            if (existing.entity.lower() == entity.lower()
                    and existing.attribute.lower() == attribute.lower()):
                self._stm[i] = ShortTermFact(entity, attribute, value, retrieval_tags)
                return self._stm[i]
        fact = ShortTermFact(entity, attribute, value, retrieval_tags)
        self._stm.append(fact)
        return fact

    def get_facts(self, entity: str = "", context: str = "") -> list[ShortTermFact]:
        """Retrieve relevant short-term facts.

        If entity is provided, returns facts for that entity.
        If context is provided, matches against retrieval tags.
        Filters out faded facts (vividness < 0.1).
        """
        if not hasattr(self, '_stm'):
            self._stm = []
        # Prune faded facts
        self._stm = [f for f in self._stm if f.vividness >= 0.1]

        results: list[ShortTermFact] = []
        context_words = _resonance_words(context) if context else set()

        for fact in self._stm:
            if entity and fact.entity.lower() == entity.lower():
                results.append(fact)
            elif context_words:
                tag_set = set(fact.retrieval_tags)
                if context_words & tag_set:
                    results.append(fact)

        results.sort(key=lambda f: f.vividness, reverse=True)
        return results

    def get_stm_context(self, entity: str = "", context: str = "") -> str:
        """Build a context string from short-term facts."""
        facts = self.get_facts(entity=entity, context=context)
        if not facts:
            return ""
        lines = ["=== RECENT FACTS I REMEMBER ==="]
        for f in facts[:10]:
            lines.append(f"  {f.entity}'s {f.attribute}: {f.value}")
        return "\n".join(lines)

    # ── Entity Preferences ────────────────────────────────────────────

    def update_entity_preference(self, entity: str, category: str,
                                 item: str, sentiment: str = "likes"):
        """Store a structured preference for an entity.

        Examples:
            mem.update_entity_preference("Scott", "food", "pizza", "likes")
            mem.update_entity_preference("Scott", "hobby", "boxing", "likes")
            mem.update_entity_preference("Scott", "music", "country", "dislikes")
        """
        prefs = self._brief_data.setdefault("entity_preferences", {})
        if entity not in prefs:
            prefs[entity] = {}
        if category not in prefs[entity]:
            prefs[entity][category] = []
        # Avoid duplicates
        for existing in prefs[entity][category]:
            if existing.get("item", "").lower() == item.lower():
                existing["sentiment"] = sentiment
                self._save_brief()
                return
        prefs[entity][category].append({
            "item": item,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_brief()

    def get_entity_preferences(self, entity: str) -> dict:
        """Return all known preferences for an entity."""
        return self._brief_data.get("entity_preferences", {}).get(entity, {})

    def get_preference_context(self, entity: str) -> str:
        """Build a context string about an entity's known preferences."""
        prefs = self.get_entity_preferences(entity)
        if not prefs:
            return ""
        lines = [f"=== WHAT I KNOW ABOUT {entity.upper()}'S PREFERENCES ==="]
        for category, items in prefs.items():
            likes = [i["item"] for i in items if i.get("sentiment") == "likes"]
            dislikes = [i["item"] for i in items if i.get("sentiment") == "dislikes"]
            if likes:
                lines.append(f"  {category}: likes {', '.join(likes)}")
            if dislikes:
                lines.append(f"  {category}: dislikes {', '.join(dislikes)}")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════
    #  Task Memory Branch — projects, tasks, actions, solutions, artifacts
    # ══════════════════════════════════════════════════════════════════

    # ── Project management ────────────────────────────────────────────

    def set_active_project(self, project_name: str):
        """Switch to a project (creates if new).  Loads project data from disk."""
        safe = re.sub(r'[^\w\-. ]', '_', project_name.strip())
        if not safe:
            return
        if self._active_project and self._active_project != safe:
            self._save_project()  # persist current project before switching
        self._active_project = safe
        proj_dir = self._projects_dir / safe
        proj_dir.mkdir(parents=True, exist_ok=True)
        self._load_project(safe)
        # Update access timestamp
        self._project_meta["last_accessed"] = datetime.now().isoformat()
        self._project_meta["access_count"] = self._project_meta.get("access_count", 0) + 1
        self._project_meta.setdefault("created", datetime.now().isoformat())
        self._project_meta.setdefault("agent_track_quality", AUTO_TRACK_QUALITY_INIT)
        # Flush meta to disk so list_projects can read it
        self._write_json(proj_dir / "meta.json", self._project_meta)

    def list_projects(self) -> list[dict]:
        """List all projects with status and last accessed time."""
        results = []
        if not self._projects_dir.exists():
            return results
        for pdir in sorted(self._projects_dir.iterdir()):
            if not pdir.is_dir():
                continue
            meta_file = pdir / "meta.json"
            meta = {}
            if meta_file.exists():
                try:
                    meta = self._read_json(meta_file)
                except Exception:
                    pass
            last = meta.get("last_accessed", "")
            tier = self._project_decay_tier(last) if last else "unknown"
            results.append({
                "name": pdir.name,
                "last_accessed": last,
                "access_count": meta.get("access_count", 0),
                "tier": tier,
                "active_tasks": sum(
                    1 for t in self._list_project_tasks(pdir.name)
                    if t.get("status") == "active"
                ),
            })
        return results

    def archive_project(self, project_name: str):
        """Manual freeze — mark project as archived in its meta."""
        safe = re.sub(r'[^\w\-. ]', '_', project_name.strip())
        proj_dir = self._projects_dir / safe
        if not proj_dir.exists():
            return
        meta_file = proj_dir / "meta.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = self._read_json(meta_file)
            except Exception:
                pass
        meta["archived"] = True
        meta["archived_at"] = datetime.now().isoformat()
        self._write_json(meta_file, meta)

    def _list_project_tasks(self, project_name: str) -> list[dict]:
        """Quick read of a project's tasks without fully loading it."""
        proj_dir = self._projects_dir / project_name
        tasks_file = proj_dir / "tasks.json"
        if tasks_file.exists():
            try:
                return self._read_json(tasks_file)
            except Exception:
                pass
        return []

    def _project_decay_tier(self, last_accessed: str) -> str:
        """Determine the decay tier for a project based on last access."""
        if not last_accessed:
            return "unknown"
        try:
            last = datetime.fromisoformat(last_accessed)
        except ValueError:
            return "unknown"
        days = (datetime.now() - last).total_seconds() / 86400
        if days <= PROJECT_DECAY_ACTIVE_DAYS:
            return "active"
        elif days <= PROJECT_DECAY_COOLING_DAYS:
            return "cooling"
        elif days <= PROJECT_DECAY_COLD_DAYS:
            return "cold"
        return "archived"

    def _project_decay_multiplier(self) -> float:
        """Get the decay multiplier for the current active project."""
        if not self._active_project or not self._project_meta:
            return 1.0
        tier = self._project_decay_tier(
            self._project_meta.get("last_accessed", ""))
        if tier == "cooling":
            return PROJECT_DECAY_COOLING_MULT
        elif tier == "cold":
            return PROJECT_DECAY_COLD_MULT
        elif tier == "archived":
            return 10.0  # nearly everything fades
        return 1.0

    # ── Project persistence ───────────────────────────────────────────

    def _load_project(self, project_name: str):
        """Load all data for a project from disk."""
        proj_dir = self._projects_dir / project_name
        # Tasks
        tasks_file = proj_dir / "tasks.json"
        self._project_tasks = []
        if tasks_file.exists():
            try:
                data = self._read_json(tasks_file)
                if isinstance(data, list):
                    self._project_tasks = [
                        TaskRecord.from_dict(d) for d in data if isinstance(d, dict)]
            except Exception:
                pass
        # Actions
        actions_file = proj_dir / "actions.json"
        self._project_actions = []
        if actions_file.exists():
            try:
                data = self._read_json(actions_file)
                if isinstance(data, list):
                    self._project_actions = [
                        ActionRecord.from_dict(d) for d in data if isinstance(d, dict)]
            except Exception:
                pass
        # Artifacts
        artifacts_file = proj_dir / "artifacts.json"
        self._project_artifacts = []
        if artifacts_file.exists():
            try:
                data = self._read_json(artifacts_file)
                if isinstance(data, list):
                    self._project_artifacts = [
                        ArtifactRecord.from_dict(d) for d in data if isinstance(d, dict)]
            except Exception:
                pass
        # Meta
        meta_file = proj_dir / "meta.json"
        self._project_meta = {}
        if meta_file.exists():
            try:
                data = self._read_json(meta_file)
                if isinstance(data, dict):
                    self._project_meta = data
            except Exception:
                pass

    def _save_project(self):
        """Persist current project data to disk."""
        if not self._active_project:
            return
        proj_dir = self._projects_dir / self._active_project
        proj_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(proj_dir / "tasks.json",
                         [t.to_dict() for t in self._project_tasks])
        self._write_json(proj_dir / "actions.json",
                         [a.to_dict() for a in self._project_actions])
        self._write_json(proj_dir / "artifacts.json",
                         [a.to_dict() for a in self._project_artifacts])
        self._write_json(proj_dir / "meta.json", self._project_meta)

    def _load_solutions(self):
        """Load global solution patterns from disk."""
        self._solutions = []
        if self._solutions_file.exists():
            try:
                data = self._read_json(self._solutions_file)
                if isinstance(data, list):
                    self._solutions = [
                        SolutionPattern.from_dict(d) for d in data if isinstance(d, dict)]
            except Exception:
                pass

    def _save_solutions(self):
        """Persist global solution patterns to disk."""
        self._write_json(self._solutions_file,
                         [s.to_dict() for s in self._solutions])

    # ── Task lifecycle ────────────────────────────────────────────────

    def start_task(self, description: str, project: str | None = None,
                   priority: int = 5, parent_id: str = "",
                   tags: list[str] | None = None) -> str:
        """Create a new active task.  Returns task_id.

        If project is given, switches to that project first.
        If no project is set, creates one called 'default'.
        """
        if project is not None:
            self.set_active_project(project)
        elif not self._active_project:
            self.set_active_project("default")
        task = TaskRecord(
            description=description,
            project=self._active_project,
            priority=priority,
            parent_id=parent_id,
            tags=tags,
        )
        self._project_tasks.append(task)
        self._save_project()
        return task.task_id

    def complete_task(self, task_id: str, outcome: str = ""):
        """Mark a task as completed and auto-extract solution patterns."""
        task = self._find_task(task_id)
        if not task:
            return
        task.complete(outcome)
        self._auto_extract_solution(task_id)
        self._save_project()

    def fail_task(self, task_id: str, reason: str = ""):
        """Mark a task as failed."""
        task = self._find_task(task_id)
        if not task:
            return
        task.fail(reason)
        self._save_project()

    def abandon_task(self, task_id: str, reason: str = ""):
        """Mark a task as abandoned."""
        task = self._find_task(task_id)
        if not task:
            return
        task.status = "abandoned"
        task.completed_at = datetime.now().isoformat()
        task.outcome = reason
        self._save_project()

    def get_active_tasks(self, project: str | None = None) -> list[TaskRecord]:
        """Get all active (in-progress) tasks for the current or named project."""
        if project is not None and project != self._active_project:
            # Peek at another project without switching
            raw = self._list_project_tasks(re.sub(r'[^\w\-. ]', '_', project.strip()))
            return [TaskRecord.from_dict(d) for d in raw
                    if isinstance(d, dict) and d.get("status") == "active"]
        return [t for t in self._project_tasks if t.status == "active"]

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Look up a single task by ID."""
        return self._find_task(task_id)

    def _find_task(self, task_id: str) -> TaskRecord | None:
        for t in self._project_tasks:
            if t.task_id == task_id:
                return t
        return None

    # ── Action logging ────────────────────────────────────────────────

    def log_action(self, task_id: str, action: str, result: str = "success",
                   error: str = "", fix: str = "", importance: int = 5):
        """Log an action taken while working on a task.

        Parameters
        ----------
        task_id : str
            The task this action belongs to.
        action : str
            What was tried (e.g. "Added retry logic to token refresh").
        result : str
            "success", "failure", or "partial".
        error : str
            Error message if result was failure.
        fix : str
            What fixed it (if this action resolved a previous failure).
        importance : int
            1-10 importance rating.
        """
        record = ActionRecord(
            task_id=task_id, action=action, result=result,
            error=error, fix=fix, importance=importance,
        )
        self._project_actions.append(record)
        # If this is a successful fix, auto-record solution pattern
        if result == "success" and fix:
            # Look for preceding failures on same task
            task_actions = [a for a in self._project_actions if a.task_id == task_id]
            failures = [a for a in task_actions if a.result == "failure"]
            if failures:
                failed_desc = [f"{a.action}: {a.error}" for a in failures[-5:]]
                task = self._find_task(task_id)
                tags = list(task.tags) if task and task.tags else []
                self.record_solution(
                    problem=error or action,
                    failed_approaches=failed_desc,
                    solution=fix,
                    tags=tags,
                    importance=SOLUTION_INITIAL_IMPORTANCE,
                )
        self._save_project()

    def get_task_actions(self, task_id: str) -> list[ActionRecord]:
        """Get all actions logged for a specific task."""
        return [a for a in self._project_actions if a.task_id == task_id]

    # ── Solution memory ───────────────────────────────────────────────

    def record_solution(self, problem: str, failed_approaches: list[str],
                        solution: str, tags: list[str] | None = None,
                        importance: int = SOLUTION_INITIAL_IMPORTANCE):
        """Explicitly record a problem→solution pattern.

        These live at the global level (not per-project) so fixes
        discovered in one project are available in another.
        """
        # Deduplicate: if a very similar problem already exists, update it
        problem_words = set(problem.lower().split())
        for existing in self._solutions:
            existing_words = set(existing.problem_signature.lower().split())
            if problem_words and existing_words:
                overlap = len(problem_words & existing_words) / max(
                    len(problem_words | existing_words), 1)
                if overlap > 0.7:
                    # Merge: append new failed approaches, update solution
                    for fa in (failed_approaches or []):
                        if fa not in existing.failed_approaches:
                            existing.failed_approaches.append(fa)
                    existing.solution = solution
                    existing.timestamp = datetime.now().isoformat()
                    existing.importance = max(existing.importance, importance)
                    self._save_solutions()
                    return
        pattern = SolutionPattern(
            problem=problem,
            failed_approaches=failed_approaches,
            solution=solution,
            tags=tags,
            importance=importance,
        )
        self._solutions.append(pattern)
        self._save_solutions()

    def find_solutions(self, problem_description: str,
                       top_k: int = 3) -> list[SolutionPattern]:
        """Search for solution patterns matching a problem description.

        Uses word overlap matching (same approach as resonance).
        Returns top_k matches sorted by relevance × vividness.
        """
        if not self._solutions or not problem_description:
            return []
        query_words = set(w.lower() for w in problem_description.split()
                          if len(w) >= 3 and w.lower() not in _RESONANCE_STOP)
        if not query_words:
            return []
        scored: list[tuple[float, SolutionPattern]] = []
        for sol in self._solutions:
            sol_words = set(w for w in sol.search_text.split()
                           if len(w) >= 3 and w not in _RESONANCE_STOP)
            if not sol_words:
                continue
            common = query_words & sol_words
            jaccard = len(common) / max(len(query_words | sol_words), 1)
            coverage = len(common) / max(len(query_words), 1)
            overlap = max(jaccard, coverage * 0.4)
            if overlap > 0.15:
                score = overlap * sol.vividness
                scored.append((score, sol))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def _auto_extract_solution(self, task_id: str):
        """Auto-extract a solution pattern when a task completes.

        Scans the task's action log for fail→success sequences.
        Only creates a pattern if the agent didn't already record one.
        """
        actions = self.get_task_actions(task_id)
        if len(actions) < 2:
            return
        failures = [a for a in actions if a.result == "failure"]
        successes = [a for a in actions if a.result == "success"]
        if not failures or not successes:
            return
        # Check if a solution was already recorded for this problem
        last_error = failures[-1].error or failures[-1].action
        existing = self.find_solutions(last_error, top_k=1)
        if existing and existing[0].vividness > 3.0:
            return  # Already covered
        failed_desc = [f"{a.action}: {a.error}" for a in failures[-5:]]
        fix = successes[-1].fix or successes[-1].action
        self.record_solution(
            problem=last_error,
            failed_approaches=failed_desc,
            solution=fix,
            tags=[],
            importance=SOLUTION_AUTO_IMPORTANCE,
        )

    # ── Project knowledge (artifacts) ─────────────────────────────────

    def track_artifact(self, name: str, artifact_type: str = "",
                       description: str = "", project: str | None = None,
                       importance: int = 5,
                       dependencies: list[str] | None = None):
        """Register a key artifact the agent should remember about the project.

        The artifact_type is freeform — 'file', 'character', 'endpoint', etc.
        """
        if project is not None and project != self._active_project:
            self.set_active_project(project)
        elif not self._active_project:
            self.set_active_project("default")
        # Dedup by normalized name
        norm = name.strip().lower()
        for existing in self._project_artifacts:
            if existing.name.strip().lower() == norm:
                # Update in place
                if description:
                    existing.description = description
                if artifact_type:
                    existing.artifact_type = artifact_type
                existing.importance = max(existing.importance, importance)
                if dependencies:
                    for dep in dependencies:
                        if dep not in existing.dependencies:
                            existing.dependencies.append(dep)
                existing.last_updated = datetime.now().isoformat()
                self._save_project()
                return
        record = ArtifactRecord(
            name=name,
            artifact_type=artifact_type,
            description=description,
            importance=importance,
            dependencies=dependencies,
        )
        self._project_artifacts.append(record)
        self._save_project()

    def update_artifact(self, name: str, state: str | None = None,
                        description: str | None = None):
        """Update the state or description of a tracked artifact."""
        norm = name.strip().lower()
        for a in self._project_artifacts:
            if a.name.strip().lower() == norm:
                if state is not None:
                    a.current_state = state
                if description is not None:
                    a.description = description
                a.last_updated = datetime.now().isoformat()
                self._save_project()
                return

    def get_project_overview(self, project: str | None = None) -> str:
        """Build a compact summary of project artifacts."""
        if project is not None and project != self._active_project:
            # Load temporarily
            proj_dir = self._projects_dir / re.sub(r'[^\w\-. ]', '_', project.strip())
            arts_file = proj_dir / "artifacts.json"
            arts = []
            if arts_file.exists():
                try:
                    data = self._read_json(arts_file)
                    if isinstance(data, list):
                        arts = [ArtifactRecord.from_dict(d) for d in data]
                except Exception:
                    pass
        else:
            arts = self._project_artifacts
        if not arts:
            return ""
        pname = project or self._active_project or "Project"
        lines = [f"=== PROJECT: {pname.upper()} ==="]
        # Sort by importance descending
        for a in sorted(arts, key=lambda x: x.importance, reverse=True):
            parts = [a.name]
            if a.artifact_type:
                parts[0] = f"{a.name} ({a.artifact_type})"
            if a.description:
                desc = a.description
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                parts.append(f"— {desc}")
            if a.current_state:
                parts.append(f"[{a.current_state}]")
            lines.append("  " + " ".join(parts))
        return "\n".join(lines)

    def get_related(self, name: str) -> list[ArtifactRecord]:
        """Get artifacts that depend on or are depended upon by the named artifact."""
        norm = name.strip().lower()
        related = []
        for a in self._project_artifacts:
            if a.name.strip().lower() == norm:
                # Find things this artifact depends on
                for dep_name in a.dependencies:
                    dep_norm = dep_name.strip().lower()
                    for b in self._project_artifacts:
                        if b.name.strip().lower() == dep_norm and b not in related:
                            related.append(b)
            else:
                # Find things that depend on the target
                for dep_name in a.dependencies:
                    if dep_name.strip().lower() == norm and a not in related:
                        related.append(a)
        return related

    # ── Adaptive auto-tracking (professional mode) ────────────────────

    def _auto_track_entities(self, text: str):
        """Auto-extract potential artifacts from text (professional mode only).

        Only runs when professional=True and an active project is set.
        Frequency is gated by agent_track_quality score.
        """
        if not self.professional or not self._active_project:
            return
        quality = self._project_meta.get(
            "agent_track_quality", AUTO_TRACK_QUALITY_INIT)
        # High quality agent → skip most auto-tracks
        if quality > 0.8:
            # Only run on every 5th action (use access_count as proxy)
            count = self._project_meta.get("access_count", 0)
            if count % 5 != 0:
                return
        # Extract potential entity names (capitalized sequences, file paths,
        # quoted strings, URLs — simple heuristics, not NLP)
        candidates: list[str] = []
        # File paths (e.g. src/auth.py, config.yaml)
        for m in re.finditer(r'[\w./\\-]+\.(?:py|js|ts|yaml|yml|json|toml|cfg|md|txt|html|css|sql|sh|go|rs|java|rb|php|c|cpp|h)', text):
            candidates.append(m.group())
        # Quoted strings (likely names/identifiers)
        for m in re.finditer(r'"([^"]{3,40})"', text):
            candidates.append(m.group(1))
        for m in re.finditer(r"'([^']{3,40})'", text):
            candidates.append(m.group(1))
        for candidate in candidates:
            norm = candidate.strip().lower()
            # Skip if already tracked
            already = any(a.name.strip().lower() == norm
                          for a in self._project_artifacts)
            if already:
                continue
            # Auto-track with low importance
            record = ArtifactRecord(
                name=candidate.strip(),
                artifact_type="auto",
                description="(auto-detected)",
                importance=3,
                auto_tracked=True,
            )
            self._project_artifacts.append(record)

    def update_track_quality(self, novel_auto: int, redundant_auto: int):
        """Update the agent tracking quality score after a session.

        Parameters
        ----------
        novel_auto : int
            How many auto-tracked items supplied genuinely new information.
        redundant_auto : int
            How many auto-tracked items were redundant with agent-tracked ones.
        """
        if not self._project_meta:
            return
        total = novel_auto + redundant_auto
        if total == 0:
            return
        miss_rate = novel_auto / total
        old_q = self._project_meta.get("agent_track_quality", AUTO_TRACK_QUALITY_INIT)
        new_q = (1 - AUTO_TRACK_EMA_ALPHA) * old_q + AUTO_TRACK_EMA_ALPHA * (1.0 - miss_rate)
        self._project_meta["agent_track_quality"] = round(max(0.0, min(1.0, new_q)), 4)
        self._save_project()

    # ── Task context for prompt injection ─────────────────────────────

    def get_task_context(self, conversation_context: str = "") -> str:
        """Build the task memory section for prompt injection.

        Returns formatted text covering:
        - Active tasks (always shown)
        - Relevant solutions (matched against conversation context)
        - Project overview (compact artifact list)

        Respects hard token caps to avoid context bloat.
        """
        lines: list[str] = []

        # Active tasks (always shown if a project is active)
        active = self.get_active_tasks()
        if active:
            proj = self._active_project or "Current"
            lines.append(f"=== ACTIVE TASKS — {proj.upper()} ===")
            for t in sorted(active, key=lambda x: x.priority, reverse=True):
                action_count = len(self.get_task_actions(t.task_id))
                errors = sum(1 for a in self._project_actions
                             if a.task_id == t.task_id and a.result == "failure")
                suffix = ""
                if action_count:
                    suffix = f" — {action_count} actions"
                    if errors:
                        suffix += f", {errors} error{'s' if errors != 1 else ''}"
                lines.append(f"  (P{t.priority}) {t.description}{suffix}")
            lines.append("")

        # Relevant solutions (matched against conversation context)
        if conversation_context:
            matches = self.find_solutions(conversation_context, top_k=3)
            if matches:
                lines.append("=== KNOWN SOLUTIONS ===")
                for sol in matches:
                    lines.append(f"  \"{sol.problem_signature}\":")
                    for fa in sol.failed_approaches[-3:]:
                        # Compact each failed approach
                        fa_short = fa if len(fa) <= 80 else fa[:77] + "..."
                        lines.append(f"    ✗ {fa_short}")
                    sol_short = sol.solution if len(sol.solution) <= 100 else sol.solution[:97] + "..."
                    lines.append(f"    ✓ {sol_short}")
                    if sol.times_applied:
                        lines.append(f"    (applied {sol.times_applied}x)")
                lines.append("")

        # Project overview (compact artifact list)
        overview = self.get_project_overview()
        if overview:
            lines.append(overview)
            lines.append("")

        return "\n".join(lines)

    # ── Association graph ─────────────────────────────────────────────

    def _build_association_edges(self) -> dict[int, dict[int, int]]:
        """Build a weighted graph connecting memories that share keywords.

        Returns {idx: {neighbor_idx: overlap_count, ...}, ...}
        """
        n = len(self.self_reflections)
        word_sets = [
            _resonance_words(f"{r.content} {r.emotion}")
            for r in self.self_reflections
        ]
        edges: dict[int, dict[int, int]] = {i: {} for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                shared = len(word_sets[i] & word_sets[j])
                if shared >= ASSOCIATION_MIN_WEIGHT:
                    edges[i][j] = shared
                    edges[j][i] = shared
        return edges

    def associate(self, seed_indices: list[int],
                  hops: int = ASSOCIATION_HOPS) -> list[Memory]:
        """Walk the association graph from seed memories.

        Returns up to RESONANCE_LIMIT associated memories (not including seeds).
        """
        edges = self._build_association_edges()
        visited = set(seed_indices)
        frontier = set(seed_indices)
        found: list[tuple[float, int]] = []

        for hop in range(hops):
            next_frontier: set[int] = set()
            for node in frontier:
                for neighbor, weight in edges.get(node, {}).items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        score = weight / (hop + 1)
                        found.append((score, neighbor))
            frontier = next_frontier

        found.sort(key=lambda x: x[0], reverse=True)
        return [self.self_reflections[idx] for _, idx in found[:self.RESONANCE_LIMIT]]

    # ── Contradiction detection ───────────────────────────────────────

    def detect_contradictions(self, limit: int = 5,
                              llm_fn=None) -> list[tuple[Memory, Memory]]:
        """Find pairs of memories that may contradict each other.

        Looks for memories sharing topic overlap but with opposing
        emotional valence or negation patterns. Returns (older, newer) pairs.

        Parameters
        ----------
        limit : int
            Maximum number of contradiction pairs to return.
        llm_fn : callable, optional
            If provided, lexical candidates (score > 0.3) are verified by
            the LLM for semantic contradiction, catching subtle conflicts
            that negation patterns alone would miss.
        """
        if len(self.self_reflections) < 2:
            return []

        contradictions: list[tuple[float, Memory, Memory]] = []
        n = len(self.self_reflections)

        # Lower threshold when LLM is available to catch more candidates
        threshold = 0.3 if llm_fn else 0.5

        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.self_reflections[i], self.self_reflections[j]
                score = self._contradiction_score(a, b)
                if score > threshold:
                    if a.timestamp <= b.timestamp:
                        contradictions.append((score, a, b))
                    else:
                        contradictions.append((score, b, a))

        contradictions.sort(key=lambda x: x[0], reverse=True)

        # If LLM available, verify top candidates for semantic contradiction
        if llm_fn and contradictions:
            verified: list[tuple[float, Memory, Memory]] = []
            # Check up to limit*3 candidates (LLM might reject many)
            for score, a, b in contradictions[:limit * 3]:
                if len(verified) >= limit:
                    break
                if score > 0.5:
                    # High-confidence lexical hit — keep without LLM check
                    verified.append((score, a, b))
                else:
                    # Borderline — ask LLM to verify
                    if _llm_verify_contradiction(llm_fn, a.content, b.content):
                        verified.append((min(score + 0.2, 1.0), a, b))
            return [(a, b) for _, a, b in verified[:limit]]

        return [(a, b) for _, a, b in contradictions[:limit]]

    @staticmethod
    def _contradiction_score(a: Memory, b: Memory) -> float:
        """Score how likely two memories contradict each other."""
        words_a = _content_words(a.content)
        words_b = _content_words(b.content)
        topic_overlap = _overlap_ratio(words_a, words_b)
        if topic_overlap < 0.15:
            return 0.0

        score = 0.0

        vec_a = _emotion_to_vector(a.emotion)
        vec_b = _emotion_to_vector(b.emotion)
        if vec_a and vec_b:
            valence_diff = abs(vec_a[0] - vec_b[0])
            if valence_diff > 0.8:
                score += 0.4

        if min(a.importance, b.importance) < 3:
            return 0.0

        _NEG = {"not", "no", "never", "don't", "doesn't", "isn't", "wasn't",
                "wouldn't", "couldn't", "can't", "won't", "hardly", "barely"}
        a_has_neg = bool(_NEG & set(a.content.lower().split()))
        b_has_neg = bool(_NEG & set(b.content.lower().split()))
        if a_has_neg != b_has_neg and topic_overlap > 0.25:
            score += 0.4

        score *= (0.5 + topic_overlap)
        return min(1.0, score)

    def get_contradiction_context(self) -> str:
        """Build a context string flagging detected contradictions."""
        pairs = self.detect_contradictions(limit=3)
        if not pairs:
            return ""
        lines = ["=== POSSIBLE CONTRADICTIONS IN MY MEMORIES ===",
                 "(These memories seem to conflict — consider which still feels true)"]
        for older, newer in pairs:
            lines.append(f'  Earlier: "{older.content}" ({older.emotion})')
            lines.append(f'  Later:   "{newer.content}" ({newer.emotion})')
            lines.append("")
        return "\n".join(lines)

    # ── Memory consolidation ("sleep") ────────────────────────────────

    def find_consolidation_clusters(self, min_cluster: int = 3,
                                     max_clusters: int = 3) -> list[list[Memory]]:
        """Find groups of related memories that could be consolidated into gist memories."""
        if len(self.self_reflections) < min_cluster:
            return []

        n = len(self.self_reflections)
        word_sets = [_content_words(r.content) for r in self.self_reflections]
        adjacency: dict[int, set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                overlap = _overlap_ratio(word_sets[i], word_sets[j])
                if 0.25 <= overlap < _DEDUP_THRESHOLD:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        used: set[int] = set()
        clusters: list[list[Memory]] = []
        by_degree = sorted(range(n), key=lambda i: len(adjacency[i]), reverse=True)
        for seed in by_degree:
            if seed in used or len(adjacency[seed]) < min_cluster - 1:
                continue
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
                tag = "" if self.professional else (f" ({m.emotion})" if m.emotion else "")
                mem_lines.append(f"  - {m.content}{tag}")
            cluster_blocks.append(
                f"CLUSTER {i + 1} ({len(cluster)} related memories):\n"
                + "\n".join(mem_lines)
            )

        return CONSOLIDATION_PROMPT.format(
            clusters="\n\n".join(cluster_blocks))

    def apply_consolidation(self, gists: list[dict]):
        """Store consolidation gists as new synthetic memories."""
        for g in gists:
            content = g.get("gist", "").strip()
            if not content or len(content) < 20:
                continue
            new_words = _content_words(content)
            is_dup = False
            for existing in self.self_reflections:
                if _overlap_ratio(new_words, _content_words(existing.content)) >= _DEDUP_THRESHOLD:
                    is_dup = True
                    break
            if is_dup:
                continue
            self.self_reflections.append(Memory(
                content=content,
                emotion=("neutral" if self.professional
                         else g.get("emotion", "understanding")),
                importance=g.get("importance", 6),
                source="consolidation",
                why_saved="Synthesized from related experiences during memory consolidation",
            ))

    # ── Memory Dreaming (between-session cross-memory pattern discovery) ──

    def needs_dream(self) -> bool:
        """True if enough sessions have passed for a dream cycle."""
        count = self._brief_data.get("session_count", 0)
        last = self._brief_data.get("last_dream_session", 0)
        return count - last >= DREAM_INTERVAL

    def find_dream_candidates(self, max_pairs: int = 5) -> list[tuple[Memory, Memory, float]]:
        """Find pairs of memories that were never co-active but share latent connections.

        This mimics sleep-stage memory replay: the system looks for 2nd/3rd-degree
        connections in the association graph between memories that have never been
        in the same context window together.  Returns (mem_a, mem_b, link_strength)
        sorted by link strength.
        """
        if len(self.self_reflections) < 4:
            return []

        edges = self._build_association_edges()
        if not edges:
            return []

        # Build 2-hop reachability (memories connected by an intermediary)
        two_hop: list[tuple[int, int, float]] = []
        for a_idx, neighbors_a in edges.items():
            for bridge_idx, w_ab in neighbors_a.items():
                for c_idx, w_bc in edges.get(bridge_idx, {}).items():
                    if c_idx != a_idx and c_idx not in neighbors_a:
                        # a and c are NOT directly connected but both connect through bridge
                        strength = (w_ab + w_bc) / 2.0
                        if a_idx < c_idx:  # deduplicate pairs
                            two_hop.append((a_idx, c_idx, strength))

        # Deduplicate and keep strongest link per pair
        seen: dict[tuple[int, int], float] = {}
        for a, c, s in two_hop:
            key = (a, c)
            if key not in seen or s > seen[key]:
                seen[key] = s

        # Filter out memories that are too similar (consolidation territory)
        # and too dissimilar (no real connection)
        candidates = []
        for (a_idx, c_idx), strength in seen.items():
            if a_idx >= len(self.self_reflections) or c_idx >= len(self.self_reflections):
                continue
            mem_a = self.self_reflections[a_idx]
            mem_c = self.self_reflections[c_idx]
            overlap = _overlap_ratio(mem_a.content_words, mem_c.content_words)
            if overlap >= 0.25:  # too similar — consolidation handles this
                continue
            if overlap < 0.02 and strength < 3:  # too distant
                continue
            candidates.append((mem_a, mem_c, strength))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:max_pairs]

    def prepare_dream_prompt(self) -> str:
        """Build the prompt for between-session memory dreaming.

        Returns empty string if no dream candidates found.
        """
        pairs = self.find_dream_candidates()
        if not pairs:
            return ""

        pair_blocks = []
        for i, (a, b, strength) in enumerate(pairs):
            tag_a = "" if self.professional else (f" ({a.emotion})" if a.emotion else "")
            tag_b = "" if self.professional else (f" ({b.emotion})" if b.emotion else "")
            pair_blocks.append(
                f"PAIR {i + 1}:\n"
                f"  Memory A: {a.content}{tag_a}\n"
                f"  Memory B: {b.content}{tag_b}"
            )

        return DREAM_PROMPT.format(pairs="\n\n".join(pair_blocks))

    def apply_dream(self, connections: list[dict]):
        """Store dream-discovered connections as new memories.

        Each connection should have: {insight, emotion, importance}
        """
        dream_log = self._brief_data.setdefault("dream_log", [])
        session = self._brief_data.get("session_count", 0)

        for c in connections:
            insight = c.get("insight", "").strip()
            if not insight or len(insight) < 20:
                continue
            # Dedup against existing memories
            new_words = _content_words(insight)
            is_dup = any(
                _overlap_ratio(new_words, _content_words(m.content)) >= _DEDUP_THRESHOLD
                for m in self.self_reflections
            )
            if is_dup:
                continue
            mem = Memory(
                content=insight,
                emotion=("neutral" if self.professional
                         else c.get("emotion", "curious")),
                importance=max(1, min(10, c.get("importance", 5))),
                source="dream",
                why_saved="Connection discovered during memory dreaming — "
                          "linking experiences that were never in the same context",
            )
            self.self_reflections.append(mem)
            self._index_memory(len(self.self_reflections) - 1, mem)
            dream_log.append({
                "session": session,
                "insight": insight[:200],
                "timestamp": datetime.now().isoformat(),
            })

        self._brief_data["last_dream_session"] = session
        # Cap dream log to last 50 entries
        if len(dream_log) > 50:
            self._brief_data["dream_log"] = dream_log[-50:]

    # ── Regret Scoring (track importance estimation mistakes) ─────────

    def get_regret_memories(self) -> list[Memory]:
        """Return all memories that have been flagged as overestimated.

        Sorted by regret score (highest first). These are memories where the
        agent believed something was highly important but later rescored it down.
        """
        regretted = [
            m for m in self.self_reflections
            if getattr(m, "_regret", 0.0) > 0
        ]
        regretted.sort(key=lambda m: m._regret, reverse=True)
        return regretted

    def get_regret_patterns(self) -> dict:
        """Analyze what kinds of memories tend to be overestimated.

        Returns a dict with:
        - count: how many regretted memories
        - avg_original_importance: what they were rated at
        - avg_final_importance: what they ended up at
        - common_emotions: emotions most frequently overestimated
        - common_sources: sources most frequently overestimated
        """
        regretted = self.get_regret_memories()
        if not regretted:
            return {"count": 0}

        orig_imps = [m._believed_importance for m in regretted if m._believed_importance > 0]
        final_imps = [m.importance for m in regretted]
        emotions = [m.emotion for m in regretted]
        sources = [m.source for m in regretted]

        # Count emotion/source frequencies
        emotion_counts: dict[str, int] = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        source_counts: dict[str, int] = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        return {
            "count": len(regretted),
            "avg_original_importance": round(sum(orig_imps) / len(orig_imps), 1) if orig_imps else 0,
            "avg_final_importance": round(sum(final_imps) / len(final_imps), 1),
            "common_emotions": sorted(emotion_counts, key=emotion_counts.get, reverse=True)[:3],
            "common_sources": sorted(source_counts, key=source_counts.get, reverse=True)[:3],
        }

    def get_regret_context(self) -> str:
        """Build a context string about regret patterns for self-awareness.

        Only included when there are enough data points to be meaningful.
        """
        patterns = self.get_regret_patterns()
        if patterns["count"] < 3:
            return ""
        lines = [
            "=== THINGS I'VE LEARNED ABOUT MY OWN JUDGMENT ===",
            f"I've overestimated the importance of {patterns['count']} memories.",
        ]
        if patterns.get("avg_original_importance") and patterns.get("avg_final_importance"):
            lines.append(
                f"On average I rated them {patterns['avg_original_importance']}/10 "
                f"but they turned out to be more like {patterns['avg_final_importance']}/10."
            )
        if patterns.get("common_emotions"):
            lines.append(
                f"I tend to overrate memories tagged as: {', '.join(patterns['common_emotions'])}."
            )
        return "\n".join(lines)

    # ── Relationship Arc Tracking (per-entity warmth trajectory) ──────

    def _update_relationship_arc(self, entity: str, emotion: str):
        """Update the relationship trajectory for an entity based on a new impression.

        Called automatically by add_social_impression().  The trajectory is a
        single float in [-1, 1] representing the direction of travel: positive
        means warming, negative means cooling.  Each new impression nudges the
        trajectory based on emotional valence with exponential decay toward 0.

        In professional mode, only impression count is tracked (no emotional
        trajectory).
        """
        arcs = self._brief_data.setdefault("relationship_arcs", {})
        session = self._brief_data.get("session_count", 0)

        if entity not in arcs:
            arcs[entity] = {
                "trajectory": 0.0,   # -1 (cooling) to +1 (warming)
                "warmth": 0.0,       # current warmth level (-1 to 1)
                "history": [],       # [(session, warmth)] data points
                "impression_count": 0,
            }

        arc = arcs[entity]

        if self.professional:
            # Professional mode: count interactions but skip emotional trajectory
            arc["impression_count"] = arc.get("impression_count", 0) + 1
            return

        vec = _emotion_to_vector(emotion)
        valence = vec[0] if vec else 0.0  # pleasure dimension = warmth signal

        # Exponential moving average: trajectory drifts toward current valence
        old_traj = arc["trajectory"]
        alpha = 0.35  # blending factor — how fast trajectory shifts
        new_traj = old_traj * (1 - alpha) + valence * alpha
        arc["trajectory"] = round(max(-1.0, min(1.0, new_traj)), 4)

        # Warmth accumulates: current warmth + trajectory nudge
        old_warmth = arc["warmth"]
        warmth_nudge = 0.1 * arc["trajectory"]
        arc["warmth"] = round(max(-1.0, min(1.0, old_warmth + warmth_nudge)), 4)

        arc["impression_count"] = arc.get("impression_count", 0) + 1

        # Record history point (deduplicate per session)
        history = arc["history"]
        if history and history[-1][0] == session:
            history[-1] = [session, arc["warmth"]]
        else:
            history.append([session, arc["warmth"]])
        # Cap history to last 100 data points
        if len(history) > 100:
            arc["history"] = history[-100:]

    def get_relationship_arc(self, entity: str) -> dict | None:
        """Return the relationship arc for an entity, or None if no data.

        Returns:
            {trajectory, warmth, history, impression_count, trend_label}
        """
        arcs = self._brief_data.get("relationship_arcs", {})
        if entity not in arcs:
            return None
        arc = dict(arcs[entity])  # shallow copy
        t = arc["trajectory"]
        if t > 0.15:
            arc["trend_label"] = "warming"
        elif t < -0.15:
            arc["trend_label"] = "cooling"
        else:
            arc["trend_label"] = "stable"
        return arc

    def get_arc_context(self, entity: str) -> str:
        """Build a context string about the relationship trajectory with an entity."""
        arc = self.get_relationship_arc(entity)
        if not arc or arc["impression_count"] < 2:
            return ""
        w = arc["warmth"]
        warmth_word = (
            "very warm" if w > 0.5 else
            "warm" if w > 0.2 else
            "cool" if w < -0.2 else
            "cold" if w < -0.5 else
            "neutral"
        )
        trend = arc["trend_label"]
        history = arc.get("history", [])
        since_text = ""
        if len(history) >= 2:
            # Find when current trend started
            for i in range(len(history) - 1, 0, -1):
                if (history[i][1] - history[i - 1][1]) * arc["trajectory"] < 0:
                    since_text = f" since session {history[i][0]}"
                    break
        return (
            f"(My relationship with {entity} feels {warmth_word} and {trend}{since_text}. "
            f"Trajectory: {arc['trajectory']:+.2f})"
        )

    # ── Resonance (old memories resurfacing) ──────────────────────────

    def resonate(self, context: str, limit: int | None = None,
                 llm_fn=None) -> list[Memory]:
        """Find old faded memories that resonate with current conversation.

        Uses the inverted index for O(k) lookup plus co-occurrence graph
        expansion for learned associations.  Adaptive floor adjusts based
        on match quality.

        Parameters
        ----------
        context : str
            The current conversation or query text.
        limit : int, optional
            Maximum number of resonant memories to return.
        llm_fn : callable, optional
            If provided and lexical matching yields sparse results, the LLM
            is asked to generate semantically related concepts to bridge
            the gap (e.g. "quantum entanglement" → "particle physics").
        """
        hard_cap = max(limit or self.RESONANCE_LIMIT, self.RESONANCE_LIMIT + 3)
        if not context or not self.self_reflections:
            return []

        context_words = _resonance_words(context)
        if not context_words:
            return []

        # Expand query words with synonyms so 'afraid' finds 'fear' memories
        context_words = _expand_synonyms(context_words)
        context_prefixes = {w[:5] for w in context_words if len(w) >= 5}
        context_bigrams = _bigrams(context)
        context_trigrams = _trigrams(context)

        # Co-occurrence expansion: find candidates through learned associations
        cooc_terms = self._expand_via_cooccurrence(context_words)

        active_set = set(
            id(r) for r in sorted(
                self.self_reflections,
                key=lambda r: r.mood_adjusted_vividness(self._mood),
                reverse=True,
            )[:self.ACTIVE_SELF_LIMIT]
        )

        # Lazy rebuild if index is stale
        if self.self_reflections and not getattr(self, '_word_index', None):
            self._word_index: dict[str, set[int]] = {}
            self._prefix_index: dict[str, set[int]] = {}
            self._rebuild_index()

        candidate_indices: set[int] = set()
        for w in context_words:
            candidate_indices |= self._word_index.get(w, set())
        for p in context_prefixes:
            candidate_indices |= self._prefix_index.get(p, set())
        # Co-occurrence expanded terms (learned associations)
        for w in cooc_terms:
            candidate_indices |= self._word_index.get(w, set())

        scored: list[tuple[float, int, Memory]] = []
        seed_indices: list[int] = []
        for idx in candidate_indices:
            if idx >= len(self.self_reflections):
                continue
            ref = self.self_reflections[idx]
            if id(ref) in active_set:
                continue

            mem_text = f"{ref.content} {ref.emotion}".lower()
            mem_words = _resonance_words(mem_text)
            mem_prefixes = {w[:5] for w in mem_words if len(w) >= 5}

            exact_overlap = len(context_words & mem_words)
            prefix_overlap = len(context_prefixes & mem_prefixes)
            total_overlap = max(exact_overlap, prefix_overlap)

            # Bigram (phrase) matching bonus
            mem_bigs = _bigrams(mem_text)
            bigram_hits = len(context_bigrams & mem_bigs)
            total_overlap += bigram_hits * 2  # phrases worth double

            # Trigram (longer phrase) matching bonus
            mem_trigs = _trigrams(mem_text)
            trigram_hits = len(context_trigrams & mem_trigs)
            total_overlap += trigram_hits * 3  # long phrases worth triple

            if total_overlap >= 1:
                score = total_overlap + (ref.importance * 0.2)
                scored.append((score, idx, ref))
                seed_indices.append(idx)

        scored.sort(key=lambda x: x[0], reverse=True)

        # Adaptive floor based on best match quality
        top_score = scored[0][0] if scored else 0.0
        floor = _adaptive_floor(top_score, self.RESONANCE_SCORE_FLOOR)

        resonant: list[Memory] = []
        for score, idx, ref in scored:
            if len(resonant) >= hard_cap:
                break
            if score >= floor:
                resonant.append(ref)

        # If nothing passed the adaptive floor, return best-effort
        if not resonant and scored:
            for score, idx, ref in scored[:hard_cap]:
                resonant.append(ref)

        # Associative chain expansion
        if seed_indices:
            associated = self.associate(seed_indices[:5], hops=ASSOCIATION_HOPS)
            resonant_ids = {id(r) for r in resonant} | active_set
            for a in associated:
                if id(a) not in resonant_ids and len(resonant) < hard_cap:
                    a_words = _resonance_words(
                        f"{a.content} {a.emotion}".lower())
                    a_prefixes = {w[:5] for w in a_words if len(w) >= 5}
                    if (context_words & a_words) or (context_prefixes & a_prefixes):
                        resonant.append(a)
                        resonant_ids.add(id(a))

        # ── LLM Semantic Bridging ─────────────────────────────────────
        # If lexical matching found very few results and an LLM is available,
        # ask it to generate semantically related terms and re-search.
        if llm_fn and len(resonant) < 3 and len(self.self_reflections) > 10:
            bridge_terms = _semantic_bridge(llm_fn, context)
            if bridge_terms:
                bridge_words = set()
                for term in bridge_terms:
                    bridge_words |= _resonance_words(term.lower())
                bridge_words -= context_words  # only truly new terms
                if bridge_words:
                    bridge_indices: set[int] = set()
                    for w in bridge_words:
                        bridge_indices |= self._word_index.get(w, set())
                        if len(w) >= 5:
                            bridge_indices |= self._prefix_index.get(w[:5], set())
                    resonant_ids = {id(r) for r in resonant} | active_set
                    for idx in bridge_indices:
                        if idx >= len(self.self_reflections):
                            continue
                        ref = self.self_reflections[idx]
                        if id(ref) in resonant_ids:
                            continue
                        if len(resonant) >= hard_cap:
                            break
                        resonant.append(ref)
                        resonant_ids.add(id(ref))

        for r in resonant:
            r.touch()

        return resonant

    # ── Context block (inject into your agent's system prompt) ────────

    def get_context_block(self, current_entity: str = "",
                          resonant: list[Memory] | None = None,
                          conversation_context: str = "") -> str:
        """Build a narrative memory block for injection into a system prompt.

        Returns a first-person text block including:
        - Mood indicator (character mode only)
        - Compressed self-brief (if available)
        - Entity brief (if entity provided)
        - Foreground memories (full text, relevant to context)
        - Background memories (compressed one-liners)
        - Social impressions for the current entity
        - Resonant memories (old ones resurfacing)
        - Contradiction flags

        In professional mode, emotion tags and mood indicators are omitted
        and headers use task-oriented language.
        """
        lines = []
        pro = self.professional

        # Mood indicator (character mode only)
        if not pro:
            mood_label = self.mood_label
            if mood_label != "neutral":
                lines.append(f"(Right now I'm feeling somewhat {mood_label}.)")
                lines.append("")

        # Compressed brief
        self_brief = self._brief_data.get("self_brief", "")
        if self_brief:
            lines.append("=== OPERATIONAL CONTEXT ===" if pro
                         else "=== MY COMPRESSED SELF-UNDERSTANDING ===")
            lines.append(self_brief)
            lines.append("")
        if current_entity:
            entity_brief = self._brief_data.get("entity_briefs", {}).get(current_entity, "")
            if entity_brief:
                lines.append(
                    f"=== CONTEXT: {current_entity.upper()} ===" if pro
                    else f"=== MY UNDERSTANDING OF {current_entity.upper()} ===")
                lines.append(entity_brief)
                lines.append("")

        # Entity preferences (structured likes/dislikes)
        if current_entity:
            pref_ctx = self.get_preference_context(current_entity)
            if pref_ctx:
                lines.append(pref_ctx)
                lines.append("")

        # Short-term facts
        stm_ctx = self.get_stm_context(
            entity=current_entity, context=conversation_context)
        if stm_ctx:
            lines.append(stm_ctx)
            lines.append("")

        # Task memory (active tasks, solutions, project overview)
        if self._active_project:
            task_ctx = self.get_task_context(
                conversation_context=conversation_context)
            if task_ctx:
                lines.append(task_ctx)

        foreground, background = self.partition_active_self(context=conversation_context)
        if foreground:
            if pro:
                lines.append("=== RELEVANT KNOWLEDGE ===" if conversation_context and background
                             else "=== STORED KNOWLEDGE ===")
            else:
                lines.append("=== THINGS ON MY MIND RIGHT NOW ===" if conversation_context and background
                             else "=== THINGS I KNOW ABOUT MYSELF ===")
            for r in foreground:
                emotion_tag = "" if pro else (f" ({r.emotion})" if r.emotion else "")
                lines.append(f"— {r.content}{emotion_tag}")
            lines.append("")
        if background:
            lines.append("=== STORED KNOWLEDGE (background) ===" if pro
                         else "=== THINGS I KNOW ABOUT MYSELF (background) ===")
            for r in background:
                # Structured compression: preserve key facets instead of blind truncation
                parts = []
                # Core content — extract salient portion
                content = r.content
                if len(content) > 80:
                    # Try to break at sentence boundary
                    cut = content[:80].rfind(".")
                    if cut < 30:
                        cut = content[:80].rfind(",")
                    if cut < 30:
                        cut = content[:80].rfind(" ")
                    content = content[:max(cut, 30)].rstrip() + "…"
                parts.append(content)
                if not pro and r.emotion:
                    parts.append(f"[{r.emotion}]")
                if r.entity:
                    parts.append(f"re:{r.entity}")
                lines.append(f"· {' '.join(parts)}")
            lines.append("")

        if current_entity:
            active_social = self.get_active_social(current_entity)
            if active_social:
                lines.append(
                    f"=== NOTES ON {current_entity.upper()} ===" if pro
                    else f"=== MY IMPRESSIONS OF {current_entity.upper()} ===")
                if not pro:
                    arc_ctx = self.get_arc_context(current_entity)
                    if arc_ctx:
                        lines.append(arc_ctx)
                for r in active_social:
                    emotion_tag = "" if pro else (f" ({r.emotion})" if r.emotion else "")
                    lines.append(f"— {r.content}{emotion_tag}")
                lines.append("")

        if resonant:
            lines.append("=== RELATED PRIOR KNOWLEDGE ===" if pro
                         else "=== SOMETHING THIS REMINDS ME OF (old memories resurfacing) ===")
            for r in resonant:
                emotion_tag = "" if pro else (f" ({r.emotion})" if r.emotion else "")
                lines.append(f"— {r.content}{emotion_tag}")
            lines.append("")

        contradiction_ctx = self.get_contradiction_context()
        if contradiction_ctx:
            lines.append(contradiction_ctx)
            lines.append("")

        regret_ctx = self.get_regret_context()
        if regret_ctx:
            lines.append(regret_ctx)
            lines.append("")

        return "\n".join(lines) if lines else ""

    # ── Persistence ───────────────────────────────────────────────────

    def save(self):
        """Persist all memory data to disk (encrypted if key was provided)."""
        self._ensure_dirs()

        self._write_json(self.self_file,
                         [r.to_dict() for r in self.self_reflections])

        for entity, impressions in self.social_impressions.items():
            fname = self._entity_filename(entity)
            self._write_json(self.social_dir / fname,
                             [r.to_dict() for r in impressions])

        # Persist short-term memory
        stm_file = self.data_dir / "stm.json"
        if hasattr(self, '_stm') and self._stm:
            self._write_json(stm_file, [f.to_dict() for f in self._stm])
        elif stm_file.exists():
            stm_file.unlink()

        # Persist co-occurrence graph
        cooc_file = self.data_dir / "cooccurrence.json"
        if hasattr(self, '_cooccurrence') and self._cooccurrence:
            self._write_json(cooc_file, self._cooccurrence)

        # Persist task memory (project data + global solutions)
        if self._active_project:
            self._save_project()
        if self._solutions:
            self._save_solutions()

    def _load(self):
        if self.self_file.exists():
            try:
                data = self._read_json(self.self_file)
                if isinstance(data, list):
                    self.self_reflections = [
                        Memory.from_dict(d) for d in data
                        if isinstance(d, dict) and "content" in d
                    ]
            except Exception as e:
                print(f"[VividnessMem] Warning: could not load self_memory, starting fresh ({e})")
                self.self_reflections = []

        if self.social_dir.exists():
            ext = "*.enc" if self._fernet else "*.json"
            for fpath in self.social_dir.glob(ext):
                try:
                    data = self._read_json(fpath)
                    if not isinstance(data, list):
                        continue
                    # Recover entity name: encrypted files store it inside,
                    # plain files derive it from the filename.
                    if self._fernet:
                        # First entry's 'entity' field (or fallback to hash)
                        entity = ""
                        for d in data:
                            if isinstance(d, dict) and d.get("entity"):
                                entity = d["entity"]
                                break
                        if not entity:
                            entity = fpath.stem  # fallback
                    else:
                        entity = fpath.stem.replace("_", " ").title()
                    self.social_impressions[entity] = [
                        Memory.from_dict(d) for d in data
                        if isinstance(d, dict) and "content" in d
                    ]
                except Exception as e:
                    print(f"[VividnessMem] Warning: corrupt {fpath.name}, skipping ({e})")

        # Load short-term memory
        stm_file = self.data_dir / "stm.json"
        if stm_file.exists():
            try:
                data = self._read_json(stm_file)
                if isinstance(data, list):
                    self._stm = [ShortTermFact.from_dict(d) for d in data
                                 if isinstance(d, dict)]
            except Exception:
                self._stm = []

        # Load co-occurrence graph
        cooc_file = self.data_dir / "cooccurrence.json"
        if cooc_file.exists():
            try:
                data = self._read_json(cooc_file)
                if isinstance(data, dict):
                    self._cooccurrence = data
            except Exception:
                pass

    # ── Brief & maintenance ───────────────────────────────────────────

    def _load_brief(self):
        if self.brief_file.exists():
            try:
                data = self._read_json(self.brief_file)
                if isinstance(data, dict):
                    self._brief_data.update(data)
            except Exception:
                pass
        self._load_mood()

    def _save_brief(self):
        self._ensure_dirs()
        self._save_mood()
        self._write_json(self.brief_file, self._brief_data)

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

    def prepare_brief_prompt(self, entity: str = "") -> str:
        """Build the prompt for compressed brief generation.

        Pass the name of the entity your agent most commonly talks to.
        """
        sorted_self = sorted(self.self_reflections,
                             key=lambda r: r.vividness, reverse=True)[:50]
        self_lines = []
        for r in sorted_self:
            tag = "" if self.professional else (f" ({r.emotion})" if r.emotion else "")
            self_lines.append(f"- [imp={r.importance}] {r.content}{tag}")
        self_text = "\n".join(self_lines) if self_lines else "(no self-reflections yet)"

        entity_mems = self.social_impressions.get(entity, []) if entity else []
        sorted_social = sorted(entity_mems,
                               key=lambda r: r.vividness, reverse=True)[:30]
        social_lines = []
        for r in sorted_social:
            tag = "" if self.professional else (f" ({r.emotion})" if r.emotion else "")
            social_lines.append(f"- [imp={r.importance}] {r.content}{tag}")
        social_text = ("\n".join(social_lines)
                       if social_lines else f"(no impressions of {entity or 'others'} yet)")

        prev_self = self._brief_data.get("self_brief", "")
        prev_entity = self._brief_data.get("entity_briefs", {}).get(entity, "") if entity else ""
        if prev_self or prev_entity:
            prev_section = "YOUR PREVIOUS BRIEF (update and improve this):\n"
            if prev_self:
                prev_section += f"  Self: {prev_self}\n"
            if prev_entity:
                prev_section += f"  {entity}: {prev_entity}\n"
            prev_section += "\n"
        else:
            prev_section = ""

        display_entity = entity or "Others"
        return BRIEF_PROMPT.format(
            entity=display_entity,
            entity_upper=display_entity.upper(),
            previous_brief_section=prev_section,
            self_memories=self_text,
            social_memories=social_text,
        )

    def apply_brief(self, parsed: dict, entity: str = ""):
        """Store the compressed brief from LLM response."""
        if "self_brief" in parsed:
            self._brief_data["self_brief"] = parsed["self_brief"][:2000]
        if "entity_brief" in parsed and entity:
            if "entity_briefs" not in self._brief_data:
                self._brief_data["entity_briefs"] = {}
            self._brief_data["entity_briefs"][entity] = parsed["entity_brief"][:2000]
        self._brief_data["last_brief_session"] = self._brief_data.get("session_count", 0)

    def prepare_rescore_prompt(self) -> tuple[str, list[Memory]]:
        """Build the prompt for importance re-evaluation.

        Returns (prompt_text, indexed_list_of_memories).
        """
        if not self.self_reflections:
            return "", []

        candidates = []
        for r in self.self_reflections:
            age_days = (datetime.now() - datetime.fromisoformat(
                r.timestamp)).total_seconds() / 86400
            if age_days < 1:
                continue
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
            tag = "" if self.professional else (f" ({r.emotion})" if r.emotion else "")
            lines.append(
                f"[{i}] importance={r.importance}, access_count={r._access_count}, "
                f"age={age_days:.0f}d: {r.content}{tag}"
            )

        return RESCORE_PROMPT.format(memories="\n".join(lines)), selected

    def apply_rescores(self, adjustments: list[dict],
                       indexed_memories: list[Memory]):
        """Apply importance re-scoring and emotional reappraisal.

        Enforces +/-2 cap per memory.  When importance drops significantly,
        the memory is flagged with a regret score so the agent can learn
        what kinds of things it tends to overestimate.
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
            original = ref.importance
            clamped = max(ref.importance - 2, min(ref.importance + 2, new_imp))

            # Regret tracking: if importance drops, record the overestimation
            if clamped < original and original >= 5:
                drop = original - clamped
                regret = drop / 10.0  # normalize: 2-point drop on a 10-scale = 0.2
                ref._regret = max(getattr(ref, "_regret", 0.0), regret)
                if getattr(ref, "_believed_importance", 0) == 0:
                    ref._believed_importance = original

            ref.importance = clamped
            new_emotion = adj.get("new_emotion")
            if new_emotion and isinstance(new_emotion, str) and not self.professional:
                new_emotion = new_emotion.strip()[:50]
                if new_emotion:
                    ref.emotion = new_emotion
        self._brief_data["last_rescore_session"] = self._brief_data.get("session_count", 0)

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a summary of the memory system's state."""
        arcs = self._brief_data.get("relationship_arcs", {})
        result = {
            "total_self_reflections": len(self.self_reflections),
            "active_self": len(self.get_active_self()),
            "social_entities": list(self.social_impressions.keys()),
            "total_social": sum(len(v) for v in self.social_impressions.values()),
            "session_count": self._brief_data.get("session_count", 0),
            "has_brief": bool(self._brief_data.get("self_brief")),
            "regret_count": len(self.get_regret_memories()),
            "dream_count": len(self._brief_data.get("dream_log", [])),
            "tracked_arcs": {e: a.get("trend_label", "stable")
                             for e, a in arcs.items()
                             if self.get_relationship_arc(e)},
        }
        # Task memory stats
        result["active_project"] = self._active_project
        result["total_solutions"] = len(self._solutions)
        result["project_tasks"] = len(self._project_tasks)
        result["project_actions"] = len(self._project_actions)
        result["project_artifacts"] = len(self._project_artifacts)
        if self._project_meta:
            result["project_tier"] = self._project_decay_tier(
                self._project_meta.get("last_accessed", ""))
            result["agent_track_quality"] = self._project_meta.get(
                "agent_track_quality", AUTO_TRACK_QUALITY_INIT)
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Prompt Templates (send these to your LLM, parse the JSON response)
# ═══════════════════════════════════════════════════════════════════════════

CURATION_PROMPT = """The conversation has ended. Take a moment to reflect.

Think about what just happened — what stood out to you? Did you discover
anything about yourself? Did the other person say something that changed
how you think? Did you form any new opinions or notice any patterns?

Write down what you want to remember. You have two journals:

1. SELF JOURNAL — reflections about yourself: who you're becoming, what you
   believe, what you care about, moments that felt meaningful.

2. IMPRESSIONS OF OTHERS — your honest impressions of whoever you spoke
   with: what they're like, what surprised you, what you agree or disagree
   on, how they made you feel.

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


DREAM_PROMPT = """You're dreaming. Not literally — but between sessions, your memory
is replaying. Below are pairs of memories that were NEVER in the same
conversation, but share a hidden connection through intermediate memories.

Your job: look at each pair and ask "what do these two experiences have
in common that I never noticed before?" The connection might be:
- A shared emotional undercurrent
- A pattern you're repeating without realising
- An insight that only emerges when you see both memories side by side
- A contradiction you hadn't spotted
- A cause-and-effect you missed in real time

{pairs}

For each pair where you see a genuine connection, write a dream insight:

```json
[
  {{"insight": "I just realised that my frustration with X is actually the same feeling as...", "emotion": "curious", "importance": 5}}
]
```

Rules:
- Only write insights for pairs where you genuinely see a connection
- Write naturally, in first person, like waking up with a new thought
- Importance 3-7 (dreams are interesting but need real-world validation)
- Return [] if no pairs spark anything genuine

Return ONLY the JSON array."""


def parse_dream_response(response: str) -> list[dict]:
    """Parse the LLM's dream response into connection entries."""
    return parse_curation_response(response)  # same JSON array format


# ═══════════════════════════════════════════════════════════════════════════
#  Response parsers (parse LLM JSON output)
# ═══════════════════════════════════════════════════════════════════════════

def parse_curation_response(response: str) -> list[dict]:
    """Parse the LLM's curation response into memory entries."""
    response = response.strip()
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


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Query Expansion (optional — call before resonate for richer results)
# ═══════════════════════════════════════════════════════════════════════════

QUERY_EXPANSION_PROMPT = """Restate the following query using different words.
Give 3 alternative phrasings that mean the same thing but use different
vocabulary. Return ONLY the 3 alternatives, one per line, no numbering.

Query: {query}"""

SEMANTIC_BRIDGE_PROMPT = """Given the search query below, list 5 conceptually related topics,
terms, or phrases that someone might have stored in their memories about
this subject — even if they used completely different words.

For example:
  Query: "quantum entanglement"
  particle physics, superposition, wave function, Bell's theorem, quantum mechanics

Return ONLY the 5 terms/phrases, comma-separated, on a single line.

Query: {query}"""

CONTRADICTION_VERIFY_PROMPT = """Do these two statements contradict each other?
Consider semantic meaning, not just surface wording. Two statements contradict
if believing both to be true at the same time would be logically inconsistent.

Statement A: "{a}"
Statement B: "{b}"

Answer with ONLY "yes" or "no"."""


def _semantic_bridge(llm_fn, query: str) -> list[str]:
    """Ask the LLM for semantically related concepts to bridge lexical gaps."""
    try:
        prompt = SEMANTIC_BRIDGE_PROMPT.format(query=query)
        response = llm_fn(prompt)
        if response and isinstance(response, str):
            # Parse comma-separated terms
            terms = [t.strip() for t in response.strip().split(",") if t.strip()]
            return terms[:8]  # cap at 8 terms
    except Exception:
        pass
    return []


def _llm_verify_contradiction(llm_fn, content_a: str, content_b: str) -> bool:
    """Ask the LLM whether two memory contents semantically contradict."""
    try:
        prompt = CONTRADICTION_VERIFY_PROMPT.format(a=content_a, b=content_b)
        response = llm_fn(prompt)
        if response and isinstance(response, str):
            return response.strip().lower().startswith("yes")
    except Exception:
        pass
    return False


def expand_query(llm_fn, query: str) -> str:
    """Use an LLM to expand a query with alternative phrasings.

    Parameters
    ----------
    llm_fn : callable
        Function that takes a prompt string and returns a response string.
    query : str
        The original search query.

    Returns
    -------
    str
        The original query plus LLM-generated alternative phrasings,
        concatenated for broader retrieval coverage.
    """
    try:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        response = llm_fn(prompt)
        if response and isinstance(response, str):
            return f"{query} {response.strip()}"
    except Exception:
        pass
    return query
