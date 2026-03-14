"""
SpecMem — Procedural Memory System for LLM Agents
===================================================

A standalone, plug-and-play procedural learning system.
Agents learn *how to do things better* through structured reflection
on task outcomes — without changing any weights or parameters.

    from SpecMem import SpecMem

Works standalone, or alongside VividnessMem for episodic enrichment:

    from SpecMem import SpecMem
    from VividnessMem import VividnessMem

    episodic  = VividnessMem("memory_data")
    procedural = SpecMem("skill_data", vividness_mem=episodic)

Features:
 • Attempt → Outcome → Reflect → Learn loop
 • 3-tier retrieval: tag match, trigger similarity, full-text fallback
 • Confidence earned from real outcomes (Bayesian-style posterior)
 • Synonym ring for semantic retrieval bridging
 • Spaced-repetition vividness decay (unused skills atrophy)
 • Active/Proven lifecycle with compression (bloat-free long-term growth)
 • Generalization via periodic reflection (cross-lesson pattern discovery)
 • Optional VividnessMem bridge for episodic context enrichment
 • LLM-in-the-loop curation prompts for reflection, compression, generalization
 • Full JSON persistence to disk
 • Zero external dependencies (pure Python 3.10+)

Philosophy:
 • The system learns by remembering what worked, what failed, and why
 • Confidence is EARNED from outcomes, not declared
 • Proven lessons compress — removing step-by-step history once reliable
 • Old proven lessons can be DEMOTED if they start failing again
 • A better method can SUPERSEDE an older proven one
 • Without VividnessMem present, runs fully standalone

Author : Kronic90  — https://github.com/Kronic90
License: MIT
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
#  Constants & helpers
# ═══════════════════════════════════════════════════════════════════════════

INITIAL_STABILITY    = 4.0    # days before ~50% vividness fade
SPACING_BONUS        = 1.6    # stability multiplier per well-spaced use
MIN_SPACING_DAYS     = 0.5    # uses closer than this don't boost stability
PROVEN_THRESHOLD     = 3      # consecutive successes to reach "proven"
DEMOTION_FAILURES    = 2      # consecutive failures to demote proven → active
RETRIEVAL_LIMIT      = 5      # max lessons returned by default
GENERALIZE_INTERVAL  = 3      # sessions between generalization sweeps

# ── Tag matching stop words — words too common to be useful tags ──────────
_TAG_STOP = frozenset(
    "the a an is are was were be been being have has had do does did "
    "will would could should shall may might can that this with from "
    "for and but not very much also just really some more into "
    "about what when where which while how their there then than "
    "your you they them".split()
)

# ── Synonym ring (same concept as VividnessMem) ──────────────────────────
_SYNONYM_GROUPS: list[frozenset[str]] = [
    frozenset({"error", "bug", "fault", "defect", "issue", "problem", "failure"}),
    frozenset({"fix", "repair", "patch", "resolve", "solve", "correct"}),
    frozenset({"deploy", "release", "publish", "ship", "launch"}),
    frozenset({"test", "verify", "validate", "check", "assert"}),
    frozenset({"build", "compile", "construct", "assemble"}),
    frozenset({"config", "configuration", "settings", "setup", "configure"}),
    frozenset({"database", "db", "datastore", "storage"}),
    frozenset({"search", "find", "locate", "lookup", "query"}),
    frozenset({"create", "generate", "produce", "craft"}),
    frozenset({"delete", "remove", "destroy", "drop", "purge"}),
    frozenset({"update", "modify", "change", "alter", "edit", "revise"}),
    frozenset({"send", "transmit", "dispatch", "deliver"}),
    frozenset({"receive", "accept", "fetch", "retrieve", "obtain"}),
    frozenset({"fast", "quick", "rapid", "swift", "speedy"}),
    frozenset({"slow", "sluggish", "gradual", "lagging"}),
    frozenset({"crash", "freeze", "hang", "stall", "lock"}),
    frozenset({"timeout", "delay", "latency", "lag"}),
    frozenset({"retry", "repeat", "reattempt", "redo"}),
    frozenset({"parse", "extract", "analyze", "process"}),
    frozenset({"format", "structure", "organize", "arrange"}),
    frozenset({"connect", "link", "attach", "bind", "join"}),
    frozenset({"disconnect", "detach", "unlink", "separate"}),
    frozenset({"encrypt", "secure", "protect", "guard"}),
    frozenset({"auth", "authenticate", "authorize", "login", "signin"}),
    frozenset({"cache", "memoize", "buffer", "store"}),
    frozenset({"log", "record", "track", "monitor", "trace"}),
    frozenset({"debug", "troubleshoot", "diagnose", "inspect"}),
    frozenset({"optimize", "improve", "enhance", "refine", "tune"}),
    frozenset({"migrate", "transfer", "move", "port", "convert"}),
    frozenset({"backup", "snapshot", "archive", "save"}),
    frozenset({"restore", "recover", "rollback", "revert"}),
    frozenset({"install", "setup", "deploy", "provision"}),
    frozenset({"scale", "resize", "expand", "grow"}),
    frozenset({"merge", "combine", "consolidate", "unify"}),
    frozenset({"split", "divide", "partition", "separate", "segment"}),
    frozenset({"retry", "backoff", "reconnect", "failover"}),
    frozenset({"clean", "tidy", "sanitize", "scrub", "purify"}),
    frozenset({"plan", "design", "architect", "blueprint", "outline"}),
    frozenset({"document", "describe", "explain", "annotate"}),
    frozenset({"succeed", "pass", "work", "complete", "accomplish"}),
    frozenset({"fail", "break", "crash", "error", "malfunction"}),
]

_SYNONYM_MAP: dict[str, frozenset[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        # Merge if a word appears in multiple groups
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


def _content_words(text: str) -> set[str]:
    """Extract meaningful (non-stop) words ≥4 chars from text."""
    return {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())} - _TAG_STOP


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_coeff(a: set[str], b: set[str]) -> float:
    """Overlap coefficient — robust when synonym expansion inflates sets.

    |A & B| / min(|A|, |B|)  — rewards any overlap without penalizing
    large expanded sets the way Jaccard does.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ═══════════════════════════════════════════════════════════════════════════
#  Attempt — a single try at a task
# ═══════════════════════════════════════════════════════════════════════════

class Attempt:
    """Record of one attempt to apply a strategy."""

    __slots__ = ("timestamp", "approach", "outcome", "diagnosis", "next_step")

    def __init__(self, approach: str, outcome: str = "pending",
                 diagnosis: str = "", next_step: str = ""):
        self.timestamp = datetime.now().isoformat()
        self.approach = approach
        self.outcome = outcome          # "success" | "failure" | "partial"
        self.diagnosis = diagnosis      # Why it worked/failed
        self.next_step = next_step      # What to try differently (failure/partial)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "approach": self.approach,
            "outcome": self.outcome,
            "diagnosis": self.diagnosis,
            "next_step": self.next_step,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Attempt":
        obj = cls.__new__(cls)
        obj.timestamp = d.get("timestamp", datetime.now().isoformat())
        obj.approach  = d.get("approach", "")
        obj.outcome   = d.get("outcome", "pending")
        obj.diagnosis = d.get("diagnosis", "")
        obj.next_step = d.get("next_step", "")
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  Lesson — the core learning unit
# ═══════════════════════════════════════════════════════════════════════════

class Lesson:
    """A single procedural lesson with earned confidence.

    Lessons start as 'active' (still learning) and progress to 'proven'
    after consistent success.  Proven lessons compress by removing the
    full attempt history and retaining only the proven_summary.  If a
    proven lesson starts failing, it demotes back to 'active'.
    """

    __slots__ = (
        "id", "title", "tags", "trigger",
        "attempts", "current_strategy", "confidence",
        "status", "times_applied", "times_succeeded",
        "consecutive_successes", "consecutive_failures",
        "proven_summary", "superseded_by",
        "created", "last_used",
        "_stability", "_access_count", "_last_access",
    )

    def __init__(self, title: str, trigger: str, tags: set[str] | None = None,
                 strategy: str = ""):
        self.id = uuid.uuid4().hex[:12]
        self.title = title
        self.tags: set[str] = tags or set()
        self.trigger = trigger
        self.attempts: list[Attempt] = []
        self.current_strategy = strategy
        self.confidence: float = 0.3     # low default — must be earned
        self.status: str = "active"      # "active" | "proven"
        self.times_applied: int = 0
        self.times_succeeded: int = 0
        self.consecutive_successes: int = 0
        self.consecutive_failures: int = 0
        self.proven_summary: str = ""
        self.superseded_by: str | None = None
        self.created = datetime.now().isoformat()
        self.last_used = self.created
        self._stability: float = INITIAL_STABILITY
        self._access_count: int = 0
        self._last_access: str = self.created

    # ── Vividness (spaced-repetition decay) ───────────────────────────

    def touch(self):
        """Record a use.  Well-spaced uses increase stability."""
        now = datetime.now()
        last = datetime.fromisoformat(self._last_access)
        gap_days = (now - last).total_seconds() / 86400
        if gap_days >= MIN_SPACING_DAYS:
            self._stability *= SPACING_BONUS
        self._access_count += 1
        self._last_access = now.isoformat()
        self.last_used = now.isoformat()

    @property
    def vividness(self) -> float:
        """How 'fresh' this lesson feels — decays without use."""
        age_days = (datetime.now()
                    - datetime.fromisoformat(self.created)
                    ).total_seconds() / 86400
        retention = math.exp(-age_days / max(self._stability, 0.1))
        # Proven lessons decay slower (floor at 0.3 retention)
        if self.status == "proven":
            retention = max(retention, 0.3)
        return self.confidence * retention

    @property
    def content_words(self) -> set[str]:
        """All meaningful words across title, trigger, strategy, summary."""
        text = f"{self.title} {self.trigger} {self.current_strategy} {self.proven_summary}"
        return _content_words(text)

    @property
    def tag_words(self) -> set[str]:
        """Tags as a clean word set."""
        return {t.lower().strip() for t in self.tags if t.strip()}

    # ── Outcome registration ──────────────────────────────────────────

    def record_attempt(self, approach: str, outcome: str,
                       diagnosis: str = "", next_step: str = "") -> Attempt:
        """Record an attempt and update confidence/counters."""
        attempt = Attempt(
            approach=approach,
            outcome=outcome,
            diagnosis=diagnosis,
            next_step=next_step,
        )
        # Proven lessons don't accumulate step-by-step history —
        # only counters matter.  If demoted, attempts start logging again.
        if self.status != "proven":
            self.attempts.append(attempt)
        self.times_applied += 1
        self.touch()

        if outcome == "success":
            self.times_succeeded += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self._update_confidence_up()
        elif outcome == "failure":
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self._update_confidence_down()
        else:  # partial
            self.consecutive_successes = 0
            # Partial doesn't reset failure streak but doesn't increment it
            self._update_confidence_partial()

        # Auto-promote to proven
        if (self.status == "active"
                and self.consecutive_successes >= PROVEN_THRESHOLD):
            self._promote()

        # Auto-demote from proven
        if (self.status == "proven"
                and self.consecutive_failures >= DEMOTION_FAILURES):
            self._demote()

        return attempt

    def _update_confidence_up(self):
        """Bayesian-ish update after success."""
        # Moves toward times_succeeded / times_applied with momentum
        empirical = self.times_succeeded / max(self.times_applied, 1)
        self.confidence += (empirical - self.confidence) * 0.3
        self.confidence = min(self.confidence, 0.99)

    def _update_confidence_down(self):
        """Bayesian-ish update after failure."""
        empirical = self.times_succeeded / max(self.times_applied, 1)
        self.confidence += (empirical - self.confidence) * 0.4  # faster drop
        self.confidence = max(self.confidence, 0.05)

    def _update_confidence_partial(self):
        """Small nudge after partial success."""
        empirical = (self.times_succeeded + 0.5) / max(self.times_applied, 1)
        self.confidence += (empirical - self.confidence) * 0.15
        self.confidence = max(0.05, min(0.99, self.confidence))

    def _promote(self):
        """Compress to proven status — trim attempt history."""
        self.status = "proven"
        # proven_summary gets written by the LLM during curation;
        # for now generate a mechanical one
        if not self.proven_summary:
            fail_count = self.times_applied - self.times_succeeded
            self.proven_summary = (
                f"Proven after {self.times_succeeded} successes "
                f"({fail_count} earlier failures). "
                f"Strategy: {self.current_strategy[:200]}"
            )
        # Clear attempt history to remove bloat
        self.attempts.clear()

    def _demote(self):
        """Revert to active — something about this stopped working."""
        self.status = "active"
        self.consecutive_successes = 0
        # Keep proven_summary as context for what used to work

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "title": self.title,
            "tags": sorted(self.tags),
            "trigger": self.trigger,
            "current_strategy": self.current_strategy,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "times_applied": self.times_applied,
            "times_succeeded": self.times_succeeded,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "proven_summary": self.proven_summary,
            "superseded_by": self.superseded_by,
            "created": self.created,
            "last_used": self.last_used,
            "stability": self._stability,
            "access_count": self._access_count,
            "last_access": self._last_access,
        }
        if self.attempts:
            d["attempts"] = [a.to_dict() for a in self.attempts]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Lesson":
        obj = cls.__new__(cls)
        obj.id                    = d.get("id", uuid.uuid4().hex[:12])
        obj.title                 = d.get("title", "")
        obj.tags                  = set(d.get("tags", []))
        obj.trigger               = d.get("trigger", "")
        obj.current_strategy      = d.get("current_strategy", "")
        obj.confidence            = d.get("confidence", 0.3)
        obj.status                = d.get("status", "active")
        obj.times_applied         = d.get("times_applied", 0)
        obj.times_succeeded       = d.get("times_succeeded", 0)
        obj.consecutive_successes = d.get("consecutive_successes", 0)
        obj.consecutive_failures  = d.get("consecutive_failures", 0)
        obj.proven_summary        = d.get("proven_summary", "")
        obj.superseded_by         = d.get("superseded_by", None)
        obj.created               = d.get("created", datetime.now().isoformat())
        obj.last_used             = d.get("last_used", obj.created)
        obj._stability            = d.get("stability", INITIAL_STABILITY)
        obj._access_count         = d.get("access_count", 0)
        obj._last_access          = d.get("last_access", obj.created)
        obj.attempts = [Attempt.from_dict(a) for a in d.get("attempts", [])]
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  SpecMem — the main procedural memory system
# ═══════════════════════════════════════════════════════════════════════════

class SpecMem:
    """Procedural memory store with outcome-driven learning.

    Parameters
    ----------
    data_dir : str or Path
        Directory for JSON persistence. Created if it doesn't exist.
    vividness_mem : object or None
        Optional VividnessMem instance for episodic enrichment.
        If None, runs fully standalone.
    """

    RETRIEVAL_LIMIT        = RETRIEVAL_LIMIT
    RETRIEVAL_SCORE_FLOOR  = 0.15   # minimum combined score to return a lesson
    LEARN_MATCH_THRESHOLD  = 0.25   # minimum score to reuse an existing lesson in learn()
    TAG_WEIGHT             = 0.35
    TRIGGER_WEIGHT         = 0.40
    FULLTEXT_WEIGHT        = 0.25

    def __init__(self, data_dir: str | Path = "skill_data",
                 vividness_mem=None):
        self.data_dir = Path(data_dir)
        self.lessons_file = self.data_dir / "lessons.json"
        self.meta_file    = self.data_dir / "meta.json"

        self.lessons: list[Lesson] = []

        # Optional VividnessMem bridge
        self._episodic = vividness_mem

        # Inverted indices for retrieval
        self._word_index:   dict[str, set[int]] = {}
        self._prefix_index: dict[str, set[int]] = {}
        self._tag_index:    dict[str, set[int]] = {}

        # Session tracking
        self._meta: dict = {
            "session_count": 0,
            "total_lessons_created": 0,
            "total_lessons_proven": 0,
        }

        self._ensure_dirs()
        self._load()
        self._rebuild_index()

    # ── Directory & persistence ───────────────────────────────────────

    def _ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        """Load lessons and metadata from disk."""
        if self.lessons_file.exists():
            try:
                raw = json.loads(self.lessons_file.read_text(encoding="utf-8"))
                self.lessons = [Lesson.from_dict(d) for d in raw]
            except (json.JSONDecodeError, KeyError):
                self.lessons = []
        if self.meta_file.exists():
            try:
                self._meta = json.loads(
                    self.meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, KeyError):
                pass

    def save(self):
        """Persist all lessons and metadata to disk."""
        self.lessons_file.write_text(
            json.dumps([l.to_dict() for l in self.lessons],
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.meta_file.write_text(
            json.dumps(self._meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Inverted index ────────────────────────────────────────────────

    def _index_lesson(self, idx: int, lesson: Lesson):
        """Add a lesson to all inverted indices."""
        # Word & prefix index over content
        words = lesson.content_words
        expanded = _expand_synonyms(words)
        for w in expanded:
            self._word_index.setdefault(w, set()).add(idx)
            if len(w) >= 5:
                self._prefix_index.setdefault(w[:5], set()).add(idx)
        # Tag index
        for tag in lesson.tag_words:
            self._tag_index.setdefault(tag, set()).add(idx)
            # Also index tag synonyms
            for syn in _SYNONYM_MAP.get(tag, frozenset()):
                self._tag_index.setdefault(syn, set()).add(idx)

    def _rebuild_index(self):
        """Rebuild all indices from scratch."""
        self._word_index.clear()
        self._prefix_index.clear()
        self._tag_index.clear()
        for i, lesson in enumerate(self.lessons):
            self._index_lesson(i, lesson)

    # ══════════════════════════════════════════════════════════════════
    #  CORE API
    # ══════════════════════════════════════════════════════════════════

    # ── 1. Add a new lesson ───────────────────────────────────────────

    def add_lesson(self, title: str, trigger: str,
                   tags: set[str] | list[str] | None = None,
                   strategy: str = "") -> Lesson:
        """Create and store a new lesson.

        Parameters
        ----------
        title : str
            Short human-readable title, e.g. "Deploy Flask app safely"
        trigger : str
            Natural language description of WHEN this applies.
            "When deploying a Python web app to production"
        tags : set or list of str
            Free-form tags for structured retrieval.
        strategy : str
            Initial known strategy (can be empty if learning from scratch).
        """
        clean_tags = set(tags) if tags else set()
        # Auto-extract additional tags from title + trigger
        auto_tags = _content_words(f"{title} {trigger}")
        clean_tags |= {t for t in auto_tags if len(t) >= 4}

        lesson = Lesson(title=title, trigger=trigger,
                        tags=clean_tags, strategy=strategy)
        self.lessons.append(lesson)
        self._index_lesson(len(self.lessons) - 1, lesson)
        self._meta["total_lessons_created"] += 1
        self.save()
        return lesson

    # ── 2. Retrieve relevant lessons ──────────────────────────────────

    def retrieve(self, task_description: str,
                 limit: int | None = None,
                 include_superseded: bool = False) -> list[tuple[Lesson, float]]:
        """Find lessons relevant to a task. Returns (lesson, score) pairs.

        Uses 3-tier retrieval:
          Tier 1: Tag matching (structured, fast)
          Tier 2: Trigger similarity (word overlap + synonyms)
          Tier 3: Full-text fallback (strategy + summary text)

        Scores are weighted and combined with confidence × vividness.
        """
        cap = limit or self.RETRIEVAL_LIMIT
        if not task_description or not self.lessons:
            return []

        task_words = _content_words(task_description)
        if not task_words:
            return []

        expanded_words = _expand_synonyms(task_words)
        task_prefixes = {w[:5] for w in expanded_words if len(w) >= 5}

        # Gather candidate indices from all index tiers
        candidate_indices: set[int] = set()

        # Tier 1: Tag index
        for w in expanded_words:
            candidate_indices |= self._tag_index.get(w, set())

        # Tier 2 + 3: Word/prefix index
        for w in expanded_words:
            candidate_indices |= self._word_index.get(w, set())
        for p in task_prefixes:
            candidate_indices |= self._prefix_index.get(p, set())

        scored: list[tuple[float, Lesson]] = []
        for idx in candidate_indices:
            if idx >= len(self.lessons):
                continue
            lesson = self.lessons[idx]

            # Skip superseded unless requested
            if lesson.superseded_by and not include_superseded:
                continue

            # Tier 1: Tag match (overlap coefficient — robust to expansion)
            tag_score = _overlap_coeff(expanded_words, lesson.tag_words)

            # Tier 2: Trigger similarity
            trigger_words = _content_words(lesson.trigger)
            trigger_expanded = _expand_synonyms(trigger_words)
            trigger_score = _overlap_coeff(expanded_words, trigger_expanded)

            # Tier 3: Full-text (strategy + summary)
            fulltext_words = lesson.content_words
            fulltext_expanded = _expand_synonyms(fulltext_words)
            fulltext_score = _overlap_coeff(expanded_words, fulltext_expanded)

            # Weighted combination
            retrieval_score = (
                tag_score     * self.TAG_WEIGHT
                + trigger_score * self.TRIGGER_WEIGHT
                + fulltext_score * self.FULLTEXT_WEIGHT
            )

            # Final score includes confidence and vividness
            final = retrieval_score * (0.4 + 0.4 * lesson.confidence
                                       + 0.2 * lesson.vividness)

            if final >= self.RETRIEVAL_SCORE_FLOOR:
                scored.append((final, lesson))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = [(lesson, score) for score, lesson in scored[:cap]]

        # Touch retrieved lessons
        for lesson, _ in results:
            lesson.touch()

        return results

    # ── 3. Record an outcome ──────────────────────────────────────────

    def record_outcome(self, lesson_id: str, approach: str, outcome: str,
                       diagnosis: str = "", next_step: str = "") -> Attempt | None:
        """Record the outcome of applying a lesson.

        Parameters
        ----------
        lesson_id : str
            The lesson's unique ID.
        approach : str
            What was actually tried.
        outcome : str
            "success", "failure", or "partial"
        diagnosis : str
            Why it worked or failed.
        next_step : str
            What to try differently next time (on failure/partial).

        Returns the Attempt, or None if lesson_id not found.
        """
        lesson = self.get_lesson(lesson_id)
        if not lesson:
            return None

        attempt = lesson.record_attempt(
            approach=approach,
            outcome=outcome,
            diagnosis=diagnosis,
            next_step=next_step,
        )

        # Update strategy on failure/partial if next_step provided
        if outcome in ("failure", "partial") and next_step:
            lesson.current_strategy = next_step

        # Update strategy on success if approach differs from current
        if outcome == "success" and approach and approach != lesson.current_strategy:
            lesson.current_strategy = approach

        # Track proven count
        if lesson.status == "proven":
            self._meta["total_lessons_proven"] = sum(
                1 for l in self.lessons if l.status == "proven")

        self.save()
        return attempt

    # ── 4. Record outcome for a new or matched lesson in one call ─────

    def learn(self, task_description: str, approach: str, outcome: str,
              diagnosis: str = "", next_step: str = "",
              title: str = "", tags: set[str] | list[str] | None = None
              ) -> tuple[Lesson, Attempt]:
        """High-level API: retrieve-or-create, then record outcome.

        If a highly relevant lesson already exists, records the outcome
        against it.  Otherwise creates a new lesson and records the
        first attempt.

        Returns (lesson, attempt).
        """
        # Try to find an existing match
        matches = self.retrieve(task_description, limit=1)
        if matches and matches[0][1] >= self.LEARN_MATCH_THRESHOLD:
            lesson = matches[0][0]
        else:
            lesson = self.add_lesson(
                title=title or task_description[:80],
                trigger=task_description,
                tags=tags,
                strategy=approach if outcome == "success" else "",
            )

        attempt = lesson.record_attempt(
            approach=approach,
            outcome=outcome,
            diagnosis=diagnosis,
            next_step=next_step,
        )

        if outcome in ("failure", "partial") and next_step:
            lesson.current_strategy = next_step
        if outcome == "success" and approach:
            lesson.current_strategy = approach

        self.save()
        return lesson, attempt

    # ── 5. Supersede a lesson with a better one ───────────────────────

    def supersede(self, old_lesson_id: str, new_lesson_id: str):
        """Mark old_lesson as superseded by new_lesson."""
        old = self.get_lesson(old_lesson_id)
        if old:
            old.superseded_by = new_lesson_id
            self.save()

    # ── Utility lookups ───────────────────────────────────────────────

    def get_lesson(self, lesson_id: str) -> Lesson | None:
        """Find a lesson by ID."""
        for l in self.lessons:
            if l.id == lesson_id:
                return l
        return None

    def get_proven(self) -> list[Lesson]:
        """Return all proven lessons."""
        return [l for l in self.lessons if l.status == "proven"]

    def get_active(self) -> list[Lesson]:
        """Return all active (still-learning) lessons."""
        return [l for l in self.lessons
                if l.status == "active" and not l.superseded_by]

    # ══════════════════════════════════════════════════════════════════
    #  SESSION & LIFECYCLE
    # ══════════════════════════════════════════════════════════════════

    def start_session(self):
        """Call at the beginning of each agent session."""
        self._meta["session_count"] = self._meta.get("session_count", 0) + 1
        self.save()

    def needs_generalization(self) -> bool:
        """True if enough sessions have passed for generalization."""
        return (self._meta.get("session_count", 0) % GENERALIZE_INTERVAL == 0
                and self._meta.get("session_count", 0) > 0)

    # ══════════════════════════════════════════════════════════════════
    #  CONTEXT BLOCK (for system prompt injection)
    # ══════════════════════════════════════════════════════════════════

    def get_context_block(self, task_description: str = "",
                          limit: int | None = None) -> str:
        """Build a text block summarizing relevant procedural knowledge.

        Inject this into your agent's system prompt before task execution.
        """
        parts: list[str] = []
        parts.append("## Procedural Memory (What I Know How To Do)")

        # Stats
        proven_count = sum(1 for l in self.lessons if l.status == "proven")
        active_count = sum(1 for l in self.lessons
                          if l.status == "active" and not l.superseded_by)
        if proven_count or active_count:
            parts.append(f"[{proven_count} proven skills, "
                         f"{active_count} still learning]")

        if not task_description:
            # General summary — list top proven lessons by vividness
            top = sorted(self.lessons,
                        key=lambda l: l.vividness, reverse=True)[:5]
            if top:
                parts.append("")
                parts.append("Top skills:")
                for l in top:
                    marker = "✓" if l.status == "proven" else "~"
                    parts.append(f"  {marker} {l.title} "
                                 f"(conf: {l.confidence:.0%})")
            return "\n".join(parts)

        # Task-specific retrieval
        matches = self.retrieve(task_description, limit=limit)
        if not matches:
            parts.append("")
            parts.append("No relevant lessons found for this task.")
            # Bridge to episodic memory if available
            episodic_ctx = self._episodic_bridge(task_description)
            if episodic_ctx:
                parts.append("")
                parts.append("Related episodic memories:")
                parts.append(episodic_ctx)
            return "\n".join(parts)

        parts.append("")
        for lesson, score in matches:
            marker = "✓ PROVEN" if lesson.status == "proven" else "~ LEARNING"
            parts.append(f"─── {marker}: {lesson.title} "
                         f"(confidence: {lesson.confidence:.0%}) ───")
            parts.append(f"When: {lesson.trigger}")

            if lesson.current_strategy:
                parts.append(f"Strategy: {lesson.current_strategy}")

            if lesson.proven_summary:
                parts.append(f"History: {lesson.proven_summary}")

            # Show recent attempts for active lessons (learning context)
            if lesson.status == "active" and lesson.attempts:
                recent = lesson.attempts[-3:]
                parts.append("Recent attempts:")
                for a in recent:
                    icon = {"success": "+", "failure": "✗",
                            "partial": "~"}.get(a.outcome, "?")
                    parts.append(f"  {icon} {a.approach[:100]}")
                    if a.diagnosis:
                        parts.append(f"    → {a.diagnosis[:120]}")
                    if a.next_step:
                        parts.append(f"    Next: {a.next_step[:120]}")

            parts.append("")

        # Bridge to episodic memory
        episodic_ctx = self._episodic_bridge(task_description)
        if episodic_ctx:
            parts.append("Related episodic memories:")
            parts.append(episodic_ctx)

        return "\n".join(parts)

    def _episodic_bridge(self, task_description: str) -> str:
        """If VividnessMem is available, pull relevant episodic context."""
        if self._episodic is None:
            return ""
        try:
            resonant = self._episodic.resonate(context=task_description)
            if not resonant:
                return ""
            lines = []
            for mem in resonant[:3]:
                content = mem.content
                if len(content) > 100:
                    cut = content[:100].rfind(" ")
                    content = content[:max(cut, 60)] + "…"
                emotion = getattr(mem, "emotion", "")
                tag = f" [{emotion}]" if emotion else ""
                lines.append(f"  · {content}{tag}")
            return "\n".join(lines)
        except Exception:
            return ""

    # ══════════════════════════════════════════════════════════════════
    #  LLM-IN-THE-LOOP PROMPTS
    # ══════════════════════════════════════════════════════════════════

    def reflection_prompt(self, task_description: str, approach: str,
                          raw_output: str = "") -> str:
        """Generate a prompt for the LLM to reflect on a task outcome.

        Feed this to your LLM, then parse the response and call
        record_outcome() or learn() with the extracted fields.
        """
        # Find any existing lessons for context
        matches = self.retrieve(task_description, limit=3)
        history_ctx = ""
        if matches:
            history_parts = []
            for lesson, score in matches:
                history_parts.append(
                    f"- '{lesson.title}' (conf: {lesson.confidence:.0%}): "
                    f"{lesson.current_strategy[:150]}")
                for a in lesson.attempts[-2:]:
                    history_parts.append(
                        f"  → {a.outcome}: {a.diagnosis[:100]}")
            history_ctx = "\nPrevious related lessons:\n" + "\n".join(history_parts)

        return f"""Reflect on this task outcome as a learning exercise.

Task: {task_description}
Approach taken: {approach}
{f"Output/result: {raw_output[:500]}" if raw_output else ""}
{history_ctx}

Analyze and respond in this EXACT JSON format:
{{
    "outcome": "success" or "failure" or "partial",
    "diagnosis": "WHY it worked or failed — root cause, not just symptoms",
    "lesson_title": "short title for this skill (3-8 words)",
    "tags": ["tag1", "tag2", "tag3"],
    "current_strategy": "the best known approach based on all evidence",
    "next_step": "what to try differently next time (only if failure/partial, else empty string)",
    "confidence_note": "how sure are you this strategy is reliable, and why"
}}
"""

    def generalization_prompt(self) -> str:
        """Generate a prompt for the LLM to find cross-lesson patterns.

        Call this periodically (check needs_generalization()).
        """
        active = self.get_active()
        proven = self.get_proven()
        all_lessons = active + proven

        if len(all_lessons) < 3:
            return ""

        lesson_summaries = []
        for l in all_lessons[:20]:  # Cap to avoid huge prompts
            status = "PROVEN" if l.status == "proven" else "learning"
            summary = l.proven_summary or l.current_strategy
            fails = l.times_applied - l.times_succeeded
            lesson_summaries.append(
                f"- [{status}] '{l.title}' (tags: {', '.join(sorted(l.tags)[:5])})\n"
                f"  Strategy: {summary[:150]}\n"
                f"  Applied {l.times_applied}x, succeeded {l.times_succeeded}x, "
                f"failed {fails}x")

        return f"""Review these procedural lessons and find patterns.

{chr(10).join(lesson_summaries)}

Look for:
1. GENERALIZATIONS: Are there 2+ lessons that share a common abstract pattern?
   If so, what's the general principle?
2. CONTRADICTIONS: Do any lessons give conflicting advice for similar situations?
3. GAPS: Based on failure patterns, is there a skill that's clearly missing?
4. OUTDATED: Any lessons that seem superseded by newer, better approaches?

Respond in this EXACT JSON format:
{{
    "generalizations": [
        {{
            "title": "general principle title",
            "trigger": "when this general principle applies",
            "strategy": "the general approach",
            "source_lessons": ["lesson_id_1", "lesson_id_2"],
            "tags": ["tag1", "tag2"]
        }}
    ],
    "contradictions": [
        {{"lesson_ids": ["id1", "id2"], "note": "what conflicts"}}
    ],
    "gaps": ["description of missing skill"],
    "outdated": ["lesson_id that should be superseded"]
}}
"""

    def compression_prompt(self, lesson: Lesson) -> str:
        """Generate a prompt for the LLM to write a proven_summary.

        Called when a lesson is about to be promoted to proven.
        The LLM summarizes the full learning trajectory into one paragraph.
        """
        attempts_text = []
        for a in lesson.attempts:
            attempts_text.append(
                f"  {a.outcome}: {a.approach[:100]}\n"
                f"    Diagnosis: {a.diagnosis[:150]}\n"
                f"    Next step: {a.next_step[:100]}" if a.next_step else
                f"  {a.outcome}: {a.approach[:100]}\n"
                f"    Diagnosis: {a.diagnosis[:150]}")

        return f"""This lesson is being promoted to PROVEN status.
Summarize the full learning journey in ONE concise paragraph.

Lesson: {lesson.title}
Trigger: {lesson.trigger}
Final strategy: {lesson.current_strategy}
Attempts ({len(lesson.attempts)} total):
{chr(10).join(attempts_text)}

Write a proven_summary that captures:
- What was tried and failed (briefly)
- What finally worked and why
- Any important caveats or conditions

Respond with ONLY the summary paragraph, no JSON wrapping.
"""

    # ══════════════════════════════════════════════════════════════════
    #  RESPONSE PARSERS
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def parse_reflection_response(raw: str) -> dict:
        """Parse the LLM's reflection JSON response.

        Returns a dict with: outcome, diagnosis, lesson_title, tags,
        current_strategy, next_step, confidence_note.
        Falls back to sensible defaults on parse failure.
        """
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            cleaned = raw.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```",
                                  cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract key fields
            return {
                "outcome": "partial",
                "diagnosis": raw[:200] if raw else "Could not parse reflection",
                "lesson_title": "",
                "tags": [],
                "current_strategy": "",
                "next_step": "",
                "confidence_note": "",
            }

    @staticmethod
    def parse_generalization_response(raw: str) -> dict:
        """Parse the LLM's generalization JSON response."""
        try:
            cleaned = raw.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```",
                                  cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return {
                "generalizations": [],
                "contradictions": [],
                "gaps": [],
                "outdated": [],
            }

    # ── Apply parsed generalization results ───────────────────────────

    def apply_generalizations(self, parsed: dict):
        """Create new lessons from generalization discoveries."""
        for gen in parsed.get("generalizations", []):
            title = gen.get("title", "")
            if not title:
                continue
            # Don't create duplicates
            existing = self.retrieve(title, limit=1)
            if existing and existing[0][1] >= 0.4:
                continue
            self.add_lesson(
                title=title,
                trigger=gen.get("trigger", ""),
                tags=gen.get("tags", []),
                strategy=gen.get("strategy", ""),
            )

        for old_id in parsed.get("outdated", []):
            # Mark as superseded (without a specific replacement yet)
            old = self.get_lesson(old_id)
            if old and not old.superseded_by:
                old.superseded_by = "outdated"

        self.save()

    # ══════════════════════════════════════════════════════════════════
    #  STATS & DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        """Return a summary of the system's current state."""
        return {
            "total_lessons": len(self.lessons),
            "active": sum(1 for l in self.lessons
                         if l.status == "active" and not l.superseded_by),
            "proven": sum(1 for l in self.lessons if l.status == "proven"),
            "superseded": sum(1 for l in self.lessons if l.superseded_by),
            "total_attempts": sum(l.times_applied for l in self.lessons),
            "total_successes": sum(l.times_succeeded for l in self.lessons),
            "overall_success_rate": (
                sum(l.times_succeeded for l in self.lessons)
                / max(sum(l.times_applied for l in self.lessons), 1)
            ),
            "avg_confidence": (
                sum(l.confidence for l in self.lessons)
                / max(len(self.lessons), 1)
            ),
            "sessions": self._meta.get("session_count", 0),
        }

    def __len__(self) -> int:
        return len(self.lessons)

    def __repr__(self) -> str:
        s = self.stats()
        return (f"SpecMem({s['total_lessons']} lessons: "
                f"{s['proven']} proven, {s['active']} active, "
                f"success rate {s['overall_success_rate']:.0%})")
