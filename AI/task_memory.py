"""
task_memory.py  —  Learning-by-doing memory for Aria & Rex.

Records what each AI did during tasks, simulations, and file operations,
along with self-reflections on outcomes and how to improve.

Stored separately from personal/social memories.
Retrieved only when relevant to the current conversation or task.

Two implementations matching each AI's memory architecture:
  - AriaTaskMemory  — Organic / vividness-based (mirrors memory_aria.py)
  - RexTaskMemory   — MemGPT core/archival split (mirrors memory_rex.py)

Storage: ai_dialogue_data/{aria,rex}/task_memories.json

Tag format (parsed from AI responses):
    [TASK_MEMORY]
    summary: What you did and what the results showed
    reflection: What you learned, what you would do differently
    keywords: comma, separated, relevant, terms
    importance: 1-10
    [/TASK_MEMORY]
"""

import json
import re
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "ai_dialogue_data"

# ─── Soft dedup helpers ────────────────────────────────────────────────────
_DEDUP_THRESHOLD = 0.80
_DEDUP_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "was",
    "are", "not", "but", "you", "your", "have", "has", "had",
    "been", "will", "would", "could", "should", "can", "did",
    "does", "just", "about", "they", "them", "what", "when",
    "something", "things", "thing", "well", "yeah", "okay",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())} - _DEDUP_STOP


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ═══════════════════════════════════════════════════════════════════════════
#  Task Entry  (shared data structure)
# ═══════════════════════════════════════════════════════════════════════════
class TaskEntry:
    """A single learning-by-doing memory."""

    def __init__(
        self,
        summary: str = "",
        reflection: str = "",
        keywords: list[str] | None = None,
        task_type: str = "general",
        importance: int = 5,
        timestamp: str = "",
        emotion: str = "",
    ):
        self.summary = summary
        self.reflection = reflection
        self.keywords = keywords or []
        self.task_type = task_type
        self.importance = importance
        self.timestamp = timestamp or datetime.now().isoformat()
        self.emotion = emotion
        # For Aria's organic decay — tracks how often this memory is accessed
        self._access_count: int = 0

    def touch(self):
        """Mark this memory as accessed (keeps it vivid for Aria's system)."""
        self._access_count += 1

    @property
    def vividness(self) -> float:
        """How 'present' this experience feels — used by Aria's organic retrieval.

        Blends importance, recency, and access frequency, just like her
        personal Reflection entries.
        """
        try:
            age_hours = (
                datetime.now() - datetime.fromisoformat(self.timestamp)
            ).total_seconds() / 3600
        except (ValueError, TypeError):
            age_hours = 0
        recency_score = max(0, 10 - (age_hours / 24))  # -1 per day
        access_bonus = min(3, self._access_count * 0.5)
        return (self.importance * 0.6) + (recency_score * 0.3) + (access_bonus * 0.1)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "reflection": self.reflection,
            "keywords": self.keywords,
            "task_type": self.task_type,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "access_count": self._access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskEntry":
        e = cls(
            summary=d.get("summary", ""),
            reflection=d.get("reflection", ""),
            keywords=d.get("keywords", []),
            task_type=d.get("task_type", "general"),
            importance=d.get("importance", 5),
            timestamp=d.get("timestamp", ""),
            emotion=d.get("emotion", ""),
        )
        e._access_count = d.get("access_count", 0)
        return e

    def __repr__(self):
        return f"[{self.task_type} | imp={self.importance}] {self.summary[:60]}"


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════
_STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "was",
    "are", "not", "but", "you", "your", "have", "has", "had",
    "been", "will", "would", "could", "should", "can", "did",
    "does", "just", "about", "they", "them", "what", "when",
    "turn", "session", "conversation", "here", "there", "also",
}


def _score_relevance(entry: TaskEntry, query_lower: str, query_words: set[str]) -> float:
    """Score how relevant a task entry is to a query string."""
    score = 0.0
    entry_text = (
        f"{entry.summary} {entry.reflection} {' '.join(entry.keywords)}"
    ).lower()

    # Explicit keyword matches (high weight)
    for kw in entry.keywords:
        if kw.lower() in query_lower:
            score += 3.0

    # Word overlap
    entry_words = set(re.findall(r"\b\w{3,}\b", entry_text)) - _STOP_WORDS
    overlap = query_words & entry_words
    score += len(overlap)

    # Importance bonus
    score += entry.importance * 0.1

    return score


# ═══════════════════════════════════════════════════════════════════════════
#  Aria's Task Memory  — Organic / vividness-based
#
#  Mirrors memory_aria.py's Reflection system:
#    - All entries stored in a single journal (flat list)
#    - Retrieval based on *vividness* (importance x recency x access)
#    - Most vivid experiences surface first — like real recall
#    - Keyword relevance used as a filter, then ranked by vividness
#    - Entries "fade" over time but high-importance ones resist fading
#    - Accessing a memory makes it more vivid (positive feedback)
# ═══════════════════════════════════════════════════════════════════════════
class AriaTaskMemory:
    """Organic task memory — experiences surface by vividness, like real recall.

    No hard cap on total entries — Aria never fully forgets.
    Only ACTIVE_LIMIT entries get injected into context (the rest remain
    searchable via recall/resonance when triggered by conversation).
    """

    ACTIVE_LIMIT = 4  # how many vivid entries to inject into context

    def __init__(self):
        self.data_dir = DATA_ROOT / "aria"
        self.file_path = self.data_dir / "task_memories.json"
        self.entries: list[TaskEntry] = []
        self._load()

    def add(self, entry: TaskEntry):
        """Journal a new experience, merging if near-duplicate exists."""
        new_words = _content_words(f"{entry.summary} {entry.reflection}")
        for existing in self.entries:
            old_words = _content_words(f"{existing.summary} {existing.reflection}")
            if _overlap_ratio(new_words, old_words) >= _DEDUP_THRESHOLD:
                # Merge: keep richer content, boost existing
                if len(entry.summary) > len(existing.summary):
                    existing.summary = entry.summary
                if len(entry.reflection) > len(existing.reflection):
                    existing.reflection = entry.reflection
                existing.importance = max(existing.importance, entry.importance)
                existing.keywords = list(set(existing.keywords) | set(entry.keywords))
                existing.touch()
                self.save()
                return
        self.entries.append(entry)
        self.save()

    def search(self, query: str, limit: int = 3) -> list[TaskEntry]:
        """Surface the most vivid relevant experiences.

        First filters by keyword relevance (must pass a minimum threshold),
        then ranks by vividness — the organic way. Accessing a memory makes
        it more vivid (just like Aria's personal memories).
        """
        if not query or not self.entries:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w{3,}\b", query_lower)) - _STOP_WORDS

        # Filter to entries with some relevance
        candidates: list[TaskEntry] = []
        for entry in self.entries:
            if _score_relevance(entry, query_lower, query_words) > 1.5:
                candidates.append(entry)

        # Rank by vividness (organic recall — recent + important + accessed wins)
        candidates.sort(key=lambda e: e.vividness, reverse=True)

        # Accessing makes memories more vivid (positive feedback loop)
        result = candidates[:limit]
        for e in result:
            e.touch()

        return result

    def get_active(self, limit: int | None = None) -> list[TaskEntry]:
        """Return the most vivid task experiences overall (no query filter)."""
        n = limit or self.ACTIVE_LIMIT
        ranked = sorted(self.entries, key=lambda e: e.vividness, reverse=True)
        return ranked[:n]

    def get_context_block(self, query: str = "") -> str:
        """Build a narrative context block (journal-style, first person)."""
        entries = self.search(query, limit=self.ACTIVE_LIMIT) if query else self.get_active()
        if not entries:
            return ""
        lines = ["=== MY TASK EXPERIENCES (what I remember doing) ==="]
        for e in entries:
            emotion_tag = f" ({e.emotion})" if e.emotion else ""
            lines.append(f"\u2014 {e.summary}{emotion_tag}")
            if e.reflection:
                lines.append(f"  \u2192 {e.reflection}")
        return "\n".join(lines)

    # ─── Action-type patterns (detect what she's about to do) ─────────
    _ACTION_SIGNALS = {
        "file_write": {"write", "create", "save", "append", "file", "document", "draft"},
        "file_read":  {"read", "open", "contents", "check", "look", "examine"},
        "code":       {"code", "script", "python", "calculate", "simulate", "algorithm", "function"},
        "simulation": {"simulate", "simulation", "model", "parameters", "variables", "test"},
        "analysis":   {"analyze", "analyse", "compare", "evaluate", "measure", "results"},
        "worldbuild": {"lore", "world", "society", "faction", "culture", "narrative"},
    }

    def recall(self, context: str, limit: int = 3) -> list[TaskEntry]:
        """Human-style task recall -- only surfaces when something triggers it.

        Like a person thinking "wait, I've done this before..." when starting
        a task that reminds them of past experience. Returns nothing if the
        current context doesn't connect to any past task.

        NOT a search engine. Works by:
          1. Detect what kind of task is happening (writing, coding, simulating...)
          2. Find past experiences with topic overlap (prefix matching, no embeddings)
          3. Only return memories with genuine connection (2+ word matches)
          4. Rank by vividness (recent + important + frequently-accessed wins)
        """
        if not context or not self.entries:
            return []

        context_lower = context.lower()
        context_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", context_lower)) - _STOP_WORDS
        if not context_words:
            return []

        # Step 1: Detect current action type from context
        active_actions: set[str] = set()
        for action_type, signals in self._ACTION_SIGNALS.items():
            if context_words & signals:
                active_actions.add(action_type)

        # Step 2: Score each memory by topic overlap + action match
        context_prefixes = {w[:5] for w in context_words if len(w) >= 5}
        scored: list[tuple[float, TaskEntry]] = []

        for entry in self.entries:
            # Only consider deliberate reflections (imp >= 5) --
            # auto-captured noise ("Written: file.md") isn't a lesson
            if entry.importance < 5:
                continue

            entry_text = f"{entry.summary} {entry.reflection} {' '.join(entry.keywords)}".lower()
            entry_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", entry_text)) - _STOP_WORDS
            entry_prefixes = {w[:5] for w in entry_words if len(w) >= 5}

            # Prefix overlap (handles plurals, conjugations)
            prefix_overlap = len(context_prefixes & entry_prefixes)
            # Explicit keyword hits (tagged by the AI itself)
            keyword_hits = sum(1 for kw in entry.keywords if kw.lower() in context_lower)

            total_match = prefix_overlap + (keyword_hits * 2)

            if total_match < 2:
                continue  # not enough connection -- don't surface

            # Action type bonus: if she's doing the same KIND of thing
            action_bonus = 0.0
            entry_action_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", entry_text))
            for action_type in active_actions:
                if entry_action_words & self._ACTION_SIGNALS[action_type]:
                    action_bonus += 1.5

            score = total_match + action_bonus + (entry.importance * 0.15)
            scored.append((score, entry))

        if not scored:
            return []

        # Step 3: Among matches, rank by vividness (organic -- recent+accessed wins)
        scored.sort(key=lambda x: (x[0], x[1].vividness), reverse=True)
        results = [entry for _, entry in scored[:limit]]

        # Touch -- she remembered this, it becomes more vivid
        for e in results:
            e.touch()

        return results

    def recall_context_block(self, context: str) -> str:
        """Build a lesson-first context block from recalled task memories.

        Only returns text if the context triggers a genuine recall.
        Leads with what she LEARNED, not just what she did.
        """
        recalled = self.recall(context)
        if not recalled:
            return ""
        lines = ["(Past experiences that feel relevant to what we're doing now:)"]
        for e in recalled:
            if e.reflection:
                # Lead with the lesson
                lines.append(f"-- I learned: {e.reflection}")
                lines.append(f"   (from: {e.summary[:80]})")
            else:
                lines.append(f"-- {e.summary}")
        return "\n".join(lines)

    # ─── Persistence (Aria) ───────────────────────────────────────────

    def save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self.entries]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = [TaskEntry.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self.entries = []

    def stats(self) -> dict:
        types: dict[str, int] = {}
        for e in self.entries:
            types[e.task_type] = types.get(e.task_type, 0) + 1
        return {
            "total_entries": len(self.entries),
            "by_type": types,
            "most_vivid": self.entries[0].summary[:60] if self.entries else "none",
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Rex's Task Memory  — MemGPT-style core/archival split
#
#  Mirrors memory_rex.py's MemoryEntry system:
#    - Core tasks: small set of high-importance entries always in context
#    - Archival tasks: everything else, searchable by keyword on demand
#    - Structured retrieval — explicit keyword search, importance-ranked
#    - Core promotion/demotion when importance thresholds change
# ═══════════════════════════════════════════════════════════════════════════
class RexTaskMemory:
    """MemGPT-style task memory — core skills always present, rest searchable."""

    CORE_LIMIT = 5       # always-in-context task memories
    ARCHIVAL_MAX = 200   # max archival entries

    def __init__(self):
        self.data_dir = DATA_ROOT / "rex"
        self.file_path = self.data_dir / "task_memories.json"
        self.core: list[TaskEntry] = []      # always injected into context
        self.archival: list[TaskEntry] = []   # searchable on demand
        self._load()

    @property
    def entries(self) -> list[TaskEntry]:
        """All entries (core + archival), for stats and browsing."""
        return self.core + self.archival

    def add(self, entry: TaskEntry):
        """Add a task memory. High-importance -> core, rest -> archival.

        If core is full but this entry outranks the weakest core entry,
        demote the weak one to archival and promote this one.
        """
        if entry.importance >= 7 and len(self.core) < self.CORE_LIMIT:
            self.core.append(entry)
        else:
            self.archival.append(entry)
            # Try to promote if this is important enough
            if entry.importance >= 7 and len(self.core) >= self.CORE_LIMIT:
                self.core.sort(key=lambda e: e.importance)
                if entry.importance > self.core[0].importance:
                    demoted = self.core.pop(0)
                    self.archival.append(demoted)
                    self.core.append(entry)
                    self.archival.remove(entry)

        # Trim archival
        if len(self.archival) > self.ARCHIVAL_MAX:
            self.archival.sort(key=lambda e: e.importance, reverse=True)
            self.archival = self.archival[:self.ARCHIVAL_MAX]

        self.save()

    def search(self, query: str, limit: int = 3) -> list[TaskEntry]:
        """Search archival task memories by keyword relevance.

        Core entries are always in context so searching them is redundant.
        This searches archival entries and returns the most relevant ones.
        """
        if not query:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"\b\w{3,}\b", query_lower)) - _STOP_WORDS

        scored: list[tuple[float, TaskEntry]] = []
        for entry in self.archival:
            score = _score_relevance(entry, query_lower, query_words)
            if score > 1.5:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    def get_context_block(self, query: str = "") -> str:
        """Build a structured context block (MemGPT-style, categorised)."""
        lines = []

        # Core always present
        if self.core:
            lines.append("=== CORE TASK MEMORY (always available) ===")
            for e in self.core:
                cat_tag = f"[{e.task_type}]" if e.task_type != "general" else ""
                lines.append(f"\u2022 {cat_tag} {e.summary} ({e.importance}/10)")
                if e.reflection:
                    lines.append(f"  Learned: {e.reflection}")

        # Relevant archival entries if query provided
        if query:
            relevant = self.search(query, limit=2)
            if relevant:
                lines.append("")
                lines.append("=== RELEVANT PAST TASKS (from archival) ===")
                for e in relevant:
                    cat_tag = f"[{e.task_type}]" if e.task_type != "general" else ""
                    lines.append(f"\u2022 {cat_tag} {e.summary} ({e.importance}/10)")
                    if e.reflection:
                        lines.append(f"  Learned: {e.reflection}")

        return "\n".join(lines) if lines else ""

    # ─── Persistence ──────────────────────────────────────────────────

    def save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "core": [e.to_dict() for e in self.core],
            "archival": [e.to_dict() for e in self.archival],
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Support both old flat format and new core/archival format
                if isinstance(data, list):
                    # Migrate from old flat format
                    all_entries = [TaskEntry.from_dict(d) for d in data]
                    self.core = [e for e in all_entries if e.importance >= 7][:self.CORE_LIMIT]
                    self.archival = [e for e in all_entries if e not in self.core]
                else:
                    self.core = [TaskEntry.from_dict(d) for d in data.get("core", [])]
                    self.archival = [TaskEntry.from_dict(d) for d in data.get("archival", [])]
            except (json.JSONDecodeError, KeyError):
                self.core = []
                self.archival = []

    def stats(self) -> dict:
        types: dict[str, int] = {}
        for e in self.entries:
            types[e.task_type] = types.get(e.task_type, 0) + 1
        return {
            "total_entries": len(self.entries),
            "core": len(self.core),
            "archival": len(self.archival),
            "by_type": types,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Tag parsing — extract [TASK_MEMORY] blocks from AI responses
# ═══════════════════════════════════════════════════════════════════════════
_RE_TASK_MEMORY = re.compile(
    r"\[TASK_MEMORY\](.*?)(?:\[/TASK_MEMORY\]|</TASK_MEMORY>)",
    re.DOTALL | re.IGNORECASE,
)


def parse_task_memory_tags(text: str) -> tuple[str, list[dict]]:
    """Parse ``[TASK_MEMORY]`` blocks from an AI response.

    Returns ``(cleaned_text, list_of_parsed_entry_dicts)``.
    Each dict may contain: summary, reflection, keywords, importance, task_type, emotion.
    """
    entries: list[dict] = []

    for match in _RE_TASK_MEMORY.finditer(text):
        block = match.group(1).strip()
        entry: dict = {}
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("summary:"):
                entry["summary"] = line[len("summary:"):].strip()
            elif low.startswith("reflection:"):
                entry["reflection"] = line[len("reflection:"):].strip()
            elif low.startswith("keywords:"):
                kw_text = line[len("keywords:"):].strip()
                entry["keywords"] = [k.strip() for k in kw_text.split(",") if k.strip()]
            elif low.startswith("importance:"):
                try:
                    entry["importance"] = int(line[len("importance:"):].strip())
                except ValueError:
                    entry["importance"] = 5
            elif low.startswith("type:"):
                entry["task_type"] = line[len("type:"):].strip()
            elif low.startswith("emotion:"):
                entry["emotion"] = line[len("emotion:"):].strip()

        if entry.get("summary"):
            entries.append(entry)

    # Remove tags from display text
    cleaned = _RE_TASK_MEMORY.sub("", text).strip()
    return cleaned, entries


def extract_keywords(text: str) -> list[str]:
    """Pull meaningful keywords from a text string (for auto-captured entries)."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    _STOP = {
        "file", "created", "saved", "chart", "bytes", "error", "tool",
        "results", "that", "this", "with", "from", "updated", "none",
        "true", "false", "calculation", "system",
    }
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w not in _STOP and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= 10:
            break
    return out
