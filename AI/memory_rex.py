"""
memory_rex.py — MemGPT-inspired structured memory system for Rex (Phi).

Two memory banks:
  1. Self Memory   — Identity traits, opinions, likes/dislikes, revelations
  2. Social Memory — Per-entity interaction memories (keyed by name)

Architecture:
  - Core Memory: small set of always-available entries injected into context
  - Archival Memory: larger searchable store on disk, queried on demand
  - All entries tagged with emotion + importance by the LLM itself

Storage: JSON files under ai_dialogue_data/rex/
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── Storage paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "ai_dialogue_data" / "rex"
SELF_FILE = DATA_DIR / "self_memory.json"
SOCIAL_DIR = DATA_DIR / "social"


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOCIAL_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Memory Entry
# ═══════════════════════════════════════════════════════════════════════════
class MemoryEntry:
    """A single structured memory block."""

    def __init__(
        self,
        content: str,
        category: str,           # e.g. "identity", "opinion", "revelation", "interaction"
        emotion: str = "",       # tagged by the LLM
        importance: int = 5,     # 1-10, tagged by the LLM
        source: str = "",        # who/what triggered this memory
        timestamp: str = "",
    ):
        self.content = content
        self.category = category
        self.emotion = emotion
        self.importance = importance
        self.source = source
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "category": self.category,
            "emotion": self.emotion,
            "importance": self.importance,
            "source": self.source,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            content=d["content"],
            category=d.get("category", ""),
            emotion=d.get("emotion", ""),
            importance=d.get("importance", 5),
            source=d.get("source", ""),
            timestamp=d.get("timestamp", ""),
        )

    def __repr__(self):
        return f"[{self.importance}/10 | {self.emotion}] {self.content[:60]}…"


# ═══════════════════════════════════════════════════════════════════════════
#  Rex Memory System (MemGPT-style)
# ═══════════════════════════════════════════════════════════════════════════
class RexMemory:
    """
    Structured memory with explicit core/archival split.

    Core Memory:  Always present in context. Small and curated.
                  Think of it as Rex's "working identity."

    Archival Memory: Everything else. Searchable by keyword.
                     Persisted to disk between sessions.
    """

    # Max entries kept in core (always injected into context)
    CORE_SELF_LIMIT = 10
    CORE_SOCIAL_LIMIT = 5  # per entity

    def __init__(self):
        _ensure_dirs()

        # ── Self Memory ──
        self.core_self: list[MemoryEntry] = []       # always in context
        self.archival_self: list[MemoryEntry] = []    # on disk, searchable

        # ── Social Memory (per entity) ──
        self.core_social: dict[str, list[MemoryEntry]] = {}
        self.archival_social: dict[str, list[MemoryEntry]] = {}

        self._load()

    # ─── Core Memory Operations ───────────────────────────────────────

    def add_self_memory(self, entry: MemoryEntry):
        """Add to self memory. High-importance goes to core, rest to archival."""
        if entry.importance >= 7 and len(self.core_self) < self.CORE_SELF_LIMIT:
            self.core_self.append(entry)
        else:
            self.archival_self.append(entry)
            # If core is full but this is more important, swap out the least important
            if entry.importance >= 7 and len(self.core_self) >= self.CORE_SELF_LIMIT:
                self.core_self.sort(key=lambda e: e.importance)
                if entry.importance > self.core_self[0].importance:
                    demoted = self.core_self.pop(0)
                    self.archival_self.append(demoted)
                    self.core_self.append(entry)
                    self.archival_self.remove(entry)  # move from archival to core

    def add_social_memory(self, entity: str, entry: MemoryEntry):
        """Add interaction memory about a specific entity."""
        entry.source = entity
        limit = self.CORE_SOCIAL_LIMIT

        if entity not in self.core_social:
            self.core_social[entity] = []
        if entity not in self.archival_social:
            self.archival_social[entity] = []

        if entry.importance >= 7 and len(self.core_social[entity]) < limit:
            self.core_social[entity].append(entry)
        else:
            self.archival_social[entity].append(entry)
            if entry.importance >= 7 and len(self.core_social[entity]) >= limit:
                self.core_social[entity].sort(key=lambda e: e.importance)
                if entry.importance > self.core_social[entity][0].importance:
                    demoted = self.core_social[entity].pop(0)
                    self.archival_social[entity].append(demoted)
                    self.core_social[entity].append(entry)
                    self.archival_social[entity].remove(entry)

    # ─── Search Archival ──────────────────────────────────────────────

    def search_self(self, keyword: str, limit: int = 5) -> list[MemoryEntry]:
        """Search archival self memories by keyword."""
        keyword_lower = keyword.lower()
        results = [
            e for e in self.archival_self
            if keyword_lower in e.content.lower()
            or keyword_lower in e.category.lower()
            or keyword_lower in e.emotion.lower()
        ]
        results.sort(key=lambda e: e.importance, reverse=True)
        return results[:limit]

    def search_social(self, entity: str, keyword: str = "", limit: int = 5) -> list[MemoryEntry]:
        """Search archival social memories for an entity."""
        entries = self.archival_social.get(entity, [])
        if keyword:
            keyword_lower = keyword.lower()
            entries = [
                e for e in entries
                if keyword_lower in e.content.lower()
                or keyword_lower in e.emotion.lower()
            ]
        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries[:limit]

    # ─── Context Block (injected into system prompt) ──────────────────

    def get_context_block(self, current_entity: str = "") -> str:
        """
        Returns a formatted text block to inject into Rex's system prompt.
        Contains core self memories + core social memories for current entity.
        """
        lines = []

        if self.core_self:
            lines.append("=== MY IDENTITY (Core Self Memory) ===")
            for e in self.core_self:
                lines.append(f"• [{e.emotion}, {e.importance}/10] {e.content}")
            lines.append("")

        if current_entity and current_entity in self.core_social:
            entries = self.core_social[current_entity]
            if entries:
                lines.append(f"=== WHAT I KNOW ABOUT {current_entity.upper()} ===")
                for e in entries:
                    lines.append(f"• [{e.emotion}, {e.importance}/10] {e.content}")
                lines.append("")

        return "\n".join(lines) if lines else ""

    # ─── Persistence ──────────────────────────────────────────────────

    def save(self):
        _ensure_dirs()

        # Self memory
        data = {
            "core": [e.to_dict() for e in self.core_self],
            "archival": [e.to_dict() for e in self.archival_self],
        }
        with open(SELF_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Social memory (one file per entity)
        all_entities = set(list(self.core_social.keys()) + list(self.archival_social.keys()))
        for entity in all_entities:
            data = {
                "core": [e.to_dict() for e in self.core_social.get(entity, [])],
                "archival": [e.to_dict() for e in self.archival_social.get(entity, [])],
            }
            safe_name = entity.lower().replace(" ", "_")
            with open(SOCIAL_DIR / f"{safe_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self):
        # Self memory
        if SELF_FILE.exists():
            with open(SELF_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.core_self = [MemoryEntry.from_dict(d) for d in data.get("core", [])]
            self.archival_self = [MemoryEntry.from_dict(d) for d in data.get("archival", [])]

        # Social memory
        if SOCIAL_DIR.exists():
            for fpath in SOCIAL_DIR.glob("*.json"):
                entity = fpath.stem.replace("_", " ").title()
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.core_social[entity] = [MemoryEntry.from_dict(d) for d in data.get("core", [])]
                self.archival_social[entity] = [MemoryEntry.from_dict(d) for d in data.get("archival", [])]

    # ─── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "core_self": len(self.core_self),
            "archival_self": len(self.archival_self),
            "social_entities": list(self.core_social.keys()),
            "total_social_core": sum(len(v) for v in self.core_social.values()),
            "total_social_archival": sum(len(v) for v in self.archival_social.values()),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Curation Prompt — asked to Rex at end of conversation
# ═══════════════════════════════════════════════════════════════════════════
CURATION_PROMPT = """The conversation has ended. Reflect on what just happened.

Decide what (if anything) is worth remembering. You have two memory banks:

1. SELF MEMORY — things about yourself: new opinions you formed, identity traits
   you noticed, things you liked or disliked, revelations, changes in how you think.

2. SOCIAL MEMORY — things about the entity you were talking to (Aria):
   impressions, patterns you noticed, things they said that stuck with you,
   agreements or disagreements that mattered.

For each memory you want to store, output it in this exact JSON format:

```json
[
  {
    "bank": "self" or "social",
    "content": "the memory in your own words",
    "category": "identity|opinion|revelation|preference|interaction|impression|pattern",
    "emotion": "what you feel about this — use your own words",
    "importance": 1-10
  }
]
```

Only store what genuinely matters to you. You can store 0 items if nothing was worth remembering.
Return ONLY the JSON array, no other text."""


def parse_curation_response(response: str) -> list[dict]:
    """Parse the LLM's curation response into memory entries."""
    # Try to extract JSON from the response
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
