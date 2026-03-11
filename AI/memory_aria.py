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
}

# ─── Storage paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "ai_dialogue_data" / "aria"
SELF_FILE = DATA_DIR / "self_memory.json"
SOCIAL_DIR = DATA_DIR / "social"


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOCIAL_DIR.mkdir(parents=True, exist_ok=True)


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
    ):
        self.content = content
        self.emotion = emotion
        self.importance = importance
        self.source = source
        self.timestamp = timestamp or datetime.now().isoformat()
        # Organic decay: memories lose vividness over time
        # but high-importance ones resist fading
        self._access_count = 0

    def touch(self):
        """Mark this memory as accessed (keeps it vivid)."""
        self._access_count += 1

    @property
    def vividness(self) -> float:
        """
        How 'present' this memory feels. Combines importance, recency, and access.
        Higher = more likely to surface in context.
        """
        age_hours = (datetime.now() - datetime.fromisoformat(self.timestamp)).total_seconds() / 3600
        recency_score = max(0, 10 - (age_hours / 24))  # loses 1 point per day
        access_bonus = min(3, self._access_count * 0.5)
        return (self.importance * 0.6) + (recency_score * 0.3) + (access_bonus * 0.1)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "emotion": self.emotion,
            "importance": self.importance,
            "source": self.source,
            "timestamp": self.timestamp,
            "access_count": self._access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reflection":
        r = cls(
            content=d["content"],
            emotion=d.get("emotion", ""),
            importance=d.get("importance", 5),
            source=d.get("source", ""),
            timestamp=d.get("timestamp", ""),
        )
        r._access_count = d.get("access_count", 0)
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
    """

    # How many reflections to inject into context
    ACTIVE_SELF_LIMIT = 8
    ACTIVE_SOCIAL_LIMIT = 5  # per entity
    RESONANCE_LIMIT = 3  # max old memories that can resurface per turn

    def __init__(self):
        _ensure_dirs()

        # ── Self Memory (identity journal) ──
        self.self_reflections: list[Reflection] = []

        # ── Social Memory (per entity) ──
        self.social_impressions: dict[str, list[Reflection]] = {}

        self._load()

    # ─── Add Memories ─────────────────────────────────────────────────

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
                return
        self.self_reflections.append(reflection)

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

    # ─── Surface Memories (by vividness) ──────────────────────────────

    def get_active_self(self) -> list[Reflection]:
        """Return the most vivid self-reflections for context injection."""
        sorted_refs = sorted(self.self_reflections, key=lambda r: r.vividness, reverse=True)
        active = sorted_refs[:self.ACTIVE_SELF_LIMIT]
        for r in active:
            r.touch()  # accessing makes it more vivid
        return active

    def get_active_social(self, entity: str) -> list[Reflection]:
        """Return the most vivid impressions of a specific entity."""
        entries = self.social_impressions.get(entity, [])
        sorted_entries = sorted(entries, key=lambda r: r.vividness, reverse=True)
        active = sorted_entries[:self.ACTIVE_SOCIAL_LIMIT]
        for r in active:
            r.touch()
        return active

    # ─── Resonance (old memories resurfacing) ─────────────────────────

    def resonate(self, context: str, limit: int | None = None) -> list[Reflection]:
        """Find old faded memories that resonate with current conversation.

        Searches ALL memories (including ones below the active threshold)
        for keyword overlap with the conversation context. Matching memories
        get a .touch() boost, potentially pulling them back into the active
        set over time — like suddenly remembering something from months ago
        because the conversation triggered it.

        Returns only memories that are NOT already in the active set.
        """
        n = limit or self.RESONANCE_LIMIT
        if not context or not self.self_reflections:
            return []

        # Extract meaningful words from conversation context
        context_words = {
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", context.lower())
        } - _RESONANCE_STOP
        if not context_words:
            return []

        # Get current active set IDs so we don't duplicate
        active_set = set(
            id(r) for r in sorted(
                self.self_reflections, key=lambda r: r.vividness, reverse=True
            )[:self.ACTIVE_SELF_LIMIT]
        )

        # Score all non-active memories by keyword overlap
        # Use prefix matching (first 5 chars) to handle plurals/conjugations
        context_prefixes = {w[:5] for w in context_words if len(w) >= 5}
        scored: list[tuple[float, Reflection]] = []
        for ref in self.self_reflections:
            if id(ref) in active_set:
                continue  # already surfaced normally
            mem_text = f"{ref.content} {ref.emotion}".lower()
            mem_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", mem_text))
            mem_prefixes = {w[:5] for w in mem_words if len(w) >= 5}
            # Count matches: exact words + prefix matches for longer words
            exact_overlap = len(context_words & mem_words)
            prefix_overlap = len(context_prefixes & mem_prefixes)
            total_overlap = max(exact_overlap, prefix_overlap)
            if total_overlap >= 2:  # need at least 2 word matches
                # Score: overlap count + importance bonus
                score = total_overlap + (ref.importance * 0.2)
                scored.append((score, ref))

        scored.sort(key=lambda x: x[0], reverse=True)
        resonant = [ref for _, ref in scored[:n]]

        # Touch resonant memories — they're being remembered
        for r in resonant:
            r.touch()

        return resonant

    # ─── Context Block (injected into system prompt) ──────────────────

    def get_context_block(self, current_entity: str = "", resonant: list[Reflection] | None = None) -> str:
        """
        Returns a narrative block to inject into Aria's system prompt.
        Written in first person, like reading from a journal.
        Optionally includes resonant memories (old ones triggered by context).
        """
        lines = []

        active_self = self.get_active_self()
        if active_self:
            lines.append("=== THINGS I KNOW ABOUT MYSELF ===")
            for r in active_self:
                emotion_tag = f" ({r.emotion})" if r.emotion else ""
                lines.append(f"— {r.content}{emotion_tag}")
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
        }


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

For each memory, output it in this exact JSON format:

```json
[
  {
    "bank": "self" or "social",
    "content": "write this naturally, like a journal entry — in your own voice",
    "emotion": "how this makes you feel — in your own words",
    "importance": 1-10
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
