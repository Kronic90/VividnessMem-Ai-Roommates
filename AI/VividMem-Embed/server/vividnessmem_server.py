"""
VividnessMem Server — REST API for SillyTavern Integration
===========================================================

Exposes VividnessMem as a FastAPI service so SillyTavern (or any HTTP client)
can store, retrieve, and manage character memories.

Usage:
    python vividnessmem_server.py [--port 5050] [--data-dir ./memory_data]

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# ── Add parent paths so we can import VividnessMem ────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # c:\Users\scott\AI
_STANDALONE = _PROJECT_ROOT / "standalone memory"
sys.path.insert(0, str(_STANDALONE))
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from VividnessMem import VividnessMem

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vividnessmem-server")

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VividnessMem Server",
    description="REST API bridge for VividnessMem — organic memory for SillyTavern",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ST runs on localhost, allow all for dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Per-character memory instances ────────────────────────────────────────
_instances: dict[str, VividnessMem] = {}
_lock = threading.Lock()
_DATA_DIR: Path = Path("./vividmem_data")


def _sanitize_name(name: str) -> str:
    """Normalize character name to a filesystem-safe key."""
    return "".join(c if c.isalnum() or c in "-_ " else "" for c in name).strip()


def _get_mem(character: str) -> VividnessMem:
    """Return (or lazily create) the VividnessMem instance for a character."""
    key = _sanitize_name(character).lower()
    if not key:
        raise HTTPException(400, "character name is required")
    with _lock:
        if key not in _instances:
            char_dir = _DATA_DIR / key
            char_dir.mkdir(parents=True, exist_ok=True)
            log.info("Loading memory for character: %s  →  %s", character, char_dir)
            _instances[key] = VividnessMem(data_dir=str(char_dir))
        return _instances[key]


# ═══════════════════════════════════════════════════════════════════════════
#  Pydantic request / response models
# ═══════════════════════════════════════════════════════════════════════════

class AddReflectionReq(BaseModel):
    character: str
    content: str
    emotion: str = "neutral"
    importance: int = Field(5, ge=1, le=10)
    source: str = "reflection"
    why_saved: str = ""

class AddImpressionReq(BaseModel):
    character: str
    entity: str
    content: str
    emotion: str = "neutral"
    importance: int = Field(5, ge=1, le=10)
    why_saved: str = ""

class AddFactReq(BaseModel):
    character: str
    entity: str
    attribute: str
    value: str

class QueryReq(BaseModel):
    character: str
    context: str = ""
    entity: str = ""

class MoodUpdateReq(BaseModel):
    character: str
    emotions: list[str]

class ProcessMessageReq(BaseModel):
    """All-in-one: process a user message, auto-extract + store + retrieve."""
    character: str
    user_name: str = "User"
    message: str
    is_user: bool = True
    emotion: str = "neutral"
    importance: int = Field(5, ge=1, le=10)
    conversation_context: str = ""

class PreferenceReq(BaseModel):
    character: str
    entity: str
    category: str
    item: str
    sentiment: str = "likes"

class BulkImportReq(BaseModel):
    character: str
    memories: list[dict]


# ═══════════════════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "VividnessMem",
        "version": "1.0.0",
        "characters_loaded": list(_instances.keys()),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Core memory operations
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/memory/reflection")
def add_reflection(req: AddReflectionReq):
    """Store a self-reflection / episodic memory."""
    mem = _get_mem(req.character)
    m = mem.add_self_reflection(
        content=req.content,
        emotion=req.emotion,
        importance=req.importance,
        source=req.source,
        why_saved=req.why_saved,
    )
    mem.save()
    return {"stored": True, "vividness": round(m.vividness, 3)}


@app.post("/api/memory/impression")
def add_impression(req: AddImpressionReq):
    """Store a social impression about an entity."""
    mem = _get_mem(req.character)
    m = mem.add_social_impression(
        entity=req.entity,
        content=req.content,
        emotion=req.emotion,
        importance=req.importance,
        why_saved=req.why_saved,
    )
    mem.save()
    return {"stored": True, "vividness": round(m.vividness, 3)}


@app.post("/api/memory/fact")
def add_fact(req: AddFactReq):
    """Store a short-term factual memory."""
    mem = _get_mem(req.character)
    f = mem.add_fact(
        entity=req.entity,
        attribute=req.attribute,
        value=req.value,
    )
    mem.save()
    return {"stored": True, "entity": f.entity, "attribute": f.attribute}


# ═══════════════════════════════════════════════════════════════════════════
#  Retrieval
# ═══════════════════════════════════════════════════════════════════════════

def _memory_to_dict(m) -> dict:
    """Serialize a Memory object for JSON responses."""
    return {
        "content": m.content,
        "emotion": m.emotion,
        "importance": m.importance,
        "vividness": round(m.vividness, 3),
        "source": getattr(m, "source", ""),
        "entity": getattr(m, "entity", ""),
        "timestamp": m.timestamp,
    }


@app.post("/api/memory/query")
def query_memories(req: QueryReq):
    """Retrieve active memories filtered by context and/or entity."""
    mem = _get_mem(req.character)
    results = []

    # Self-reflections
    active = mem.get_active_self(context=req.context)
    results.extend([_memory_to_dict(m) for m in active])

    # Social impressions
    if req.entity:
        social = mem.get_active_social(req.entity)
        results.extend([_memory_to_dict(m) for m in social])

    # Resonant old memories
    if req.context:
        resonant = mem.resonate(req.context)
        results.extend([_memory_to_dict(m) for m in resonant])

    # Short-term facts
    facts = mem.get_facts(entity=req.entity, context=req.context)
    for f in facts:
        results.append({
            "content": f"{f.entity}: {f.attribute} = {f.value}",
            "emotion": "neutral",
            "importance": 5,
            "vividness": round(f.vividness, 3),
            "source": "fact",
            "entity": f.entity,
            "timestamp": f.timestamp,
        })

    return {"character": req.character, "count": len(results), "memories": results}


@app.get("/api/memory/context/{character}")
def get_context_block(character: str, entity: str = "", conversation_context: str = ""):
    """Get the full formatted context block ready for system prompt injection."""
    mem = _get_mem(character)
    block = mem.get_context_block(
        current_entity=entity,
        conversation_context=conversation_context,
    )
    return {"character": character, "context_block": block}


# ═══════════════════════════════════════════════════════════════════════════
#  Process message (all-in-one for SillyTavern)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/memory/process")
def process_message(req: ProcessMessageReq):
    """
    All-in-one endpoint for SillyTavern:
    1. Stores the message as a memory (reflection or impression)
    2. Updates mood from emotion
    3. Retrieves relevant context block for prompt injection
    """
    mem = _get_mem(req.character)

    # Store the message as a memory
    if req.is_user:
        # User's message → social impression about the user
        mem.add_social_impression(
            entity=req.user_name,
            content=req.message,
            emotion=req.emotion,
            importance=req.importance,
            why_saved=f"User said during conversation",
        )
    else:
        # Character's own message → self-reflection
        mem.add_self_reflection(
            content=req.message,
            emotion=req.emotion,
            importance=req.importance,
            source="dialogue",
            why_saved="Character's own dialogue",
        )

    # Update mood
    if req.emotion and req.emotion != "neutral":
        mem.update_mood_from_conversation([req.emotion])

    # Build context block
    context_block = mem.get_context_block(
        current_entity=req.user_name if req.is_user else "",
        conversation_context=req.conversation_context,
    )

    mem.save()

    return {
        "character": req.character,
        "context_block": context_block,
        "mood": mem.mood_label,
        "mood_pad": list(mem.mood),
        "stats": mem.stats(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Mood
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/memory/mood/{character}")
def get_mood(character: str):
    mem = _get_mem(character)
    return {
        "character": character,
        "mood_label": mem.mood_label,
        "mood_pad": list(mem.mood),
    }


@app.post("/api/memory/mood")
def update_mood(req: MoodUpdateReq):
    mem = _get_mem(req.character)
    mem.update_mood_from_conversation(req.emotions)
    mem.save()
    return {
        "character": req.character,
        "mood_label": mem.mood_label,
        "mood_pad": list(mem.mood),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Preferences
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/memory/preference")
def add_preference(req: PreferenceReq):
    mem = _get_mem(req.character)
    mem.update_entity_preference(
        entity=req.entity,
        category=req.category,
        item=req.item,
        sentiment=req.sentiment,
    )
    mem.save()
    return {"stored": True}


@app.get("/api/memory/preferences/{character}/{entity}")
def get_preferences(character: str, entity: str):
    mem = _get_mem(character)
    return {
        "character": character,
        "entity": entity,
        "preferences": mem.get_entity_preferences(entity),
        "context": mem.get_preference_context(entity),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Relationship arcs
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/memory/arc/{character}/{entity}")
def get_relationship_arc(character: str, entity: str):
    mem = _get_mem(character)
    arc = mem.get_relationship_arc(entity)
    return {
        "character": character,
        "entity": entity,
        "arc": arc,
        "context": mem.get_arc_context(entity) if arc else "",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Session management
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/memory/session/{character}/bump")
def bump_session(character: str):
    """Start a new session — bumps session counter, checks for briefs/dreams."""
    mem = _get_mem(character)
    count = mem.bump_session()
    mem.save()
    return {
        "character": character,
        "session_count": count,
        "needs_brief": mem.needs_brief(),
        "needs_dream": mem.needs_dream(),
        "needs_rescore": mem.needs_rescore(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Stats & management
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/memory/stats/{character}")
def get_stats(character: str):
    mem = _get_mem(character)
    return {"character": character, **mem.stats()}


@app.get("/api/memory/characters")
def list_characters():
    """List all characters that have memory data on disk."""
    chars = []
    if _DATA_DIR.exists():
        for d in sorted(_DATA_DIR.iterdir()):
            if d.is_dir():
                chars.append(d.name)
    return {"characters": chars}


@app.post("/api/memory/import")
def bulk_import(req: BulkImportReq):
    """Import a list of memories at once."""
    mem = _get_mem(req.character)
    count = 0
    for item in req.memories:
        kind = item.get("type", "reflection")
        if kind == "reflection":
            mem.add_self_reflection(
                content=item.get("content", ""),
                emotion=item.get("emotion", "neutral"),
                importance=item.get("importance", 5),
                source=item.get("source", "import"),
            )
            count += 1
        elif kind == "impression":
            mem.add_social_impression(
                entity=item.get("entity", "Unknown"),
                content=item.get("content", ""),
                emotion=item.get("emotion", "neutral"),
                importance=item.get("importance", 5),
            )
            count += 1
        elif kind == "fact":
            mem.add_fact(
                entity=item.get("entity", "Unknown"),
                attribute=item.get("attribute", ""),
                value=item.get("value", ""),
            )
            count += 1
    mem.save()
    return {"character": req.character, "imported": count}


@app.post("/api/memory/consolidate/{character}")
def trigger_consolidation(character: str):
    """Find consolidation clusters and return the prompt for LLM processing."""
    mem = _get_mem(character)
    clusters = mem.find_consolidation_clusters()
    if not clusters:
        return {"character": character, "clusters": 0, "prompt": None}
    prompt = mem.prepare_consolidation_prompt()
    return {
        "character": character,
        "clusters": len(clusters),
        "prompt": prompt,
    }


@app.post("/api/memory/dream/{character}")
def trigger_dream(character: str):
    """Find dream candidates and return the prompt for LLM processing."""
    mem = _get_mem(character)
    if not mem.needs_dream():
        return {"character": character, "candidates": 0, "prompt": None}
    candidates = mem.find_dream_candidates()
    if not candidates:
        return {"character": character, "candidates": 0, "prompt": None}
    prompt = mem.prepare_dream_prompt()
    return {
        "character": character,
        "candidates": len(candidates),
        "prompt": prompt,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VividnessMem REST Server")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", default="./vividmem_data",
                        help="Root directory for per-character memory storage")
    args = parser.parse_args()

    global _DATA_DIR
    _DATA_DIR = Path(args.data_dir).resolve()
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Memory data directory: %s", _DATA_DIR)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
