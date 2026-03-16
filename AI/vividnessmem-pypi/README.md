# VividnessMem

**Organic episodic memory for AI agents — no RAG, no embeddings, no vector DB.**

A standalone memory system that gives LLM-based agents genuine long-term persistence. Memories decay naturally over time, but emotionally significant or frequently recalled ones resist fading — inspired by how human episodic memory actually works.

Single file. No external services. Zero required dependencies.

## Install

```bash
pip install vividnessmem
```

For optional AES encryption at rest:

```bash
pip install vividnessmem[encryption]
```

## Quick Start

```python
from vividnessmem import VividnessMem

mem = VividnessMem("./my_agent_memory")

# Store memories with emotion and importance (1-10)
mem.add_self_reflection("I really enjoyed that conversation about astronomy", "happy", 8)
mem.add_self_reflection("The debugging session was frustrating but educational", "determined", 6)

# Store social impressions (isolated per entity)
mem.add_social_impression("Alex", "Alex explains things clearly and patiently", "grateful", 7)

# Store structured facts (updates in place when values change)
mem.add_fact("Alex", "favorite_color", "blue")
mem.add_fact("Alex", "favorite_color", "green")  # Updates, doesn't duplicate

# Track preferences
mem.update_entity_preference("Alex", "music", "jazz", "likes")

# Retrieve by resonance (keyword + trigram + co-occurrence matching)
matches = mem.resonate("tell me about that astronomy discussion")

# Get a context block to inject into your LLM prompt
context = mem.get_context_block(current_entity="Alex", conversation_context="astronomy")

# Persist to disk
mem.save()
```

## What Makes It Different

| Feature | VividnessMem | Typical RAG |
|---|---|---|
| **Infrastructure** | Single Python file, JSON persistence | Vector DB + embedding model |
| **Retrieval** | Inverted index + trigrams + co-occurrence | Cosine similarity |
| **Memory decay** | Spaced-repetition forgetting curve | None (everything persists equally) |
| **Emotional salience** | Importance × recency × mood = vividness | N/A |
| **Mood-congruent recall** | Current mood biases which memories surface | N/A |
| **Deduplication** | Soft merge at 80% Jaccard overlap | Manual or none |
| **Contradiction detection** | Lexical + optional LLM semantic verification | N/A |
| **Concurrent access** | File locking + atomic writes | N/A |
| **Scaling** | 2.7x latency increase 100→5000 memories | 47x for TF-IDF RAG |
| **Parameter-aware retrieval** | Per-parameter memory scanning for tool calls | N/A |
| **Dependencies** | Zero (stdlib only) | Embedding model + vector DB |

## Benchmark Results

Evaluated on **Mem2ActBench** (5-seed average, 100 items per seed, Gemma 3 12B Q4_K_M local, 14,094 sessions indexed):

| Metric | No Memory | Embedding (MiniLM-L6-v2) | VividnessMem |
|---|:---:|:---:|:---:|
| **Tool Accuracy** | 0.053 | 0.440 | **0.470 ± 0.024** |
| **F1** | 0.118 | 0.512 | **0.523 ± 0.011** |
| **Precision** | 0.130 | 0.559 | **0.527 ± 0.010** |
| **Recall** | 0.113 | 0.519 | **0.561 ± 0.020** |
| **BLEU-1** | 0.273 | 0.641 | **0.614 ± 0.008** |

VividnessMem outperforms embedding-based retrieval by **+3 pp** on Tool Accuracy and **+1.1 pp** on F1, with lower variance across seeds. Uses parameter-aware retrieval and zero vector infrastructure.

**Per difficulty level (5-seed average):**

| Level | Embedding TA | VividnessMem TA |
|---|:---:|:---:|
| L1 (simple recall) | 0.528 | **0.573** |
| L2 (multi-hop) | 0.111 | **0.213** |
| L3 (temporal) | 0.400 | 0.384 |
| L4 (complex reasoning) | **0.733** | 0.504 |

Evaluated on **MemoryBench WritingPrompts** (50 items):

| Condition | METEOR |
|---|:---:|
| No Memory | 0.092 |
| VividnessMem | **0.141** |

VividnessMem provides a **+52.7% relative gain** over no-memory on narrative recall.

## The Vividness Formula

```
vividness = (importance × 0.6) + (recency × 0.3) + (access_bonus × 0.1)
```

- **Importance (60%)** — how significant the event was (1–10)
- **Recency (30%)** — exponential decay with spaced-repetition stability growth
- **Access bonus (10%)** — capped at 3.0, grows with each recall

Mood alignment provides an additional boost/penalty based on emotional congruence.

## Full Feature List

- **Vividness-ranked recall** with spaced-repetition decay
- **Mood-congruent memory** (40+ emotions mapped to PAD 3D space)
- **Associative chains** (multi-hop graph walk on shared concepts)
- **Resonance** with bigram/trigram matching and co-occurrence expansion
- **Adaptive relevance floor** (dynamic threshold based on corpus density)
- **Soft deduplication** (80% Jaccard merge)
- **Contradiction detection** (negation + emotional reversal patterns)
- **Memory consolidation** (cluster related memories into gist insights)
- **Short-term facts** (entity-attribute-value triples, update in place)
- **Entity preferences** (tracked likes/dislikes per entity)
- **Compressed briefs** (periodic self-summaries for stable identity)
- **Importance re-scoring** with emotional reappraisal
- **Metacognitive rationale** (why_saved, why_importance, why_emotion)
- **Relationship arc tracking** (per-entity warmth trajectory)
- **Inverted word/prefix index** for O(k) resonance lookup
- **LLM semantic bridging** for `resonate()` (optional — pass `llm_fn` to bridge vocabulary gaps)
- **LLM contradiction verification** for `detect_contradictions()` (optional — catches semantic conflicts)
- **Cross-platform file locking** with atomic writes for safe concurrent access
- **Synonym ring** for semantic bridging
- **Optional AES encryption** at rest (Fernet + PBKDF2)
- **Full JSON persistence** to disk

## API Reference

### Core

```python
mem = VividnessMem(directory, encryption_key=None)  # Load or create
mem.save()                                           # Persist to disk
mem.stats()                                          # Usage statistics
```

### Storing Memories

```python
mem.add_self_reflection(content, emotion, importance)
mem.add_social_impression(entity, content, emotion, importance)
mem.add_fact(entity, attribute, value)               # Updates in place
mem.update_entity_preference(entity, category, item, sentiment)
```

### Retrieving

```python
mem.get_active_self()                                # Top vivid self-memories
mem.get_active_social(entity)                        # Top vivid for entity
mem.resonate(query, llm_fn=None)                     # Keyword/trigram + optional LLM bridging
mem.get_context_block(current_entity=None, conversation_context=None)
mem.get_facts(entity=None)                           # Structured facts
mem.get_entity_preferences(entity)                   # Preference map
```

### Maintenance

```python
mem.update_mood_from_conversation(emotions)          # Shift mood state
mem.detect_contradictions(llm_fn=None)               # Lexical + optional LLM verification
mem.find_consolidation_clusters(min_cluster=3)       # Related memory groups
mem.find_dream_candidates()                          # Between-session patterns
mem.get_relationship_arc(entity)                     # Warmth trajectory
```

## Requirements

- Python 3.9+
- No required dependencies (stdlib only)
- Optional: `cryptography` for AES encryption (`pip install vividnessmem[encryption]`)

## License

MIT

## Links

- [GitHub Repository](https://github.com/Kronic90/VividnessMem-Ai-Roommates)
- [Full Documentation & Test Results](https://github.com/Kronic90/VividnessMem-Ai-Roommates)
