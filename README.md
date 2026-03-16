# VividnessMem — Organic Episodic Memory for AI Agents

A standalone, zero-dependency memory system for AI agents that gives them genuine long-term persistence — without RAG, without embeddings, without vector databases. Memories decay naturally over time, but emotionally significant or frequently recalled ones resist fading, inspired by how human episodic memory actually works.

**Single file. No external services. Drop it into any project.**

## Install

```bash
pip install vividnessmem
```

For optional AES encryption at rest:

```bash
pip install vividnessmem[encryption]
```

Or grab the single file directly from [`AI/standalone memory/VividnessMem.py`](AI/standalone%20memory/VividnessMem.py) — no dependencies required.

## Why VividnessMem?

Most AI memory systems bolt a vector database onto an LLM and call it a day. That solves retrieval but misses everything that makes memory feel *real*: emotional salience, natural forgetting, mood-dependent recall, contradiction awareness, and the way remembering something changes the memory itself.

VividnessMem takes a different approach:

- **No embeddings** — retrieval uses an inverted index with bigram/trigram matching and co-occurrence expansion, not cosine similarity
- **No vector DB** — it's a single Python file that persists to JSON. Zero infrastructure
- **Memories have weight** — importance, emotion, recency, and access patterns all determine what surfaces
- **Memories decay** — spaced-repetition forgetting curve means unused memories fade, well-timed recalls strengthen them
- **Mood colours recall** — current emotional state biases which memories surface, just like in humans
- **Contradictions are flagged** — the system detects when the agent holds opposing beliefs
- **Deduplication is soft** — near-identical memories merge rather than stacking, reinforcing recurring themes

## Benchmark Results

VividnessMem has been evaluated on two established benchmarks using a local Gemma 3 12B model (Q4_K_M quantisation, no API calls).

### Mem2ActBench (100-item evaluation)

Mem2ActBench tests whether a memory system can store conversational sessions and later retrieve the right information to select correct tools/actions. 14,094 sessions indexed, 100 question-answer pairs evaluated across difficulty levels L1–L4.

| Metric | No Memory | VividnessMem |
|---|:---:|:---:|
| **Tool Accuracy** | 0.100 | **0.450** |
| **F1** | 0.158 | **0.527** |
| **Precision** | 0.174 | **0.554** |
| **Recall** | 0.151 | **0.529** |
| **BLEU-1** | 0.302 | **0.627** |

**Per difficulty level:**

| Level | No Memory TA | VividnessMem TA | No Memory F1 | VividnessMem F1 |
|---|:---:|:---:|:---:|:---:|
| L1 (n=53) | 0.075 | **0.547** | 0.098 | **0.571** |
| L2 (n=27) | 0.074 | **0.185** | 0.199 | **0.327** |
| L3 (n=5) | 0.000 | **0.400** | 0.133 | **0.633** |
| L4 (n=15) | 0.267 | **0.600** | 0.305 | **0.694** |

VividnessMem shows the strongest gains on harder questions (L3/L4), where multi-hop retrieval and contextual understanding matter most.

### MemoryBench — WritingPrompts (50-item evaluation)

MemoryBench tests narrative memory: given a set of writing-prompt conversations as training context, can the system recall details accurately? Measured by METEOR score.

| Condition | METEOR |
|---|:---:|
| No Memory | 0.125 |
| VividnessMem | **0.159** |
| Embedding (all-MiniLM-L6-v2) | 0.159 |

VividnessMem matches embedding-based retrieval on narrative recall — without using any embeddings at all.

## How It Works

### Vividness Formula

Every memory has a **vividness score** that determines whether it surfaces in context:

```
vividness = (importance × 0.6) + (recency × 0.3) + (access_bonus × 0.1)
```

- **Importance (60%)** — how significant the event was rated (1–10), set at creation time
- **Recency (30%)** — exponential decay: `10 × e^(-age_days / stability)`. New memories start with stability = 3 days. Well-spaced access increases stability (Ebbinghaus-inspired spaced repetition)
- **Access bonus (10%)** — capped at 3.0, grows by 0.5 each time the memory is recalled

Mood-congruent recall adds a further adjustment: memories whose emotional tone aligns with the current mood get a slight vividness boost, while emotionally incongruent memories are slightly suppressed.

### Retrieval Pipeline

Retrieval is **not** vector search. It works in layers:

1. **Active set** — the top-K most vivid memories are always injected into context (bounded by `ACTIVE_SELF_LIMIT=8`)
2. **Resonance** — keyword overlap between the current conversation and stored memory text, using prefix matching to handle conjugation. Enhanced with bigram (20% lift) and trigram (30% lift) matching for phrase-level precision
3. **Co-occurrence expansion** — a graph of word co-occurrence patterns learned from stored memories. When a query mentions "coffee", co-occurrence might expand to also search for "morning" or "cafe" if those words frequently appeared together
4. **Adaptive floor** — the minimum relevance score adjusts dynamically based on corpus density, preventing both over-retrieval in dense topics and under-retrieval in sparse ones
5. **Associative chains** — when a memory resurfaces, connected memories (sharing ≥2 content words) are traversed via a multi-hop graph walk, pulling in conceptually related memories that didn't directly match

This is associative recall, not semantic search. It's imperfect by design — just like human memory.

### Memory Types

VividnessMem stores several kinds of memory:

- **Self-reflections** — the agent's observations about itself, its values, and its experiences
- **Social impressions** — observations about other entities (people, other AIs), isolated per-entity
- **Short-term facts** — structured entity-attribute-value triples (e.g. "Alex → favorite_color → blue") that update in place when values change
- **Entity preferences** — tracked likes/dislikes per entity per category, with sentiment updates
- **Compressed briefs** — periodic AI-written self-summaries injected above individual memories for stable identity

### Key Features

| Feature | What it does |
|---|---|
| **Soft deduplication** | Near-duplicate memories merge (80% Jaccard threshold) instead of stacking. Reinforces recurring themes without consuming slots |
| **Spaced-repetition decay** | Exponential forgetting curve. Stability grows with well-spaced access; rapid re-access doesn't game the system |
| **Mood-congruent recall** | 40+ emotions mapped to PAD (Pleasure-Arousal-Dominance) 3D space. Current mood biases which memories surface |
| **Contradiction detection** | Opposing memories flagged automatically. Same-topic memories with conflicting emotional valence or negation patterns are surfaced for resolution |
| **Memory consolidation** | Related memory clusters (3+ memories, 25-80% overlap) are identified for synthesis into higher-level "gist" memories |
| **Emotional reappraisal** | During periodic re-scoring, emotion tags can be updated — initial anger may soften into understanding over time |
| **Metacognitive rationale** | Each memory stores *why* it was saved, *why* it has this importance, and *why* it has this emotion tag |
| **Bounded context** | No hard cap on total memories, but only `ACTIVE_LIMIT` entries get injected into the prompt (~1,277 tokens worst case). The rest exist on disk, reachable via resonance |

## Verified: 233 Tests, 0 Failures

The memory system has been rigorously tested across four suites:

| Suite | Tests | What it covers |
|---|:---:|---|
| **Robustness** | 63 | Adversarial inputs, corruption recovery, save/load integrity, scale stress (1,001 memories: 1ms active retrieval, 7ms resonance), cross-entity isolation, soft dedup, temporal coherence |
| **Behavioral Integration** | 29 | End-to-end persistence across simulated process restarts — session persistence, unprompted recall, cross-session resonance, identity continuity |
| **Brief & Rescore** | 69 | Compressed-brief generation, context injection, retrospective importance re-scoring lifecycle |
| **Advanced Memory** | 72 | Mood-congruent recall, spaced-repetition decay, emotional reappraisal, contradiction detection, memory consolidation, associative chains |

All test code and results are in the [`AI/tests/`](AI/tests/) directory.

## Quick Start

### Via pip (recommended)

```python
from vividnessmem import VividnessMem

mem = VividnessMem("./my_agent_memory")

# Store memories with emotion and importance
mem.add_self_reflection("I really enjoyed that conversation about astronomy", "happy", 8)
mem.add_self_reflection("The debugging session was frustrating but educational", "determined", 6)

# Store social impressions (isolated per entity)
mem.add_social_impression("Alex", "Alex explains things clearly and patiently", "grateful", 7)

# Store structured facts (updates in place)
mem.add_fact("Alex", "favorite_color", "blue")
mem.add_fact("Alex", "favorite_color", "green")  # Updates, doesn't duplicate

# Track preferences
mem.update_entity_preference("Alex", "music", "jazz", "likes")

# Retrieve by resonance (keyword/trigram/co-occurrence matching)
matches = mem.resonate("tell me about that astronomy discussion")

# Get the context block to inject into your LLM prompt
context = mem.get_context_block(current_entity="Alex", conversation_context="astronomy")

# Persist to disk
mem.save()
```

### Via single file

Copy [`VividnessMem.py`](AI/standalone%20memory/VividnessMem.py) into your project and import directly:

```python
from VividnessMem import VividnessMem
```

No dependencies beyond Python's standard library.

## AI Roommates — The Test Environment

To stress-test VividnessMem in practice, two AI agents with different memory architectures share a living space:

- **Aria** — uses VividnessMem (organic, vividness-ranked, no RAG)
- **Rex** — uses a MemGPT-style structured memory (core/archival split)

Both agents talk autonomously, curate their own memories, and maintain persistent identities across sessions. The full application includes a PyQt5 GUI, sandboxed code execution, web browsing, vision capabilities, and a shared message board.

## Repository Structure

```
README.md               This file
LICENSE                  MIT license
AI/
  standalone memory/
    VividnessMem.py      The standalone memory system — drop this into your own project
    howtouse.txt         Quick usage reference
  ai_roommates.py        Main app: GUI, conversation engine, model loading
  memory_aria.py         Aria's memory (VividnessMem-based)
  memory_rex.py          Rex's memory (MemGPT-style)
  task_memory.py         Long-term task/technique memory
  web_tools.py           Web search, page reading, image fetching
  message_board.py       Threaded message board
  tests/                 Full test suites with results
  benchmarks/            Internal benchmark suite (RAG/MemGPT comparison)
  benchmark_results/     Mem2ActBench and MemoryBench result files
```

## Known Limitations

- **No semantic understanding** — retrieval is lexical (keywords, bigrams, trigrams, co-occurrence). "Quantum entanglement" finds "quantum" but won't find "particle physics" unless co-occurrence has learned the link
- **No concurrent access** — single-threaded file I/O. Multiple processes hitting the same memory files would race
- **Scale ceiling untested beyond 5K** — performance is excellent at 5K memories but months of heavy use could push beyond this
- **Mood feedback loop** — bounded by decay to prevent spiralling, but the decay rate is a reasonable default rather than empirically optimised
- **Contradiction detection is lexical** — catches negation patterns and emotional reversals on shared topics, but subtle semantic contradictions will be missed
- **Consolidation quality depends on the LLM** — gist generation is only as good as the model doing the synthesis

## Latest Bug Fixes
- **Negative Memory Fixation** - fixed a bug that caused agents to fixate on negative memories, resulting in constant angry/sad agents
- **Context Overcrowding** - fixed a bug that causes agent to lose focus on current tasks due to irrelevant memories taking priority
- **Spaced-Repetion** - Fixed a bug causing agents memories to never fade, this stopped the Ebbinghaus decay from working as intended with certain memories


## License

MIT
