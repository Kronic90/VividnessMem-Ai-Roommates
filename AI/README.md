# VividnessMem — Organic Episodic Memory for AI Agents

A standalone, zero-dependency memory system for AI agents that gives them genuine long-term persistence — without RAG, without embeddings, without vector databases. Memories decay naturally over time, but emotionally significant or frequently recalled ones resist fading, inspired by how human episodic memory actually works.

**Single file. No external services. Drop it into any project.**

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
- **Neurochemistry simulation** — five neurotransmitters modulate encoding, retrieval, and emotional processing in real time

## NeuroChemistry System *(new in v1.0.7)*

A simulated neurochemical layer that modulates memory encoding, retrieval, and mood in real time — inspired by how dopamine, cortisol, serotonin, oxytocin, and norepinephrine actually regulate human cognition.

```python
mem = VividnessMem("./my_agent")
mem.chemistry.enabled = True

# Events trigger realistic chemical cascades
mem.chemistry.on_event("surprise_positive")       # Dopamine spike → encoding boost
mem.chemistry.on_event("social_bonding")           # Oxytocin surge → social memory boost
mem.chemistry.on_event("threat_detected")          # Cortisol + norepinephrine → hypervigilance

# Check current chemical state
state = mem.chemistry.get_state()
# → dopamine: 0.65 ▲ | cortisol: 0.28 ~ | serotonin: 0.58 ~ | oxytocin: 0.55 ▲ | norepinephrine: 0.48 ~
```

### 5 Neurotransmitters → 9 Cognitive Modifiers

| Chemical | Modifier | Effect |
|---|---|---|
| **Dopamine** | Encoding boost | High DA → memories stored more strongly (+30% max) |
| **Dopamine** | Novelty seeking | Elevated DA → preference for novel/surprising memories |
| **Cortisol** | Retrieval inhibition | High cortisol → harder to recall non-urgent memories |
| **Cortisol** | Emotional narrowing | Stress narrows recall to threat-relevant content |
| **Serotonin** | Mood stability | High 5-HT → less mood-congruent bias in recall |
| **Serotonin** | Patience factor | Affects temporal discounting in memory weighting |
| **Oxytocin** | Social memory boost | High OT → social/relational memories surface more easily |
| **Oxytocin** | Trust bias | Elevated OT → positive social memories weighted higher |
| **Norepinephrine** | Alertness factor | High NE → faster retrieval, broader associative spread |

### 10 Built-in Event Profiles

`surprise_positive`, `social_bonding`, `threat_detected`, `achievement`, `loss_grief`, `creative_flow`, `conflict`, `relaxation`, `learning_breakthrough`, `rejection`

Each event triggers a unique combination of chemical changes with realistic rise/decay dynamics. Chemicals naturally return to baseline over time via exponential decay.

### Emotion-to-Chemistry Mapping

When memories are stored with emotion tags (happy, anxious, grateful, etc.), the system automatically triggers the corresponding chemical cascade — so the act of *remembering* something emotional changes the agent's neurochemical state, just like in humans.

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
| **NeuroChemistry** | 5 neurotransmitters modulate encoding strength, retrieval bias, emotional narrowing, social memory boost, and alertness in real time |

## SillyTavern Extension

VividnessMem ships with a full SillyTavern extension for use with any LLM backend. The extension provides:

- **Automatic memory extraction** from conversations (user and AI messages)
- **Context injection** into prompts with live token count
- **`{{vividmem}}`** and **`{{vividmood}}`** macros for custom prompt template placement
- **OOC message filtering** — messages starting with `((`, `//`, `[OOC]` etc. are ignored
- **Browse, delete, export, and wipe** memories from the UI
- **Manual memory notes** — add memories directly
- **Consolidation and Dream triggers** — run memory maintenance from the UI
- **Per-chat or cross-chat** memory scoping
- **Token budget control** — set max context tokens for memory injection

See the [ST Extension README](VividMem-Embed/README.md) for setup instructions and API reference.

## Improved Performance with VividEmbed

For scenarios where you want the best of both worlds — organic memory dynamics *plus* semantic similarity — [VividEmbed](https://github.com/Kronic90/VividEmbed) is a fine-tuned embedding model built specifically for VividnessMem:

```bash
pip install vividembed
```

VividEmbed adds a hybrid vector layer on top of VividnessMem's associative retrieval, trained on 58 special tokens and 10 memory-specific objectives. It's fully optional — VividnessMem works great without it.

## Verified: 664 Tests, 0 Failures

The memory system has been rigorously tested across multiple suites:

| Suite | Tests | What it covers |
|---|:---:|---|
| **Robustness** | 63 | Adversarial inputs, corruption recovery, save/load integrity, scale stress (1,001 memories: 1ms active retrieval, 7ms resonance), cross-entity isolation, soft dedup, temporal coherence |
| **Behavioral Integration** | 29 | End-to-end persistence across simulated process restarts — session persistence, unprompted recall, cross-session resonance, identity continuity |
| **Brief & Rescore** | 69 | Compressed-brief generation, context injection, retrospective importance re-scoring lifecycle |
| **Advanced Memory** | 72 | Mood-congruent recall, spaced-repetition decay, emotional reappraisal, contradiction detection, memory consolidation, associative chains |
| **Professional Mode** | 190 | Professional context management, task-based memory, cross-session task recall |
| **NeuroChemistry** | 241 | Chemical cascades, event profiles, cognitive modifiers, emotion mapping, decay dynamics, integration with memory encoding/retrieval |

All test code and results are in the [`tests/`](tests/) directory.

## Quick Start

```python
from VividnessMem import VividnessMem

mem = VividnessMem("./my_agent_memory")

# Store memories with emotion and importance
mem.add_self_reflection("I really enjoyed that conversation about astronomy", "happy", 8)
mem.add_self_reflection("The debugging session was frustrating but educational", "determined", 6)

# Store social impressions (isolated per entity)
mem.add_social_impression("Alex", "Alex explains things clearly and patiently", "grateful", 7)

# Store structured facts (updates in place)
mem.add_fact("Alex", "favorite_color", "blue")
mem.add_fact("Alex", "favorite_color", "green")  # Updates, doesn't duplicate

# Retrieve what matters right now
context = mem.get_active_context(current_mood="curious")

# Search by resonance (associative keyword matching)
results = mem.get_resonant_memories("that time we talked about stars")

# Enable neurochemistry
mem.chemistry.enabled = True
mem.chemistry.on_event("learning_breakthrough")   # Dopamine + serotonin surge
state = mem.chemistry.get_state()                  # Check chemical levels
```

## Install

**PyPI (library only):**
```bash
pip install vividnessmem
```

**From source (full repo with tests, benchmarks, and ST extension):**
```bash
git clone https://github.com/Kronic90/VividnessMem-Ai-Roommates.git
```

**SillyTavern extension:** See [VividMem-Embed/README.md](VividMem-Embed/README.md) for setup.

## Project Structure

```
├── standalone memory/VividnessMem.py   # The memory system (~4,000 lines)
├── VividMem-Embed/                     # SillyTavern integration
│   ├── server/                         # FastAPI REST server
│   ├── st-extension/                   # SillyTavern extension (JS/CSS)
│   └── README.md                       # ST extension docs
├── vividnessmem-pypi/                  # PyPI package source
├── tests/                              # 664 tests across 6 suites
└── benchmarks/                         # Mem2ActBench, MemoryBench scripts
```

## License

PolyForm Noncommercial 1.0.0 — free for personal and non-commercial use.

---

**v1.0.7** | [PyPI](https://pypi.org/project/vividnessmem/) | [SillyTavern Extension](VividMem-Embed/README.md) | [VividEmbed](https://github.com/Kronic90/VividEmbed)
