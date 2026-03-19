<p align="center">
  <img src="https://raw.githubusercontent.com/Kronic90/VividnessMem-Ai-Roommates/main/AI/vividnessmem-pypi/VividnessLogo.png" alt="VividnessMem Logo" width="400"/>
</p>

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

### Mem2ActBench (5-seed averaged evaluation)

Mem2ActBench tests whether a memory system can store conversational sessions and later retrieve the right information to select correct tools/actions. 14,094 sessions indexed, 100 question-answer pairs evaluated per seed across difficulty levels L1–L4. Results averaged over 5 random seeds (42, 123, 7, 99, 256).

| Metric | No Memory | Embedding (MiniLM-L6-v2) | VividnessMem |
|---|:---:|:---:|:---:|
| **Tool Accuracy** | 0.053 | 0.440 | **0.470 ± 0.024** |
| **F1** | 0.118 | 0.512 | **0.523 ± 0.011** |
| **Precision** | 0.130 | 0.559 | **0.527 ± 0.010** |
| **Recall** | 0.113 | 0.519 | **0.561 ± 0.020** |
| **BLEU-1** | 0.273 | 0.641 | **0.614 ± 0.008** |

**Per difficulty level (5-seed average):**

| Level | No Memory TA | Embedding TA | VividnessMem TA |
|---|:---:|:---:|:---:|
| L1 (simple recall) | 0.025 | 0.528 | **0.573** |
| L2 (multi-hop) | 0.069 | 0.111 | **0.213** |
| L3 (temporal) | 0.000 | **0.400** | 0.384 |
| L4 (complex reasoning) | 0.178 | **0.733** | 0.504 |

VividnessMem outperforms embedding-based retrieval (all-MiniLM-L6-v2 with cosine similarity) on overall Tool Accuracy by **+3 percentage points** and on F1 by **+1.1 pp**, with much lower variance across seeds (±0.024 vs single-run). Uses parameter-aware retrieval to match tool arguments against stored memories — zero embeddings, zero vector infrastructure. The largest gap is on L2 (multi-hop) questions where VividnessMem scores 0.213 vs Embedding’s 0.111 — a **+10.2 pp** advantage.

### MemoryBench — WritingPrompts (50-item evaluation)

MemoryBench tests narrative memory: given a set of writing-prompt conversations as training context, can the system recall details accurately? Measured by METEOR score.

| Condition | METEOR |
|---|:---:|
| No Memory | 0.092 |
| VividnessMem | **0.141** |

VividnessMem provides a **+52.7% relative gain** over no-memory on narrative recall.

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

See the [ST Extension README](AI/VividMem-Embed/README.md) for setup instructions and API reference.

## Improved Performance with VividEmbed

For scenarios where you want the best of both worlds — organic memory dynamics *plus* semantic similarity — [VividEmbed](https://github.com/Kronic90/VividEmbed) is a fine-tuned embedding model built specifically for VividnessMem:

```bash
pip install vividembed
```

VividEmbed adds a hybrid vector layer on top of VividnessMem's associative retrieval, trained on 58 special tokens and 10 memory-specific objectives. It's fully optional — VividnessMem works great without it.

## Verified: 664 Tests, 0 Failures

The memory system has been rigorously tested across six suites:

| Suite | Tests | What it covers |
|---|:---:|---|
| **Robustness** | 63 | Adversarial inputs, corruption recovery, save/load integrity, scale stress (1,001 memories: 1ms active retrieval, 7ms resonance), cross-entity isolation, soft dedup, temporal coherence |
| **Behavioral Integration** | 29 | End-to-end persistence across simulated process restarts — session persistence, unprompted recall, cross-session resonance, identity continuity |
| **Brief & Rescore** | 69 | Compressed-brief generation, context injection, retrospective importance re-scoring lifecycle |
| **Advanced Memory** | 72 | Mood-congruent recall, spaced-repetition decay, emotional reappraisal, contradiction detection, memory consolidation, associative chains |
| **Professional Mode** | 190 | Professional context management, task-based memory, cross-session task recall |
| **NeuroChemistry** | 241 | Chemical cascades, event profiles, cognitive modifiers, emotion mapping, decay dynamics, integration with memory encoding/retrieval |

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

# Enable neurochemistry (v1.0.7)
mem.chemistry.enabled = True
mem.chemistry.on_event("learning_breakthrough")   # Dopamine + serotonin surge
state = mem.chemistry.get_state()                  # Check chemical levels
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
  VividMem-Embed/
    server/
      vividnessmem_server.py   FastAPI REST server for SillyTavern integration
      requirements.txt         Python dependencies (fastapi, uvicorn, pydantic)
    st-extension/
      manifest.json            SillyTavern extension manifest
      index.js                 Client-side extension (auto-store, emotion detection, mood badge)
      style.css                Extension styles (emotion-coloured memory cards, mood badge)
    README.md                  Full SillyTavern extension documentation
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

## How to Use in SillyTavern

VividnessMem integrates with [SillyTavern](https://github.com/SillyTavern/SillyTavern) as a third-party extension, giving your characters organic long-term memory with emotion-aware recall, natural forgetting, and mood-driven personality.

### 1. Start the Memory Server

```bash
cd AI/VividMem-Embed/server
pip install -r requirements.txt
python vividnessmem_server.py --port 5050
```

The server creates a `vividmem_data/` directory for per-character memory storage. Each character gets isolated memory — switching characters in SillyTavern automatically switches memory contexts.

### 2. Install the SillyTavern Extension

Copy the extension into SillyTavern's third-party extensions folder:

```powershell
# Windows — adjust the SillyTavern path to your install location
Copy-Item -Recurse AI\VividMem-Embed\st-extension\* "C:\path\to\SillyTavern\public\scripts\extensions\third-party\VividnessMem\"
```

```bash
# Linux / macOS
cp -r AI/VividMem-Embed/st-extension/ /path/to/SillyTavern/public/scripts/extensions/third-party/VividnessMem/
```

### 3. Enable and Configure

1. Open SillyTavern in your browser
2. Open the **Extensions** panel (puzzle piece icon)
3. Find **VividnessMem** and toggle it on
4. Set the **Server URL** to `http://127.0.0.1:5050` (default)
5. Click **Test Connection** — a green dot confirms it's working

### What Happens Next

Once enabled, the extension works automatically:

- **Every message** you send is stored as a social impression with auto-detected emotion and importance
- **Every character reply** is stored as a self-reflection in that character's memory
- **Before each generation**, the most relevant memories are injected into the system prompt — decayed by time, biased by the character's current mood
- **A mood badge** appears next to the character name showing their current emotional state
- **Relationship arcs** build over time — warmth, trajectory, interaction history

The extension settings panel lets you toggle auto-storage, emotion detection, context injection, and browse stored memories. See [`AI/VividMem-Embed/README.md`](AI/VividMem-Embed/README.md) for the full API reference, troubleshooting, and advanced configuration.

## Known Limitations

- **Scale ceiling untested beyond 5K** — performance is excellent at 5K memories but months of heavy use could push beyond this
- **Consolidation quality depends on the LLM** — gist generation is only as good as the model doing the synthesis

## Latest Bug Fixes

- **Negative Memory Fixation** — fixed a bug that caused agents to fixate on negative memories, resulting in constant angry/sad agents
- **Context Overcrowding** — fixed a bug that caused agents to lose focus on current tasks due to irrelevant memories taking priority
- **Spaced-Repetition Decay** — fixed a bug causing agent memories to never fade, which stopped the Ebbinghaus decay from working as intended with certain memories
- **Consolidation Duplicate Append** — fixed a bug where `apply_consolidation()` appended each gist memory twice, doubling consolidated memories in the store
- **Mood Feedback Loop** — bounded mood decay to prevent emotional spiralling; the decay rate is now tuned to prevent runaway positive/negative loops
- **No Semantic Understanding** — `resonate()` now accepts an optional `llm_fn` parameter for LLM-powered semantic bridging. When lexical matching returns few results, the LLM generates conceptually related terms to bridge the vocabulary gap (e.g. "quantum entanglement" now finds memories about "particle physics")
- **Lexical-Only Contradiction Detection** — `detect_contradictions()` now accepts an optional `llm_fn` parameter. Borderline lexical candidates are verified by the LLM for semantic contradiction, catching subtle conflicts that negation patterns alone would miss
- **No Concurrent Access** — file I/O now uses cross-platform file locking (`msvcrt` on Windows, `fcntl` on Unix) with atomic writes (temp file + rename) to prevent corruption from concurrent access

## License

MIT
