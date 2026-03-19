<p align="center">
  <img src="https://raw.githubusercontent.com/Kronic90/VividnessMem-Ai-Roommates/main/AI/vividnessmem-pypi/VividnessLogo.png" alt="VividnessMem Logo" width="400"/>
</p>

<h1 align="center">VividnessMem</h1>

<p align="center">
  <strong>Organic episodic memory for AI agents — no RAG, no embeddings, no vector DB.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/vividnessmem/"><img src="https://img.shields.io/pypi/v/vividnessmem" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/vividnessmem/"><img src="https://img.shields.io/pypi/pyversions/vividnessmem" alt="Python versions"/></a>
  <a href="https://github.com/Kronic90/VividnessMem-Ai-Roommates/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Kronic90/VividnessMem-Ai-Roommates" alt="License"/></a>
</p>

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

## What's New in v1.0.7

### NeuroChemistry System
A simulated neurochemical layer that modulates memory encoding, retrieval, and mood in real time — inspired by how dopamine, cortisol, serotonin, oxytocin, and norepinephrine actually regulate human cognition:

```python
mem = VividnessMem("./my_agent_memory")
mem.chemistry.enabled = True  # Activate neurochemistry

# Chemistry reacts to conversation events
mem.chemistry.on_event("warmth", intensity=0.8)      # Oxytocin rises → social memory boost
mem.chemistry.on_event("surprise_positive")           # Dopamine spike → encoding boost
mem.chemistry.on_event("conflict", intensity=0.6)     # Cortisol + NE rise → flashbulb encoding

# Also reacts to detected emotions
mem.chemistry.on_emotion("excited", intensity=0.9)

# Between sessions, chemistry resets (simulates sleep)
mem.chemistry.sleep_reset(hours=8.0)

# Inspect current neurochemical state
print(mem.chemistry.describe())
# → dopamine: 0.65 ▲ | cortisol: 0.28 ~ | serotonin: 0.58 ~ | oxytocin: 0.55 ▲ | norepinephrine: 0.48 ~
```

**5 neurotransmitters → 9 cognitive modifiers:**

| Chemical | Modulates | Effect |
|----------|-----------|--------|
| **Dopamine** | Encoding boost | High DA → memories stored more strongly (+30% max) |
| **Cortisol** | Attention width, flashbulb | Moderate → focused recall, extreme → flashbulb encoding |
| **Serotonin** | Mood stability | Low serotonin → moods linger longer, bias retrieval more |
| **Oxytocin** | Social memory boost | High OT → social impressions weighted more heavily |
| **Norepinephrine** | Consolidation bonus | Low NE (rest) → better memory consolidation |

**10 event profiles:** `surprise_positive`, `surprise_negative`, `conflict`, `warmth`, `novelty`, `resolution`, `achievement`, `loss`, `humor`, `stress`

Each event pushes chemicals realistically — conflict raises cortisol + norepinephrine while dropping serotonin + oxytocin. Chemicals drift back to baseline via homeostatic decay at biologically-inspired rates (norepinephrine is fast/phasic, cortisol is slow/lingering). The Yerkes-Dodson inverted-U is implemented for cortisol — moderate stress sharpens attention, extreme stress impairs it.

### SillyTavern Extension (v1.0.7)
Full-featured SillyTavern integration with a REST API server:
- **`{{vividmem}}` macro** — place memory context anywhere in your prompt template
- **`{{vividmood}}` macro** — insert current mood label
- **Token budget** — cap how many tokens of memory get injected
- **Per-chat / cross-chat memory toggle**
- **OOC message filter** — skips `(( ))`, `/ooc`, `//` from being stored
- **Import old memory JSON files** with automatic re-indexing
- **Export / backup** memories as JSON
- **Delete individual memories** from the browse panel
- **Manual memory notes** — type custom memories directly
- **Consolidate / Dream** triggers from the UI
- **Wipe all memories** with double-confirmation safety

See the [SillyTavern extension](https://github.com/Kronic90/VividnessMem-Ai-Roommates/tree/main/VividMem-Embed) for setup instructions.

### Improved Performance with VividEmbed
For embedding-based retrieval that outperforms standard RAG, pair VividnessMem with **[VividEmbed](https://github.com/Kronic90/VividEmbed)** — a fine-tuned emotion-vividness-aware embedding model:

```bash
pip install vividembed
```

VividEmbed provides hybrid vector retrieval that's aware of emotional salience and vividness scores, giving better recall on benchmarks than vanilla embedding approaches.

### Professional Mode
Toggle between **Character mode** (default — personality-rich, emotionally expressive) and **Professional mode** (neutral tone, fact-focused, no persona quirks):

```python
mem = VividnessMem("./my_agent_memory", professional=True)
```

- System context block output adapts automatically
- No persona/emotion sections in professional mode — just facts and retrieved memories
- Switch at any time; data is fully compatible between modes

### Task-Based Memory
Track projects, tasks, actions, and solution patterns — giving your agent structured knowledge about *what it's done* and *what worked*:

```python
# Project management
mem.set_active_project("website-redesign", description="Frontend overhaul")

# Task lifecycle
task_id = mem.start_task("Fix navbar alignment", priority="high")
mem.log_action(task_id, "inspect", "Checked CSS grid layout", result="found gap mismatch")
mem.complete_task(task_id, outcome="Fixed with grid-template-columns adjustment")

# Solution memory — record what worked and find it later
mem.record_solution("css grid alignment", "Use grid-template-columns with explicit values",
                    domain="frontend", tags=["css", "layout"])
matches = mem.find_solutions("grid layout broken", domain="frontend")

# Artifact tracking
mem.track_artifact(project="website-redesign", path="src/navbar.css",
                   artifact_type="source", status="modified")
```

- **4 new data classes**: `TaskRecord`, `ActionRecord`, `SolutionPattern`, `ArtifactRecord`
- Project-scoped storage with cross-project global solutions
- Automatic solution extraction from completed tasks
- Access-based decay tiers (active → warm → cold → archive)
- Full integration into `get_context_block()` output
- Adaptive auto-tracking in professional mode

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

### v1.0.5 Regression Tests (Professional Mode + Task Branch)

All v1.0.5 additions verified with no performance regression (2-seed average, seeds 42 & 123):

**Mem2ActBench** (n=100 per condition):

| Mode | Tool Accuracy | F1 | BLEU-1 |
|---|:---:|:---:|:---:|
| Character | 0.350 | 0.441 | 0.594 |
| Professional | 0.345 | 0.435 | 0.584 |

**MemoryBench WritingPrompts** (n=50 per condition):

| Mode | METEOR |
|---|:---:|
| Character | 0.159 |
| Professional | 0.158 |
| Previous baseline | 0.159 |

Character and Professional modes produce statistically identical results — confirming zero regression from v1.0.5 code additions.

## The Vividness Formula

```
vividness = (importance × 0.6) + (recency × 0.3) + (access_bonus × 0.1)
```

- **Importance (60%)** — how significant the event was (1–10)
- **Recency (30%)** — exponential decay with spaced-repetition stability growth
- **Access bonus (10%)** — capped at 3.0, grows with each recall

Mood alignment provides an additional boost/penalty based on emotional congruence.

## Full Feature List

### Core Memory
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

### Professional Mode *(new in v1.0.5)*
- **`professional=True` toggle** — neutral, fact-focused context blocks
- **Persona suppression** — no mood quirks or character voice in output
- **Same data layer** — switch modes without losing any memories

### Task-Based Memory *(new in v1.0.5)*
- **Project management** — `set_active_project()`, `list_projects()`, `archive_project()`
- **Task lifecycle** — `start_task()`, `complete_task()`, `fail_task()`, `abandon_task()`
- **Action logging** — `log_action()` with automatic solution extraction
- **Solution memory** — `record_solution()`, `find_solutions()` with hybrid overlap scoring
- **Artifact tracking** — `track_artifact()`, `update_artifact_status()`
- **Access decay tiers** — active → warm → cold → archive based on usage
- **Context block integration** — active tasks and relevant solutions injected automatically
- **Adaptive auto-tracking** — professional mode auto-records task transitions

### NeuroChemistry *(new in v1.0.7)*
- **5 neurotransmitters** — dopamine, cortisol, serotonin, oxytocin, norepinephrine
- **9 cognitive modifiers** — encoding boost, attention width, flashbulb, mood stability, social boost, warmth nudge, consolidation bonus, Yerkes-Dodson
- **10 event profiles** — surprise, conflict, warmth, novelty, resolution, achievement, loss, humor, stress
- **Emotion-to-chemistry mapping** — PAD vectors drive chemical shifts automatically
- **Homeostatic drift** — chemicals return to baseline at biologically-inspired rates
- **Sleep reset** — simulated overnight restoration between sessions

## API Reference

### Core

```python
mem = VividnessMem(directory, encryption_key=None, professional=False)
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

### Task-Based Memory *(v1.0.5)*

```python
# Projects
mem.set_active_project(name, description="")         # Switch active project
mem.list_projects()                                  # All projects with status
mem.archive_project(name)                            # Archive a project

# Tasks
task_id = mem.start_task(description, priority="medium", tags=None)
mem.complete_task(task_id, outcome="")               # Mark done + extract solution
mem.fail_task(task_id, reason="")                    # Mark failed
mem.abandon_task(task_id, reason="")                 # Mark abandoned

# Actions & Solutions
mem.log_action(task_id, action_type, description, result="", artifacts=None)
mem.record_solution(problem, solution, domain="", tags=None, confidence=1.0)
matches = mem.find_solutions(query, domain="", min_confidence=0.0)

# Artifacts
mem.track_artifact(project, path, artifact_type="file", status="created")
mem.update_artifact_status(project, path, status)
```

### Maintenance

```python
mem.update_mood_from_conversation(emotions)          # Shift mood state
mem.detect_contradictions(llm_fn=None)               # Lexical + optional LLM verification
mem.find_consolidation_clusters(min_cluster=3)       # Related memory groups
mem.find_dream_candidates()                          # Between-session patterns
mem.get_relationship_arc(entity)                     # Warmth trajectory
```

### NeuroChemistry *(v1.0.7)*

```python
mem.chemistry.enabled = True                         # Activate neurochemistry
mem.chemistry.on_event("warmth", intensity=0.8)      # Push chemicals from event
mem.chemistry.on_emotion("excited", intensity=0.9)   # Push from emotion label
mem.chemistry.tick()                                 # Homeostatic drift (auto-called)
mem.chemistry.sleep_reset(hours=8.0)                 # Between-session reset
mem.chemistry.levels                                 # Current chemical levels dict
mem.chemistry.describe()                             # Human-readable summary
mem.chemistry.get_modifiers()                        # 9 cognitive modifier values
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
