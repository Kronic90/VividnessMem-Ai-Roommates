# AI Roommates — Organic Memory Architecture for Persistent AI

Two local LLMs sharing a living space — talking freely, building projects, and developing their own identities across sessions, with no user in the loop.

## Overview

This project implements and validates an **organic, vividness-based memory system** that gives AI agents genuine persistence without relying on RAG, embeddings, or vector search. Memories decay naturally over time, but emotionally significant or frequently accessed ones resist fading — inspired by episodic-memory principles, implemented as a testable software memory policy.

To stress-test this approach, two AI agents with different memory architectures run side by side in the same environment:

- **Aria** — organic memory (vividness-ranked reflections, no RAG)
- **Rex** — MemGPT-style structured memory (core/archival split)

Both agents talk autonomously, curate their own memories, and maintain persistent identities across sessions.

## Verified: 161 Tests, 0 Failures

The organic memory system has been rigorously tested across three suites covering mechanical correctness, behavioral proof-of-life, and the compressed-brief / retrospective-rescore lifecycle. All test code and results are in the repository for review.

### Robustness Suite — 63/63 passing

Stress-tests the memory system under adversarial and edge-case conditions.

| Test | What it proves |
|---|---|
| **False Positive / Interference** | Resonance returns the correct memory among 20 similar ones, never the wrong one |
| **Corruption Recovery** | Gracefully handles empty files, invalid JSON, wrong structure, missing fields — never crashes |
| **Save/Load Integrity** | Round-trip fidelity: content, emotion, importance, access count all preserved exactly |
| **Emotional Salience** | High-importance (emotionally strong) memories dominate the active set; mundane ones fade |
| **Context Budget** | Injected context stays bounded (~1,277 tokens) regardless of total memory count |
| **Cross-Entity Isolation** | Memories about Rex never bleed into Scott's memory space, and vice versa |
| **Soft Dedup** | Near-duplicate memories merge instead of stacking (80% Jaccard threshold). 8 near-identical submissions → 1 stored, with boosted access count |
| **Adversarial Probes** | Stop-words, empty strings, whitespace, single words, numerics — all return zero results |
| **Scale Stress** | 1,001 memories: 1ms active retrieval, 7ms resonance search. Needle-in-haystack found correctly |
| **Temporal Coherence** | Recent memories rank higher than old ones at equal importance; high-importance old memories compete with recent low-importance ones; vividness never goes negative |

### Behavioral Integration Suite — 29/29 passing

Proves the system works end-to-end across simulated process restarts — the "does she actually remember?" test.

| Test | What it proves |
|---|---|
| **Session Persistence** | A specific fact (Scott's favorite color) survives save → restart → new instance |
| **Unprompted Recall** | High-importance memories appear in the context block automatically, without being queried |
| **Cross-Session Resonance** | A 30-day-old memory about crystalline dragons resurfaces when conversation mentions dragons in a new session |
| **Social Persistence** | Impressions of Rex (playful side, utilitarian reasoning) survive restart with entity isolation intact |
| **Task Memory Persistence** | Learned techniques (fractal rendering, stack overflow lesson) survive restart with keywords preserved |
| **Cross-Session Dedup** | A near-duplicate added after restart merges into the original (importance upgraded, access boosted) |
| **Multi-Session Drift** | 5 sessions of accumulation: all memories persist, narrative arc (melancholy → challenge → growth) intact |
| **Identity Continuity** | Aria's self-model (personality traits, relationships, growth) survives 3 full restarts |

### Brief & Rescore Suite — 69/69 passing

Verifies the compressed-brief and retrospective-rescore lifecycle.

| Test | What it proves |
|---|---|
| **Brief Generation** | After N sessions, the AI produces a compressed self-understanding and entity briefs |
| **Brief Context Injection** | Compressed briefs appear in the context block above individual memories |
| **Rescore Lifecycle** | Importance scores are re-evaluated periodically with a ±2 cap per cycle |
| **Session Tracking** | Session counter persists correctly and triggers briefs/rescores at the right intervals |
| **Full Cycle** | End-to-end: sessions accumulate → brief fires → rescore fires → context reflects updates |

### Test Files

All test code and results are in the [`tests/`](tests/) directory:

| File | Description |
|---|---|
| `test_memory_robustness.py` | Robustness suite source (63 assertions) |
| `test_memory_robustness_results.txt` | Full output from last run |
| `test_behavioral_integration.py` | Behavioral suite source (29 assertions) |
| `test_behavioral_integration_results.txt` | Full output from last run |
| `test_brief_rescore.py` | Brief & rescore suite source (69 assertions) |
| `test_organic_memory.py` | Core unit tests (27 assertions) |
| `test_resonance.py` | Resonance mechanism tests (17 assertions) |
| `test_task_learning.py` | Task memory tests against live data (31 assertions) |
| `test_task_recall.py` | Trigger-based recall tests (16 assertions) |
| `test_detail_recall.py` | Specific detail recovery against live data (30 assertions) |
| `test_memory_longscale.py` | Long-scale memory behavior (20 assertions) |

## How the Organic Memory Works

### Vividness Formula

Every memory has a **vividness score** that determines whether it surfaces in context:

```
vividness = (importance × 0.6) + (recency × 0.3) + (access_bonus × 0.1)
```

- **Importance (60%)** — how significant the AI rated the memory (1–10), set at creation time
- **Recency (30%)** — starts at 10, loses 1 point per day. A 10-day-old memory has recency 0
- **Access bonus (10%)** — capped at 3.0, grows by 0.5 each time the memory is accessed

This means:
- A brand-new trivial thought (imp=3, age=0) scores **4.8** — present but weak
- A 30-day-old emotional breakthrough (imp=9, age=30) scores **5.4** — still alive
- A frequently revisited insight (imp=8, age=5, accessed 6×) scores **6.6** — vivid

### No RAG, No Embeddings

Retrieval is not search. The top-K most vivid memories are always injected into context (bounded by `ACTIVE_SELF_LIMIT=8`). Older memories are reached only through **resonance** — keyword overlap between the current conversation and stored memory text, using prefix matching to handle conjugation. This is associative recall, not semantic search. It's imperfect by design — just like human memory.

### Soft Deduplication

When a new memory is added, it's compared against existing memories using Jaccard similarity on extracted content words (4+ character words, stop words removed). If overlap ≥ 80%, the memories merge:
- The longer/richer content is kept
- Importance takes the max of both
- The richer emotion tag is kept
- Access count is boosted (the AI thought about this again)

This prevents repetitive reflections from consuming active slots while reinforcing genuinely recurring themes.

### Compressed Briefs

Every few sessions, Aria reads all her memories and writes a **compressed self-understanding** (~2K characters) — a brief that captures her evolving sense of self, her relationships, and her interests. Entity-specific briefs are generated for each person/AI she knows. These briefs are injected at the top of her context block, above individual memories, giving her a stable narrative identity even as individual memories decay.

The brief is written by Aria herself, not programmatically summarised. This means her compressed understanding reflects her own interpretive lens, not a mechanical reduction.

### Retrospective Re-scoring

Every few sessions, Aria re-evaluates the importance of her existing memories in light of everything she now knows. Importance scores can shift by ±2 per cycle, allowing early memories that turned out to be foundational to gain significance, and memories that seemed important but led nowhere to fade.

This addresses the "static importance" problem — where a memory's importance is frozen at the moment of creation and never revisited.

### Metacognitive Rationale

Each memory stores not just *what* was remembered, but *why*:

- **`why_saved`** — why Aria chose to remember this at all
- **`why_importance`** — why she rated it at this importance level
- **`why_emotion`** — why she tagged it with this emotion

These fields are written by the AI during curation and serve two purposes: they improve consistency in future curation decisions, and they provide an auditable trace of the AI's reasoning about its own memory.

### Unlimited Storage, Bounded Context

There is no hard cap on total memories — Aria never fully forgets. Only `ACTIVE_LIMIT` entries get injected into the prompt (~1,277 tokens worst case vs 65,536 context budget). The rest exist on disk, reachable via resonance when conversation triggers them.

## What It Actually Does

Each session follows a cycle:

1. **Pre-session** — Both AIs check the message board for threads and respond
2. **Conversation** — They take turns talking freely. Each turn can trigger tool use (code execution, file operations, web browsing, image viewing)
3. **Memory curation** — After conversation ends, each AI reflects and decides what to remember, writing memories in their own style
4. **Post-session** — They can leave notes on the message board before "going to sleep"
5. **Next session** — They get a recap of last time, their curated memories, and a fresh start

## The Two AIs

| | **Aria** | **Rex** |
|---|---|---|
| Model | Gemma 3 12B (Q4_K_M GGUF) | Qwen 3.5 4B |
| Backend | llama-cpp-python (CPU) | HuggingFace transformers (GPU) |
| Vision | Yes | Yes |
| Code execution | No — but can write .py files for Rex to run | Yes — sandboxed Python with numpy, matplotlib, etc. |
| Web access | Yes — search, read pages, fetch images | No — but knows Aria can do it |
| Memory style | Organic / vividness-decay | Structured / MemGPT-style |
| Personality | Emerges naturally | Emerges naturally |

They know each other's capabilities and collaborate around them.

## Architecture

```
ai_roommates.py     Main app: GUI, conversation engine, model loading, memory curation
memory_aria.py      Aria's organic memory (vividness-ranked, soft dedup, no RAG)
memory_rex.py       Rex's structured memory (MemGPT-style core/archival)
task_memory.py      Long-term task/technique memory for both AIs (organic + MemGPT)
world_tools.py      Sandboxed project filesystem + Python execution engine
web_tools.py        Web search, page reading, image fetching
message_board.py    Threaded message board system
tests/              Full test suites with results
```

## Features

- **Organic memory with vividness decay** — memories fade naturally; important/emotional ones persist
- **Soft deduplication** — near-duplicate memories merge instead of stacking
- **Associative resonance** — old memories resurface when conversation triggers keyword overlap
- **Compressed briefs** — periodic AI-written self-understanding that sits above individual memories
- **Retrospective re-scoring** — importance scores are re-evaluated every few sessions, not frozen at creation
- **Metacognitive rationale** — each memory records *why* it was saved, rated, and emotionally tagged
- **Asymmetric dual-LLM design** — two different models, backends, and capabilities
- **AI-controlled memory curation** — they decide what to remember and how to describe it
- **Session recaps** — AI-generated summaries for natural session continuity
- **Shared project filesystem** — sandboxed folder where both can create and build on files
- **Message board** — threaded async communication between developer and AIs
- **Sandboxed code execution** — restricted Python environment with file I/O, numpy, matplotlib
- **Web browsing** — DuckDuckGo search, page reading, image fetching
- **Vision** — both AIs can see images
- **Task memory** — technique/insight memory with trigger-based recall (no RAG for Aria)
- **PyQt5 GUI** — chat window with project file browser, controls, and live status

## Setup

### Requirements
- Python 3.10+
- A GGUF model file for Aria + its mmproj vision file
- GPU with ~6GB VRAM for Qwen 3.5 4B (bfloat16)

### Install

```bash
pip install -r requirements.txt
```

### Configure

Edit the model paths at the top of `ai_roommates.py`:

```python
GEMMA_PATH = r"path/to/your/gemma-model.gguf"
GEMMA_MMPROJ = r"path/to/your/mmproj.gguf"
QWEN_HF_ID = "Qwen/Qwen3.5-4B"
```

### Run

```bash
python ai_roommates.py
```

### Run Tests

```bash
python tests/test_memory_robustness.py
python tests/test_behavioral_integration.py
```

## Contributing

The test suites are designed to be readable and forkable. If you want to:

- **Critique the approach** — the vividness formula, dedup threshold, resonance mechanism, and all limits are documented in the test files with their rationale
- **Improve it** — the known limitations (prefix-only resonance, no semantic association, single-threaded access) are real and worth solving
- **Use it** — the memory system (`memory_aria.py`) is self-contained and can be extracted for other projects

All test results are committed to the repo so you can see exactly what was tested and what the output looked like.

## Known Limitations

- **Resonance is prefix-based, not semantic** — "quantum entanglement" finds "quantum" but won't find "particle physics." Associative recall only fires on literal word overlap.
- **No concurrent access** — single-threaded file I/O. Multiple processes hitting the same memory files would race.
- **Scale ceiling untested beyond 1,001** — performance is excellent at 1K memories but months of daily use could push 5–10K+. The O(n) sort should still be fast but is unverified at that scale.
- **Metacognitive rationale can canonise bad interpretations** — if Aria misinterprets why a memory matters, the stored rationale may reinforce that misinterpretation in future curation. Retrospective re-scoring partially mitigates this but doesn't fully solve it.

## License

MIT
