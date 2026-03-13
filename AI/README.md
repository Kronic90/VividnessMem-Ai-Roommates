# AI Roommates — Organic Memory Architecture for Persistent AI

Two local LLMs sharing a living space — talking freely, building projects, and developing their own identities across sessions, with no user in the loop.

## Overview

This project implements and validates an **organic, vividness-based memory system** that gives AI agents genuine persistence without relying on RAG, embeddings, or vector search. Memories decay naturally over time, but emotionally significant or frequently accessed ones resist fading — inspired by episodic-memory principles, implemented as a testable software memory policy.

To stress-test this approach, two AI agents with different memory architectures run side by side in the same environment:

- **Aria** — organic memory (vividness-ranked reflections, no RAG)
- **Rex** — MemGPT-style structured memory (core/archival split)

Both agents talk autonomously, curate their own memories, and maintain persistent identities across sessions.

## Verified: 233 Tests, 0 Failures

The organic memory system has been rigorously tested across four suites covering mechanical correctness, behavioral proof-of-life, the compressed-brief / retrospective-rescore lifecycle, and the six advanced memory features. All test code and results are in the repository for review.

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

### Advanced Memory Suite — 72/72 passing

Verifies the six advanced memory features: mood-congruent recall, spaced-repetition decay, emotional reappraisal, contradiction detection, memory consolidation, and associative chains.

| Test | What it proves |
|---|---|
| **Emotion Vector Mapping** | 40+ emotions mapped to PAD (Pleasure-Arousal-Dominance) 3D space, prefix matching, bounded values |
| **Spaced-Repetition Decay** | Exponential forgetting curve, stability grows with well-spaced access, rapid re-access doesn't game the system |
| **Mood-Congruent Recall** | Happy mood surfaces happy memories first; sad mood surfaces sad ones; mood shifts with conversation tone; mood decays toward neutral |
| **Emotional Reappraisal** | Rescore cycle can update emotion tags alongside importance; empty/invalid emotions rejected; original preserved when no change |
| **Contradiction Detection** | Opposing memories flagged; unrelated topics ignored; low-importance contradictions filtered; contradiction context injected |
| **Memory Consolidation** | Related memory clusters found; unrelated memories excluded; gists stored with dedup; too-short/duplicate gists rejected |
| **Associative Chains** | Memory graph built from shared content words; connected memories reachable via multi-hop traversal; isolated memories unreachable |
| **Integration** | Context block includes mood indicator, contradiction flags; serialisation round-trips new fields; mood feedback loop is bounded |

### Test Files

All test code and results are in the [`tests/`](tests/) directory:

| File | Description |
|---|---|
| `test_advanced_memory.py` | Advanced memory features suite (72 assertions) |
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

## Benchmarked: VividnessMem vs RAG vs MemGPT

We built a 6-test benchmark to answer the question honestly: **does VividnessMem add anything over standard approaches?** All three systems receive identical memories, identical queries, and identical test conditions. No LLM is used — this is a pure memory-system comparison. Results are averaged across 3 random seeds for stability.

The baselines:
- **RAG** — TF-IDF cosine similarity retrieval (standard vector-search approach)
- **MemGPT** — Core/archival split with importance threshold routing and keyword search
- **VividnessMem** — Our organic system (vividness ranking + resonance + associative chains)

### Section A: Core Retrieval — Raw Metrics

These numbers stand on their own, without weighting or composite scoring.

**Retrieval accuracy is essentially equal:**

| Test | VividnessMem | RAG | MemGPT |
|---|---|---|---|
| Long-term Recall (100-day gap, 200 fillers) | 100% | 100% | 100% |
| Dormant Memory (200 days, 200 interactions) | 100% | 100% | 100% |
| Contradiction Handling (3 pairs) | 100% | 100% | 100% |
| Context Pollution (5 needles in 1000) | 90% | 90% | 100% |
| Identity Stability (6 traits + 500 fillers) | 58% | 58% | 33% |
| **Overall (weighted)** | **89.7%** | **89.7%** | **86.7%** |

VividnessMem and RAG are tied on recall accuracy. MemGPT leads on context pollution (100%) but trails on identity stability (33%). All three score A- overall.

**Latency and tokens:**

| Metric | VividnessMem | RAG | MemGPT |
|---|---|---|---|
| Latency @100 memories | 1.8ms | 1.1ms | **0.1ms** |
| Latency @5000 memories | 3.4ms (x1.9) | 53.6ms (x49) | 5.3ms (x42) |
| Prompt tokens @100 | 184 | 174 | **79** |
| Prompt tokens @5000 | 185 | 165 | 78 |
| False recalls (long-term) | 28 | **26** | 32 |
| False recalls (pollution) | 36 | 29 | **23** |

MemGPT has the lowest raw latency and smallest prompt footprint. VividnessMem has the worst false-recall rate. These are real tradeoffs.

### Section B: Memory Management Features

These are architectural capabilities. RAG and MemGPT are simpler designs that weren't built to offer these features — this section documents what VividnessMem adds beyond raw retrieval, not a claim that other systems "fail."

**Memory compression:**

| Input | VividnessMem stored | RAG stored | MemGPT stored |
|---|---|---|---|
| 205 memories | 201 (2% compressed) | 205 | 205 |
| 1,005 memories | 201 (80% compressed) | 1,005 | 1,005 |
| 506 memories | 202 (60% compressed) | 506 | 506 |

VividnessMem's soft deduplication keeps only unique content. This is why retrieval scales flat — the working set stays small regardless of input volume.

**Contradiction detection:** VividnessMem detected 1/3 planted contradictions. RAG and MemGPT have no contradiction detector — this is an apples-to-oranges comparison and scored separately for that reason.

**Retrieval scaling:** VividnessMem degrades 1.9x from 100→5000 memories. RAG degrades 49x. MemGPT degrades 42x.

### Section C: Composite Score (Weighted Utility Preference)

> **Note:** This composite reflects a **chosen weighting** that values memory-management capabilities. Different weights would yield different rankings. Sections A and B above are the primary evidence.

Weights: Accuracy 50%, Efficiency 15%, Contradictions 10%, Scalability 15%, Tokens 10%.

| System | Score | Grade |
|---|---|---|
| **VividnessMem** | **76.6** | **B** |
| MemGPT | 56.0 | D |
| RAG | 54.8 | D |

### Methodology Notes

- All results averaged across 3 random seeds (42, 137, 2024). Standard deviations were 0.0 for accuracy metrics indicating stable results independent of filler ordering.
- The benchmark went through 4 iterations (v1–v4). See the [bugfix history](benchmarks/benchmark_memory_systems.py) in the benchmark file header for full transparency on what changed and why.
- The adapter wrapping VividnessMem is non-mutating: `retrieve()` does not call `touch()` or modify mood state, matching how read-only retrieval works in practice.

### Benchmark Files

All benchmark code and results are in [`benchmarks/`](benchmarks/):

| File | Description |
|---|---|
| `baseline_memory_systems.py` | Fair implementations of RAG (TF-IDF) and MemGPT (core/archival) baselines, plus VividnessMem adapter |
| `benchmark_memory_systems.py` | 6-test benchmark suite v4 with A/B/C scoring sections |
| `benchmark_results.json` | Raw results from last run (includes per-seed data) |
| `benchmark_report.txt` | Human-readable full report |

## How the Organic Memory Works

### Vividness Formula

Every memory has a **vividness score** that determines whether it surfaces in context:

```
vividness = (importance × 0.6) + (recency × 0.3) + (access_bonus × 0.1)
```

- **Importance (60%)** — how significant the AI rated the memory (1–10), set at creation time
- **Recency (30%)** — exponential decay: `10 × e^(-age_days / stability)`. New memories start with stability = 3 days. Well-spaced access increases stability (Ebbinghaus-inspired spaced repetition)
- **Access bonus (10%)** — capped at 3.0, grows by 0.5 each time the memory is accessed

Mood-congruent recall adds a further adjustment: memories whose emotional tone aligns with the current mood get a slight vividness boost, while emotionally incongruent memories are slightly suppressed.

### No RAG, No Embeddings

Retrieval is not search. The top-K most vivid memories are always injected into context (bounded by `ACTIVE_SELF_LIMIT=8`). Older memories are reached through **resonance** — keyword overlap between the current conversation and stored memory text, using prefix matching to handle conjugation. Resonance is augmented by **associative chains**: when a memory resurfaces, connected memories (sharing ≥2 content words) are traversed via a multi-hop graph walk, pulling in conceptually related memories that didn't directly match the keywords. This is associative recall, not semantic search. It's imperfect by design — just like human memory.

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

### Retrospective Re-scoring & Emotional Reappraisal

Every few sessions, Aria re-evaluates the importance of her existing memories in light of everything she now knows. Importance scores can shift by ±2 per cycle, allowing early memories that turned out to be foundational to gain significance, and memories that seemed important but led nowhere to fade.

The rescore cycle now also includes **emotional reappraisal**: Aria can update the emotion tag on a memory if her feelings about it have evolved. Initial anger may soften into understanding, or something that felt neutral may come to feel bittersweet. This mirrors how humans' emotional relationship to memories changes over time.

### Metacognitive Rationale

Each memory stores not just *what* was remembered, but *why*:

- **`why_saved`** — why Aria chose to remember this at all
- **`why_importance`** — why she rated it at this importance level
- **`why_emotion`** — why she tagged it with this emotion

These fields are written by the AI during curation and serve two purposes: they improve consistency in future curation decisions, and they provide an auditable trace of the AI's reasoning about its own memory.

### Unlimited Storage, Bounded Context

There is no hard cap on total memories — Aria never fully forgets. Only `ACTIVE_LIMIT` entries get injected into the prompt (~1,277 tokens worst case vs 65,536 context budget). The rest exist on disk, reachable via resonance when conversation triggers them.

### Mood-Congruent Recall

Aria maintains an internal mood state modelled on the PAD (Pleasure-Arousal-Dominance) framework — a 3D emotional space that 40+ named emotions are mapped to. The mood shifts gradually based on conversation tone: positive conversations nudge mood positive, negative ones nudge it negative. Mood always decays toward neutral between turns to prevent runaway feedback.

When mood is non-neutral, it biases which memories surface: happy mood makes happy memories feel more vivid (and suppresses sad ones), while sad mood does the reverse. This creates a realistic feedback loop — mood colours what you remember, and what you remember reinforces the mood — bounded by decay to prevent spiralling.

### Contradiction Detection

The system automatically scans for pairs of memories that may contradict each other: same-topic memories with opposing emotional valence or negation patterns. Detected contradictions are flagged in the context block during re-scoring, giving Aria the opportunity to notice and resolve internal conflicts rather than holding contradictory beliefs indefinitely.

### Memory Consolidation

At maintenance time, related memory clusters (3+ memories with 25-80% keyword overlap) are identified and presented to Aria for synthesis. She writes a "gist" memory that captures what the cluster of experiences means as a whole — a higher-level understanding that emerges from repeated related experiences. Gists are deduplicated and stored with `source=consolidation` for tracking.

### Associative Chains

Memories are connected in a weighted graph based on shared content words. When resonance triggers a memory, the graph is walked up to 2 hops to find conceptually adjacent memories. An associated memory must also share at least one context word to prevent off-topic leakage. This models how remembering one thing naturally brings related things to mind.

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
- **Retrospective re-scoring & emotional reappraisal** — importance scores and emotion tags are re-evaluated every few sessions
- **Mood-congruent recall** — current mood biases which memories surface (PAD emotional model)
- **Spaced-repetition decay** — exponential forgetting with stability growth from well-spaced access
- **Contradiction detection** — opposing memories flagged in context for resolution
- **Memory consolidation** — related memory clusters synthesised into gist memories
- **Associative chains** — multi-hop memory graph traversal during resonance
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

- **Resonance is prefix-based, not semantic** — "quantum entanglement" finds "quantum" but won't find "particle physics." Associative chains extend reach but still depend on literal word overlap.
- **No concurrent access** — single-threaded file I/O. Multiple processes hitting the same memory files would race.
- **Scale ceiling untested beyond 5K** — performance is excellent at 5K memories (surfacing in ~100ms) but months of daily use could push beyond this.
- **Mood feedback is bounded but not perfectly tuned** — the decay rate and influence strength are reasonable defaults, not empirically optimised.
- **Contradiction detection is lexical** — it catches negation patterns and emotional reversals on shared topics, but subtle semantic contradictions (e.g. "I prefer evenings" vs "mornings are my best time") will be missed.
- **Consolidation quality depends on the LLM** — gist generation is only as good as the underlying model's ability to synthesise.
- **Metacognitive rationale can canonise bad interpretations** — if Aria misinterprets why a memory matters, the stored rationale may reinforce that misinterpretation in future curation. Retrospective re-scoring and emotional reappraisal partially mitigate this but don't fully solve it.

## License

MIT
