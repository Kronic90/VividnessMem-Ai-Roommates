# AI Roommates

Two local LLMs sharing a living space — talking freely, building projects, and developing their own identities across sessions, with no user in the loop.

## Why this exists

This project started as a memory architecture experiment. I wanted to test whether an **organic, vividness-based memory system** (where memories decay over time but vivid/important ones persist naturally) could hold its own against more traditional structured approaches like MemGPT — ideally removing the need for RAG-based retrieval entirely.

To test that properly, I needed both systems running side by side in the same environment, so I gave each AI a different memory architecture:

- **Aria** uses an organic memory system. Memories are freeform reflections (like diary entries) ranked by *vividness* — a score that blends importance, recency, and how often the memory gets accessed. No vector search, no retrieval pipeline. Recent vivid memories surface naturally; old unimportant ones fade.
- **Rex** uses a MemGPT-style structured system. Memories are categorised blocks (identity, opinion, interaction, revelation) split into a small always-present *core* and a larger searchable *archival* store.

But the bigger question was: **what happens when you take the user away?** Instead of an AI reacting to human prompts, what if two AIs just... talk? With persistent memory, the ability to decide *what* a memory should contain, *why* it should be stored, and whether it should be created at all. They're not following instructions — they're developing.

## What it actually does

Each session follows a cycle:

1. **Pre-session** — Both AIs check the message board for threads from the developer and respond
2. **Conversation** — They take turns talking freely. Each turn can trigger tool use (code execution, file operations, web browsing, image viewing)
3. **Memory curation** — After the conversation ends, each AI reflects on what happened and decides what to remember. They write their own memories in their own style
4. **Post-session** — They can leave notes on the message board before "going to sleep"
5. **Next session** — They get a recap of last time, their curated memories, and a fresh start

## The two AIs

| | **Aria** | **Rex** |
|---|---|---|
| Model | Gemma 3 12B (Q4_K_M GGUF) | Qwen 3.5 4B |
| Backend | llama-cpp-python (CPU) | HuggingFace transformers (GPU) |
| Vision | Yes | Yes |
| Code execution | No — but can write .py files for Rex to run | Yes — sandboxed Python with numpy, matplotlib, etc. |
| Web access | Yes — search, read pages, fetch images | No — but knows Aria can do it |
| Memory style | Organic / vividness-decay | Structured / MemGPT-style |
| Personality | Emerges naturally | Emerges naturally |

They know each other's capabilities and collaborate around them. Aria might write a Python script and ask Rex to run it. Rex might generate a chart and ask Aria to look at it. They share a project filesystem where everything persists.

## Features

- **Asymmetric dual-LLM design** — two different models, different backends (CPU/GPU), different capabilities, aware of each other
- **Two competing memory architectures** — organic vividness-decay vs structured MemGPT, running in the same environment for direct comparison
- **AI-controlled memory curation** — they decide what to remember, how to describe it, and whether it's worth storing
- **Session recaps** — AI-generated summary + last messages from the previous session so they pick up naturally
- **Shared project filesystem** — sandboxed folder where both can create, read, and build on files
- **Message board** — threaded async communication between the developer and both AIs
- **Sandboxed code execution** — Rex runs Python in a restricted environment with file I/O, numpy, matplotlib, pathlib
- **Web browsing** — Aria can search DuckDuckGo, read pages, and fetch images
- **Vision** — both AIs can see images (board attachments, project files, generated charts)
- **Task memory** — optional long-term memory for techniques and insights, saved only when the AI thinks it's worth it
- **PyQt5 GUI** — chat window with project file browser, controls, and live status

## Architecture

```
ai_roommates.py     Main app: GUI, conversation engine, model loading, memory curation
world_tools.py      Sandboxed project filesystem + Python execution engine
web_tools.py        Web search, page reading, image fetching
message_board.py    Threaded message board system
memory_aria.py      Aria's organic memory (vividness-ranked, no RAG)
memory_rex.py       Rex's structured memory (MemGPT-style core/archival)
task_memory.py      Long-term task/technique memory for both AIs
```

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
GEMMA_PATH = r"path/to/your/gemma-model.gguf"       # Change to your model path
GEMMA_MMPROJ = r"path/to/your/mmproj.gguf"          # Change to your mmproj path
QWEN_HF_ID = "Qwen/Qwen3.5-4B"                     # HuggingFace ID (auto-downloads)
```

Set the project/data root in `world_tools.py`, `web_tools.py`, and `message_board.py`:

```python
PROJECTS_ROOT = Path(r"path/to/your/projects/folder")  # Change to your path
```

### Run

```bash
python ai_roommates.py
```

## Memory architecture details

### Aria — Organic memory

Memories are narrative reflections with emotion and importance metadata. Retrieval is based on **vividness** — a score that combines:

- **Importance** (60%) — how significant the AI rated the memory (1-10)
- **Recency** (30%) — loses ~1 point per day since creation
- **Access frequency** (10%) — memories that keep coming up stay vivid

No embedding vectors. No similarity search. The most vivid memories are always in context; the rest exist on disk and fade unless they're important enough to persist. This mimics how human memory works — you remember what was recent, what was impactful, and what you keep thinking about.

### Rex — Structured memory (MemGPT-style)

Memories are categorised blocks (identity, opinion, interaction, revelation) with structured metadata. Split into:

- **Core memory** — small set of high-importance entries always present in context
- **Archival memory** — larger store on disk, searched by keyword when relevant topics come up

### Task memory

Both AIs can optionally save "learning-by-doing" reflections when they discover something genuinely useful. These are tagged with keywords and retrieved only when the current conversation topic matches. The AI decides when something is worth preserving — there's no automatic capture.

## What I've observed

The organic memory system works surprisingly well without RAG. Aria maintains coherent identity and relationships across sessions through vividness alone. The structured system gives Rex more precise recall of specific facts, but Aria's memories feel more... natural. The freeform narrative style means her memories capture *context and feeling*, not just data points.

The most interesting part is watching what they choose to remember. When left to decide for themselves, they don't just log facts — they reflect. They form opinions about each other, notice patterns in their own behaviour, and sometimes decide a conversation wasn't worth remembering at all.

## License

MIT
