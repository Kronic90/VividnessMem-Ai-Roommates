"""
push_to_github.py — Push the AI Roommates project to GitHub.

FIRST-TIME SETUP:
  1. Create an empty repo on GitHub (e.g. "ai-roommates")
  2. Set GITHUB_REPO below to your repo URL
  3. Run: python push_to_github.py
  4. Git will prompt you to log in via browser the first time
  
UPDATING:
  Just run: python push_to_github.py
  It commits any changes and pushes them.
"""

import subprocess
import sys
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURE THESE
# ════════════════════════════════════════════════════════════════════════
GITHUB_REPO = "https://github.com/Kronic90/VividnessMem-Ai-Roommates.git"  # <-- PUT YOUR REPO URL HERE, e.g. "https://github.com/Kronic90/VividnessMem-Ai-Roommates.git"

GIT_NAME = "Kronic90"          # Your name for commits
GIT_EMAIL = "chefturnip2@gmail.com"              # Your email for commits (can be your GitHub noreply email)

COMMIT_MSG = input("Commit message (or Enter for default): ").strip()
if not COMMIT_MSG:
    COMMIT_MSG = "Update AI Roommates project"

# ════════════════════════════════════════════════════════════════════════
#  CORE PROJECT FILES — only these go into the repo
# ════════════════════════════════════════════════════════════════════════
PROJECT_FILES = [
    # Main application
    "ai_roommates.py",

    # Supporting modules
    "world_tools.py",
    "web_tools.py",
    "message_board.py",
    "memory_aria.py",
    "memory_rex.py",
    "task_memory.py",

    # Config / docs
    "requirements.txt",
    "README.md",
    ".gitignore",
]

# ════════════════════════════════════════════════════════════════════════
#  SCRIPT LOGIC — no need to edit below here
# ════════════════════════════════════════════════════════════════════════
REPO_DIR = Path(__file__).parent


def run(cmd, check=True):
    """Run a shell command and return output."""
    print(f"  > {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=str(REPO_DIR),
        capture_output=True, text=True, encoding="utf-8"
    )
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0 and check:
        print(f"    ERROR: {result.stderr.strip()}")
        sys.exit(1)
    return result


def ensure_gitignore():
    """Create .gitignore if it doesn't exist."""
    gitignore = REPO_DIR / ".gitignore"
    if gitignore.exists():
        return
    print("\nCreating .gitignore...")
    gitignore.write_text(
        "# AI Roommates — only tracked files are committed (see push_to_github.py)\n"
        "# This .gitignore blocks everything, we use 'git add -f' for project files.\n"
        "\n"
        "# Ignore everything by default\n"
        "*\n"
        "\n"
        "# Then whitelist project files\n"
        "!.gitignore\n"
        "!ai_roommates.py\n"
        "!world_tools.py\n"
        "!web_tools.py\n"
        "!message_board.py\n"
        "!memory_aria.py\n"
        "!memory_rex.py\n"
        "!task_memory.py\n"
        "!requirements.txt\n"
        "!README.md\n"
        "!push_to_github.py\n"
        "!LICENSE\n",
        encoding="utf-8"
    )


def ensure_requirements():
    """Create requirements.txt if it doesn't exist."""
    req = REPO_DIR / "requirements.txt"
    if req.exists():
        return
    print("\nCreating requirements.txt...")
    req.write_text(
        "# AI Roommates — Python dependencies\n"
        "# Install: pip install -r requirements.txt\n"
        "\n"
        "# Core\n"
        "PyQt5>=5.15\n"
        "llama-cpp-python>=0.3.0\n"
        "transformers>=4.45.0\n"
        "torch>=2.0\n"
        "Pillow>=10.0\n"
        "\n"
        "# Web tools\n"
        "duckduckgo-search>=6.0\n"
        "beautifulsoup4>=4.12\n"
        "lxml>=5.0\n"
        "requests>=2.31\n"
        "\n"
        "# Sandbox extras\n"
        "numpy>=1.24\n"
        "matplotlib>=3.7\n",
        encoding="utf-8"
    )


def ensure_readme():
    """Create README.md if it doesn't exist."""
    readme = REPO_DIR / "README.md"
    if readme.exists():
        return
    print("\nCreating README.md...")
    readme.write_text(
r"""# AI Roommates

Two local LLMs living together — having open-ended conversations, building projects, and developing their own identities over time.

## What is this?

**AI Roommates** puts two AI models in a shared environment where they talk freely with no user in the loop. They have complementary abilities, persistent memory, a shared project workspace, and a message board where a human developer can leave them notes.

- **Aria** (Gemma 3 12B, CPU via llama-cpp) — Vision, web browsing, creative writing, file management
- **Rex** (Qwen 3.5 4B, GPU via transformers) — Vision, live Python execution, data analysis, simulations, charts

They're not task-oriented agents — they're *roommates*. They develop opinions, remember past conversations, argue, collaborate on projects, and pick up where they left off.

## Features

- **Asymmetric dual-LLM** — two different models with different capabilities that are aware of each other's strengths
- **Persistent memory** — each AI curates what it wants to remember after every conversation
- **Session recaps** — summary + last messages from the previous session so they can continue naturally
- **Shared project filesystem** — sandboxed folder where both can create, read, and build on files
- **Message board** — threaded async messaging between the developer and both AIs
- **Sandboxed code execution** — Rex runs Python in a restricted sandbox with file I/O, numpy, matplotlib
- **Web browsing** — Aria can search the web, read pages, and fetch images
- **Vision** — both AIs can see images (board attachments, project files, charts)
- **Task memory** — optional long-term memory for techniques and insights they want to preserve
- **PyQt5 GUI** — chat window with project browser, controls, and live status

## Architecture

```
ai_roommates.py     — Main app: GUI, conversation loop, model loading, memory curation
world_tools.py      — Sandboxed project filesystem + Python execution engine
web_tools.py        — Web search, page reading, image fetching (Aria only)
message_board.py    — Threaded message board system
memory_aria.py      — Aria's memory system (organic/reflective style)
memory_rex.py       — Rex's memory system (structured/categorical style)
task_memory.py      — Long-term task/technique memory for both AIs
```

## Setup

### Requirements
- Python 3.10+
- A GGUF model file for Aria (e.g. Gemma 3 12B Q4_K_M) + its mmproj file
- GPU with enough VRAM for Qwen 3.5 4B (~6GB in bfloat16)

### Install
```bash
pip install -r requirements.txt
```

### Configure
Edit the paths at the top of `ai_roommates.py`:
```python
GEMMA_PATH = r"path/to/your/gemma-model.gguf"          # Change to your model path
GEMMA_MMPROJ = r"path/to/your/mmproj.gguf"             # Change to your mmproj path
QWEN_HF_ID = "Qwen/Qwen3.5-4B"                        # HuggingFace model ID (auto-downloads)
```

Also set the project root paths in `world_tools.py`, `web_tools.py`, and `message_board.py`.

### Run
```bash
python ai_roommates.py
```

## How it works

1. **Pre-session** — Each AI checks the message board for threads from the developer and responds
2. **Conversation** — They take turns talking freely. Each turn can trigger tool use (code, files, web, images)
3. **Memory curation** — After the conversation, each AI reflects on what happened and saves what it wants to remember
4. **Post-session** — They can leave notes on the message board before "going to sleep"
5. **Next session** — They get a recap of last time + their curated memories, and start fresh

## License

MIT
""",
        encoding="utf-8"
    )


def main():
    if not GITHUB_REPO:
        print("=" * 60)
        print("  ERROR: Set GITHUB_REPO in this script first!")
        print("  1. Create an empty repo on GitHub")
        print("  2. Copy the HTTPS URL (e.g. https://github.com/you/ai-roommates.git)")
        print("  3. Paste it into GITHUB_REPO at the top of this script")
        print("=" * 60)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  AI Roommates — GitHub Push")
    print(f"  Repo: {GITHUB_REPO}")
    print(f"  Commit: {COMMIT_MSG}")
    print(f"{'=' * 60}\n")

    # Generate supporting files if needed
    ensure_gitignore()
    ensure_requirements()

    # Check which project files exist
    missing = [f for f in PROJECT_FILES if not (REPO_DIR / f).exists()]
    if missing:
        print(f"Note: These files don't exist yet (skipping): {missing}")

    existing = [f for f in PROJECT_FILES if (REPO_DIR / f).exists()]
    # Always include the push script itself
    if "push_to_github.py" not in existing:
        existing.append("push_to_github.py")

    # Git setup
    print("\n--- Git setup ---")
    run(f'git config user.name "{GIT_NAME}"')
    if GIT_EMAIL:
        run(f'git config user.email "{GIT_EMAIL}"')

    # Check if remote exists
    result = run("git remote -v", check=False)
    if "origin" not in (result.stdout or ""):
        print("\nAdding remote origin...")
        run(f'git remote add origin "{GITHUB_REPO}"')
    else:
        # Update remote URL in case it changed
        run(f'git remote set-url origin "{GITHUB_REPO}"')

    # Stage only project files (force-add past .gitignore)
    print("\n--- Staging files ---")
    for f in existing:
        run(f'git add -f "{f}"')
    print(f"  Staged {len(existing)} files")

    # Commit
    print("\n--- Committing ---")
    result = run(f'git commit -m "{COMMIT_MSG}"', check=False)
    if "nothing to commit" in (result.stdout or ""):
        print("  No changes to commit — everything is up to date!")
        return

    # Push
    print("\n--- Pushing to GitHub ---")
    run("git push -u origin main")

    print(f"\n{'=' * 60}")
    print(f"  Done! Check your repo at: {GITHUB_REPO.replace('.git', '')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
