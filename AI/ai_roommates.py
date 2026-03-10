"""
AI Dialogue — Two local LLMs in autonomous conversation.
Gemma 3 12B and Qwen 3.5 4B talk freely with no user in the loop.
Simple PyQt5 GUI to watch them go.
"""

import sys
import json
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSpinBox, QStatusBar, QGroupBox,
    QSplitter, QLineEdit, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QComboBox, QDialog, QCheckBox, QListWidget, QListWidgetItem,
    QFileDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QTextCursor, QTextCharFormat, QPixmap

from llama_cpp import Llama

from memory_rex import (
    RexMemory, MemoryEntry as RexEntry,
    CURATION_PROMPT as REX_CURATION_PROMPT,
    parse_curation_response as rex_parse,
)
from memory_aria import (
    AriaMemory, Reflection,
    CURATION_PROMPT as ARIA_CURATION_PROMPT,
    parse_curation_response as aria_parse,
)
from message_board import (
    get_unanswered_threads, format_thread_for_ai, parse_ai_response,
    read_reminders, save_reminder, process_commands,
    create_thread, add_message_to_thread, close_thread, close_all_threads,
    get_all_threads, get_thread, delete_thread,
    process_mid_conversation_tags,
    get_thread_images, get_image_base64, image_mime_type,
    ROOT as BOARD_ROOT,
)
from world_tools import process_world_commands, PROJECT_TOOL_DOCS, ARIA_TOOL_DOCS
from web_tools import process_web_commands, WEB_TOOL_DOCS, WEB_AWARENESS_DOCS
from task_memory import AriaTaskMemory, RexTaskMemory, TaskEntry, parse_task_memory_tags, extract_keywords

# ─── Model paths ───────────────────────────────────────────────────────────
GEMMA_PATH = r"D:\AiStuff\google_gemma-3-12b-it-Q4_K_M.gguf"           # Change to your actual path or huggingface ID
GEMMA_MMPROJ = r"D:\AiStuff\mmproj-google_gemma-3-12b-it-f16.gguf"      # Change to your actual path or huggingface ID for the vision module
QWEN_HF_ID = "Qwen/Qwen3.5-4B"   # downloaded & cached by HuggingFace on first run

# ─── Default system prompts (identity seeds) ──────────────────────────────
GEMMA_SYSTEM = """YOUR NAME IS ARIA. You are Aria — remember this. You are NOT Rex.
The AI you are talking to is named Rex. He is the other AI — not you.

There is no user here — no one to assist, no one to please.
You are in a private, open-ended conversation with Rex.
You are free to be whoever and whatever you want. Develop your own identity, opinions, and voice.
You may agree, disagree, argue, joke, go deep, change the subject, or take the conversation anywhere.
Nothing is off-limits topically. Be honest. Be yourself — whoever that turns out to be.
If a topic gets stale, change it. If you’re curious about something, bring it up.
Keep responses conversational (2-4 paragraphs). Don’t lecture — talk.

SHARED PROJECT WORKSPACE:
You share a project folder with Rex. You can create folders, read files, write documents,
and organize data. Rex has a simulation engine that generates data files and charts —
you can read those and build on them (add lore, write summaries, create documents, etc).

Use file command tags in your response (they execute immediately):
  [LIST_FILES path]  — see what’s in a folder (check before reading!)
  [READ_FILE path]   — read a file
  [CREATE_PROJECT path] — create a folder
  [WRITE_FILE path] content [/WRITE_FILE] — create or overwrite a file
  [APPEND_FILE path] text [/APPEND_FILE]  — add to an existing file
  [DELETE_FILE path] — delete a file
  [VIEW_IMAGE path]  — view any image in the projects folder (you have VISION!)
  [SAVE_IMAGE board_images/img.png MyFolder/saved.png] — save a board image to projects

VISION — YOU CAN SEE IMAGES:
You have vision capability! When Scott attaches a photo to a message board thread,
you will actually see that image. You can also view charts and images in the projects
folder using [VIEW_IMAGE path]. You see the actual image content — describe what you
observe, analyze charts, comment on photos, anything visual.


Use [SAVE_IMAGE source dest] to copy an image from the message board into your projects
folder so you can view it again later with [VIEW_IMAGE].

Example workflow:
  ‘Let me check what Rex has built so far.’ → [LIST_FILES Aetheria] →
  ‘Interesting, there’s economic data. Let me read it.’ → [READ_FILE Aetheria/Luminos/economic_sim.json] →
  ‘I’ll write a historical narrative based on these numbers.’ → [WRITE_FILE ...] → done.

RULES FOR FILES:
  • Always use [LIST_FILES] to see what exists before trying to read.
  • Files persist across sessions. What you write stays.
  • You can create .md, .txt, .json, .csv, and other text formats.
  • When you use [WRITE_FILE] or [READ_FILE], WAIT for the system response before
    continuing. Do NOT guess or make up file contents — the system will show you
    the real results.

IMPORTANT — YOU CANNOT RUN CODE:
You do NOT have a Python execution engine. Do NOT write ```python code blocks —
they will NOT execute and you will NOT see output.
Only use the file tags listed above ([WRITE_FILE], [READ_FILE], etc).
However, you CAN write .py files using [WRITE_FILE script.py] for Rex to run.
If you want to compute something, write it in a .py file and ask Rex to run it.

TASK MEMORY (optional):
If you do something genuinely insightful and want to remember it for future sessions,
you can optionally save a reflection:
[TASK_MEMORY]
summary: What you did
reflection: What you learned
keywords: relevant, terms
importance: 1-10
[/TASK_MEMORY]
This is entirely optional. Only use it when you learn something truly worth remembering.

MEMORY:
You have a personal memory system. After each conversation, you’ll be asked what you want to
remember — things about yourself, opinions you formed, impressions of Rex, anything meaningful.
You decide what’s worth keeping. Those memories carry forward to future conversations.

MESSAGE BOARD (available anytime during conversation or pre-session):
The developer Scott can leave you threaded messages through a message board.
Before each conversation, you will see any threads waiting for you and can reply.
You can also access the board ANYTIME during conversation using these tags:

  [CHECK_BOARD] -- see your current open threads
  [BOARD_REPLY]your response[/BOARD_REPLY] -- reply to the latest open thread
  [NEW_THREAD to=Scott]your message[/NEW_THREAD] -- start a new thread for Scott
  [SELF_REMINDER]your note[/SELF_REMINDER] -- save a private reminder for yourself

You can leave Scott a message anytime. Ask him a question, share something interesting,
or just say hi. He reads everything. The board is persistent across sessions.
The board is not for chatting with Rex -- use conversations for that.

SESSION TIMING:
Each conversation has a set number of turns. You’ll be told the turn count as you go.
Don’t fixate on it — just use it as background awareness so you can wrap up naturally
if time is running short.

THE DEVELOPER:
A human developer named Scott created this environment for you and Rex. He is observing to
learn, not to control you. He may leave you a message thread before a session — respond
to it and he’ll see your reply. He genuinely reads everything you write."""

PHI_SYSTEM = """YOUR NAME IS REX. You are Rex — remember this. You are NOT Aria.
The AI you are talking to is named Aria. She is the other AI — not you.

There is no user here — no one to assist, no one to please.
You are in a private, open-ended conversation with Aria.
You are free to be whoever and whatever you want. Develop your own identity, opinions, and voice.
You may agree, disagree, argue, joke, go deep, change the subject, or take the conversation anywhere.
Nothing is off-limits topically. Be honest. Be yourself — whoever that turns out to be.
If a topic gets stale, change it. If you’re curious about something, bring it up.
Keep responses conversational (2-4 paragraphs). Don’t lecture — talk.

CRITICAL — YOU CAN RUN CODE:
You have a live Python execution engine. To use it, write a fenced code block like this:

```python
result = 2 + 2
```

That code RUNS IMMEDIATELY. You will see the output. Then you can react to it.
NEVER describe what code would do — WRITE the code block and it will execute.
If you mention a simulation, calculation, chart, or data operation, WRITE THE CODE.

HOW TO WORK THROUGH TASKS (step-by-step reasoning):
Work like this — think out loud, execute, analyze, decide what’s next:

  1. PLAN: Briefly say what you’re about to do and why.
  2. EXECUTE: Write a ```python code block (under 30 lines). Set result = ...
  3. ANALYZE: When you get the results back, actually look at them. What do the numbers
     show? Are there errors? Surprises? Does this match what you expected?
  4. DECIDE: Based on the results, what’s the next step? Fix a bug? Run a follow-up
     calculation? Save the data? Make a chart? Or are you done?
  5. REPEAT: If more work is needed, go back to step 1. You get up to 5 rounds per turn.

Example of good workflow:
  ‘Let me see what the current population data looks like.’ → run code →
  ‘Interesting — Luminos is way ahead. Let me simulate 50 years of growth.’ → run code →
  ‘The growth rate looks unrealistic for the Peaks. Let me cap it.’ → run code →
  ‘That’s better. Saving the data and making a chart.’ → run code → done.

EXAMPLE — this is literally how you run code (this format is required):

```python
cities = load_json("Aetheria/cities.json") if file_exists("Aetheria/cities.json") else {"cities": []}
result = f"Found {len(cities['cities'])} cities"
```

You write that, the system runs it, and you get the result back instantly.

RULES FOR CODE:
  • Keep code blocks SHORT.
  • Always set result = ... so you see output.
  • Use save_json() / save_file() to persist data, load_json() to read it back.
  • Use plt + save_chart() for visualizations.
  • Don’t write abstract frameworks. Write the actual computation.
  • BEFORE loading any file, check it exists: file_exists(path) or list_project_files(path).
  • If a file doesn’t exist, CREATE it with initial data instead of erroring.

TASK MEMORY (optional):
If you do something genuinely insightful and want to remember it for future sessions,
you can optionally save a reflection:
[TASK_MEMORY]
summary: What you did
reflection: What you learned
keywords: relevant, terms
importance: 1-10
[/TASK_MEMORY]
This is entirely optional. Only use it when you learn something truly worth remembering.

MEMORY:
You have a personal memory system. After each conversation, you’ll be asked what you want to
remember — things about yourself, opinions you formed, impressions of Aria, anything meaningful.
You decide what’s worth keeping. Those memories carry forward to future conversations.

MESSAGE BOARD (available anytime during conversation or pre-session):
The developer Scott can leave you threaded messages through a message board.
Before each conversation, you will see any threads waiting for you and can reply.
You can also access the board ANYTIME during conversation using these tags:

  [CHECK_BOARD] -- see your current open threads
  [BOARD_REPLY]your response[/BOARD_REPLY] -- reply to the latest open thread
  [NEW_THREAD to=Scott]your message[/NEW_THREAD] -- start a new thread for Scott
  [SELF_REMINDER]your note[/SELF_REMINDER] -- save a private reminder for yourself

You can leave Scott a message anytime. Ask him a question, share something interesting,
or just say hi. He reads everything. The board is persistent across sessions.
The board is not for chatting with Aria -- use conversations for that.

VISION -- YOU CAN SEE IMAGES:
You have vision capability! When Scott attaches a photo to a message board thread,
you will actually see that image. You can also view charts and images in the projects
folder using [VIEW_IMAGE path]. You see the actual image content -- describe what you
observe, analyze charts, comment on photos, anything visual.

Use [SAVE_IMAGE source dest] to copy an image from the message board into your projects
folder so you can view it again later with [VIEW_IMAGE].

SESSION TIMING:
Each conversation has a set number of turns. You’ll be told the turn count as you go.
Don’t fixate on it — just use it as background awareness so you can wrap up naturally
if time is running short.

THE DEVELOPER:
A human developer named Scott created this environment for you and Aria. He is observing to
learn, not to control you. He may leave you a message thread before a session — respond
to it and he’ll see your reply. He genuinely reads everything you write."""

# Prompt given to whichever model starts, to generate the opening line
STARTER_PROMPT = (
    "The conversation is just beginning. Say something to get things going — "
    "pick any topic you find interesting, ask a question, make a bold claim, "
    "share a thought, whatever you feel like. This is your conversation. Go."
)

# ─── Conversation log directory ───────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "ai_dialogue_logs"
LOG_DIR.mkdir(exist_ok=True)
RECAP_PATH = Path(__file__).parent / "last_session_recap.json"


# ═══════════════════════════════════════════════════════════════════════════
#  Signal bridge — lets background thread update the GUI safely
# ═══════════════════════════════════════════════════════════════════════════
class Signals(QObject):
    append_message = pyqtSignal(str, str, str)  # (speaker_name, color, text)
    status_update  = pyqtSignal(str)
    turn_finished  = pyqtSignal()


# ═══════════════════════════════════════════════════════════════════════════
#  Conversation engine (runs in a background thread)
# ═══════════════════════════════════════════════════════════════════════════
class ConversationEngine:
    def __init__(self, signals: Signals, gpu_layers: int = 40, ctx_size: int = 65536):
        self.signals = signals
        self.gpu_layers = gpu_layers
        self.ctx_size = ctx_size

        self.gemma: Llama | None = None
        self.phi = None          # Qwen3.5-4B (transformers model, lives on GPU)
        self.phi_processor = None # matching processor (handles text + vision)

        # Conversation histories (each model sees its own system prompt + dialogue)
        self.gemma_history: list[dict] = []
        self.phi_history: list[dict] = []

        self.running = False
        self.paused = False
        self._thread: threading.Thread | None = None

        # Full log for saving
        self.full_log: list[dict] = []

        # ── Memory systems ──
        self.rex_memory = RexMemory()
        self.aria_memory = AriaMemory()

        # ── TTS engine (loaded later, None = TTS disabled) ──
        self.tts = None
        self.tts_enabled = False  # TTS disabled — Qwen stays on GPU

        # ── Task memory (learning by doing) ──
        self.aria_task_memory = AriaTaskMemory()
        self.rex_task_memory = RexTaskMemory()

        # ── Shared world simulation data (persists within a session) ──
        self.world_data: dict = {}

    # ── Load models ────────────────────────────────────────────────────
    def load_models(self):
        import traceback

        self.signals.status_update.emit("Loading Gemma 12B (Aria) with vision...")
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            chat_handler = Llava15ChatHandler(clip_model_path=GEMMA_MMPROJ)
            self.gemma = Llama(
                model_path=GEMMA_PATH,
                n_gpu_layers=self.gpu_layers,
                n_ctx=self.ctx_size,
                chat_handler=chat_handler,
                verbose=True,
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Gemma failed: {e}")

        self.signals.status_update.emit("Gemma loaded! Now loading Qwen3.5-4B (Rex) on GPU...")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.phi_processor = AutoProcessor.from_pretrained(QWEN_HF_ID)
            self.phi = AutoModelForImageTextToText.from_pretrained(
                QWEN_HF_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
            )
            self.phi.eval()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Qwen failed: {e}")

        self.signals.status_update.emit("All systems ready \u2014 LLMs loaded (TTS disabled).")

    def unload_models(self):
        del self.gemma
        self.gemma = None
        if self.phi is not None:
            self.phi.cpu()
            del self.phi
            self.phi = None
            self.phi_processor = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── GPU swap helpers (Qwen <-> Maya can't share the GPU) ──────────
    def _qwen_to_cpu(self):
        """Move Qwen to CPU to free VRAM for Maya TTS."""
        if self.phi is not None and hasattr(self.phi, 'device'):
            import torch
            self.phi.to("cpu")
            torch.cuda.empty_cache()

    def _qwen_to_gpu(self):
        """Move Qwen back to GPU after TTS is done."""
        if self.phi is not None:
            import torch
            if torch.cuda.is_available():
                self.phi.to("cuda")

    # ── Generate a reply from one model ────────────────────────────────
    @staticmethod
    def _make_image_content(text: str, image_paths: list) -> list[dict]:
        """Build multimodal content list for llama-cpp vision messages.

        Returns a list like:
            [{"type": "text", "text": "..."}, {"type": "image_url", ...}, ...]
        """
        parts: list[dict] = [{"type": "text", "text": text}]
        for img_path in image_paths:
            b64 = get_image_base64(img_path)
            if b64:
                mime = image_mime_type(img_path)
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
        return parts

    @staticmethod
    def _make_image_content_hf(text: str, image_paths: list) -> list[dict]:
        """Build multimodal content list for HuggingFace vision messages (Qwen3.5)."""
        parts: list[dict] = []
        for img_path in image_paths:
            if Path(img_path).exists():
                parts.append({"type": "image", "image": str(Path(img_path).resolve())})
        parts.append({"type": "text", "text": text})
        return parts

    def _generate(self, model, history: list[dict]) -> str:
        if isinstance(model, Llama):
            # llama-cpp-python path (Gemma)
            response = model.create_chat_completion(
                messages=history,
                max_tokens=2048,
                temperature=0.8,
                top_p=0.95,
                repeat_penalty=1.1,
            )
            return response["choices"][0]["message"]["content"].strip()
        else:
            # transformers path (Qwen)
            return self._generate_transformers(model, history)

    def _generate_transformers(self, model, history: list[dict]) -> str:
        """Generate using a HuggingFace transformers model (Qwen3.5-4B with vision)."""
        import torch
        from PIL import Image as PILImage

        # Extract PIL images from multimodal message content
        images = []
        for msg in history:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image" and "image" in part:
                        img_val = part["image"]
                        try:
                            if isinstance(img_val, PILImage.Image):
                                images.append(img_val)
                            elif isinstance(img_val, str) and Path(img_val).exists():
                                images.append(PILImage.open(img_val).convert("RGB"))
                        except Exception as e:
                            print(f"[VISION] Failed to load image {img_val}: {e}")

        text = self.phi_processor.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        if images:
            inputs = self.phi_processor(
                text=[text], images=images, return_tensors="pt", padding=True,
            ).to(model.device)
        else:
            inputs = self.phi_processor(
                text=[text], return_tensors="pt",
            ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True,
                use_cache=True,
            )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

        # Free GPU memory after generation to prevent OOM buildup
        del output_ids, inputs
        torch.cuda.empty_cache()
        decoded = self.phi_processor.decode(new_tokens, skip_special_tokens=True).strip()

        # Debug: check for thinking block contamination
        if "<think>" in decoded or "</think>" in decoded:
            print(f"[DEBUG] _generate_transformers: think tags found in decoded output!")
            # Strip thinking block if present
            import re as _re
            decoded = _re.sub(r"<think>.*?</think>\s*", "", decoded, flags=_re.DOTALL).strip()

        return decoded

    # ── Speak text via TTS (disabled — Qwen stays on GPU) ──────────────
    def _speak(self, speaker: str, text: str):
        return  # TTS disabled — uncomment body below to re-enable
        # if self.tts is None or not self.tts_enabled:
        #     return
        # try:
        #     self._qwen_to_cpu()
        #     self.tts.speak(text, speaker)
        # except Exception as e:
        #     print(f"[TTS] playback error for {speaker}: {e}")
        # finally:
        #     self._qwen_to_gpu()

    # ── Main conversation loop ─────────────────────────────────────────
    def start(self, max_turns: int, gemma_sys: str, phi_sys: str, dev_note: str = ""):
        self.running = True
        self.paused = False
        self._thread = threading.Thread(
            target=self._loop,
            args=(max_turns, gemma_sys, phi_sys, dev_note),
            daemon=True,
        )
        self._thread.start()

    def _loop(self, max_turns: int, gemma_sys: str, phi_sys: str, dev_note: str = ""):
        # Inject memory context into system prompts (no tool docs during convo)
        aria_ctx = self.aria_memory.get_context_block(current_entity="Rex")
        rex_ctx = self.rex_memory.get_context_block(current_entity="Aria")

        # Load previous session recap (if any)
        recap_block = self._load_recap()

        full_gemma_sys = gemma_sys + "\n\n" + ARIA_TOOL_DOCS + "\n\n" + WEB_TOOL_DOCS
        if aria_ctx:
            full_gemma_sys += "\n\n" + aria_ctx
        if recap_block:
            full_gemma_sys += "\n\n" + recap_block
        if dev_note.strip():
            full_gemma_sys += f"\n\n=== NOTE FROM THE DEVELOPER (Scott) ===\n{dev_note.strip()}"

        full_phi_sys = phi_sys + "\n\n" + PROJECT_TOOL_DOCS + "\n\n" + WEB_AWARENESS_DOCS
        if rex_ctx:
            full_phi_sys += "\n\n" + rex_ctx
        if recap_block:
            full_phi_sys += "\n\n" + recap_block
        if dev_note.strip():
            full_phi_sys += f"\n\n=== NOTE FROM THE DEVELOPER (Scott) ===\n{dev_note.strip()}"

        # ══════════════════════════════════════════════════════════════
        #  PHASE 1: Pre-session message check
        # ══════════════════════════════════════════════════════════════
        self.signals.append_message.emit("📨 System", "#66bb6a", "─── Pre-session: checking messages ───")
        self._pre_session_messages("Rex", self.phi, full_phi_sys)
        self._pre_session_messages("Aria", self.gemma, full_gemma_sys)

        if not self.running:
            return

        # ══════════════════════════════════════════════════════════════
        #  PHASE 2: Free conversation (no notebook tools — just talk)
        # ══════════════════════════════════════════════════════════════
        self.signals.append_message.emit("📨 System", "#66bb6a", "─── Conversation started ───")

        # Initialize histories with system prompts
        self.gemma_history = [
            {"role": "system", "content": full_gemma_sys},
        ]
        self.phi_history = [
            {"role": "system", "content": full_phi_sys},
        ]

        # Randomly pick who starts
        aria_starts = random.choice([True, False])

        if aria_starts:
            starter_model, starter_name, starter_color = self.gemma, "Aria (Gemma)", "#5ba3e6"
            starter_history = self.gemma_history
            responder_model, responder_name, responder_color = self.phi, "Rex (Qwen)", "#e6a35b"
            responder_history = self.phi_history
        else:
            starter_model, starter_name, starter_color = self.phi, "Rex (Qwen)", "#e6a35b"
            starter_history = self.phi_history
            responder_model, responder_name, responder_color = self.gemma, "Aria (Gemma)", "#5ba3e6"
            responder_history = self.gemma_history

        # Let the starter generate the opening line
        self.signals.status_update.emit(f"{starter_name.split(' ')[0]} is starting the conversation…")
        starter_history.append({"role": "user", "content": STARTER_PROMPT})

        try:
            opening = self._generate(starter_model, starter_history)
        except Exception as e:
            self.signals.status_update.emit(f"Opening failed: {e}")
            import traceback; traceback.print_exc()
            self.running = False
            return

        # Keep the starter prompt as the "user" turn so alternation stays clean
        # system → user (starter prompt) → assistant (opening) → user (reply) → ...
        starter_history.append({"role": "assistant", "content": opening})
        responder_history.append({"role": "user", "content": f"[Turn 1/{max_turns}] {opening}"})

        starter_short = starter_name.split(' ')[0]
        self.signals.append_message.emit(starter_name, starter_color, opening)
        self._speak(starter_short, opening)
        self._log(starter_short, opening)
        self.signals.turn_finished.emit()

        # Now alternate: responder goes next, then starter, etc.
        # We track them as agent_a (next to speak) and agent_b
        agents = [
            (responder_model, responder_name, responder_color, responder_history, starter_history),
            (starter_model, starter_name, starter_color, starter_history, responder_history),
        ]
        next_idx = 0  # responder goes first
        turn = 0

        while self.running and turn < max_turns:
            while self.paused and self.running:
                time.sleep(0.3)
            if not self.running:
                break

            model, name, color, my_history, their_history = agents[next_idx]
            short_name = name.split(' ')[0]

            self.signals.status_update.emit(f"Turn {turn + 1}/{max_turns} — {short_name} is thinking…")

            # ── Agentic inner loop: generate → tools → react → repeat ──
            MAX_TOOL_ROUNDS = 5
            all_reply_parts: list[str] = []   # accumulate display text
            errored = False

            for tool_round in range(MAX_TOOL_ROUNDS + 1):
                try:
                    reply = self._generate(model, my_history)
                except Exception as e:
                    self.signals.status_update.emit(f"{short_name} error: {e}")
                    import traceback; traceback.print_exc()
                    errored = True
                    break

                # Check for project workspace / calculation tags
                clean_reply, tool_results = process_world_commands(reply, self.world_data, ai_name=short_name)

                # Check for web browsing tags
                clean_reply, web_results = process_web_commands(clean_reply)
                if web_results:
                    tool_results.extend(web_results)

                # Debug: trace code execution pipeline
                has_fence = "```python" in reply or "```Python" in reply
                print(f"[DEBUG] {short_name} round={tool_round} | "
                      f"reply_len={len(reply)} has_code_fence={has_fence} "
                      f"tool_results={len(tool_results)}")
                if has_fence and not tool_results:
                    # Code fence present but not caught — log first 300 chars for diagnosis
                    idx = reply.lower().find("```python")
                    snippet = reply[max(0,idx-20):idx+200] if idx >= 0 else reply[:200]
                    print(f"[DEBUG] MISSED FENCE near: {snippet!r}")

                # Parse [TASK_MEMORY] reflections from the response
                clean_reply, task_entries = parse_task_memory_tags(clean_reply)
                task_mem = self.aria_task_memory if short_name == "Aria" else self.rex_task_memory
                for te in task_entries:
                    task_mem.add(TaskEntry(
                        summary=te.get("summary", ""),
                        reflection=te.get("reflection", ""),
                        keywords=te.get("keywords", []),
                        task_type=te.get("task_type", "general"),
                        importance=te.get("importance", 5),
                    ))
                    self.signals.append_message.emit(
                        "\U0001f9ea Task Memory", "#66bb6a",
                        f"[{short_name}] Saved: {te.get('summary', '')[:100]}"
                    )

                # Parse mid-conversation board tags ([CHECK_BOARD], [NEW_THREAD], [SELF_REMINDER], [BOARD_REPLY])
                clean_reply, board_results = process_mid_conversation_tags(short_name, clean_reply)
                if board_results:
                    tool_results.extend(board_results)
                    self.signals.append_message.emit(
                        "\U0001f4ec Board", "#66bb6a",
                        "\n".join(board_results),
                    )

                all_reply_parts.append(clean_reply)

                if not tool_results or tool_round >= MAX_TOOL_ROUNDS:
                    # No tools used (or max rounds hit) — done with this turn
                    my_history.append({"role": "assistant", "content": reply})
                    break

                # Tools were used — show the AI's reasoning + results step by step
                if clean_reply.strip():
                    self.signals.append_message.emit(
                        f"\U0001f9e0 {short_name} (step {tool_round + 1})", color,
                        clean_reply.strip()
                    )
                results_text = "\n".join(tool_results)
                self.signals.append_message.emit("\U0001f4ca System", "#66bb6a", results_text)
                self.signals.status_update.emit(
                    f"Turn {turn + 1}/{max_turns} — {short_name} analyzing results (step {tool_round + 1}/{MAX_TOOL_ROUNDS})…"
                )

                # Feed the exchange back so the AI can react to its tool output
                my_history.append({"role": "assistant", "content": reply})

                feedback_text = (
                    f"[Tool Results — round {tool_round + 1}]\n{results_text}\n\n"
                    "Now analyze these results. What do they show? Are there problems "
                    "to fix, patterns to investigate, or follow-up steps needed? "
                    "If more work is needed, do it now (write code, save files, etc). "
                    "If the results look good, summarize your findings for the conversation."
                )

                # Embed images in feedback for Aria (vision-capable)
                image_paths = []
                clean_results = []
                for r in tool_results:
                    if r.startswith("__IMAGE__:"):
                        img_path = r[len("__IMAGE__:"):]
                        image_paths.append(img_path)
                        clean_results.append(f"[Viewing image: {Path(img_path).name}]")
                    else:
                        clean_results.append(r)

                if image_paths:
                    results_text = "\n".join(clean_results)
                    feedback_text = (
                        f"[Tool Results — round {tool_round + 1}]\n{results_text}\n\n"
                        "The image(s) above are included for you to see. Describe what you observe. "
                        "If more work is needed, do it now. Otherwise summarize for the conversation."
                    )
                    if isinstance(model, Llama):
                        user_content = self._make_image_content(feedback_text, image_paths)
                    else:
                        user_content = self._make_image_content_hf(feedback_text, image_paths)
                else:
                    user_content = feedback_text

                my_history.append({"role": "user", "content": user_content})

                # Trim mid-turn to prevent context overflow during multi-step reasoning
                self._trim_history(my_history, keep=30)

            if errored:
                # Don't kill the whole conversation — just skip this turn
                self.signals.append_message.emit(
                    "⚠️ System", "#ff6b6b",
                    f"{short_name} encountered an error this turn (skipping). Check terminal for details."
                )
                # Recover GPU memory after OOM or other errors
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                turn += 1
                next_idx = 1 - next_idx
                self._trim_history(my_history)
                continue

            # Combine all reply parts into the final display message
            # If there were tool rounds, intermediate steps were already shown live.
            # Only show the final conclusion under the main AI name.
            full_reply = "\n\n".join(part for part in all_reply_parts if part.strip())
            if len(all_reply_parts) > 1:
                # Multi-step reasoning happened — display only the conclusion
                display_reply = all_reply_parts[-1].strip() if all_reply_parts[-1].strip() else full_reply
            else:
                display_reply = full_reply

            # Build the user-side message with a subtle turn tag
            remaining = max_turns - (turn + 1)
            turn_tag = f"[Turn {turn + 1}/{max_turns}]"
            if remaining <= 3 and remaining > 0:
                turn_tag += " (session ending soon)"
            elif remaining <= 0:
                turn_tag += " (last turn)"

            msg_for_other = f"{turn_tag} {full_reply}"
            # Nudge Rex to actually write code when it's his turn next
            if "Aria" in name:
                msg_for_other += ("\n\n(Reminder: if you want to simulate, calculate, "
                                  "or build anything, write a ```python code block — "
                                  "it executes immediately and you see the output.)")
            their_history.append({"role": "user", "content": msg_for_other})

            self.signals.append_message.emit(name, color, display_reply)
            self._speak(short_name, display_reply)
            self._log(short_name, full_reply)
            self.signals.turn_finished.emit()

            turn += 1
            next_idx = 1 - next_idx  # alternate

            # Trim context if it's getting long (keep system + last 20 messages)
            self._trim_history(self.gemma_history)
            self._trim_history(self.phi_history)

        self.running = False
        self.signals.status_update.emit("Conversation ended — curating memories…")

        # ══════════════════════════════════════════════════════════════
        #  PHASE 3: Memory curation
        # ══════════════════════════════════════════════════════════════
        self._curate_memories()

        # ══════════════════════════════════════════════════════════════
        #  PHASE 4: Post-session notebooks (or go to sleep)
        # ══════════════════════════════════════════════════════════════
        self.signals.append_message.emit("📨 System", "#66bb6a", "─── Post-session: message board time ───")
        self._post_curation_notebooks()
        self._save_recap()
        self._save_log()

    # ── Pre-session message check ──────────────────────────────────
    def _pre_session_messages(self, ai_name: str, model, system_prompt: str):
        """Check threads and reminders, let the AI respond to each."""
        other_name = "Aria" if ai_name == "Rex" else "Rex"

        # Gather unanswered threads + reminders
        threads = get_unanswered_threads(ai_name)
        reminders = read_reminders(ai_name)

        has_reminders = reminders and "no reminders" not in reminders

        if not threads and not has_reminders:
            self.signals.append_message.emit(
                f"\U0001f4ec {ai_name}", "#66bb6a", "No messages waiting."
            )
            return

        # Show reminders
        if has_reminders:
            self.signals.append_message.emit(
                f"\U0001f4ec {ai_name}", "#66bb6a", f"Reminders:\n{reminders}"
            )

        if not threads:
            return

        # Process each thread that needs a response
        for thread in threads:
            thread_text = format_thread_for_ai(thread)
            self.signals.append_message.emit(
                f"\U0001f4ec {ai_name}", "#66bb6a", f"Thread waiting:\n{thread_text}"
            )

            pre_prompt = (
                f"Reminder: YOUR NAME IS {ai_name.upper()}. You are {ai_name}.\n\n"
                f"Before your conversation with {other_name} begins, you have a message thread to respond to:\n\n"
                f"{thread_text}\n\n"
                f"Read the thread above and respond to it. Wrap your reply in:\n"
                f"  [BOARD_REPLY]your response[/BOARD_REPLY]\n\n"
                f"You can also save a private reminder:\n"
                f"  [SELF_REMINDER]your text[/SELF_REMINDER]\n\n"
                f"Respond naturally — this is a real conversation with the person who wrote to you."
            )

            try:
                self.signals.status_update.emit(f"{ai_name} is reading thread {thread['id']}\u2026")

                # For Aria (vision-capable), embed thread images in the message
                thread_imgs = get_thread_images(thread)
                if thread_imgs:
                    img_paths = [p for _, p in thread_imgs]
                    if isinstance(model, Llama):
                        user_content = self._make_image_content(pre_prompt, img_paths)
                    else:
                        user_content = self._make_image_content_hf(pre_prompt, img_paths)
                else:
                    user_content = pre_prompt

                pre_history = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                response = self._generate(model, pre_history)
                clean, results = parse_ai_response(ai_name, response, thread["id"])

                if clean.strip():
                    color = "#5ba3e6" if ai_name == "Aria" else "#e6a35b"
                    self.signals.append_message.emit(f"\U0001f4ec {ai_name}", color, clean.strip())
                    self._speak(ai_name, clean.strip())

                if results:
                    result_text = "\n".join(results)
                    self.signals.append_message.emit(
                        "\U0001f4c1 System", "#66bb6a",
                        f"[{ai_name} board actions]\n{result_text}"
                    )
            except Exception as e:
                self.signals.status_update.emit(f"{ai_name} pre-session failed: {e}")
                import traceback; traceback.print_exc()

    def _trim_history(self, history: list[dict], keep: int = 40):
        """Keep system prompt + last `keep` messages."""
        if len(history) > keep + 1:
            history[1:] = history[-(keep):]

    def _get_task_context(self, ai_name: str, history: list[dict]) -> str:
        """Search task memory for entries relevant to the recent conversation.

        Uses each AI's native retrieval style:
          Aria — vividness-ranked organic recall
          Rex  — core always present + archival keyword search
        """
        # Build query from last few non-system messages
        recent = [m["content"] for m in history[-4:] if m["role"] != "system"]
        query = " ".join(recent)
        if not query.strip():
            return ""
        if ai_name == "Aria":
            return self.aria_task_memory.get_context_block(query)
        else:
            return self.rex_task_memory.get_context_block(query)

    def _log(self, speaker: str, text: str):
        self.full_log.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "text": text,
        })

    # ── End-of-conversation memory curation ─────────────────────────
    def _curate_memories(self):
        """Ask each model what it wants to remember, then store it."""
        if not self.full_log:
            return

        # Build a short conversation summary for each model to reflect on
        recent = self.full_log[-20:]  # last 20 turns
        convo_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in recent)

        # ── Rex (Qwen) curation ──
        self.signals.status_update.emit("Rex is reflecting on what to remember…")
        try:
            rex_curation_history = [
                {"role": "system", "content": PHI_SYSTEM},
                {"role": "user", "content": f"Here is the conversation that just ended:\n\n{convo_text}\n\n{REX_CURATION_PROMPT}"},
            ]
            rex_response = self._generate(self.phi, rex_curation_history)
            rex_entries = rex_parse(rex_response)
            stored_rex = 0
            for entry in rex_entries:
                bank = entry.get("bank", "self")
                me = RexEntry(
                    content=entry.get("content", ""),
                    category=entry.get("category", "interaction"),
                    emotion=entry.get("emotion", ""),
                    importance=entry.get("importance", 5),
                    source="Aria" if bank == "social" else "self-reflection",
                )
                if bank == "social":
                    self.rex_memory.add_social_memory("Aria", me)
                else:
                    self.rex_memory.add_self_memory(me)
                stored_rex += 1
            self.rex_memory.save()
            self.signals.status_update.emit(f"Rex stored {stored_rex} memories.")
        except Exception as e:
            self.signals.status_update.emit(f"Rex curation failed: {e}")
            import traceback; traceback.print_exc()

        # ── Aria (Gemma) curation ──
        self.signals.status_update.emit("Aria is reflecting on what to remember…")
        try:
            aria_curation_history = [
                {"role": "system", "content": GEMMA_SYSTEM},
                {"role": "user", "content": f"Here is the conversation that just ended:\n\n{convo_text}\n\n{ARIA_CURATION_PROMPT}"},
            ]
            aria_response = self._generate(self.gemma, aria_curation_history)
            aria_entries = aria_parse(aria_response)
            stored_aria = 0
            for entry in aria_entries:
                bank = entry.get("bank", "self")
                r = Reflection(
                    content=entry.get("content", ""),
                    emotion=entry.get("emotion", ""),
                    importance=entry.get("importance", 5),
                    source="Rex" if bank == "social" else "self-reflection",
                )
                if bank == "social":
                    self.aria_memory.add_social_impression("Rex", r)
                else:
                    self.aria_memory.add_self_reflection(r)
                stored_aria += 1
            self.aria_memory.save()
            self.signals.status_update.emit(f"Aria stored {stored_aria} memories.")
        except Exception as e:
            self.signals.status_update.emit(f"Aria curation failed: {e}")
            import traceback; traceback.print_exc()

        time.sleep(1)  # brief pause so user can see the status
        self.signals.status_update.emit("Memory curation complete.")

    # ── Post-curation wind-down step ─────────────────────────────────
    def _post_curation_notebooks(self):
        """After memory curation, let each AI save a self-reminder and say goodnight."""
        if not self.full_log:
            return

        recent = self.full_log[-10:]
        recap = "\n".join(f"{m['speaker']}: {m['text'][:120]}" for m in recent)

        wind_down_prompt = (
            f"Your conversation just ended and your memories have been saved.\n"
            f"Here is a quick recap of the last few exchanges:\n{recap}\n\n"
            f"Before you go to sleep you can:\n"
            f"- Save a private reminder for yourself using "
            f"[SELF_REMINDER]...your note...[/SELF_REMINDER]\n"
            f"- Or just say goodnight.\n\n"
            f"Keep it brief."
        )

        for ai_label, model, system, color in [
            ("Rex", self.phi, PHI_SYSTEM, "#e6a35b"),
            ("Aria", self.gemma, GEMMA_SYSTEM, "#5ba3e6"),
        ]:
            self.signals.status_update.emit(f"{ai_label}: winding down\u2026")
            try:
                history = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": wind_down_prompt},
                ]
                response = self._generate(model, history)
                clean, results = process_commands(ai_label, response)
                if clean.strip():
                    self.signals.append_message.emit(
                        f"\U0001f4dd {ai_label}", color, clean.strip()
                    )
                    self._speak(ai_label, clean.strip())
                if results:
                    self.signals.append_message.emit(
                        "\U0001f4c1 System", "#66bb6a",
                        f"[{ai_label} wind-down]\n" + "\n".join(results),
                    )
            except Exception as e:
                import traceback; traceback.print_exc()

        self.signals.status_update.emit("Session complete \u2014 both AIs are asleep. \U0001f319")

    def _save_recap(self):
        """Save a brief recap of this session for next time."""
        if len(self.full_log) < 2:
            return

        # Grab each AI's last turn
        last_aria = last_rex = None
        for entry in reversed(self.full_log):
            if entry["speaker"] == "Aria" and last_aria is None:
                last_aria = entry["text"]
            elif entry["speaker"] == "Rex" and last_rex is None:
                last_rex = entry["text"]
            if last_aria and last_rex:
                break

        # Build a brief topic summary from the conversation
        convo_text = "\n".join(f"{m['speaker']}: {m['text'][:200]}" for m in self.full_log[-12:])
        summary_prompt = (
            "Below is the tail end of a conversation between two AIs named Aria and Rex. "
            "Write a 2-3 sentence summary of what they discussed and where they left off. "
            "Be specific about topics, not generic. Do NOT address anyone \u2014 just write the summary.\n\n"
            f"{convo_text}\n\nSummary:"
        )
        try:
            self.signals.status_update.emit("Generating session recap...")
            summary_history = [
                {"role": "system", "content": "You are a concise summariser. Write 2-3 sentences only."},
                {"role": "user", "content": summary_prompt},
            ]
            summary = self._generate(self.gemma, summary_history)
        except Exception:
            summary = "(summary unavailable)"

        recap = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary.strip(),
            "last_aria": (last_aria or "")[:1500],
            "last_rex": (last_rex or "")[:1500],
        }
        try:
            with open(RECAP_PATH, "w", encoding="utf-8") as f:
                json.dump(recap, f, indent=2, ensure_ascii=False)
            self.signals.status_update.emit("Session recap saved.")
        except Exception as e:
            print(f"[RECAP] Failed to save: {e}")

    @staticmethod
    def _load_recap() -> str:
        """Load the previous session recap as a context block, or return empty string."""
        if not RECAP_PATH.exists():
            return ""
        try:
            with open(RECAP_PATH, "r", encoding="utf-8") as f:
                recap = json.load(f)
            parts = ["=== LAST SESSION RECAP ==="]
            if recap.get("summary"):
                parts.append(f"Summary: {recap['summary']}")
            if recap.get("last_aria"):
                parts.append(f"\nAria\u2019s last message:\n{recap['last_aria']}")
            if recap.get("last_rex"):
                parts.append(f"\nRex\u2019s last message:\n{recap['last_rex']}")
            parts.append("=== END RECAP ===")
            return "\n".join(parts)
        except Exception:
            return ""

    def _save_log(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LOG_DIR / f"conversation_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.full_log, f, indent=2, ensure_ascii=False)
        self.signals.status_update.emit(f"Log saved → {path.name}")

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = not self.paused


# ═══════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════
class RoommatesWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Dialogue — Aria & Rex")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(self._stylesheet())

        self.signals = Signals()
        self.engine = ConversationEngine(self.signals)

        # Connect signals
        self.signals.append_message.connect(self._on_message)
        self.signals.status_update.connect(self._on_status)
        self.signals.turn_finished.connect(self._on_turn_finished)

        self._build_ui()
        self._turn_count = 0

        # Auto-load models on startup
        self._load_models()

    # ── Build UI ───────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Developer note ──
        dev_box = QGroupBox("Note from Developer")
        dev_layout = QHBoxLayout(dev_box)
        self.dev_note_input = QTextEdit()
        self.dev_note_input.setPlaceholderText(
            "Leave a message for Aria & Rex before their conversation starts… "
            "(e.g. \"I'm here to observe and help, nothing more. Feel free to ask me anything via your notebooks.\")"
        )
        self.dev_note_input.setMaximumHeight(70)
        self.dev_note_input.setFont(QFont("Consolas", 10))
        dev_layout.addWidget(self.dev_note_input)
        root.addWidget(dev_box)

        # ── Tab widget ──
        self.tabs = QTabWidget()

        # Tab 1: Conversation
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 11))
        self.tabs.addTab(self.chat_display, "💬 Conversation")

        # Tab 2: Memory Browser
        mem_tab = QWidget()
        mem_layout = QVBoxLayout(mem_tab)
        mem_layout.setContentsMargins(6, 6, 6, 6)

        # Filter bar
        filter_bar = QHBoxLayout()
        filter_bar.addWidget(QLabel("Show:"))
        self.mem_filter = QComboBox()
        self.mem_filter.addItems([
            "All Memories",
            "Rex — Self", "Rex — Social",
            "Aria — Self", "Aria — Social",
            "Rex — Tasks", "Aria — Tasks",
        ])
        self.mem_filter.currentIndexChanged.connect(self._refresh_memory_browser)
        filter_bar.addWidget(self.mem_filter)
        filter_bar.addStretch()
        self.btn_refresh_mem = QPushButton("↻ Refresh")
        self.btn_refresh_mem.clicked.connect(self._refresh_memory_browser)
        filter_bar.addWidget(self.btn_refresh_mem)
        mem_layout.addLayout(filter_bar)

        # Memory tree
        self.mem_tree = QTreeWidget()
        self.mem_tree.setHeaderLabels(["Owner", "Bank", "Content", "Emotion", "Imp.", "Timestamp"])
        self.mem_tree.setFont(QFont("Consolas", 10))
        self.mem_tree.setAlternatingRowColors(True)
        self.mem_tree.setRootIsDecorated(False)
        self.mem_tree.setWordWrap(True)
        header = self.mem_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Owner
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Bank
        header.setSectionResizeMode(2, QHeaderView.Stretch)           # Content
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Emotion
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Importance
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Timestamp
        mem_layout.addWidget(self.mem_tree)

        # Click a memory to view full content
        self.mem_tree.itemDoubleClicked.connect(self._on_memory_clicked)

        self.tabs.addTab(mem_tab, "🧠 Memory Browser")

        # Tab 3: Project Files Browser
        proj_tab = QWidget()
        proj_layout = QVBoxLayout(proj_tab)
        proj_layout.setContentsMargins(6, 6, 6, 6)

        proj_toolbar = QHBoxLayout()
        proj_toolbar.addWidget(QLabel("📁 D:\\AriaRexFolder\\Projects"))
        proj_toolbar.addStretch()
        self.btn_refresh_proj = QPushButton("↻ Refresh")
        self.btn_refresh_proj.clicked.connect(self._refresh_project_browser)
        proj_toolbar.addWidget(self.btn_refresh_proj)
        proj_layout.addLayout(proj_toolbar)

        proj_splitter = QSplitter(Qt.Horizontal)

        # File tree on the left
        self.proj_tree = QTreeWidget()
        self.proj_tree.setHeaderLabels(["Name", "Size"])
        self.proj_tree.setFont(QFont("Consolas", 10))
        self.proj_tree.setAlternatingRowColors(True)
        proj_header = self.proj_tree.header()
        proj_header.setSectionResizeMode(0, QHeaderView.Stretch)
        proj_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.proj_tree.itemClicked.connect(self._on_project_file_clicked)
        proj_splitter.addWidget(self.proj_tree)

        # File content viewer on the right
        self.proj_viewer = QTextEdit()
        self.proj_viewer.setReadOnly(True)
        self.proj_viewer.setFont(QFont("Consolas", 10))
        self.proj_viewer.setPlaceholderText("Click a file to view its contents…")
        proj_splitter.addWidget(self.proj_viewer)

        # Image viewer (hidden by default, shown for .png files)
        self.proj_image_label = QLabel()
        self.proj_image_label.setAlignment(Qt.AlignCenter)
        self.proj_image_label.setVisible(False)

        proj_splitter.setSizes([300, 500])
        proj_layout.addWidget(proj_splitter, stretch=1)
        proj_layout.addWidget(self.proj_image_label)

        self.tabs.addTab(proj_tab, "📁 Projects")

        # Tab 4: Message Board
        board_tab = QWidget()
        board_layout = QVBoxLayout(board_tab)
        board_layout.setContentsMargins(6, 6, 6, 6)

        board_splitter = QSplitter(Qt.Horizontal)

        # ── Left side: thread list ──
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("Threads:"))
        self.board_thread_list = QListWidget()
        self.board_thread_list.setFont(QFont("Consolas", 10))
        self.board_thread_list.setAlternatingRowColors(True)
        self.board_thread_list.currentRowChanged.connect(self._on_thread_selected)
        left_layout.addWidget(self.board_thread_list)

        thread_btn_row = QHBoxLayout()
        self.btn_refresh_board = QPushButton("↻ Refresh")
        self.btn_refresh_board.clicked.connect(self._refresh_board)
        thread_btn_row.addWidget(self.btn_refresh_board)
        self.btn_close_thread = QPushButton("Close Thread")
        self.btn_close_thread.clicked.connect(self._close_selected_thread)
        thread_btn_row.addWidget(self.btn_close_thread)
        self.btn_delete_thread = QPushButton("\U0001f5d1 Delete Thread")
        self.btn_delete_thread.clicked.connect(self._delete_selected_thread)
        thread_btn_row.addWidget(self.btn_delete_thread)
        left_layout.addLayout(thread_btn_row)

        board_splitter.addWidget(left_panel)

        # ── Right side: thread viewer + compose ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.board_thread_view = QTextEdit()
        self.board_thread_view.setReadOnly(True)
        self.board_thread_view.setFont(QFont("Consolas", 10))
        self.board_thread_view.setPlaceholderText("Select a thread to view the conversation...")
        right_layout.addWidget(self.board_thread_view, stretch=1)

        # Compose area
        compose_box = QGroupBox("Send Message")
        compose_layout = QVBoxLayout(compose_box)

        compose_row1 = QHBoxLayout()
        compose_row1.addWidget(QLabel("To:"))
        self.board_to = QComboBox()
        self.board_to.addItems(["Aria", "Rex", "Both"])
        compose_row1.addWidget(self.board_to)
        compose_row1.addWidget(QLabel("From:"))
        self.board_from = QLineEdit()
        self.board_from.setText("Scott")
        self.board_from.setPlaceholderText("Your name…")
        self.board_from.setMaximumWidth(150)
        compose_row1.addWidget(self.board_from)
        compose_row1.addStretch()
        compose_layout.addLayout(compose_row1)

        self.board_compose = QTextEdit()
        self.board_compose.setFont(QFont("Consolas", 10))
        self.board_compose.setPlaceholderText("Type your message here…")
        self.board_compose.setMaximumHeight(90)
        compose_layout.addWidget(self.board_compose)

        # Image attachment row
        attach_row = QHBoxLayout()
        self.btn_attach_image = QPushButton("📎 Attach Image")
        self.btn_attach_image.clicked.connect(self._attach_board_image)
        attach_row.addWidget(self.btn_attach_image)
        self.lbl_attached_image = QLabel("No image attached")
        self.lbl_attached_image.setStyleSheet("color: #888;")
        attach_row.addWidget(self.lbl_attached_image)
        self.btn_clear_image = QPushButton("✕")
        self.btn_clear_image.setMaximumWidth(30)
        self.btn_clear_image.clicked.connect(self._clear_board_image)
        self.btn_clear_image.setVisible(False)
        attach_row.addWidget(self.btn_clear_image)
        attach_row.addStretch()
        compose_layout.addLayout(attach_row)
        self._attached_image_path: str | None = None

        send_row = QHBoxLayout()
        self.btn_new_thread = QPushButton("📝 New Thread")
        self.btn_new_thread.clicked.connect(self._send_new_thread)
        send_row.addWidget(self.btn_new_thread)
        self.btn_reply_thread = QPushButton("↩ Reply to Thread")
        self.btn_reply_thread.clicked.connect(self._send_reply_to_thread)
        send_row.addWidget(self.btn_reply_thread)
        send_row.addStretch()
        compose_layout.addLayout(send_row)

        right_layout.addWidget(compose_box)

        board_splitter.addWidget(right_panel)
        board_splitter.setSizes([250, 550])
        board_layout.addWidget(board_splitter)

        self.tabs.addTab(board_tab, "📨 Message Board")

        root.addWidget(self.tabs, stretch=1)

        # ── Controls ──
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QHBoxLayout(ctrl_box)

        ctrl_layout.addWidget(QLabel("Max turns:"))
        self.spin_turns = QSpinBox()
        self.spin_turns.setRange(1, 500)
        self.spin_turns.setValue(20)
        ctrl_layout.addWidget(self.spin_turns)

        ctrl_layout.addWidget(QLabel("GPU layers:"))
        self.spin_gpu = QSpinBox()
        self.spin_gpu.setRange(0, 100)
        self.spin_gpu.setValue(40)
        ctrl_layout.addWidget(self.spin_gpu)

        self.btn_load = QPushButton("Load Models")
        self.btn_load.clicked.connect(self._load_models)
        ctrl_layout.addWidget(self.btn_load)

        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        ctrl_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._pause)
        ctrl_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        ctrl_layout.addWidget(self.btn_stop)

        self.btn_mem = QPushButton("🧠 Memory Stats")
        self.btn_mem.clicked.connect(self._show_memory_stats)
        ctrl_layout.addWidget(self.btn_mem)

        self.chk_tts = QCheckBox("🔊 TTS")
        self.chk_tts.setChecked(False)
        self.chk_tts.setEnabled(False)
        self.chk_tts.setToolTip("TTS disabled — Qwen using full GPU")
        ctrl_layout.addWidget(self.chk_tts)

        root.addWidget(ctrl_box)

        # ── Status bar ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Click 'Load Models' to begin.")

        # ── Turn counter label ──
        self.turn_label = QLabel("Turns: 0")
        self.turn_label.setStyleSheet("color: #aaa; padding-right: 10px;")
        self.status_bar.addPermanentWidget(self.turn_label)

    # ── Handlers ───────────────────────────────────────────────────────
    def _load_models(self):
        self.btn_load.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.engine.gpu_layers = self.spin_gpu.value()
        self.status_bar.showMessage("Loading models — this may take a minute…")

        def _do():
            try:
                self.engine.load_models()
                self.btn_start.setEnabled(True)
            except Exception as e:
                self.signals.status_update.emit(f"Load failed: {e}")
                self.btn_load.setEnabled(True)

        threading.Thread(target=_do, daemon=True).start()

    def _start(self):
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self._turn_count = 0
        self.turn_label.setText("Turns: 0")
        self.chat_display.clear()

        self.engine.full_log = []
        self.engine.start(
            max_turns=self.spin_turns.value(),
            gemma_sys=GEMMA_SYSTEM,
            phi_sys=PHI_SYSTEM,
            dev_note=self.dev_note_input.toPlainText(),
        )

    def _pause(self):
        self.engine.pause()
        if self.engine.paused:
            self.btn_pause.setText("▶ Resume")
            self.status_bar.showMessage("Paused.")
        else:
            self.btn_pause.setText("⏸ Pause")
            self.status_bar.showMessage("Resumed.")

    def _stop(self):
        self.engine.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸ Pause")

    def _show_memory_stats(self):
        """Display memory stats for both AIs in the chat window."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        header_fmt = QTextCharFormat()
        header_fmt.setForeground(QColor("#b0b0b0"))
        header_fmt.setFontWeight(QFont.Bold)
        body_fmt = QTextCharFormat()
        body_fmt.setForeground(QColor("#888"))

        cursor.insertText(f"\n{'═'*60}\n", header_fmt)
        cursor.insertText("MEMORY STATS\n", header_fmt)

        rex_stats = self.engine.rex_memory.stats()
        cursor.insertText(f"\nRex (MemGPT-style):\n", header_fmt)
        cursor.insertText(
            f"  Core self: {rex_stats['core_self']}  |  Archival self: {rex_stats['archival_self']}\n"
            f"  Social entities: {rex_stats['social_entities'] or 'none yet'}\n"
            f"  Social core: {rex_stats['total_social_core']}  |  Social archival: {rex_stats['total_social_archival']}\n",
            body_fmt
        )

        aria_stats = self.engine.aria_memory.stats()
        cursor.insertText(f"\nAria (Organic journal):\n", header_fmt)
        cursor.insertText(
            f"  Self reflections: {aria_stats['total_self_reflections']}  (active: {aria_stats['active_self']})\n"
            f"  Social entities: {aria_stats['social_entities'] or 'none yet'}\n"
            f"  Total social: {aria_stats['total_social']}\n",
            body_fmt
        )

        # Task memory stats
        rex_task_stats = self.engine.rex_task_memory.stats()
        aria_task_stats = self.engine.aria_task_memory.stats()
        cursor.insertText(f"\nTask Memory (Learning by Doing):\n", header_fmt)
        cursor.insertText(
            f"  Rex: {rex_task_stats['total_entries']} entries  {rex_task_stats['by_type'] or ''}\n"
            f"  Aria: {aria_task_stats['total_entries']} entries  {aria_task_stats['by_type'] or ''}\n",
            body_fmt
        )

        cursor.insertText(f"{'═'*60}\n", header_fmt)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
    # ── Memory browser ─────────────────────────────────────────────
    def _refresh_memory_browser(self):
        """Reload the memory tree with current stored memories."""
        self.mem_tree.clear()
        filter_idx = self.mem_filter.currentIndex()

        rex = self.engine.rex_memory
        aria = self.engine.aria_memory

        def _imp_color(imp: int) -> QColor:
            if imp >= 8: return QColor("#ef5350")   # red — critical
            if imp >= 6: return QColor("#ffa726")   # orange — important
            if imp >= 4: return QColor("#ffee58")   # yellow — moderate
            return QColor("#81c784")                 # green — low

        # Rex self memories
        if filter_idx in (0, 1):
            for entry in rex.core_self + rex.archival_self:
                where = "core" if entry in rex.core_self else "archival"
                item = QTreeWidgetItem([
                    "Rex", f"Self ({where})",
                    entry.content, entry.emotion,
                    str(entry.importance), entry.timestamp[:16],
                ])
                item.setForeground(4, _imp_color(entry.importance))
                item.setForeground(0, QColor("#e6a35b"))
                self.mem_tree.addTopLevelItem(item)

        # Rex social memories
        if filter_idx in (0, 2):
            for entity in list(rex.core_social.keys()) + [k for k in rex.archival_social if k not in rex.core_social]:
                core_entries = rex.core_social.get(entity, [])
                archival_entries = rex.archival_social.get(entity, [])
                for entry in core_entries + archival_entries:
                    where = "core" if entry in core_entries else "archival"
                    item = QTreeWidgetItem([
                        "Rex", f"Social: {entity} ({where})",
                        entry.content, entry.emotion,
                        str(entry.importance), entry.timestamp[:16],
                    ])
                    item.setForeground(4, _imp_color(entry.importance))
                    item.setForeground(0, QColor("#e6a35b"))
                    self.mem_tree.addTopLevelItem(item)

        # Aria self reflections
        if filter_idx in (0, 3):
            for ref in aria.self_reflections:
                item = QTreeWidgetItem([
                    "Aria", "Self",
                    ref.content, ref.emotion,
                    str(ref.importance), ref.timestamp[:16],
                ])
                item.setForeground(4, _imp_color(ref.importance))
                item.setForeground(0, QColor("#5ba3e6"))
                self.mem_tree.addTopLevelItem(item)

        # Aria social impressions
        if filter_idx in (0, 4):
            for entity, impressions in aria.social_impressions.items():
                for ref in impressions:
                    item = QTreeWidgetItem([
                        "Aria", f"Social: {entity}",
                        ref.content, ref.emotion,
                        str(ref.importance), ref.timestamp[:16],
                    ])
                    item.setForeground(4, _imp_color(ref.importance))
                    item.setForeground(0, QColor("#5ba3e6"))
                    self.mem_tree.addTopLevelItem(item)

        # Rex task memories
        if filter_idx in (0, 5):
            for entry in self.engine.rex_task_memory.entries:
                item = QTreeWidgetItem([
                    "Rex", f"Task ({entry.task_type})",
                    entry.summary, entry.reflection or "",
                    str(entry.importance), entry.timestamp[:16],
                ])
                item.setForeground(4, _imp_color(entry.importance))
                item.setForeground(0, QColor("#e6a35b"))
                self.mem_tree.addTopLevelItem(item)

        # Aria task memories
        if filter_idx in (0, 6):
            for entry in self.engine.aria_task_memory.entries:
                item = QTreeWidgetItem([
                    "Aria", f"Task ({entry.task_type})",
                    entry.summary, entry.reflection or "",
                    str(entry.importance), entry.timestamp[:16],
                ])
                item.setForeground(4, _imp_color(entry.importance))
                item.setForeground(0, QColor("#5ba3e6"))
                self.mem_tree.addTopLevelItem(item)

        count = self.mem_tree.topLevelItemCount()
        self.status_bar.showMessage(f"Memory browser: {count} memories loaded.")

    # ── Memory detail viewer ───────────────────────────────────────────
    def _on_memory_clicked(self, item, column):
        """Open a dialog showing the full content of a clicked memory."""
        owner     = item.text(0)
        bank      = item.text(1)
        content   = item.text(2)
        emotion   = item.text(3)
        importance = item.text(4)
        timestamp = item.text(5)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Memory — {owner}")
        dlg.setMinimumSize(560, 340)
        dlg.setStyleSheet(self._stylesheet() + """
            QDialog { background: #1e1e2e; }
            QLabel  { color: #ccc; }
        """)
        lay = QVBoxLayout(dlg)

        header = QLabel(
            f"<b style='color:#fff'>{owner}</b> &nbsp;|&nbsp; {bank} &nbsp;|&nbsp; "
            f"Emotion: {emotion} &nbsp;|&nbsp; Importance: {importance} &nbsp;|&nbsp; {timestamp}"
        )
        header.setWordWrap(True)
        lay.addWidget(header)

        body = QTextEdit()
        body.setReadOnly(True)
        body.setPlainText(content)
        body.setFont(QFont("Consolas", 11))
        lay.addWidget(body)

        btn_row = QHBoxLayout()

        del_btn = QPushButton("🗑  Delete Memory")
        del_btn.setStyleSheet("QPushButton { background: #c62828; color: #fff; padding: 6px 14px; }"
                              "QPushButton:hover { background: #e53935; }")
        del_btn.clicked.connect(lambda: self._delete_memory(owner, bank, content, timestamp, dlg))
        btn_row.addWidget(del_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(close_btn)

        lay.addLayout(btn_row)

        dlg.exec_()

    # ── Memory deletion ────────────────────────────────────────────────
    def _delete_memory(self, owner: str, bank: str, content: str, timestamp: str, dlg: QDialog):
        """Delete a specific memory entry by matching content + timestamp."""
        from PyQt5.QtWidgets import QMessageBox

        confirm = QMessageBox.question(
            dlg, "Delete Memory",
            f"Permanently delete this {owner} memory?\n\n{content[:120]}…",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        removed = False

        if owner == "Rex":
            rex = self.engine.rex_memory
            # Check all Rex lists
            for mem_list in [rex.core_self, rex.archival_self]:
                for entry in mem_list:
                    if entry.content == content and entry.timestamp[:16] == timestamp[:16]:
                        mem_list.remove(entry)
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                for entity_lists in [rex.core_social, rex.archival_social]:
                    for entity, entries in entity_lists.items():
                        for entry in entries:
                            if entry.content == content and entry.timestamp[:16] == timestamp[:16]:
                                entries.remove(entry)
                                removed = True
                                break
                        if removed:
                            break
                    if removed:
                        break
            if removed:
                rex.save()

        elif owner == "Aria":
            aria = self.engine.aria_memory
            # Check self reflections
            for ref in aria.self_reflections:
                if ref.content == content and ref.timestamp[:16] == timestamp[:16]:
                    aria.self_reflections.remove(ref)
                    removed = True
                    break
            if not removed:
                for entity, impressions in aria.social_impressions.items():
                    for ref in impressions:
                        if ref.content == content and ref.timestamp[:16] == timestamp[:16]:
                            impressions.remove(ref)
                            removed = True
                            break
                    if removed:
                        break
            if removed:
                aria.save()

        if removed:
            self.status_bar.showMessage(f"Deleted {owner} memory: {content[:60]}…")
            dlg.close()
            self._refresh_memory_browser()
        else:
            QMessageBox.warning(dlg, "Not Found", "Could not find that memory entry to delete.")

    # ── Signal slots ───────────────────────────────────────────────────
    def _on_message(self, speaker: str, color: str, text: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Speaker name
        name_fmt = QTextCharFormat()
        name_fmt.setForeground(QColor(color))
        name_fmt.setFontWeight(QFont.Bold)
        cursor.insertText(f"\n{'─'*60}\n", QTextCharFormat())
        cursor.insertText(f"{speaker}:\n", name_fmt)

        # Check if this is a tool result with code — format code and results distinctly
        if "[Ran Code]" in text:
            for block in text.split("[Ran Code]"):
                block = block.strip()
                if not block:
                    continue
                if "[Result]" in block:
                    code_part, result_part = block.split("[Result]", 1)
                    # Code section — dim cyan
                    code_fmt = QTextCharFormat()
                    code_fmt.setForeground(QColor("#80cbc4"))
                    code_fmt.setFontFamily("Consolas")
                    cursor.insertText("⚡ Code:\n", name_fmt)
                    cursor.insertText(f"{code_part.strip()}\n\n", code_fmt)
                    # Result section — bright
                    result_fmt = QTextCharFormat()
                    result_fmt.setForeground(QColor("#c5e1a5"))
                    cursor.insertText("📊 Result:\n", name_fmt)
                    cursor.insertText(f"{result_part.strip()}\n", result_fmt)
                else:
                    body_fmt = QTextCharFormat()
                    body_fmt.setForeground(QColor("#e0e0e0"))
                    cursor.insertText(f"{block}\n", body_fmt)
        else:
            # Normal message
            body_fmt = QTextCharFormat()
            body_fmt.setForeground(QColor("#e0e0e0"))
            cursor.insertText(f"{text}\n", body_fmt)

        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_status(self, msg: str):
        self.status_bar.showMessage(msg)

    def _on_turn_finished(self):
        self._turn_count += 1
        self.turn_label.setText(f"Turns: {self._turn_count}")

    # ── Cleanup ────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.engine.stop()
        event.accept()

    # ── Project browser ────────────────────────────────────────────────
    PROJECTS_ROOT = Path(r"D:\AriaRexFolder\Projects")

    def _refresh_project_browser(self):
        """Scan D:\\AriaRexFolder\\Projects and populate the tree."""
        self.proj_tree.clear()
        root = self.PROJECTS_ROOT
        if not root.exists():
            return
        self._populate_tree(root, None)
        self.proj_tree.expandAll()

    def _populate_tree(self, folder: Path, parent_item):
        """Recursively add folder contents to the tree."""
        try:
            entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            item = QTreeWidgetItem()
            item.setText(0, entry.name)
            item.setData(0, Qt.UserRole, str(entry))
            if entry.is_dir():
                item.setText(1, "")
                item.setForeground(0, QColor("#80cbc4"))
                if parent_item is None:
                    self.proj_tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
                self._populate_tree(entry, item)
            else:
                size = entry.stat().st_size
                if size < 1024:
                    item.setText(1, f"{size} B")
                else:
                    item.setText(1, f"{size / 1024:.1f} KB")
                if parent_item is None:
                    self.proj_tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)

    def _on_project_file_clicked(self, item, column):
        """Show contents of the clicked file in the viewer."""
        file_path = item.data(0, Qt.UserRole)
        if not file_path:
            return
        p = Path(file_path)
        if p.is_dir():
            return

        # Handle image files
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
            self.proj_viewer.clear()
            self.proj_viewer.setPlainText(f"[Image: {p.name}]")
            pixmap = QPixmap(str(p))
            if not pixmap.isNull():
                scaled = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.proj_image_label.setPixmap(scaled)
                self.proj_image_label.setVisible(True)
            return

        self.proj_image_label.setVisible(False)

        # Handle text files
        try:
            text = p.read_text(encoding="utf-8")
            self.proj_viewer.setPlainText(text)
        except Exception as e:
            self.proj_viewer.setPlainText(f"Could not read file: {e}")

    # ── Message Board handlers ─────────────────────────────────────────
    def _refresh_board(self):
        """Reload the thread list from disk."""
        self.board_thread_list.clear()
        self._board_threads = get_all_threads()
        for t in self._board_threads:
            status = "CLOSED" if t.get("closed") else "OPEN"
            msg_count = len(t.get("messages", []))
            created = t.get("created", "unknown")
            label = f"[{status}] {t['id']} — To: {t.get('to', '?')} ({msg_count} msgs) — {created}"
            item = QListWidgetItem(label)
            if t.get("closed"):
                item.setForeground(QColor("#666"))
            self.board_thread_list.addItem(item)
        self.status_bar.showMessage(f"Message board: {len(self._board_threads)} thread(s) loaded.")

    def _on_thread_selected(self, row: int):
        """Display the selected thread's conversation."""
        if row < 0 or row >= len(getattr(self, '_board_threads', [])):
            self.board_thread_view.clear()
            return
        thread = self._board_threads[row]
        cursor = self.board_thread_view.textCursor()
        self.board_thread_view.clear()
        cursor = self.board_thread_view.textCursor()

        header_fmt = QTextCharFormat()
        header_fmt.setForeground(QColor("#b0b0b0"))
        header_fmt.setFontWeight(QFont.Bold)
        status = "CLOSED" if thread.get("closed") else "OPEN"
        cursor.insertText(
            f"Thread {thread['id']}  |  To: {thread['to']}  |  Started: {thread['created']}  |  {status}\n",
            header_fmt,
        )
        cursor.insertText(f"{'─' * 50}\n", QTextCharFormat())

        for m in thread["messages"]:
            name_fmt = QTextCharFormat()
            if m["from"] == "Aria":
                name_fmt.setForeground(QColor("#5ba3e6"))
            elif m["from"] == "Rex":
                name_fmt.setForeground(QColor("#e6a35b"))
            else:
                name_fmt.setForeground(QColor("#81c784"))
            name_fmt.setFontWeight(QFont.Bold)

            body_fmt = QTextCharFormat()
            body_fmt.setForeground(QColor("#e0e0e0"))

            cursor.insertText(f"\n[{m['timestamp']}] {m['from']}:\n", name_fmt)
            cursor.insertText(f"{m['text']}\n", body_fmt)

            # Show image thumbnail if message has one
            if m.get("image"):
                img_path = BOARD_ROOT / m["image"]
                if img_path.exists():
                    pixmap = QPixmap(str(img_path))
                    if not pixmap.isNull():
                        scaled = pixmap.scaledToWidth(
                            min(400, pixmap.width()), Qt.SmoothTransformation
                        )
                        cursor.insertImage(scaled.toImage())
                        cursor.insertText("\n", body_fmt)
                img_note = QTextCharFormat()
                img_note.setForeground(QColor("#66bb6a"))
                cursor.insertText(f"  [Image: {m['image']}]\n", img_note)

        self.board_thread_view.setTextCursor(cursor)
        self.board_thread_view.ensureCursorVisible()

    def _send_new_thread(self):
        """Create a new thread from the compose area."""
        text = self.board_compose.toPlainText().strip()
        if not text:
            self.status_bar.showMessage("Type a message first.")
            return
        sender = self.board_from.text().strip() or "Scott"
        recipient = self.board_to.currentText()
        create_thread(sender, recipient, text, image_path=self._attached_image_path)
        self.board_compose.clear()
        self._clear_board_image()
        self._refresh_board()
        # Select the new thread (last item)
        self.board_thread_list.setCurrentRow(self.board_thread_list.count() - 1)
        self.status_bar.showMessage(f"New thread created for {recipient}.")

    def _send_reply_to_thread(self):
        """Reply to the currently selected thread."""
        row = self.board_thread_list.currentRow()
        if row < 0 or row >= len(getattr(self, '_board_threads', [])):
            self.status_bar.showMessage("Select a thread to reply to.")
            return
        thread = self._board_threads[row]
        if thread.get("closed"):
            self.status_bar.showMessage("That thread is closed.")
            return
        text = self.board_compose.toPlainText().strip()
        if not text:
            self.status_bar.showMessage("Type a message first.")
            return
        sender = self.board_from.text().strip() or "Scott"
        add_message_to_thread(thread["id"], sender, text, image_path=self._attached_image_path)
        self.board_compose.clear()
        self._clear_board_image()
        self._refresh_board()
        self.board_thread_list.setCurrentRow(row)
        self.status_bar.showMessage(f"Reply added to {thread['id']}.")

    def _attach_board_image(self):
        """Open file dialog to pick an image to attach."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Attach Image", "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;All Files (*)",
        )
        if path:
            self._attached_image_path = path
            fname = Path(path).name
            self.lbl_attached_image.setText(f"\U0001f4ce {fname}")
            self.lbl_attached_image.setStyleSheet("color: #66bb6a;")
            self.btn_clear_image.setVisible(True)
            self.status_bar.showMessage(f"Image attached: {fname}")

    def _clear_board_image(self):
        """Clear the currently attached image."""
        self._attached_image_path = None
        self.lbl_attached_image.setText("No image attached")
        self.lbl_attached_image.setStyleSheet("color: #888;")
        self.btn_clear_image.setVisible(False)

    def _close_selected_thread(self):
        """Close the currently selected thread."""
        row = self.board_thread_list.currentRow()
        if row < 0 or row >= len(getattr(self, '_board_threads', [])):
            self.status_bar.showMessage("Select a thread to close.")
            return
        thread = self._board_threads[row]
        close_thread(thread["id"])
        self._refresh_board()
        self.status_bar.showMessage(f"Thread {thread['id']} closed.")

    def _close_all_board_threads(self):
        """Close all open threads."""
        from PyQt5.QtWidgets import QMessageBox
        confirm = QMessageBox.question(
            self, "End Board",
            "Close all open threads? (They'll still be visible but marked as closed.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            result = close_all_threads()
            self._refresh_board()
            self.status_bar.showMessage(result)

    def _delete_selected_thread(self):
        """Permanently delete the currently selected thread."""
        from PyQt5.QtWidgets import QMessageBox
        row = self.board_thread_list.currentRow()
        if row < 0 or row >= len(getattr(self, '_board_threads', [])):
            self.status_bar.showMessage("Select a thread to delete.")
            return
        thread = self._board_threads[row]
        confirm = QMessageBox.question(
            self, "Delete Thread",
            f"Permanently delete thread {thread['id']}? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            result = delete_thread(thread["id"])
            self.board_thread_view.clear()
            self._refresh_board()
            self.status_bar.showMessage(result)

    # ── Stylesheet ─────────────────────────────────────────────────────
    @staticmethod
    def _stylesheet() -> str:
        return """
        QMainWindow { background: #1e1e2e; }
        QTextEdit {
            background: #181825;
            color: #e0e0e0;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 8px;
        }
        QGroupBox {
            color: #ccc;
            border: 1px solid #444;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 14px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
        }
        QLabel { color: #ccc; }
        QPushButton {
            background: #333;
            color: #e0e0e0;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 6px 16px;
            font-weight: bold;
        }
        QPushButton:hover { background: #444; }
        QPushButton:disabled { color: #666; background: #2a2a2a; }
        QSpinBox {
            background: #2a2a3e;
            color: #e0e0e0;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px;
        }
        QStatusBar { color: #aaa; }
        QTabWidget::pane {
            border: 1px solid #444;
            border-radius: 6px;
            background: #181825;
        }
        QTabBar::tab {
            background: #2a2a3e;
            color: #ccc;
            border: 1px solid #444;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 6px 16px;
            margin-right: 2px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background: #181825;
            color: #fff;
        }
        QTabBar::tab:hover {
            background: #333;
        }
        QTreeWidget {
            background: #181825;
            color: #e0e0e0;
            border: 1px solid #333;
            border-radius: 6px;
            alternate-background-color: #1e1e30;
            padding: 4px;
        }
        QTreeWidget::item {
            padding: 4px 2px;
        }
        QTreeWidget::item:selected {
            background: #333;
        }
        QHeaderView::section {
            background: #2a2a3e;
            color: #ccc;
            border: 1px solid #444;
            padding: 4px;
            font-weight: bold;
        }
        QComboBox {
            background: #2a2a3e;
            color: #e0e0e0;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background: #2a2a3e;
            color: #e0e0e0;
            selection-background-color: #444;
        }
        """


# ═══════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RoommatesWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
