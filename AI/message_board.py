"""
message_board.py — Threaded message board for Scott (developer) to communicate
with Aria and Rex.

Scott posts messages addressed to Aria, Rex, or Both.  Each message starts a
thread.  AIs respond to threads before conversations begin.  Scott can continue
a thread by adding follow-up messages.  Threads stay open and visible until
Scott closes them with the "End Board" action.

Storage:  D:\\AriaRexFolder\\message_board.json

Thread schema:
    {
        "id": "t-001",
        "from": "Scott",
        "to": "Aria" | "Rex" | "Both",
        "created": "2026-03-08 14:30",
        "closed": false,
        "messages": [
            {"from": "Scott", "text": "...", "timestamp": "..."},
            {"from": "Aria",  "text": "...", "timestamp": "..."},
            ...
        ]
    }

AIs respond using [BOARD_REPLY]...[/BOARD_REPLY] tags, and can save private
self-reminders with [SELF_REMINDER]...[/SELF_REMINDER].
"""

import base64
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

# ─── Root folder ───────────────────────────────────────────────────────────
ROOT = Path(r"D:\AriaRexFolder")
BOARD_FILE = ROOT / "message_board.json"
BOARD_IMAGES_DIR = ROOT / "board_images"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _ensure_structure():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("Aria", "Rex"):
        folder = ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        rem = folder / "reminders.md"
        if not rem.exists():
            rem.write_text(f"# {name}'s Reminders\n\n", encoding="utf-8")
    BOARD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not BOARD_FILE.exists():
        BOARD_FILE.write_text("[]", encoding="utf-8")


_ensure_structure()


# ═══════════════════════════════════════════════════════════════════════════
#  Board I/O
# ═══════════════════════════════════════════════════════════════════════════

def _load_board() -> list[dict]:
    try:
        return json.loads(BOARD_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_board(board: list[dict]):
    BOARD_FILE.write_text(
        json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _next_id(board: list[dict]) -> str:
    if not board:
        return "t-001"
    nums = []
    for t in board:
        try:
            tid = t["id"]
            if isinstance(tid, int):
                nums.append(tid)
            else:
                nums.append(int(tid.split("-")[1]))
        except (IndexError, ValueError, AttributeError):
            pass
    return f"t-{(max(nums) + 1) if nums else 1:03d}"


# ═══════════════════════════════════════════════════════════════════════════
#  Public API — called by GUI and engine
# ═══════════════════════════════════════════════════════════════════════════

def create_thread(sender: str, recipient: str, text: str, image_path: str | None = None) -> dict:
    """Start a new thread.  Returns the created thread dict."""
    board = _load_board()
    tid = _next_id(board)
    msg = {
        "from": sender,
        "text": text.strip(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if image_path:
        stored = _store_board_image(image_path, tid)
        if stored:
            msg["image"] = stored
    thread = {
        "id": tid,
        "from": sender,
        "to": recipient,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "closed": False,
        "messages": [msg],
    }
    board.append(thread)
    _save_board(board)
    return thread


def add_message_to_thread(thread_id: str, sender: str, text: str, image_path: str | None = None) -> str:
    """Append a message to an existing thread.  Returns status string."""
    board = _load_board()
    for t in board:
        if t["id"] == thread_id:
            msg = {
                "from": sender,
                "text": text.strip(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            if image_path:
                stored = _store_board_image(image_path, thread_id)
                if stored:
                    msg["image"] = stored
            t["messages"].append(msg)
            _save_board(board)
            return f"Reply added to thread {thread_id}."
    return f"Thread {thread_id} not found."


def close_thread(thread_id: str) -> str:
    """Mark a thread as closed."""
    board = _load_board()
    for t in board:
        if t["id"] == thread_id:
            t["closed"] = True
            _save_board(board)
            return f"Thread {thread_id} closed."
    return f"Thread {thread_id} not found."


def close_all_threads() -> str:
    """Close every open thread."""
    board = _load_board()
    count = 0
    for t in board:
        if not t.get("closed"):
            t["closed"] = True
            count += 1
    _save_board(board)
    return f"Closed {count} thread(s)."


def delete_thread(thread_id: str) -> str:
    """Permanently remove a thread from the board."""
    board = _load_board()
    for i, t in enumerate(board):
        if t["id"] == thread_id:
            board.pop(i)
            _save_board(board)
            return f"Thread {thread_id} deleted."
    return f"Thread {thread_id} not found."


def get_open_threads() -> list[dict]:
    """Return all open (not closed) threads."""
    return [t for t in _load_board() if not t.get("closed")]


def get_all_threads() -> list[dict]:
    """Return all threads (including closed)."""
    return _load_board()


def get_thread(thread_id: str) -> dict | None:
    for t in _load_board():
        if t["id"] == thread_id:
            return t
    return None


def get_threads_for(ai_name: str, open_only: bool = True) -> list[dict]:
    """Return threads addressed to *ai_name* or 'Both'."""
    board = _load_board()
    threads = []
    for t in board:
        if open_only and t.get("closed"):
            continue
        if t["to"] == ai_name or t["to"] == "Both":
            threads.append(t)
    return threads


def get_unanswered_threads(ai_name: str) -> list[dict]:
    """Return open threads for *ai_name* where the AI hasn't replied yet
    (or there are new messages since their last reply)."""
    threads = get_threads_for(ai_name, open_only=True)
    unanswered = []
    for t in threads:
        msgs = t.get("messages", [])
        if not msgs:
            continue
        # Find last message from this AI
        last_ai_idx = -1
        for i, m in enumerate(msgs):
            if m["from"] == ai_name:
                last_ai_idx = i
        # If no reply yet, or there's a newer non-AI message after their last reply
        if last_ai_idx < 0 or last_ai_idx < len(msgs) - 1:
            unanswered.append(t)
    return unanswered


def format_thread_for_ai(thread: dict) -> str:
    """Format a thread as readable text for an AI prompt."""
    lines = [f"--- Thread {thread['id']} (To: {thread.get('to', '?')}, Started: {thread.get('created', 'unknown')}) ---"]
    for m in thread.get("messages", []):
        lines.append(f"  [{m['timestamp']}] {m['from']}: {m['text']}")
        if m.get("image"):
            lines.append(f"  [Image: {m['image']}]")
    lines.append("---")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Image helpers
# ═══════════════════════════════════════════════════════════════════════════

def _store_board_image(source_path: str, thread_id: str) -> str | None:
    """Copy an image to board_images/ and return relative path, or None."""
    src = Path(source_path)
    if not src.exists() or src.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    BOARD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_name = f"{thread_id}_{ts}_{src.name}"
    dest = BOARD_IMAGES_DIR / dest_name
    shutil.copy2(str(src), str(dest))
    return f"board_images/{dest_name}"


def get_thread_images(thread: dict) -> list[tuple[int, Path]]:
    """Return [(msg_index, abs_path), ...] for messages that have images."""
    results = []
    for i, m in enumerate(thread.get("messages", [])):
        if m.get("image"):
            abs_path = ROOT / m["image"]
            if abs_path.exists():
                results.append((i, abs_path))
    return results


def get_image_base64(image_path) -> str | None:
    """Read an image file and return a base64 string."""
    p = Path(image_path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")


def image_mime_type(image_path) -> str:
    """Return the MIME type for an image based on extension."""
    ext = Path(image_path).suffix.lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    }.get(ext, "image/png")


# ═══════════════════════════════════════════════════════════════════════════
#  Self-reminders (kept from old system)
# ═══════════════════════════════════════════════════════════════════════════

def save_reminder(owner: str, content: str) -> str:
    path = ROOT / owner / "reminders.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n---\n**[{timestamp}]**\n{content.strip()}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
    return "Reminder saved."


def read_reminders(owner: str) -> str:
    path = ROOT / owner / "reminders.md"
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        return text if text else "(no reminders yet)"
    return "(no reminders yet)"


# ═══════════════════════════════════════════════════════════════════════════
#  AI response parsing
# ═══════════════════════════════════════════════════════════════════════════

_RE_BOARD_REPLY = re.compile(
    r"\[BOARD_REPLY\](.*?)(?:\[/BOARD_REPLY\]|</BOARD_REPLY>)",
    re.IGNORECASE | re.DOTALL,
)
_RE_REMINDER = re.compile(
    r"\[SELF_REMINDER\](.*?)(?:\[/SELF_REMINDER\]|</SELF_REMINDER>)",
    re.IGNORECASE | re.DOTALL,
)
_RE_CHECK_BOARD = re.compile(
    r"\[CHECK_BOARD\]",
    re.IGNORECASE,
)
_RE_NEW_THREAD = re.compile(
    r"\[NEW_THREAD\s+to=(\w+)\](.*?)(?:\[/NEW_THREAD\]|</NEW_THREAD>)",
    re.IGNORECASE | re.DOTALL,
)
_RE_SAVE_IMAGE = re.compile(
    r"\[SAVE_IMAGE\s+(.*?)\s+(.*?)\]",
    re.IGNORECASE,
)


def parse_ai_response(ai_name: str, raw_text: str, thread_id: str) -> tuple[str, list[str]]:
    """Parse an AI's response to a message board thread.

    Extracts [BOARD_REPLY] and [SELF_REMINDER] tags.
    Returns (clean_text, results).
    """
    results: list[str] = []

    # BOARD_REPLY — add to the thread
    for match in _RE_BOARD_REPLY.finditer(raw_text):
        reply_text = match.group(1).strip()
        if reply_text:
            status = add_message_to_thread(thread_id, ai_name, reply_text)
            results.append(f"[Board] {status}")
    raw_text = _RE_BOARD_REPLY.sub("", raw_text)

    # SELF_REMINDER
    for match in _RE_REMINDER.finditer(raw_text):
        result = save_reminder(ai_name, match.group(1))
        results.append(f"[Reminder] {result}")
    raw_text = _RE_REMINDER.sub("", raw_text)

    clean_text = raw_text.strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text, results


# ═══════════════════════════════════════════════════════════════════════════
#  Backward compat — legacy callers
# ═══════════════════════════════════════════════════════════════════════════

def process_commands(speaker: str, raw_text: str) -> tuple[str, list[str]]:
    """Legacy wrapper — parse self-reminder tags from AI responses."""
    results: list[str] = []

    for match in _RE_REMINDER.finditer(raw_text):
        result = save_reminder(speaker, match.group(1))
        results.append(f"[Reminder] {result}")
    raw_text = _RE_REMINDER.sub("", raw_text)

    clean_text = raw_text.strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text, results


def get_tool_docs(speaker: str) -> str:
    """Legacy — return minimal docs for self-reminders only."""
    return (
        "You can save a private reminder for yourself:\n"
        "  [SELF_REMINDER]your text[/SELF_REMINDER]\n"
        "This is only visible to you and persists across sessions."
    )


def process_mid_conversation_tags(ai_name: str, raw_text: str) -> tuple[str, list[str]]:
    """Parse board-related tags from a mid-conversation AI response.

    Handles: [CHECK_BOARD], [NEW_THREAD to=Scott]...[/NEW_THREAD],
             [SELF_REMINDER]...[/SELF_REMINDER], [BOARD_REPLY]...[/BOARD_REPLY].

    Returns (cleaned_text, list_of_result_strings).
    """
    results: list[str] = []

    # CHECK_BOARD — show open threads for this AI
    for _match in _RE_CHECK_BOARD.finditer(raw_text):
        threads = get_threads_for(ai_name, open_only=True)
        if threads:
            summaries = []
            for t in threads:
                last = t["messages"][-1] if t["messages"] else None
                last_info = f" (last: {last['from']})" if last else ""
                summaries.append(f"  {t['id']} — To: {t['to']}, msgs: {len(t['messages'])}{last_info}")
            results.append("[Board Check]\n" + "\n".join(summaries))
        else:
            results.append("[Board Check] No open threads for you right now.")
    raw_text = _RE_CHECK_BOARD.sub("", raw_text)

    # NEW_THREAD — AI posts a new thread (usually to Scott)
    for match in _RE_NEW_THREAD.finditer(raw_text):
        recipient = match.group(1).strip()
        body = match.group(2).strip()
        if body:
            thread = create_thread(ai_name, recipient, body)
            results.append(f"[Board] New thread {thread['id']} created for {recipient}.")
    raw_text = _RE_NEW_THREAD.sub("", raw_text)

    # BOARD_REPLY — reply to latest open thread for this AI
    for match in _RE_BOARD_REPLY.finditer(raw_text):
        reply_text = match.group(1).strip()
        if reply_text:
            threads = get_threads_for(ai_name, open_only=True)
            if threads:
                tid = threads[-1]["id"]
                status = add_message_to_thread(tid, ai_name, reply_text)
                results.append(f"[Board] {status}")
            else:
                results.append("[Board] No open threads to reply to.")
    raw_text = _RE_BOARD_REPLY.sub("", raw_text)

    # SELF_REMINDER
    for match in _RE_REMINDER.finditer(raw_text):
        result = save_reminder(ai_name, match.group(1))
        results.append(f"[Reminder] {result}")
    raw_text = _RE_REMINDER.sub("", raw_text)

    # SAVE_IMAGE — copy a board image to the projects folder
    for match in _RE_SAVE_IMAGE.finditer(raw_text):
        source_rel = match.group(1).strip()
        dest_rel = match.group(2).strip()
        result = _copy_image_to_projects(source_rel, dest_rel)
        results.append(f"[Image] {result}")
    raw_text = _RE_SAVE_IMAGE.sub("", raw_text)

    clean_text = raw_text.strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text, results


def _copy_image_to_projects(source_rel: str, dest_rel: str) -> str:
    """Copy an image from board_images/ (relative to ROOT) to Projects/."""
    from world_tools import PROJECTS_ROOT
    src = ROOT / source_rel
    if not src.exists():
        return f"Source not found: {source_rel}"
    if src.suffix.lower() not in IMAGE_EXTENSIONS:
        return f"Not a recognized image format: {source_rel}"
    dst = (PROJECTS_ROOT / dest_rel.replace("\\", "/").lstrip("/")).resolve()
    if not str(dst).startswith(str(PROJECTS_ROOT.resolve())):
        return "Invalid destination path."
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return f"Image saved to projects: {dest_rel}"


def check_messages(reader: str) -> str:
    """Legacy compat — returns empty since threads replace old messages."""
    return "(no new messages)"
