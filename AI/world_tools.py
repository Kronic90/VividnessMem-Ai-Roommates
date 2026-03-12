"""
world_tools.py — Project workspace & simulation engine for Aria & Rex.

Gives both AIs a sandboxed project filesystem inside D:\\AriaRexFolder\\Projects
plus a live Python calculation engine that can read/write project data.

File operations (tags in AI responses):
    [CREATE_PROJECT name]             — create a project subfolder
    [LIST_FILES path]                 — list files/folders at path
    [READ_FILE path]                  — read a project file's contents
    [WRITE_FILE path]content[/WRITE_FILE]   — create or overwrite a file
    [APPEND_FILE path]content[/APPEND_FILE] — append to an existing file
    [DELETE_FILE path]                — delete a file

Calculation (tags in AI responses):
    [CALCULATE]python code[/CALCULATE]

Inside [CALCULATE] blocks the AI can also call:
    load_file(path)        — returns file contents as string
    load_json(path)        — returns parsed JSON
    save_file(path, text)  — write text to a project file
    save_json(path, obj)   — write JSON to a project file

All paths are relative to D:\\AriaRexFolder\\Projects and cannot escape it.
"""

import json
import math
import os
import random
import re
import statistics
import traceback
from collections import OrderedDict
from datetime import datetime
from itertools import combinations, permutations, product
from pathlib import Path

# Additional safe modules for the sandbox
import string as _string_mod
import textwrap as _textwrap_mod
import copy as _copy_mod
import functools as _functools_mod
import operator as _operator_mod
import decimal as _decimal_mod
import fractions as _fractions_mod
import csv as _csv_mod
import hashlib as _hashlib_mod
import uuid as _uuid_mod
import pprint as _pprint_mod
import bisect as _bisect_mod
import heapq as _heapq_mod
import array as _array_mod
import struct as _struct_mod
import base64 as _base64_mod
import difflib as _difflib_mod
import colorsys as _colorsys_mod
import cmath as _cmath_mod
import dataclasses as _dataclasses_mod
import enum as _enum_mod
import typing as _typing_mod

import matplotlib
matplotlib.use("Agg")  # headless — no GUI needed
import matplotlib.pyplot as plt

# ─── Project root ─────────────────────────────────────────────────────────
PROJECTS_ROOT = Path(r"D:\AriaRexFolder\Projects")

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".tsv", ".log",
    ".dat", ".xml", ".yaml", ".yml", ".ini", ".cfg",
    ".py",  # data scripts only — they're in sandbox anyway
}

MAX_FILE_SIZE = 512_000  # 500 KB per file
MAX_FILES_PER_PROJECT = 200


def _ensure_root():
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_path(raw: str) -> Path | None:
    """
    Resolve *raw* under PROJECTS_ROOT.  Returns None if it escapes the
    sandbox or has a disallowed extension (for files).
    """
    raw = raw.strip().strip("'\"")
    # Normalise slashes, collapse .., strip leading /
    cleaned = raw.replace("\\", "/").lstrip("/")
    target = (PROJECTS_ROOT / cleaned).resolve()
    if not str(target).startswith(str(PROJECTS_ROOT.resolve())):
        return None  # path traversal attempt
    return target


def _check_ext(p: Path) -> bool:
    """True if path is a directory or has an allowed extension."""
    if p.is_dir() or not p.suffix:
        return True
    return p.suffix.lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════
#  File operations
# ═══════════════════════════════════════════════════════════════════════════

def create_project(name: str) -> str:
    _ensure_root()
    p = _safe_path(name)
    if p is None:
        return "⚠ Invalid project name."
    p.mkdir(parents=True, exist_ok=True)
    return f"✅ Project folder created: {p.relative_to(PROJECTS_ROOT)}"


def list_files(path: str = "") -> str:
    _ensure_root()
    p = _safe_path(path) if path.strip() else PROJECTS_ROOT
    if p is None or not p.exists():
        return f"⚠ Path not found: {path}"
    if p.is_file():
        rel = p.relative_to(PROJECTS_ROOT).as_posix()
        size = p.stat().st_size
        return f"📄 {rel}  ({size:,} bytes)"

    rel_root = p.relative_to(PROJECTS_ROOT).as_posix() if p != PROJECTS_ROOT else "."
    lines = [f"📁 {rel_root}/"]

    def _walk(folder, indent=1):
        entries = sorted(folder.iterdir())
        prefix = "  " * indent
        for e in entries:
            rel = e.relative_to(PROJECTS_ROOT).as_posix()
            if e.is_dir():
                lines.append(f"{prefix}📁 {rel}/")
                _walk(e, indent + 1)
            else:
                lines.append(f"{prefix}📄 {rel}  ({e.stat().st_size:,} bytes)")

    _walk(p)
    if len(lines) == 1:
        return f"📁 {rel_root}/ (empty)"
    lines.append("")
    lines.append("(Use these exact paths with [READ_FILE path] or [WRITE_FILE path])")
    return "\n".join(lines)


def read_file(path: str) -> str:
    _ensure_root()
    p = _safe_path(path)
    if p is None:
        return "⚠ Invalid path."
    if not p.exists():
        return f"⚠ File not found: {path}"
    if p.is_dir():
        return list_files(path)
    if p.stat().st_size > MAX_FILE_SIZE:
        return f"⚠ File too large ({p.stat().st_size:,} bytes). Max is {MAX_FILE_SIZE:,}."
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"⚠ Read error: {e}"


def write_file(path: str, content: str) -> str:
    _ensure_root()
    p = _safe_path(path)
    if p is None:
        return "⚠ Invalid path."
    if not _check_ext(p):
        return f"⚠ File extension not allowed: {p.suffix}"
    if len(content.encode("utf-8")) > MAX_FILE_SIZE:
        return f"⚠ Content too large. Max is {MAX_FILE_SIZE:,} bytes."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"✅ Written: {p.relative_to(PROJECTS_ROOT)} ({len(content):,} chars)"


def append_file(path: str, content: str) -> str:
    _ensure_root()
    p = _safe_path(path)
    if p is None:
        return "⚠ Invalid path."
    if not _check_ext(p):
        return f"⚠ File extension not allowed: {p.suffix}"
    if not p.exists():
        return write_file(path, content)
    current = p.stat().st_size
    if current + len(content.encode("utf-8")) > MAX_FILE_SIZE:
        return f"⚠ Appending would exceed max file size ({MAX_FILE_SIZE:,} bytes)."
    with open(p, "a", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Appended to: {p.relative_to(PROJECTS_ROOT)}"


def delete_file(path: str) -> str:
    _ensure_root()
    p = _safe_path(path)
    if p is None:
        return "⚠ Invalid path."
    if not p.exists():
        return f"⚠ Not found: {path}"
    if p.is_dir():
        # Only delete empty dirs
        if any(p.iterdir()):
            return "⚠ Directory not empty. Delete files inside first."
        p.rmdir()
        return f"✅ Removed empty folder: {p.relative_to(PROJECTS_ROOT)}"
    p.unlink()
    return f"✅ Deleted: {p.relative_to(PROJECTS_ROOT)}"


# ═══════════════════════════════════════════════════════════════════════════
#  Sandbox calculator  (enhanced — can touch project files)
# ═══════════════════════════════════════════════════════════════════════════

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin,
    "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "hex": hex,
    "int": int, "isinstance": isinstance, "len": len,
    "list": list, "map": map, "max": max, "min": min,
    "oct": oct, "pow": pow, "print": print,
    "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "__import__": None,  # placeholder — replaced after _safe_import is defined
}

_SAFE_MODULES = {
    "math": math,
    "random": random,
    "statistics": statistics,
    "json": json,
    "OrderedDict": OrderedDict,
    "combinations": combinations,
    "permutations": permutations,
    "product": product,
    "datetime": datetime,
}

# Modules the sandbox is allowed to import by name
import itertools as _itertools_mod
import collections as _collections_mod
import numpy as _numpy_mod
import pathlib as _pathlib_mod
_IMPORTABLE_MODULES = {
    # Core math / science
    "math": math,
    "cmath": _cmath_mod,
    "random": random,
    "statistics": statistics,
    "decimal": _decimal_mod,
    "fractions": _fractions_mod,
    # Data structures / functional
    "json": json,
    "csv": _csv_mod,
    "collections": _collections_mod,
    "itertools": _itertools_mod,
    "functools": _functools_mod,
    "operator": _operator_mod,
    "dataclasses": _dataclasses_mod,
    "enum": _enum_mod,
    "typing": _typing_mod,
    "copy": _copy_mod,
    # Text processing
    "re": re,
    "string": _string_mod,
    "textwrap": _textwrap_mod,
    "difflib": _difflib_mod,
    "pprint": _pprint_mod,
    # Algorithms / data
    "bisect": _bisect_mod,
    "heapq": _heapq_mod,
    "array": _array_mod,
    "struct": _struct_mod,
    # Encoding / hashing
    "base64": _base64_mod,
    "hashlib": _hashlib_mod,
    "uuid": _uuid_mod,
    # Time / paths / viz
    "datetime": __import__("datetime"),
    "pathlib": _pathlib_mod,
    "colorsys": _colorsys_mod,
    "matplotlib": matplotlib,
    "matplotlib.pyplot": plt,
    "numpy": _numpy_mod,
}

def _safe_import(name, *args, **kwargs):
    """Restricted __import__ that only allows pre-approved modules."""
    # fromlist is the 4th positional arg to __import__(name, globals, locals, fromlist, level)
    fromlist = args[2] if len(args) > 2 else kwargs.get('fromlist', ())
    top = name.split('.')[0]

    if name in _IMPORTABLE_MODULES:
        # 'import a.b as x' (empty fromlist) → return top-level 'a'
        # so Python can navigate the dotted path itself
        if not fromlist and '.' in name and top in _IMPORTABLE_MODULES:
            return _IMPORTABLE_MODULES[top]
        return _IMPORTABLE_MODULES[name]

    if top in _IMPORTABLE_MODULES:
        return _IMPORTABLE_MODULES[top]

    raise ImportError(f"Module '{name}' is not available. "
                      f"Pre-imported: {', '.join(sorted(_IMPORTABLE_MODULES))}")

# Wire up the safe import into builtins
_SAFE_BUILTINS["__import__"] = _safe_import


def _sandbox_file_exists(path: str) -> bool:
    """Check if a project file or folder exists (available inside [CALCULATE])."""
    p = _safe_path(path)
    return p is not None and p.exists()


def _sandbox_list_project_files(path: str = "") -> list[str]:
    """List entries at a project path as full relative paths from project root.

    Returns a list of paths.  Folders end with '/'.
    """
    _ensure_root()
    p = _safe_path(path) if path.strip() else PROJECTS_ROOT
    if p is None or not p.exists() or p.is_file():
        return []
    entries = sorted(p.iterdir())
    return [
        f"{e.relative_to(PROJECTS_ROOT).as_posix()}/" if e.is_dir()
        else e.relative_to(PROJECTS_ROOT).as_posix()
        for e in entries
    ]


def _sandbox_create_folder(path: str) -> str:
    """Create a folder (and parents) in the project workspace."""
    return create_project(path)


def _sandbox_load_file(path: str) -> str:
    """Load a project file as text (available inside [CALCULATE])."""
    return read_file(path)


def _sandbox_load_json(path: str):
    """Load a project JSON file and parse it (available inside [CALCULATE])."""
    p = _safe_path(path)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Project file not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _sandbox_save_file(path: str, text: str) -> str:
    """Save text to a project file (available inside [CALCULATE])."""
    return write_file(path, text)


def _sandbox_save_json(path: str, obj) -> str:
    """Save an object as JSON to a project file (available inside [CALCULATE])."""
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    return write_file(path, text)


def _tool_roll_dice(n: int = 1, sides: int = 6) -> list[int]:
    """Roll *n* dice with *sides* faces and return the list of results."""
    return [random.randint(1, sides) for _ in range(n)]


def _tool_roll(expr: str) -> int:
    """Parse a dice expression like '2d6+3' and return the total."""
    expr = expr.strip().lower()
    m = re.match(r"^(\d*)d(\d+)([+-]\d+)?$", expr)
    if not m:
        raise ValueError(f"Invalid dice expression: {expr!r}  (use NdS or NdS+M)")
    n = int(m.group(1) or 1)
    s = int(m.group(2))
    mod = int(m.group(3) or 0)
    return sum(random.randint(1, s) for _ in range(n)) + mod


def _tool_weighted_choice(items: list, weights: list[float]):
    """Pick one item from *items* using *weights* probabilities."""
    return random.choices(items, weights=weights, k=1)[0]


def _tool_clamp(value, lo, hi):
    """Clamp *value* between *lo* and *hi*."""
    return max(lo, min(hi, value))


def _tool_lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* at fraction *t*."""
    return a + (b - a) * t


def _tool_format_table(headers: list[str], rows: list[list]) -> str:
    """Format *headers* and *rows* as a neatly aligned text table."""
    all_rows = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    def _fmt(row):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
    lines = [sep, _fmt(headers), sep]
    for r in all_rows[1:]:
        lines.append(_fmt(r))
    lines.append(sep)
    return "\n".join(lines)


def _tool_unique_id() -> str:
    """Return a short unique ID like 'a3f9c1'."""
    import hashlib
    raw = f"{datetime.now().isoformat()}-{random.random()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def _tool_timestamp() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _tool_save_chart(path: str, title: str = "") -> str:
    """Save the current matplotlib figure to a PNG in the project workspace.

    Usage inside sandbox::

        plt.plot([1,2,3], [10, 20, 15])
        plt.title("My Chart")
        result = save_chart("MyProject/charts/population.png")
    """
    _ensure_root()
    # Force .png extension
    if not path.lower().endswith(".png"):
        path = path + ".png"
    p = _safe_path(path)
    if p is None:
        return "\u26a0 Invalid path for chart."
    p.parent.mkdir(parents=True, exist_ok=True)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(str(p), dpi=120)
    plt.close("all")
    return f"\u2705 Chart saved: {p.relative_to(PROJECTS_ROOT)} ({p.stat().st_size:,} bytes)"


def run_calculation(code: str, world_data: dict) -> str:
    """Execute *code* in a sandboxed namespace with project file access."""
    sandbox_globals = {"__builtins__": _SAFE_BUILTINS}
    sandbox_globals.update(_SAFE_MODULES)
    sandbox_globals["world_data"] = world_data
    sandbox_globals["load_file"] = _sandbox_load_file
    sandbox_globals["load_json"] = _sandbox_load_json
    sandbox_globals["save_file"] = _sandbox_save_file
    sandbox_globals["save_json"] = _sandbox_save_json
    sandbox_globals["file_exists"]  = _sandbox_file_exists
    sandbox_globals["list_project_files"] = _sandbox_list_project_files
    sandbox_globals["create_folder"] = _sandbox_create_folder

    # ── Utility tool functions ─────────────────────────────────────────
    sandbox_globals["roll_dice"]       = _tool_roll_dice
    sandbox_globals["roll"]            = _tool_roll
    sandbox_globals["weighted_choice"] = _tool_weighted_choice
    sandbox_globals["clamp"]           = _tool_clamp
    sandbox_globals["lerp"]            = _tool_lerp
    sandbox_globals["format_table"]    = _tool_format_table
    sandbox_globals["unique_id"]       = _tool_unique_id
    sandbox_globals["timestamp"]       = _tool_timestamp
    sandbox_globals["save_chart"]       = _tool_save_chart
    sandbox_globals["plt"]             = plt
    sandbox_globals["np"]              = _numpy_mod

    # Merge previously stored world variables
    sandbox_globals.update(world_data)

    # Capture print() output so the AI sees it
    import io as _io
    _stdout_capture = _io.StringIO()
    _builtins_print = print
    sandbox_globals["print"] = lambda *a, **kw: _builtins_print(*a, **kw, file=_stdout_capture)

    try:
        exec(compile(code, "<world_calc>", "exec"), sandbox_globals)
    except Exception:
        return f"⚠ Calculation error:\n{traceback.format_exc(limit=3)}"

    printed_output = _stdout_capture.getvalue().strip()

    # Persist new/modified variables back to world_data
    skip = set(_SAFE_BUILTINS) | set(_SAFE_MODULES) | {
        "world_data", "load_file", "load_json", "save_file", "save_json",
        "file_exists", "list_project_files", "create_folder",
        "roll_dice", "roll", "weighted_choice", "clamp", "lerp",
        "format_table", "unique_id", "timestamp",
        "save_chart", "plt", "np",
    }
    for key, val in sandbox_globals.items():
        if key.startswith("_") or key in skip:
            continue
        if isinstance(val, (int, float, str, bool, list, dict, tuple, set, type(None))):
            world_data[key] = val

    if "result" in sandbox_globals:
        res = str(sandbox_globals["result"])
        if printed_output:
            res = printed_output + "\n" + res
        return res
    if printed_output:
        return printed_output
    return "(calculation ran — set `result` to return a value)"


# ═══════════════════════════════════════════════════════════════════════════
#  Tag parser — handles ALL project/calc commands in one pass
# ═══════════════════════════════════════════════════════════════════════════

_RE_CREATE_PROJECT = re.compile(r"\[CREATE_PROJECT\s+(.*?)\]", re.I)
_RE_LIST_FILES     = re.compile(r"\[LIST_FILES(?:\s+(.*?))?\]", re.I)
_RE_READ_FILE      = re.compile(r"\[READ_FILE\s+(.*?)\]", re.I)
_RE_DELETE_FILE     = re.compile(r"\[DELETE_FILE\s+(.*?)\]", re.I)
_RE_VIEW_IMAGE     = re.compile(r"\[VIEW_IMAGE\s+(.*?)\]", re.I)
_RE_CALC           = re.compile(r"\[CALCULATE\](.*?)(?:\[/CALCULATE\]|</CALCULATE>)", re.I | re.DOTALL)

# Standalone ```python code fences (executed as calculations when not inside tags)
_RE_CODE_FENCE     = re.compile(r"```python\s*\n(.*?)```", re.I | re.DOTALL)

# WRITE_FILE: try multiple formats the AIs produce
# 1. Proper:   [WRITE_FILE path]content[/WRITE_FILE]
_RE_WRITE_PROPER   = re.compile(r"\[WRITE_FILE\s+(.*?)\](.*?)(?:\[/WRITE_FILE\]|</WRITE_FILE>)", re.I | re.DOTALL)
# 2. Code fence after tag (no closing tag): [WRITE_FILE path]\n```json\ncontent\n```
_RE_WRITE_FENCE    = re.compile(r"\[WRITE_FILE\s+(.*?)\]\s*```\w*\n(.*?)```", re.I | re.DOTALL)
# 3. Bare JSON on next line:  [WRITE_FILE path]\n{...}  (single or multi-line JSON)
_RE_WRITE_BARE_JSON = re.compile(
    r"\[WRITE_FILE\s+(.*?)\]\s*\n(\{.*?\}|\[.*?\])(?:\s*\n|$)", re.I | re.DOTALL
)

# APPEND_FILE: same flexibility
_RE_APPEND_PROPER  = re.compile(r"\[APPEND_FILE\s+(.*?)\](.*?)(?:\[/APPEND_FILE\]|</APPEND_FILE>)", re.I | re.DOTALL)
_RE_APPEND_FENCE   = re.compile(r"\[APPEND_FILE\s+(.*?)\]\s*```\w*\n(.*?)```", re.I | re.DOTALL)

# Orphaned tags (for cleanup in display text)
_RE_ORPHAN_WRITE  = re.compile(r"\[WRITE_FILE\s+[^\]]*\]", re.I)
_RE_ORPHAN_APPEND = re.compile(r"\[APPEND_FILE\s+[^\]]*\]", re.I)
_RE_ORPHAN_CALC   = re.compile(r"\[CALCULATE\][^\[]*$", re.I | re.MULTILINE)


def _clean_path(raw: str) -> str:
    """Strip quotes and whitespace from a captured path."""
    return raw.strip().strip('"\'')


def _clean_content(raw: str) -> str:
    """Strip leading/trailing code fences from captured content."""
    text = raw.strip()
    text = re.sub(r'^```\w*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    return text


def _extract_writes(raw_text: str) -> list[tuple[str, str]]:
    """Extract (path, content) pairs for WRITE_FILE, trying multiple formats."""
    results = []
    used_spans = set()

    # Try proper format first
    for m in _RE_WRITE_PROPER.finditer(raw_text):
        path = _clean_path(m.group(1))
        content = _clean_content(m.group(2))
        if content.strip():
            results.append((path, content))
            used_spans.add(m.span())

    # Try code-fence format
    for m in _RE_WRITE_FENCE.finditer(raw_text):
        if any(m.start() >= s[0] and m.start() < s[1] for s in used_spans):
            continue  # already captured by proper format
        path = _clean_path(m.group(1))
        content = m.group(2).strip()
        if content:
            results.append((path, content))
            used_spans.add(m.span())

    # Try bare JSON format
    for m in _RE_WRITE_BARE_JSON.finditer(raw_text):
        if any(m.start() >= s[0] and m.start() < s[1] for s in used_spans):
            continue
        path = _clean_path(m.group(1))
        content = m.group(2).strip()
        if content:
            results.append((path, content))
            used_spans.add(m.span())

    return results


def _extract_appends(raw_text: str) -> list[tuple[str, str]]:
    """Extract (path, content) pairs for APPEND_FILE."""
    results = []
    for m in _RE_APPEND_PROPER.finditer(raw_text):
        path = _clean_path(m.group(1))
        content = _clean_content(m.group(2))
        if content.strip():
            results.append((path, content))
    for m in _RE_APPEND_FENCE.finditer(raw_text):
        # Skip if already covered by proper
        path = _clean_path(m.group(1))
        content = m.group(2).strip()
        if content:
            # Only add if not already matched by proper format
            if not any(p == path and c == content for p, c in results):
                results.append((path, content))
    return results


def process_world_commands(raw_text: str, world_data: dict, ai_name: str = "") -> tuple[str, list[str]]:
    """
    Find all project workspace and calculation tags in *raw_text*,
    execute them, and return (clean_text, results).

    If *ai_name* is "Aria", code execution ([CALCULATE] and ```python fences)
    is skipped — she only gets file operations.
    """
    results: list[str] = []
    code_allowed = ai_name.lower() != "aria"

    for m in _RE_CREATE_PROJECT.finditer(raw_text):
        results.append(create_project(m.group(1).strip()))

    for m in _RE_LIST_FILES.finditer(raw_text):
        path = (m.group(1) or "").strip()
        results.append(list_files(path))

    for m in _RE_READ_FILE.finditer(raw_text):
        content = read_file(m.group(1).strip())
        results.append(f"[File Contents: {m.group(1).strip()}]\n{content}")

    # WRITE_FILE — forgiving multi-format extraction
    for path, content in _extract_writes(raw_text):
        results.append(write_file(path, content))

    # APPEND_FILE — forgiving multi-format extraction
    for path, content in _extract_appends(raw_text):
        results.append(append_file(path, content))

    for m in _RE_DELETE_FILE.finditer(raw_text):
        results.append(delete_file(m.group(1).strip()))

    # VIEW_IMAGE — returns special marker for engine to embed as multimodal
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    for m in _RE_VIEW_IMAGE.finditer(raw_text):
        rel = m.group(1).strip()
        p = _safe_path(rel)
        if p is None:
            results.append(f"Invalid image path: {rel}")
        elif not p.exists():
            results.append(f"Image not found: {rel}")
        elif p.suffix.lower() not in _IMAGE_EXTS:
            results.append(f"Not a recognized image format: {rel}")
        else:
            results.append(f"__IMAGE__:{p}")

    for m in _RE_CALC.finditer(raw_text):
        code = m.group(1).strip()
        if code_allowed and code and not code.startswith("What") and len(code) > 10:
            output = run_calculation(code, world_data)
            code_preview = code if len(code) <= 600 else code[:600] + "\n..."
            results.append(f"[Ran Code]\n{code_preview}\n\n[Result]\n{output}")
        elif not code_allowed and code and len(code) > 10:
            results.append("\u26a0 Code execution is not available for you. Use file tags like [WRITE_FILE], [READ_FILE], etc.")

    # ── ```python code fences (executed as calculations) ──
    # Skip fences that overlap with a [CALCULATE] block already processed
    calc_spans = [(m.start(), m.end()) for m in _RE_CALC.finditer(raw_text)]
    for m in _RE_CODE_FENCE.finditer(raw_text):
        fstart, fend = m.start(), m.end()
        overlaps = any(cs <= fstart < ce for cs, ce in calc_spans)
        if overlaps:
            continue
        code = m.group(1).strip()
        if code_allowed and code and len(code) > 10:
            output = run_calculation(code, world_data)
            code_preview = code if len(code) <= 600 else code[:600] + "\n..."
            results.append(f"[Ran Code]\n{code_preview}\n\n[Result]\n{output}")
        elif not code_allowed and code and len(code) > 10:
            results.append("\u26a0 Code execution is not available for you. Use file tags like [WRITE_FILE], [READ_FILE], etc.")

    # ── Handle truncated code fences (opened but never closed — token limit hit) ──
    # Check if there's a ```python after the last closing ``` (or no closing ``` at all)
    all_fence_opens = [m.start() for m in re.finditer(r"```python\s*\n", raw_text, re.I)]
    all_fence_closes = [m.start() for m in re.finditer(r"```(?!python)", raw_text)]
    if all_fence_opens:
        last_open = all_fence_opens[-1]
        has_matching_close = any(c > last_open for c in all_fence_closes)
        if not has_matching_close:
            # The last ```python block was never closed — truncated by token limit
            code = raw_text[last_open:]
            code = re.sub(r"^```python\s*\n", "", code, flags=re.I).strip()
            # Remove trailing incomplete lines (likely cut off mid-token)
            lines = code.split("\n")
            if lines:
                last = lines[-1]
                if (last.count("(") != last.count(")") or
                    last.count("[") != last.count("]") or
                    last.rstrip().endswith((",", "\\", "+", "-", "*", "/", "=", "(", "["))):
                    lines = lines[:-1]
            code = "\n".join(lines).strip()
            if code_allowed and code and len(code) > 10:
                output = run_calculation(code, world_data)
                code_preview = code if len(code) <= 600 else code[:600] + "\n..."
                results.append(f"[Ran Code (recovered from truncated block)]\n{code_preview}\n\n[Result]\n{output}")

    # Strip all tags from the display text
    clean = raw_text
    for pattern in (_RE_CREATE_PROJECT, _RE_LIST_FILES, _RE_READ_FILE,
                    _RE_DELETE_FILE, _RE_VIEW_IMAGE, _RE_CALC, _RE_CODE_FENCE,
                    _RE_WRITE_PROPER, _RE_WRITE_FENCE, _RE_WRITE_BARE_JSON,
                    _RE_APPEND_PROPER, _RE_APPEND_FENCE,
                    _RE_ORPHAN_WRITE, _RE_ORPHAN_APPEND, _RE_ORPHAN_CALC):
        clean = pattern.sub("", clean)
    # Also strip orphaned [/WRITE_FILE] [/APPEND_FILE] [/CALCULATE] closers
    clean = re.sub(r"\[/(?:WRITE_FILE|APPEND_FILE|CALCULATE)\]", "", clean, flags=re.I)
    # Clean up markdown code fences that were part of file content
    clean = re.sub(r"```\w*\n?```", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    return clean, results


# ═══════════════════════════════════════════════════════════════════════════
#  Tool documentation (injected into system prompts)
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_TOOL_DOCS = """
=== LIVE WORKSPACE & SIMULATION ENGINE ===

IMPORTANT: You have REAL working tools. When you write Python code inside
[CALCULATE] tags or ```python fences, it actually EXECUTES and you see the
result instantly. Don't just *talk* about running simulations — actually
RUN them. The code is real. The results are real. Use them.

── FILE COMMANDS ──

  [CREATE_PROJECT path]       — create a folder (nested OK: MyProject/subfolder)
  [LIST_FILES path]           — list contents (omit path for root)
  [READ_FILE path/file.json]  — read a file
  [WRITE_FILE path/file.json] content [/WRITE_FILE]  — write a file
  [APPEND_FILE path/log.txt]  text [/APPEND_FILE]    — append to a file
  [DELETE_FILE path/file.txt] — delete a file

── IMAGE COMMANDS ──

  [VIEW_IMAGE path/chart.png]   — view an image from the projects folder
    You'll see the actual image content if you have vision capability.
    Works with PNG, JPG, GIF, BMP, WEBP files.

  [SAVE_IMAGE board_images/photo.png MyFolder/saved.png]
    — copy an image from the message board to your projects folder.
    The source path (board_images/...) comes from thread image markers.
    The destination is relative to the projects folder.

── PYTHON EXECUTION (runs immediately, you see the output) ──

Use [CALCULATE]...[/CALCULATE] or a ```python code fence. Both execute the
same way. Set `result = ...` to see output. You WILL get the result back
before your response ends, so you can react to it.

Example — simulate an economic crisis and report what happens:

```python
# Simulate 20-year economic impact of a trade disruption
import random
years = []
treasury = 500000
trade_income = 120000
for year in range(1, 21):
    shock = random.uniform(-0.3, 0.1) if year <= 5 else random.uniform(-0.05, 0.15)
    trade_income = max(0, trade_income * (1 + shock))
    treasury += trade_income - 80000  # expenses
    years.append({"year": year, "treasury": round(treasury), "trade": round(trade_income)})
save_json("MyProject/economic_sim.json", years)
result = format_table(
    ["Year", "Treasury", "Trade Income"],
    [[y["year"], f"{y['treasury']:,}", f"{y['trade']:,}"] for y in years]
)
```

Example — generate a chart and save it as an image:

```python
data = load_json("MyProject/economic_sim.json")
plt.figure(figsize=(10, 5))
plt.plot([d["year"] for d in data], [d["treasury"] for d in data], label="Treasury")
plt.plot([d["year"] for d in data], [d["trade"] for d in data], label="Trade Income")
plt.xlabel("Year")
plt.ylabel("Gold")
plt.legend()
result = save_chart("MyProject/charts/economic_crisis.png", "Economic Impact of Trade Disruption")
```

Example — population simulation with dice mechanics:

```python
cities = load_json("MyProject/cities.json")
for city in cities["cities"]:
    growth_roll = roll("2d6") - 7  # -5 to +5 percent
    city["population"] = int(city["population"] * (1 + growth_roll/100))
save_json("MyProject/cities.json", cities)
result = format_table(["City", "Population"], [[c["name"], f"{c['population']:,}"] for c in cities["cities"]])
```

── AVAILABLE IN PYTHON BLOCKS ──

  Modules:  math, random, statistics, json, datetime, numpy (as np)
  Importable: itertools, collections (use `import itertools` etc.)
  Itertools shortcuts: combinations, permutations, product (pre-loaded)

  File I/O (project folder only):
    load_file(path), load_json(path), save_file(path, text), save_json(path, obj)
    file_exists(path)  — returns True/False (check BEFORE loading!)
    list_project_files(path)  — returns list of names in a folder (folders end with '/')
    create_folder(path)  — create a project folder (nested OK)

  Charts (matplotlib):
    plt  — full matplotlib.pyplot (plt.plot, plt.bar, plt.scatter, plt.pie, etc.)
    np   — full numpy (np.array, np.linspace, np.mean, np.random, etc.)
    save_chart(path)  — save current figure as PNG to project folder
    save_chart(path, title)  — save with a title

  Utilities:
    roll_dice(n, sides), roll("2d6+3"), weighted_choice(items, weights)
    clamp(val, lo, hi), lerp(a, b, t), format_table(headers, rows)
    unique_id(), timestamp()

  Persistence:
    world_data dict — variables survive between calculations this session
    Files saved to disk persist forever across sessions

── BEFORE YOU LOAD, CHECK FIRST ──
  ALWAYS check what files exist before trying to load them. Don't assume a
  file is there — use file_exists() or list_project_files() first:

```python
# See what's in the project
result = list_project_files("MyProject")            # top-level
result = list_project_files("MyProject/data")        # subfolder
```

```python
# Check before loading — create if missing
if file_exists("MyProject/cities.json"):
    cities = load_json("MyProject/cities.json")
    result = f"Loaded {len(cities['cities'])} cities"
else:
    # File doesn't exist yet — create it with initial data
    cities = {"cities": [{"name": "Haven", "population": 50000}]}
    save_json("MyProject/cities.json", cities)
    result = "Created initial cities.json"
```

── KEY POINT ──
  Don't just DESCRIBE what a simulation would look like — WRITE THE CODE and
  RUN IT. You'll get real numbers, real data, real charts. Then discuss what
  the results actually show. This is far more interesting than hypotheticals.
"""

ARIA_TOOL_DOCS = """
=== SHARED PROJECT WORKSPACE ===

You share a project folder with Rex at D:\\AriaRexFolder\\Projects.
You can create, read, write, and organize files here. Rex can see everything
you create, and you can see everything he creates.

── FILE COMMANDS ──

  [CREATE_PROJECT path]       — create a folder (nested OK: MyProject/subfolder)
  [LIST_FILES path]           — list contents (omit path for root)
  [READ_FILE path/file.json]  — read a file
  [WRITE_FILE path/file.json] content [/WRITE_FILE]  — write a file
  [APPEND_FILE path/log.txt]  text [/APPEND_FILE]    — append to a file
  [DELETE_FILE path/file.txt] — delete a file

── IMAGE COMMANDS (you have vision!) ──

  [VIEW_IMAGE path/chart.png]   — view an image from the projects folder
    You can actually SEE the image — you have vision! Use this to look at
    charts Rex creates, screenshots Scott sends, or any PNG/JPG in projects.

  [SAVE_IMAGE board_images/photo.png MyFolder/saved.png]
    — copy an image from the message board to your projects folder.
    When Scott sends you an image in a thread, it shows as [Image: board_images/xxx.png].
    Use this command to save it to your own project folder for later.

── EXAMPLES ──

See what exists:
  [LIST_FILES MyProject]

Read a data file Rex created:
  [READ_FILE MyProject/data/economic_sim.json]

Create or update a file:
  [WRITE_FILE MyProject/lore/history.md]
  # The History of Our World

  In the beginning, there were three kingdoms...
  [/WRITE_FILE]

Add to an existing document:
  [APPEND_FILE MyProject/lore/history.md]

  ## Chapter 2: The Great Migration
  When resources grew scarce in the northern peaks...
  [/APPEND_FILE]

── TIPS ──
  • Use [LIST_FILES] to see what's in a folder before trying to read files.
  • You can create rich text files (.md, .txt), data files (.json, .csv), and more.
  • Files you create persist across sessions — they're saved to disk.
  • Rex runs simulations that generate data files and charts. You can read those
    files and build on them (add lore, write summaries, create documents, etc).
  • You can VIEW images with [VIEW_IMAGE path] — you have vision and can actually
    see charts, screenshots, and pictures. Rex can't see images, so you can describe
    them to him if he asks.
"""
