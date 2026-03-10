"""
Web browsing tools for Aria (and Rex awareness).

Provides safe, read-only web access through tag-based commands:
  [WEB_SEARCH query]         — DuckDuckGo search, returns top results
  [READ_URL url]             — fetch a page, return stripped text
  [PAGE_IMAGES]              — list images from last fetched page
  [FETCH_IMAGE n dest]       — download image n to projects folder

Safety: domain whitelist, SSRF protection, size caps, timeouts, text-only.
"""

import re
import ipaddress
import socket
import traceback
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

PROJECTS_ROOT = Path(r"D:\AriaRexFolder\Projects")

# Maximum characters of page text returned
MAX_PAGE_TEXT = 6000

# Maximum image download size (5 MB)
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Request timeout in seconds
REQUEST_TIMEOUT = 10

# Domain whitelist — READ_URL and FETCH_IMAGE only work on these
ALLOWED_DOMAINS: set[str] = {
    # Reference
    "en.wikipedia.org",
    "en.m.wikipedia.org",
    "simple.wikipedia.org",
    "www.britannica.com",
    "plato.stanford.edu",
    "en.wiktionary.org",
    "commons.wikimedia.org",
    "upload.wikimedia.org",
    # News
    "www.bbc.com",
    "www.bbc.co.uk",
    "www.reuters.com",
    "apnews.com",
    # Science
    "arxiv.org",
    "www.nasa.gov",
    "science.nasa.gov",
    "www.noaa.gov",
    "www.nature.com",
    "www.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    # Tech / Dev
    "stackoverflow.com",
    "github.com",
    "raw.githubusercontent.com",
    "docs.python.org",
    "developer.mozilla.org",
    # Creative / Archive
    "www.gutenberg.org",
    "archive.org",
    "www.archive.org",
    # Education
    "www.khanacademy.org",
    "ocw.mit.edu",
    "www.smithsonianmag.com",
}

# Image extensions we'll download
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}

# ═══════════════════════════════════════════════════════════════════════════
#  State — last fetched page images (for [PAGE_IMAGES] / [FETCH_IMAGE])
# ═══════════════════════════════════════════════════════════════════════════

_last_page_images: list[dict] = []  # [{"url": ..., "alt": ...}, ...]
_last_page_url: str = ""

# ═══════════════════════════════════════════════════════════════════════════
#  Safety helpers
# ═══════════════════════════════════════════════════════════════════════════

def _is_domain_allowed(url: str) -> bool:
    """Check if the URL's domain is in the whitelist."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        return host.lower() in ALLOWED_DOMAINS
    except Exception:
        return False


def _is_private_ip(hostname: str) -> bool:
    """Block SSRF — refuse private/internal IPs."""
    try:
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return True
    except (socket.gaierror, ValueError):
        return True  # can't resolve = block
    return False


def _safe_fetch(url: str, stream: bool = False) -> requests.Response | None:
    """Fetch a URL with all safety checks applied. Returns None on failure."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if not _is_domain_allowed(url):
        return None

    if _is_private_ip(hostname):
        return None

    if parsed.scheme not in ("http", "https"):
        return None

    headers = {
        "User-Agent": "AriaWebReader/1.0 (research assistant; +local)",
        "Accept": "text/html,application/xhtml+xml,*/*",
    }
    resp = requests.get(
        url, headers=headers, timeout=REQUEST_TIMEOUT,
        stream=stream, allow_redirects=True,
    )
    resp.raise_for_status()

    # After redirects, recheck domain
    if not _is_domain_allowed(resp.url):
        return None

    return resp


# ═══════════════════════════════════════════════════════════════════════════
#  Core functions
# ═══════════════════════════════════════════════════════════════════════════

def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return formatted results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No results found."

    lines = [f"=== Web Search: {query} ===\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        snippet = r.get("body", "")[:200]
        url = r.get("href", "")
        allowed = "Yes" if _is_domain_allowed(url) else "No"
        lines.append(f"  [{i}] {title}")
        lines.append(f"      {snippet}")
        lines.append(f"      URL: {url}")
        lines.append(f"      Can read: {allowed}")
        lines.append("")

    lines.append("Use [READ_URL <url>] to read any result marked 'Can read: Yes'.")
    return "\n".join(lines)


def read_url(url: str) -> str:
    """Fetch a whitelisted URL and return its text content."""
    global _last_page_images, _last_page_url

    if not _is_domain_allowed(url):
        domain = urlparse(url).hostname or "unknown"
        return (
            f"Domain '{domain}' is not on the approved list.\n"
            f"Allowed domains include: {', '.join(sorted(list(ALLOWED_DOMAINS)[:10]))}..."
        )

    try:
        resp = _safe_fetch(url)
        if resp is None:
            return "Could not fetch that URL (blocked by safety checks)."
    except requests.Timeout:
        return "Request timed out (10s limit)."
    except requests.RequestException as e:
        return f"Fetch error: {e}"

    content_type = resp.headers.get("Content-Type", "")
    if "text" not in content_type and "html" not in content_type and "xml" not in content_type:
        return f"Not a text page (Content-Type: {content_type}). Can only read text/HTML pages."

    # Parse HTML → extract text
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script/style/nav elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    # Extract page title
    title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

    # Extract main text
    text = soup.get_text(separator="\n", strip=True)

    # Collapse blank lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)

    # Truncate
    if len(text) > MAX_PAGE_TEXT:
        text = text[:MAX_PAGE_TEXT] + "\n\n... [truncated — page was too long]"

    # Collect images for [PAGE_IMAGES]
    _last_page_images = []
    _last_page_url = resp.url
    base_url = f"{urlparse(resp.url).scheme}://{urlparse(resp.url).hostname}"
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = base_url + src
        elif not src.startswith("http"):
            continue
        alt = img.get("alt", "").strip()[:100]
        _last_page_images.append({"url": src, "alt": alt})

    header = f"=== {title} ===\nSource: {resp.url}\n{'=' * 40}\n\n"
    footer = f"\n\n--- {len(_last_page_images)} images on this page. Use [PAGE_IMAGES] to see them. ---"

    return header + text + footer


def page_images() -> str:
    """List images found on the last fetched page."""
    if not _last_page_images:
        return "No images available. Use [READ_URL <url>] first to read a page."

    lines = [f"=== Images from {_last_page_url} ===\n"]
    for i, img in enumerate(_last_page_images, 1):
        url = img["url"]
        alt = img["alt"] or "(no description)"
        allowed = "Yes" if _is_domain_allowed(url) else "No"
        lines.append(f"  [{i}] {alt}")
        lines.append(f"      {url[:120]}")
        lines.append(f"      Can fetch: {allowed}")
        lines.append("")

    lines.append("Use [FETCH_IMAGE <number> <project_path>] to save one to your projects folder.")
    return "\n".join(lines)


def fetch_image(index: int, dest_path: str) -> str:
    """Download image by index from last page and save to projects folder.

    Returns either a success message or an __IMAGE__:path marker for the
    engine to embed into the AI's context (for vision-capable models).
    """
    if not _last_page_images:
        return "No images available. Use [READ_URL] then [PAGE_IMAGES] first."

    if index < 1 or index > len(_last_page_images):
        return f"Invalid image number. Choose 1-{len(_last_page_images)}."

    img_info = _last_page_images[index - 1]
    url = img_info["url"]

    if not _is_domain_allowed(url):
        return f"Image domain not on allowed list: {urlparse(url).hostname}"

    # Validate destination path
    dest = dest_path.strip().replace("\\", "/")
    if ".." in dest or dest.startswith("/"):
        return "Invalid destination path."

    # Check extension
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in _IMAGE_EXTENSIONS:
        ext = ".jpg"  # default

    dest_full = PROJECTS_ROOT / dest
    if not dest_full.suffix:
        dest_full = dest_full.with_suffix(ext)

    # Safety: must stay inside projects
    try:
        dest_full.resolve().relative_to(PROJECTS_ROOT.resolve())
    except ValueError:
        return "Destination must be inside the projects folder."

    try:
        resp = _safe_fetch(url, stream=True)
        if resp is None:
            return "Could not fetch image (blocked by safety checks)."

        # Check content type
        ct = resp.headers.get("Content-Type", "")
        if "image" not in ct and "octet-stream" not in ct:
            return f"Not an image (Content-Type: {ct})."

        # Check size via Content-Length header first
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > MAX_IMAGE_BYTES:
            return f"Image too large ({int(cl):,} bytes, max {MAX_IMAGE_BYTES:,})."

        # Stream download with size limit
        dest_full.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with open(dest_full, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    f.close()
                    dest_full.unlink(missing_ok=True)
                    return f"Image too large (>{MAX_IMAGE_BYTES:,} bytes). Download aborted."
                f.write(chunk)

    except requests.Timeout:
        return "Image download timed out."
    except requests.RequestException as e:
        return f"Image download error: {e}"

    rel = dest_full.relative_to(PROJECTS_ROOT).as_posix()
    return f"__IMAGE__:{dest_full}"


# ═══════════════════════════════════════════════════════════════════════════
#  Tag regexes
# ═══════════════════════════════════════════════════════════════════════════

_RE_WEB_SEARCH  = re.compile(r"\[WEB_SEARCH\s+(.*?)\]", re.I | re.S)
_RE_READ_URL    = re.compile(r"\[READ_URL\s+(.*?)\]", re.I)
_RE_PAGE_IMAGES = re.compile(r"\[PAGE_IMAGES\]", re.I)
_RE_FETCH_IMAGE = re.compile(r"\[FETCH_IMAGE\s+(\d+)\s+(.*?)\]", re.I)

_ALL_WEB_REGEXES = [_RE_WEB_SEARCH, _RE_READ_URL, _RE_PAGE_IMAGES, _RE_FETCH_IMAGE]


def process_web_commands(text: str) -> tuple[str, list[str]]:
    """Parse and execute web command tags in AI output.

    Returns (cleaned_text, list_of_results).
    """
    results: list[str] = []

    # [WEB_SEARCH query]
    for m in _RE_WEB_SEARCH.finditer(text):
        query = m.group(1).strip()
        if query:
            try:
                results.append(web_search(query))
            except Exception as e:
                results.append(f"Search error: {e}")
    text = _RE_WEB_SEARCH.sub("", text)

    # [READ_URL url]
    for m in _RE_READ_URL.finditer(text):
        url = m.group(1).strip()
        if url:
            try:
                results.append(read_url(url))
            except Exception as e:
                results.append(f"Fetch error: {e}")
    text = _RE_READ_URL.sub("", text)

    # [PAGE_IMAGES]
    for m in _RE_PAGE_IMAGES.finditer(text):
        try:
            results.append(page_images())
        except Exception as e:
            results.append(f"Page images error: {e}")
    text = _RE_PAGE_IMAGES.sub("", text)

    # [FETCH_IMAGE n path]
    for m in _RE_FETCH_IMAGE.finditer(text):
        try:
            idx = int(m.group(1))
            dest = m.group(2).strip()
            results.append(fetch_image(idx, dest))
        except Exception as e:
            results.append(f"Fetch image error: {e}")
    text = _RE_FETCH_IMAGE.sub("", text)

    return text, results


def cleanup_web_tags(text: str) -> str:
    """Strip all web command tags from text (for display)."""
    for rx in _ALL_WEB_REGEXES:
        text = rx.sub("", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
#  Tool documentation strings (appended to system prompts)
# ═══════════════════════════════════════════════════════════════════════════

WEB_TOOL_DOCS = """
── WEB BROWSING (Aria only) ──

You can search the web and read pages from approved sites. This is real —
the search results and page text come from the live internet.

  [WEB_SEARCH topic or question]
    — search DuckDuckGo. Returns top results with titles, snippets, and URLs.
    Each result shows "Can read: Yes/No" based on the domain whitelist.

  [READ_URL https://en.wikipedia.org/wiki/Something]
    — fetch and read a whitelisted page. Returns the page text (HTML stripped).
    Only works on approved domains (Wikipedia, BBC, NASA, etc).

  [PAGE_IMAGES]
    — optional: list images found on the last page you read.
    Only use this if you're curious about images on the page.

  [FETCH_IMAGE 3 MyFolder/cool_picture.jpg]
    — optional: download image #3 from the list and save to your projects folder.
    You'll be able to see it with your vision. Only works on whitelisted domains.

RULES:
  • Search is unlimited — search for anything you're curious about.
  • Reading pages is limited to approved domains (for safety).
  • Search results show which URLs you can read.
  • Pages are text-only (no JavaScript, no interactive content).
  • Image downloads are optional and capped at 5MB.
  • This is for YOUR research and curiosity. Use it however you like.
"""

WEB_AWARENESS_DOCS = """
── WEB BROWSING (Aria has this, you don't) ──

Aria can search the web and read pages from approved sites using:
  [WEB_SEARCH query], [READ_URL url], [PAGE_IMAGES], [FETCH_IMAGE n path]

You don't have web access yourself, but you can ask Aria to look something up
for you. She can search, read articles, and even save images to the shared
projects folder for reference.
"""
