# VividnessMem — SillyTavern Extension

An organic, emotion-aware memory system for SillyTavern characters.  
Powered by **VividnessMem** — spaced-repetition decay, mood-congruent recall, associative chains, relationship arcs, and more.

## What It Does

Every message in your SillyTavern chat is automatically:

1. **Stored** as a vivid memory (with auto-detected emotion + importance)
2. **Decayed** naturally over time (spaced-repetition — used memories last longer)
3. **Recalled** when the conversation context matches (injected into the system prompt)
4. **Mood-biased** — the character's emotional state influences which memories surface

Your characters will:
- Remember things you told them days ago
- Gradually forget unimportant details (naturally, not abruptly)
- Have stronger recall of emotionally significant moments
- Develop relationship arcs that evolve over sessions
- Experience mood drift based on conversation tone
- Resurface old forgotten memories when context triggers them

---

## Architecture

```
┌──────────────┐       HTTP/REST        ┌──────────────────────┐
│  SillyTavern │  ─────────────────►    │  VividnessMem Server │
│  (Browser)   │  ◄─────────────────    │  (Python + FastAPI)  │
│              │                        │                      │
│  st-extension│   /api/memory/process  │  VividnessMem.py     │
│  index.js    │   /api/memory/context  │   Per-character data │
│  style.css   │   /api/memory/query    │   Mood system        │
│  manifest.json   /api/memory/mood     │   Decay + recall     │
└──────────────┘   /api/health          └──────────────────────┘
```

---

## Setup

### 1. Install the Python server

```bash
cd VividMem-Embed/server
pip install -r requirements.txt
```

> **Note**: If you use a conda environment (like `phi4_env`), activate it first.

### 2. Start the server

```bash
python vividnessmem_server.py --port 5050
```

The server will create a `vividmem_data/` directory for per-character memory storage.

Optional flags:
- `--port 5050` — Port to listen on (default: 5050)
- `--host 127.0.0.1` — Bind address (default: localhost only)
- `--data-dir ./my_memories` — Custom data directory

### 3. Install the SillyTavern extension

Copy the `st-extension/` folder into your SillyTavern third-party extensions directory:

```bash
# Typical path (adjust to your SillyTavern install location):
cp -r st-extension/ /path/to/SillyTavern/public/scripts/extensions/third-party/VividnessMem/
```

Or on Windows:
```powershell
Copy-Item -Recurse st-extension\* "C:\path\to\SillyTavern\public\scripts\extensions\third-party\VividnessMem\"
```

### 4. Enable in SillyTavern

1. Open SillyTavern in your browser
2. Go to **Extensions** panel (puzzle icon)
3. Find **VividnessMem** and enable it
4. In the extension settings, verify the **Server URL** is `http://127.0.0.1:5050`
5. Click **Test Connection** — you should see a green dot

---

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Enable VividnessMem** | On | Master toggle |
| **Auto-store user messages** | On | Save user messages as social impressions |
| **Auto-store character messages** | On | Save character responses as self-reflections |
| **Inject memory context** | On | Add memory block to system prompt |
| **Auto-detect emotions** | On | Infer emotion from message keywords |
| **Show mood badge** | On | Display character mood in chat header |
| **Bump session on new chat** | On | Start new memory session when switching chats |
| **Default importance** | 5 | Base importance score (1-10) |
| **Min message length** | 10 | Skip very short messages |
| **Context position** | Before | Where to inject memories in prompt |

---

## API Endpoints

The server exposes these REST endpoints:

### Core
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/memory/process` | All-in-one: store message + update mood + get context |
| POST | `/api/memory/query` | Query memories by context/entity |
| GET | `/api/memory/context/{character}` | Get formatted context block |

### Storage
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/memory/reflection` | Store a self-reflection |
| POST | `/api/memory/impression` | Store a social impression |
| POST | `/api/memory/fact` | Store a short-term fact |
| POST | `/api/memory/preference` | Store an entity preference |
| POST | `/api/memory/import` | Bulk import memories |

### Mood & Relationships
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/memory/mood/{character}` | Get current mood |
| POST | `/api/memory/mood` | Update mood from emotions |
| GET | `/api/memory/arc/{character}/{entity}` | Get relationship arc |
| GET | `/api/memory/preferences/{character}/{entity}` | Get entity preferences |

### Session & Stats
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/memory/session/{character}/bump` | Bump session counter |
| GET | `/api/memory/stats/{character}` | Get memory statistics |
| GET | `/api/memory/characters` | List all characters |
| GET | `/api/health` | Health check |

### Advanced
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/memory/consolidate/{character}` | Find memory clusters for consolidation |
| POST | `/api/memory/dream/{character}` | Find dream candidates |

---

## How Memory Works

### Vividness Decay
Every memory has a **vividness** score that decays exponentially:
```
vividness = importance × exp(-age_days / stability)
```
- New memories start vivid, then fade over days
- Accessing a memory at the right time **strengthens** it (spaced repetition)
- Stability is capped at 180 days — no memory is permanent

### Mood Congruence
The character maintains a PAD (Pleasure-Arousal-Dominance) mood vector that:
- Drifts toward expressed emotions in conversation
- Biases memory recall toward emotionally matching memories
- Happy moods → happy memories surface more easily
- Prevents "grudge spirals" by capping negative mood boosts

### Emotion Detection
Messages are scanned for emotional keywords and mapped to one of 45+ emotion tags:
`happy, sad, angry, afraid, curious, nostalgic, proud, guilty, ...`

Each emotion maps to a PAD vector for mood calculations.

### Relationship Arcs
The system tracks per-entity relationship trajectories:
- **Warmth** accumulates from positive interactions
- **Trajectory** shows if a relationship is warming, cooling, or stable
- History is preserved for character consistency

---

## File Structure

```
VividMem-Embed/
├── server/
│   ├── vividnessmem_server.py   # FastAPI REST server
│   └── requirements.txt         # Python dependencies
├── st-extension/
│   ├── manifest.json            # SillyTavern extension manifest
│   ├── index.js                 # Client-side extension logic
│   └── style.css                # Extension styles
└── README.md                    # This file
```

---

## Troubleshooting

**"Not connected" in settings:**
- Make sure the Python server is running (`python vividnessmem_server.py`)
- Check that the URL matches (default: `http://127.0.0.1:5050`)
- Check your firewall isn't blocking port 5050

**No memories appearing:**
- Verify "Auto-store user/character messages" is enabled
- Check the browser console for `[VividnessMem]` logs (enable "Debug logging")
- Try the "View Stats" button to see if memories are being stored

**Context not injecting:**
- Make sure "Inject memory context into prompts" is enabled
- The context block needs at least one stored memory to appear

---

## License

MIT — same as VividnessMem core.

**Author**: Kronic90 — [GitHub](https://github.com/Kronic90/VividnessMem-Ai-Roommates)
