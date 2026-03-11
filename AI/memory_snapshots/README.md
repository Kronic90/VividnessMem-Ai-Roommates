# Memory Snapshots

Real memory files from live Aria + Rex sessions. These are the actual memories both AIs created through conversation — not test data.

**Snapshot date:** March 11, 2026 (after ~10 sessions)

## Aria (Gemma 12B — Organic/Vividness Memory)

| File | Description | Entries |
|------|-------------|---------|
| `aria/self_memory.json` | Identity journal — who she's becoming | 1 |
| `aria/social/rex.json` | Her impressions of Rex | 2 |
| `aria/social/scott.json` | Her impressions of Scott (the developer) | 1 |
| `aria/task_memories.json` | Task learning — what she's built and learned | 48 |

Each self/social memory has:
- `content` — freeform journal entry in her own voice
- `emotion` — tagged by her, in her own words
- `importance` — 1-10, rated by her at creation time
- `access_count` — how many times resonance has pulled this memory back up

## Rex (Qwen3.5 4B — MemGPT-Style Memory)

| File | Description |
|------|-------------|
| `rex/self_memory.json` | Core (always in context) + Archival (searchable) |
| `rex/social/aria.json` | His impressions of Aria — also core + archival |
| `rex/task_memories.json` | Task learning entries |

Rex uses a structured core/archival split instead of vividness scoring. High-importance memories go to core (always visible), others go to archival (surfaced on demand).

## What to look for

- **Aria's emotion tags** are freeform and specific: "Complex — gratitude mixed with self-consciousness", "Wary but respectful"
- **Rex's emotion tags** tend toward analytical: "chilled but analytical", "intrigued and satisfied"
- **Task memories** show the Aetheria worldbuilding project they've been collaborating on
- **Access counts** show which memories keep getting triggered by conversation (resonance in action)
- Rex's `importance: 10` memory about the Council of Tuners being "infinitely more effective" at coercion — that was his idea, not prompted

These are the files our test suites (233 tests) validate against.
