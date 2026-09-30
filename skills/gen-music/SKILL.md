---
name: gen-music
description: "Music/song requested: interview, lyrics, approval, render. Verified under qwen3.5:9b — after SKILL.md edits re-test via software-development/local-llm-skill-testing."
version: 2.1.1
platforms: [windows]
metadata:
  hermes:
    tags: [music, song, gen-music, ace-step, comfyui, audio-generation, lyrics]
    category: creative
---

# Music Generation — ACE-Step (Local ComfyUI)

Generate a song in **4 steps, in order**. Step 3 (approval) is never skipped.
All logic (server/queue/VRAM checks, genre presets, workflow build) lives in
`scripts/render_song.py` — the model only conducts the interview and passes
parameters. Never build workflow JSON by hand, never call ComfyUI directly.

## Step 1 — Extract first, ask ONLY what's missing

Parse the user's opening request BEFORE asking. Extract: tema (atau lirik
lengkap), genre, durasi, vokal & bahasa. Skill invoked with a full
description → treat it as the spec, do NOT re-ask.

- All required fields present → NO interview. Confirm spec in ONE line.
- Complete lyrics given → show them, go STRAIGHT to Step 3 (approval still
  required).
- Missing/vague → `clarify`, only the missing fields. `clarify` takes ONE
  question per call — several missing fields = SEQUENTIAL clarify calls,
  one per field (a multi-question single call fails with "Question text
  is required"):
  1. Tema lagunya tentang apa? (free text)
  2. Judul? (free text, boleh "bebas")
  3. Genre? (choices: Pop ballad / Pop ceria / Rock / Lo-fi / Electronic /
     Dangdut-pop)
  4. Durasi? (choices: 30 / 60 / 120 detik)
  5. Vokal? (choices: Pria – Indonesia / Wanita – Indonesia / Pria – English /
     Instrumental)

Required: tema atau lirik, genre, durasi, vokal & bahasa. Never ask: BPM,
key, seed, tags — the script derives them from the genre preset.

## Step 2 — Draft lyrics

- Plain-text section tags: `[Intro]` `[Verse 1]` `[Pre-Chorus]` `[Chorus]`
  `[Verse 2]` `[Bridge]` `[Final Chorus]`. No markdown.
- Length follows duration: ≤30s → 1 verse + 1 chorus (4–8 lines). 60s →
  verse + pre-chorus + chorus + verse 2 + chorus. 120s → full structure.
- Write in the chosen language; short singable lines.
- Instrumental → no lyrics file needed.
- Show full lyrics with title on top.

## Step 3 — Approval (REQUIRED)

`clarify`: "Liriknya sudah oke untuk di-render?" (choices: "Ya, lanjut
render" / "Revisi"). Revisi → fix, show again, ask again. Never render
unapproved lyrics.

## Step 4 — Render ONE command, echo reply VERBATIM

1. `write_file` approved lyrics to `<workspace>/lyrics.txt` (skip if
   instrumental). If write_file warns a sibling session modified this
   lyrics.txt, re-read the file first — concurrent sessions share the
   path; overwrite without reading can render another session's lyrics.
2. Run (script lives in THIS skill's `scripts/` dir):

```bash
python "<skill_dir>/scripts/render_song.py" \
  --lyrics-file <workspace>/lyrics.txt \
  --genre <pop-ballad|pop-ceria|rock|lofi|electronic|dangdut-pop> \
  --vocal <male|female|instrumental> \
  --language <id|en> \
  --title "<Title>" --duration <seconds> --seed -1 \
  --output-dir <workspace>/songs
```

The script checks server/queue/VRAM itself, resolves BPM/keyscale/tags from
the genre, builds the workflow, submits, waits, downloads the MP3. Modes:
`--check` (preflight only), `--presets` (list genre presets). On failure it
prints `{"status":"error","step":...,"error":...}` — show `error`, fix,
retry once. See troubleshooting below.

**Run mode:** a render plus first model load can exceed the foreground
`terminal` cap (600s). Run renders with `terminal(background=true,
notify_on_complete=true)` and collect via `process(action='wait')`;
foreground only for known-fast re-runs. `wait` may clamp to a short per-call
timeout — call it repeatedly until `exited`; repeated "still running"
timeouts on a live render are normal, not a hang.

**Delivery (anti-fabrication + embed rules):**
- NEVER present a MEDIA path/seed/file not produced by a real successful run
  in THIS conversation. Run failed → say so.
- JSON `reply` field = the COMPLETE final answer. Echo VERBATIM, quotes
  included, first line, nothing appended (no caption, no metadata line —
  extra text after the path breaks the embed; metadata lives in JSON:
  `seed`, `bpm`, `tags`).
- Report seed in a SEPARATE sentence after the reply. Offer follow-ups
  (variasi seed baru, genre lain) also after.
- Lost the JSON output → re-run with same params to regenerate.

## Troubleshooting

| `step` in JSON | Fix |
|---|---|
| server_check | ComfyUI down — ask user to start it. |
| queue | Job already running — wait, retry. |
| vram | Free VRAM yourself before asking the user: `curl -s http://127.0.0.1:11434/api/ps` shows which model holds it; unload with `curl -s http://127.0.0.1:11434/api/generate -d '{"model": "<name>", "keep_alive": 0}'` (an idle model reloads automatically when next used). Confirm with nvidia-smi, retry. If ComfyUI's own cache is the hog, `curl -X POST http://127.0.0.1:8188/free -d '{"unload_models": true, "free_memory": true}'`. |
| submit, HTTP 400 "class_type not found" | Custom node missing — report node name. |
| execute, KSampler 'NoneType' | Should not happen — script builds the graph. If it does, report as bug. |
| execute, node interrupted (e.g. `TextEncodeAceStepAudio1.5`) | Usually OOM during encoder/model load, not a script bug. Diagnose: `curl -s http://127.0.0.1:8188/history/<prompt_id>` → look for `execution_interrupted` + `node_type`. Free VRAM (see vram row), then retry ONCE with the SAME seed — the retry succeeded after VRAM was freed in practice. |
| wait, timeout | Re-run with `--timeout 1800` or shorter duration. First render slow (~2–4 min) is normal. |

## Notes

- Genre presets + BPM/keyscale/tags live in the script (`--presets` prints
  them). Language `id` supported.
- seed `-1` = random; report actual seed from JSON for variants.
- Python MUST be invoked with `C:/Users/...` paths; MSYS `/c/Users/...`
  breaks under git-bash (`can't open file 'C:\c\Users\...'`).
- Lyrics file must be UTF-8, exactly the approved text.