---
name: gen-image
description: "Image request: interview, translate prompt, render, embed (Flux on ComfyUI). Verified under qwen3.5:9b — after SKILL.md edits re-test via software-development/local-llm-skill-testing."
version: 1.6.1
platforms: [windows]
metadata:
  hermes:
    tags: [image, gen-image, flux, comfyui, text-to-image, prompt-engineering]
    category: creative
---

# Image Generation — Flux (Local ComfyUI)

Generate an image in **4 fixed steps, in order, mechanically**. Step 1 is
extract-then-ask: never re-ask what the user already provided.

All logic lives in `scripts/render_image.py` — the model only interviews and
passes parameters. Never build workflow JSON by hand, never call ComfyUI
directly.

**Reply discipline (mandatory, small models):** every turn ≤5 short lines,
ONE action per turn. Never output option tables, emoji decorations, or
background notes (model names, paths, VRAM, "workflow patterns"). This
SKILL.md is the ONLY procedure — ignore any other setup notes from memory
or elsewhere. If every required field is already known, your FIRST turn is
the confirm line + the Step 3 command. Short replies never get truncated.

**Model is auto-detected by the script — never hardcode a filename.** The
skill supports both ComfyUI layouts:

| Layout | Where the file lives | Graph used |
|---|---|---|
| A — checkpoint | `models/checkpoints/*.safetensors` | `CheckpointLoaderSimple` |
| B — bare UNET | `models/diffusion_models/*.safetensors` | `UNETLoader` + `DualCLIPLoader(type=flux)` + `VAELoader` |

Both layouts may coexist on one machine (e.g. a flux schnell **checkpoint**
AND a bare `flux1-dev` UNET). Auto-detect prefers a flux checkpoint (fast,
4 steps) over the bare UNET; pass `--model flux1-dev.safetensors` explicitly
for the full-quality dev render (20 steps, `fp8_e4m3fn`, needs `clip_l` +
`t5xxl_fp16` + the `ae` VAE — much slower first load). If generation fails
with `ckpt_name: '...' not in []`, the model lives in the other layout and
the script will resolve it on its own — just re-run.

Environment: ComfyUI at `http://127.0.0.1:8188` (user starts it manually),
RTX 5080/5090 Laptop, 16–24GB VRAM. Max safe size: 1536 px on the long side.

Note: this machine may also run LM Studio + Ollama holding VRAM. Unload an
idle Ollama model before a render (Step 0) — a 24GB dev UNET needs the room.

## Graph shape (matches the official ComfyUI template)

The script's layout-B graph mirrors `templates/flux_dev_unet_template.json`
(shipped with this skill, from ComfyUI's Flux.1 Dev example):

```
UNETLoader ─┬─> KSampler.model
DualCLIPLoader(type=flux) -> CLIPTextEncode -> FluxGuidance -> KSampler.positive
                          └-> ConditioningZeroOut             -> KSampler.negative
VAELoader -> VAEDecode -> SaveImage
```

Rules that follow from the template — do NOT deviate:

- `cfg` stays at **1.0** on Flux. The real quality knob is **`--guidance`**
  (FluxGuidance, default 3.5). Raising cfg above 1.0 is off-distribution for
  Flux and degrades output.
- Because cfg is 1.0, a text negative has **zero** effect. The template uses
  `ConditioningZeroOut` of the positive encode instead — keep that node rather
  than adding a second CLIPTextEncode for the negative (it also saves a full
  T5 encode on every run). `--negative-mode auto` does the right thing;
  `--negative-mode text` only makes sense if you deliberately set cfg > 1.0.
- Seed range is 64-bit (`0 … 2^63-1`), not 32-bit.
- `DualCLIPLoader` takes `device: "default"` as an optional input.

## Step 0 — Server & VRAM check

```
curl -s --max-time 5 http://127.0.0.1:8188/system_stats
curl -s --max-time 5 http://127.0.0.1:8188/queue
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

- `/system_stats` fails → STOP: "ComfyUI belum jalan, tolong start dulu."
- `/queue` shows `queue_running` or `queue_pending` non-empty → STOP: a job
  is already running. Tell the user, wait for it to finish (poll `/queue`).
  Never submit on top of a running queue.
- `memory.used` > 12000 MiB → VRAM is crowded (usually Ollama holding a
  model). Ask the user before proceeding, or run:
  `curl -s http://127.0.0.1:11434/api/generate -d '{"model": "<model>", "keep_alive": 0}'`
  to unload the idle Ollama model first. GPU util ~1% during a "running"
  job = CPU-offload stall: interrupt it (`curl -X POST .../interrupt`) and
  free VRAM (`curl -X POST .../free -d '{"unload_models": true, "free_memory": true}'`),
  then re-submit.

## Step 1 — Extract first, ask ONLY what's missing

**Parse the user's opening request BEFORE asking anything.** Extract every
field already given: tipe gambar (style), prompt, ratio, plus extras (title,
size, seed, negative requests). If the user invoked the skill explicitly
(e.g. `/gen-image <description>`), treat that message as a complete request
spec — do NOT re-ask what it already answers.

- All required fields present → NO interview at all. Confirm the extracted
  spec in ONE short line (style / prompt summary / ratio) and go to Step 2.
- Something missing or vague (e.g. only "buat gambar keren") → use the
  `clarify` tool — it takes ONE question per call, so several missing
  fields = SEQUENTIAL clarify calls, one per field (a multi-question
  single call fails with "Question text is required"). Ask ONLY the
  missing fields:

1. Tipe gambar? (choices: Realistic / Anime / Cartoon / Watercolor — the
   "Other" free-text row covers "3d character" and "hand drawing"; map any
   answer to a `--style` value in Step 3)
2. Promptnya apa? (free text; user may write in any language)
3. Mau ratio berapa? (choices: 1:1 / 16:9 / 9:16)

Required: style, prompt, ratio. Optional (never ask): title (default
"image"), seed (default -1).

**Worked example — spec complete, render immediately (turn 1):**

User: "woman in Japan 9:16 realistic"

Reply line 1 (confirm): `Realistic · woman in Japan · 9:16`
Reply line 2 (this bash block, with `<skill_dir>`/`<workspace>` replaced by
real paths):

```bash
python "<skill_dir>/scripts/render_image.py" \
  --prompt "woman in Japan" \
  --style realistic \
  --ratio "9:16" \
  --title "Japan Woman" --seed -1 \
  --output-dir "<workspace>/images"
```
```

## Step 2 — Translate the prompt (the script appends style, you don't)

1. If the user's prompt is not English, translate it to English first
   (Flux is English-trained; always render from English). The `--prompt`
   value must be 100% English — no Indonesian words in the render command.
2. Keep it as the user's subject/scene words; do NOT add style keywords —
   pass `--style` instead and the script appends the correct style row +
   quality tail mechanically. Never add artist names or keyword spam.
3. SHOW the translated English prompt to the user in your reply before
   rendering.
4. NEVER fabricate or predict a MEDIA path/filename — a MEDIA line comes
   ONLY from the script's JSON `reply` after a real, successful run
   (Step 3). Before echoing any MEDIA line, CHECK the JSON `status`:
   `status=error` → say the render failed, show `error`, fix, retry once —
   NEVER emit a MEDIA line for a failed run.
5. Negative prompt: the script already includes a standard negative
   (blurry, bad anatomy, watermark, text, ...). Only pass `--negative` for
   user-requested exclusions. Flux runs at cfg 1.0; the real knob is
   `--guidance` (default 3.5) — prompt quality is what matters.

## Step 3 — Render (one command, no hand-built JSON)

```bash
python "<skill_dir>/scripts/render_image.py" \
  --prompt "<TRANSLATED ENGLISH PROMPT from Step 2>" \
  --style "<realistic|anime|cartoon|watercolor|3d character|hand drawing>" \
  --ratio "<1:1|16:9|9:16>" \
  --title "<short title>" --seed -1 \
  --output-dir "<workspace>/images"
```

QUOTE every path argument (`--output-dir` included — Windows user dirs have
spaces; unquoted args split in the shell and the run fails with a usage
error). Add `--guidance N`, `--steps N`, `--width/--height`, or
`--model <file>` only when you need to override the auto-detected defaults.

The script checks the server, auto-detects the model layout, builds the
workflow, submits, waits, downloads the PNG, and prints ONE JSON line:
`{"status": "success", "file": [...], "seed": N, "layout": "unet", "style_label": "...", "media_line": "MEDIA:\"...\"", "reply": "MEDIA:\"...\""}`.
The MEDIA path is DOUBLE-QUOTED (e.g. `MEDIA:"C:/Users/<user>/images/x.png"`)
— quoting is REQUIRED so the desktop renderer parses paths containing spaces
(the desktop regex treats an unquoted path with spaces as plain text, which is
why past embeds silently failed on machines whose username has a space).
`reply` is exactly the MEDIA line, with no metadata appended (see Step 4).
On `status=error` show `error`, fix the cause, retry once.
First render after ComfyUI start is slower (model load: minutes for a 24GB
dev UNET at 1344x768, 20 steps); later runs are faster.

## Step 4 — Deliver (verbatim echo, DO NOT compose your own sentence)

- NEVER present a MEDIA path, seed, or file that was not produced by a real,
  successful script run in THIS conversation. If the run failed or never
  happened, say so — never fabricate a plausible-looking path or seed.

- The script's JSON output contains a **`reply`** field. That field is the
  COMPLETE final answer text and is intentionally **nothing but the MEDIA
  line**.
- **Output the `reply` field VERBATIM as your reply** — every character
  unchanged, INCLUDING the double quotes around the path. Do NOT retype the
  filename, do NOT write "Open ...", do NOT create your own sentence, do NOT
  wrap it in a code block. Any rewrite breaks the embed and the user sees
  nothing. The quotes are part of the syntax, not decoration.
- **Do NOT append anything to the MEDIA line** — no "Style: ... | Seed: ..."
  line, no caption, no blank line before it. Small echo models (qwen 9B and
  friends) concatenate whatever follows the MEDIA line into the path, and the
  image then fails to load. Metadata lives in the JSON fields
  (`seed`, `style_label`, `size`), never in the reply text. If you want to
  report the seed, put it in a SEPARATE sentence after the MEDIA line, on the
  same message but clearly apart — safest is to mention it only in the
  follow-up sentence, not adjacent to the path.
- The `MEDIA:` line MUST be the first line of your reply and stand alone.
  The script now emits it **quoted** — `MEDIA:"C:/Users/<user>/.../x.png"`
  — because the desktop renderer matches media paths with a `\S+` regex: an
  unquoted path containing spaces (a Windows user dir with a space in it) is
  NOT matched and renders as dead text instead of an embed. Quoted paths
  match on desktop AND gateway. Echo the `reply` field as-is, quotes included.
- Offer follow-ups in a SEPARATE sentence after the reply block: new seed
  (variasi), ubah prompt/style, ratio lain.
- If you already rendered but lost the JSON output, re-run the script with
  the same parameters and seed to regenerate it (fast once models are warm).

## Troubleshooting

| Symptom | Fix |
|---|---|
| ComfyUI not reachable | Ask the user to start ComfyUI (Step 0). |
| HTTP 400 `value_not_in_list` (ckpt_name/unet_name) | The model moved folders — the script auto-detects both layouts; re-run it. Never hand-edit the graph JSON. |
| HTTP 400 "class_type not found" | Node missing — report the node name. |
| Out of memory | Lower resolution (`--width/--height`, multiples of 16), or add `--weight-dtype fp8_e4m3fn`. |
| First render slow | Normal — loading a 24GB UNET into VRAM takes minutes. |
| Timeout | Re-run with `--timeout 600`; the model stays warm afterwards. |
| Job never appears to progress | Check the queue directly (Step 0); a stall usually means VRAM is held by Ollama/LM Studio. |
| Result looks washed out / over-cooked | Adjust `--guidance` (3.0–4.0 is normal), not `--cfg`. |
| MEDIA line shows as plain text, no inline image | Path not quoted or model appended text to it — always echo `reply` verbatim; the script emits `MEDIA:"path"` (quoted) because desktop paths can contain spaces. |
| Render killed at ~10 min / shell timeout | Foreground `terminal` caps at 600s. Run renders with `terminal(background=true, notify_on_complete=true)`, collect via `process(action='wait')`. `wait` may clamp to a short per-call timeout — repeat until `exited`; repeated "still running" timeouts on a live render are normal, not a hang. |
| New machine / fresh ComfyUI install | Run the full smoke test in `references/fresh-install-verification.md` (models on disk, node registry via `/object_info`, VRAM sanity, 2 smoke renders at seed 42). A 16GB card showing ~14.7GB used with an empty queue is ComfyUI's own VRAM cache, not Ollama — check `http://127.0.0.1:11434/api/ps` before blaming VRAM. |

## Neighbouring skill — do not confuse the two

- `gen-image` (this skill) → a NEW image from text only: `--prompt` + `--style`
  + `--ratio`, Flux auto-detected, style row and quality tail appended by the
  script.
- `gen-edit-image` → EDITS a picture the user supplies (uploaded into the chat
  or given as a path). Different model (Flux.2 Klein 4B, 4 steps), different
  graph (ReferenceLatent), and a different prompt contract: ONE short
  imperative English sentence, no style row, no quality tail.

If the user attached/provided an image and wants it changed, that is
`gen-edit-image`. If they want a picture made from a description, it is this
skill. Getting this wrong wastes a render and confuses the user.
