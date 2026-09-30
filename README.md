# ComfyUI Local Generation Skills

Three Hermes Agent skills that drive a **local ComfyUI** server (no cloud API):
text-to-image, music/song generation, and instruction-based image editing.

| Skill | What it does | ComfyUI models it needs |
|---|---|---|
| `gen-image` | Text → image (Flux) | a Flux checkpoint **or** bare `flux1-dev` UNET + `clip_l` + `t5xxl_fp16` + `ae` VAE |
| `gen-music` | Lyrics + tags → song (ACE-Step 1.5) | `acestep_v1.5_turbo` + `qwen_0.6b_ace15` + `qwen_4b_ace15` + `ace_1.5_vae` |
| `gen-edit-image` | Instruction edit of an uploaded photo (Flux.2 Klein 4B) | `flux-2-klein-4b-fp8` + `qwen_3_4b` (as `type=flux2`) + `flux2-vae` |

Every skill is **stdlib-only Python** — no `pip install` needed. Each ships a
single script that checks the server, builds the workflow, submits it, waits,
downloads the result, and prints one JSON line whose `reply` field is the
`MEDIA:"<path>"` line the agent echoes back.

## Requirements

1. **ComfyUI** running locally at `http://127.0.0.1:8188` (start it yourself;
   these skills never launch or install it).
2. **The model files** for whichever skill you use, in ComfyUI's own model
   folders (`models/diffusion_models`, `models/text_encoders`, `models/vae`,
   or `models/checkpoints`). File names are auto-detected against the running
   server via `/object_info`; use `--model/--clip/--vae` to override.
3. **A GPU with enough VRAM** for the model you pick — 16 GB comfortably runs
   all three; the ACE-Step and Wan-class models are the heaviest.

## Install

### Option A — copy the folders (works everywhere)

```bash
cp -r skills/gen-image skills/gen-music skills/gen-edit-image \
      ~/.hermes/skills/creative/
```

Windows (PowerShell / cmd): copy them to
`%LOCALAPPDATA%\hermes\skills\creative\`.

Restart the session (or `/reload-skills`) and the three skills are live.

### Option B — install from this GitHub repo

Push this repo, then on the target machine:

```bash
hermes skills tap add <owner>/<repo>
hermes skills search gen-edit-image
hermes skills install <owner>/<repo>/skills/gen-edit-image
```

Install the other two the same way. A tap install copies the whole skill
directory, `scripts/` included, so nothing extra to fetch.

### Option C — one file at a time (no repo)

`hermes skills install <https://…/SKILL.md>` fetches the SKILL.md **plus** any
support file the body explicitly references, so the script and template come
along — as long as they sit under `references/`, `templates/`, `scripts/`, or
`assets/` (they do).

## Verify the install

No ComfyUI needed for the offline check:

```bash
python ~/.hermes/skills/creative/gen-edit-image/scripts/selfcheck.py
# -> graph wiring OK / multi-reference wiring OK / upload OK / all checks passed
```

With ComfyUI running, one real smoke render per skill:

```bash
python ~/.hermes/skills/creative/gen-image/scripts/render_image.py \
  --prompt "a red apple on a wooden table" --style realistic --ratio 1:1 \
  --title smoke --seed 42 --output-dir "$HOME/comfy_test/images"

python ~/.hermes/skills/creative/gen-music/scripts/render_song.py \
  --lyrics-file /path/to/lyrics.txt \
  --tags "pop ballad, acoustic guitar, female vocals" \
  --title smoke_song --duration 30 --bpm 90 --seed 42 \
  --output-dir "$HOME/comfy_test/songs"

python ~/.hermes/skills/creative/gen-edit-image/scripts/edit_image.py \
  --image "/path/to/some/photo.jpg" \
  --prompt "Change the car body color to glossy black." \
  --title smoke_edit --seed 42 --output-dir "$HOME/comfy_test/edits"
```

Each prints one JSON line. `"status": "success"` **plus** a real file on disk
= verified. `status: error` carries `step` + `error`; fix and retry once.

## Windows notes (these skills were built on Windows)

- **Quote every path argument.** Windows user dirs often contain a space; an
  unquoted `--output-dir` splits in the shell and the run dies on a usage error.
- **Invoke Python with forward-slash paths** (`C:/Users/<user>/...`). Passing a
  git-bash MSYS path (`/c/Users/...`) to `python.exe` arrives unmapped —
  `can't open file 'C:\c\Users\...'`.
- **Long renders**: run with the terminal tool in background mode
  (`background=true, notify_on_complete=true`) and collect with
  `process(action='wait')` — a cold 24 GB model load exceeds a foreground cap.
- **VRAM** is shared. Before a heavy render, unload an idle Ollama model
  (`curl -s http://127.0.0.1:11434/api/generate -d '{"model":"<name>","keep_alive":0}'`)
  and check `nvidia-smi`. A 16 GB card showing ~14 GB used with an empty queue
  is ComfyUI's own cache, not a leak.

## How the skills are written

They target small local models (a 9B-class model runs them fine). The pattern,
if you want to write your own in the same shape:

1. Mechanical numbered steps, one action per turn.
2. Interview/clarify only the fields the user did not already give.
3. **One** script command per render — the model never hand-builds workflow JSON.
4. The script returns a `reply` field the model echoes verbatim (breaks if the
   model retypes it).
5. An explicit anti-fabrication rule: never present a path, seed, or file that
   no real successful run produced.
