---
name: gen-edit-image
description: "Edit an existing image from an instruction (Flux.2 Klein 4B on local ComfyUI): user uploads the picture in chat, script uploads it to ComfyUI, renders the edit, embeds the result. Use for color swaps, object replacement, background changes, 'ubah X jadi Y'. (Text-to-image from scratch is the separate gen-image skill.)"
version: 1.0.0
platforms: [windows]
metadata:
  hermes:
    tags: [image, image-edit, gen-edit-image, flux2, klein, comfyui, instruction-edit]
    category: creative
---

# Gen Edit Image — Flux.2 Klein (Local ComfyUI)

Instruction-following image edit in **3 fixed steps, in order**. The user
drops/uploads the source picture straight into the chat; the script uploads
it to ComfyUI, renders, and downloads the result.

Related: `gen-image` makes a NEW image from text; this skill EDITS one the
user already has. Never guess — if no image was provided, use this skill's
Step 1 rule (ask for the image), do not fall back to `gen-image`.

All logic lives in `scripts/edit_image.py`; `scripts/selfcheck.py` is the
offline check. Never build workflow JSON by hand, never call ComfyUI directly.

**Reply discipline (mandatory, small models):** every turn ≤5 short lines,
ONE action per turn. No option tables, no emoji, no background notes (model
names, paths, VRAM). This SKILL.md is the ONLY procedure. If the image path
and the instruction are already known, your FIRST turn is the confirm line
plus the Step 2 command.

- ComfyUI at `http://127.0.0.1:8188` (user starts it manually).
- Needs these files (auto-resolved by the script from `/object_info`):
  `flux-2-klein-4b-fp8.safetensors` (diffusion_models), `qwen_3_4b.safetensors`
  (text_encoders, loaded as `type=flux2`), `flux2-vae.safetensors` (vae).
- Graph: `UNETLoader → CFGGuider(cfg 1.0)`, prompt → `ReferenceLatent` (positive),
  `ConditioningZeroOut → ReferenceLatent` (negative), `Flux2Scheduler` **4 steps**,
  `euler`, `RandomNoise`, `SamplerCustomAdvanced`, input resized to 1 MP by
  `ImageScaleToTotalPixels` and its size drives the output latent. Do NOT
  hand-build or tweak the graph — the script owns it.

## Step 0 — Server & queue check

```
curl -s --max-time 5 http://127.0.0.1:8188/system_stats
curl -s --max-time 5 http://127.0.0.1:8188/queue
```

- `/system_stats` fails → STOP: "ComfyUI belum jalan, tolong start dulu."
- `queue_running`/`queue_pending` non-empty → STOP, tell the user, poll
  `/queue` until free. Never submit on top of a running queue.
- A 16GB card reporting ~14GB used with an empty queue is ComfyUI's own cache,
  not a blocker.

## Step 1 — Get the source image path + the instruction

**Source image — where the path comes from (do NOT ask the user to type a path):**

1. The desktop app attaches images itself. An attached image arrives in your
   context as a block like
   `[The user attached an image: <description>]`
   `[You can examine it with vision_analyze using image_url: C:\...\file.jpg]`.
   **That `image_url:` value IS the local path — pass it to `--image`.**
2. If the user instead gave a path, an `@file:`/`@image:` ref, or an http(s)
   URL, use that. Quoted paths with spaces are fine.
3. Only if there is genuinely no image (no attachment, no path, no URL) do you
   ask: "Kirim/upload gambarnya dulu ya (drag ke chat), atau kasih path-nya."

**Instruction — what to ask if missing:** one `clarify` call, free text:
"Mau diubah apanya? (contoh: ganti warna tas jadi biru, ganti background jadi pantai)".
Skip it entirely when the user's message already states the change.

Required: image path, instruction. Optional (never ask): seed (default -1),
output dir, `--megapixels` (default 1.0), title (default `edit`).

## Step 2 — Render (one command)

Translate the instruction to **English** first and SHOW it in your reply.
Full imperative sentence, concrete, one change per run — the 4B distilled
model follows short literal instructions best ("Change the bag color to
blue."), and degrades on long paragraphs of style keywords. No style rows,
no quality tails (unlike gen-image — do NOT add them here).

**Multi-image (background/style transfer):** `--image` takes ONE OR MORE
paths. The FIRST is the photo being edited (its size drives the output), each
extra one is an additional reference the prompt can name ("...from the second
image"). Extras are chained through further `ReferenceLatent` nodes, i.e.
they land in the same `reference_latents` list — real multi-reference
conditioning, no custom node needed.

```bash
python "<skill_dir>/scripts/edit_image.py" \
  --image "<edit target>" ["<extra reference>" ...] \
  --prompt "<ENGLISH INSTRUCTION>" \
  --title "<short title>" --seed -1 \
  --output-dir "<workspace>/images"
```

**Colour bleed is the failure mode of a reference-image swap.** Asking
"Replace the background with the reference's studio background." once gave a
studio background AND a blue car — the model copied the reference's colour
onto the subject. Fix that worked (same seed-free re-run): freeze the subject
first, then describe the change — "Keep the car body pearl white and
unchanged. Only replace the road and trees background with the plain blue
studio background from the second image." Always vision-check a multi-image
result for exactly this; if the subject got recoloured, say so and re-run
with the subject pinned.

QUOTE every path argument (this machine's user dir has spaces). Add
`--megapixels N` or `--steps N` only to override (steps 4 is the distilled
sweet spot; more steps does not help). Default timeout 600s.

Run it with `terminal(background=true, notify_on_complete=true)` when the
render may exceed the 600s foreground cap, then collect with
`process(action='wait')`.

The script prints ONE JSON line:
`{"status": "success", "file": [...], "seed": N, "model": "...", "steps": 4,
"source": "hermes_edit_<hash>.jpg", "media_line": "MEDIA:\"...\"", "reply": "MEDIA:\"...\""}`.
On `status=error` show `error`, fix, retry once — never emit a MEDIA line.

## Step 3 — Deliver (verbatim echo)

- Output the JSON **`reply`** field VERBATIM: one line, `MEDIA:"C:/..."`,
  double quotes included, first line of the reply, nothing appended (no
  "Seed: N", no caption, no code block). Rewrites break the embed.
- NEVER present a path/seed not produced by a real successful run in THIS
  conversation. If the run failed, say it failed.
- Optional follow-up, separate sentence: variasi seed lain, ubah instruksi,
  atau `--megapixels` lebih besar untuk detail.
- **`status: success` is not proof the edit landed.** Before presenting,
  `vision_analyze` the output PNG and confirm the named object actually
  changed. Real case: "Change the bag color to blue." returned a successful
  render of a subject who owns no bag — the model recolored the shirt instead.
  Say so honestly ("objek X tidak ada di gambar; model malah ganti Y") and
  offer a corrected instruction. Never present it as the requested edit.

Evidence for every claim above (verified renders + seeds + timing, upload
mechanics, the attachment-path flow, self-check coverage):
`references/verified-runs.md`.

## Troubleshooting

**Before any of these:** if the failure smells like the *driving model*
forgetting the skill rules, losing the source path, or truncating its own
reply, check the Ollama context window first — on Ollama 0.32 the `/v1`
endpoint ignores per-request `num_ctx` and every Hermes session runs at 4096,
so this SKILL.md plus a vision description can silently evict the earlier
turns. See `devops/local-llm-providers`.

| Symptom | Fix |
|---|---|
| ComfyUI not reachable | Ask the user to start it (Step 0). |
| `cannot resolve Flux.2 [...]` | Model files moved — the script auto-detects; re-run. Report file names if it persists. |
| HTTP 400 `value_not_in_list` | Same as above; never hand-edit the graph. |
| Edit ignored / output ≈ input | Instruction too vague or too long. Rewrite as one short imperative English sentence naming the object and the new value. |
| Out of memory | Lower `--megapixels` (0.5) or pass a smaller source image. |
| Edit leaks into unrelated areas | Expected with a 4B distilled model at 4 steps; accept it or re-run with `--seed` variations and pick the best. |
| Output soft / over-saturated vs source | Input is downscaled to 1 MP. Raise `--megapixels 2` for a sharper result. |
| `upload failed` | Source file unreadable or >100 MB; check the path. |
| Timeout | Re-run with `--timeout 600`; models stay warm. |
| Edited code (graph/upload) | Offline check, no ComfyUI needed: `python "<skill_dir>/scripts/selfcheck.py"` — asserts graph wiring + a multipart upload against a throwaway fake server. Then do one real smoke render. |
| Script edits seem to have no effect | The script exists in ONE place: `<skill_dir>/scripts/edit_image.py`. Do NOT copy it next to ComfyUI's data dir — `demo/ComfyUI/user/` holds `comfyui.db`, so it is data, not a code folder. A stale second copy means the run executes the old code and the symptom looks like the edit was ignored. |
| Named object not in the image | The 4 B model edits *something* plausible instead of failing. Re-aim the instruction at an object that exists, or tell the user the edit isn't possible for that photo. |
| Reference image's colour bled onto the subject | Pin the subject in the prompt ("Keep the car body pearl white and unchanged. Only replace ..."), then re-run. See Step 2. |
