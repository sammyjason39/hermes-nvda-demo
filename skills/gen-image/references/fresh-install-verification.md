# Fresh-install / Migration Smoke Test (ComfyUI + gen skills)

End-to-end verification that a new ComfyUI install can actually render through
the gen-image and gen-music skill scripts. Run in order. ~2 min of checks +
2 smoke renders. Proven working Sep 2026 (RTX 5080 Laptop 16GB, ComfyUI 0.37.0,
install root `C:\Users\<user>\demo\ComfyUI`).

## 1. Server up

    curl -s --max-time 5 http://127.0.0.1:8188/system_stats

Expect `comfyui_version` + `devices[]` with VRAM totals. Fails → ComfyUI not started, stop.

## 2. Models on disk (install root: `C:\Users\<user>\demo\ComfyUI`)

gen-image needs: flux UNET in `models/diffusion_models/` (layout B) OR a
checkpoint in `models/checkpoints/` (layout A); plus `clip_l.safetensors` +
`t5xxl_fp16.safetensors` in `models/text_encoders/` and `ae.safetensors` in
`models/vae/`.

gen-music needs: `acestep_v1.5_turbo.safetensors` (diffusion_models),
`qwen_0.6b_ace15.safetensors` + `qwen_4b_ace15.safetensors` (text_encoders),
`ace_1.5_vae.safetensors` (vae).

    ls models/diffusion_models/ models/text_encoders/ models/vae/ models/checkpoints/

## 3. Nodes registered (no custom nodes needed — all built-in)

Fetch `/object_info` (write the JSON under `$HOME`, not `/c/Users` — permission
denied) and check presence of:
IMAGE: `UNETLoader DualCLIPLoader VAELoader FluxGuidance KSampler
ConditioningZeroOut CLIPTextEncode VAEDecode EmptyLatentImage SaveImage`
ACE: `TextEncodeAceStepAudio(1.5) EmptyAceStepLatentAudio ModelSamplingAuraFlow`
Also print the loader `choices` lists — confirms filenames are registered
regardless of which folder layout the install uses.

## 4. VRAM sanity

    curl -s http://127.0.0.1:11434/api/ps                                   # Ollama loaded models
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

`"models":[]` + no real compute apps = nothing holds VRAM. A fresh 16GB card
can still show ~14.7GB used with an empty queue: that is ComfyUI's own VRAM
cache, not a leak — proceed.

## 5. Smoke renders (seed 42, small, fast)

Image (auto-detect picks schnell checkpoint 4 steps when present; flux-dev
23.8GB UNET load is slow on 16GB — use the checkpoint path for smoke tests):

    python "C:/Users/<user>/AppData/Local/hermes/skills/creative/gen-image/scripts/render_image.py" \
      --prompt "a red apple on a wooden table, soft natural lighting" --ratio "1:1" \
      --style-label realistic --title smoke_test --seed 42 \
      --output-dir "C:/Users/<user>/comfy_test/images" --width 768 --height 768

Music (30s):

    python "C:/Users/<user>/AppData/Local/hermes/skills/creative/gen-music/scripts/render_song.py" \
      --lyrics-file lyrics.txt \
      --tags "pop ballad, acoustic guitar, warm, female vocals, indonesian" \
      --title smoke_test_song --duration 30 --bpm 90 --language id \
      --keyscale "A minor" --seed 42 --output-dir "C:/Users/<user>/comfy_test/songs"

Both print one JSON line; `status=success` + real file on disk = verified.
The emitted `reply`/`media_line` is now `MEDIA:"<quoted path>"` — quoting is
required because the desktop renderer's `\S+` path regex does not match
unquoted paths containing spaces (e.g. a `C:/Users/<user>/...` path whose user
name has a space); quoted
paths embed inline (image / audio player) on the desktop surface.

## Pitfalls

- Invoke python with `C:/Users/...` forward-slash paths ONLY. git-bash/MSYS
  `/c/Users/...` paths reach python.exe unmapped: `can't open file 'C:\c\Users\...'`.
- Write temp files (lyrics, objinfo JSON) under `$HOME`, not `/c/Users` root (EACCES).
- `search_files` pattern `*` returns 0 hits for dir inventory — use `find`.
- Run long renders with `terminal(background=true, notify_on_complete=true)` +
  `process(action='wait')`; foreground timeout for a cold 23.8GB load is too short.