#!/usr/bin/env python3
"""Render a song with ACE-Step 1.5 (turbo) on a local ComfyUI server.

Does EVERYTHING: server/queue/VRAM checks, genre preset resolution
(BPM/keyscale/tags), workflow build, submit, wait, download. Stdlib only.

Usage:
  python render_song.py --check                      # preflight only, JSON
  python render_song.py --presets                    # list genre presets, JSON
  python render_song.py --lyrics-file lyrics.txt --genre pop-ceria \
      --title "Senja" --duration 60 --language id --vocal female --seed -1 \
      --output-dir ./songs

Args:
  --lyrics-file   UTF-8 lyrics file (omit = instrumental)
  --genre         preset: pop-ballad|pop-ceria|rock|lofi|electronic|dangdut-pop
  --tags          free style tags (overrides preset tags)
  --vocal         male|female|instrumental (default female)
  --bpm --keyscale --language --seed --title --duration --output-dir
  --timeout       max wait seconds (default 900)
  --host          ComfyUI URL (default http://127.0.0.1:8188)
  --check         preflight only: server + queue + VRAM
  --presets       print preset table and exit

Prints ONE JSON object on stdout:
  success: {"status":"success","file":[...],"seed":N,"tags":"...",
            "media_line":"MEDIA:\"...\"","reply":"MEDIA:\"...\""}
  error:   {"status":"error","step":"...","error":"..."}
Exit codes: 0 ok, 2 server down, 3 submit rejected, 4 timeout, 5 execution
error, 6 no output, 7 queue busy, 8 vram low.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DIFFUSION = "acestep_v1.5_turbo.safetensors"
CLIP_1 = "qwen_0.6b_ace15.safetensors"
CLIP_2 = "qwen_4b_ace15.safetensors"
VAE = "ace_1.5_vae.safetensors"
MIN_VRAM_FREE_MB = 4000

# genre -> bpm / keyscale / base tags (ACE keyscale values, 35 total)
PRESETS = {
    "pop-ballad":  {"bpm": 85,  "keyscale": "C major", "tags": "pop ballad, acoustic guitar, piano, warm, emotional"},
    "pop-ceria":   {"bpm": 110, "keyscale": "G major", "tags": "pop ceria, bright, catchy, claps, upbeat"},
    "rock":        {"bpm": 140, "keyscale": "E minor", "tags": "rock, distorted guitars, live drums, powerful"},
    "lofi":        {"bpm": 80,  "keyscale": "F major", "tags": "lo-fi, mellow, vinyl crackle, chill"},
    "electronic":  {"bpm": 124, "keyscale": "A minor", "tags": "electronic, synth, four-on-the-floor, energetic"},
    "dangdut-pop": {"bpm": 105, "keyscale": "A minor", "tags": "dangdut-pop, kendang, flute, catchy"},
}


def http_json(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fail(code, step, msg, **extra):
    print(json.dumps({"status": "error", "step": step, "error": msg, **extra}))
    sys.exit(code)


def preflight(host):
    """Server + queue + VRAM. Returns (ok, info, err_step, err_msg)."""
    try:
        stats = http_json(host + "/system_stats", timeout=10)
    except Exception as e:
        return False, {}, "server_check", \
            f"ComfyUI not reachable at {host}: {e}. Ask the user to start ComfyUI first."
    info = {"server": "up", "comfyui_version": stats.get("system", {}).get("comfyui_version")}
    try:
        q = http_json(host + "/queue", timeout=10)
        busy = bool(q.get("queue_running") or q.get("queue_pending"))
        info["queue_busy"] = busy
        if busy:
            return False, info, "queue", \
                "ComfyUI queue busy (another job running). Wait for it to finish, then retry."
    except Exception as e:
        return False, info, "queue", f"queue check failed: {e}"
    for dev in stats.get("devices", []):
        if dev.get("type") == "cuda":
            free = dev.get("vram_free", 0) // (1024 * 1024)
            info["vram_free_mb"] = free
            if free < MIN_VRAM_FREE_MB:
                return False, info, "vram", \
                    f"VRAM low: {free}MB free (< {MIN_VRAM_FREE_MB}). Ask the user to free VRAM (e.g. unload the idle Ollama model), then retry."
            break
    return True, info, None, None


def resolve_style(args):
    """Merge genre preset + explicit overrides + vocal into final tags/bpm/keyscale."""
    p = PRESETS.get(args.genre, {}) if args.genre else {}
    if args.genre and not p:
        fail(3, "style", f"unknown genre '{args.genre}'. Valid: {', '.join(PRESETS)}")
    tags = args.tags or p.get("tags", "pop")
    if args.vocal == "instrumental":
        tags += ", instrumental"
    else:
        tags += f", {args.vocal} vocals"
    bpm = args.bpm if args.bpm is not None else p.get("bpm", 110)
    keyscale = args.keyscale or p.get("keyscale", "C major")
    return tags, bpm, keyscale


def build_workflow(tags, lyrics, seed, bpm, duration, timesig, language, keyscale, prefix):
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": DIFFUSION, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": CLIP_1, "clip_name2": CLIP_2,
                         "type": "ace", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "ModelSamplingAuraFlow",
              "inputs": {"model": ["1", 0], "shift": 3.0}},
        "5": {"class_type": "TextEncodeAceStepAudio1.5",
              "inputs": {"clip": ["2", 0], "tags": tags, "lyrics": lyrics,
                         "seed": seed, "bpm": int(bpm), "duration": float(duration),
                         "timesignature": str(timesig), "language": language,
                         "keyscale": keyscale, "generate_audio_codes": True,
                         "cfg_scale": 2.0, "temperature": 0.85,
                         "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "6": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyAceStep1.5LatentAudio",
              "inputs": {"seconds": float(duration), "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": 8, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0, "model": ["4", 0],
                         "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecodeAudio",
              "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveAudioMP3",
               "inputs": {"audio": ["9", 0], "filename_prefix": prefix,
                          "quality": "V0"}},
    }


def main():
    ap = argparse.ArgumentParser(description="Render a song via ACE-Step 1.5 on local ComfyUI")
    ap.add_argument("--lyrics-file", help="UTF-8 lyrics file (omit = instrumental)")
    ap.add_argument("--genre", help=f"preset key: {', '.join(PRESETS)}")
    ap.add_argument("--tags", help="style tags (overrides preset)")
    ap.add_argument("--vocal", default="female", choices=["male", "female", "instrumental"])
    ap.add_argument("--title", default="song")
    ap.add_argument("--duration", type=float, default=30.0, help="Seconds")
    ap.add_argument("--bpm", type=int, default=None)
    ap.add_argument("--language", default="id", help="ACE language code: id, en, ...")
    ap.add_argument("--keyscale", default=None)
    ap.add_argument("--timesignature", default="4", choices=["2", "3", "4", "6"])
    ap.add_argument("--seed", type=int, default=-1, help="-1 = random")
    ap.add_argument("--host", default="http://127.0.0.1:8188")
    ap.add_argument("--output-dir", default="./songs")
    ap.add_argument("--timeout", type=int, default=900, help="Max seconds to wait")
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--presets", action="store_true", help="list presets")
    args = ap.parse_args()

    if args.presets:
        print(json.dumps({"status": "success", "presets": PRESETS}))
        return
    if args.check:
        ok, info, step, msg = preflight(args.host)
        print(json.dumps({"status": "success" if ok else "error",
                          "step": step, "error": msg, **info}))
        sys.exit(0 if ok else 1)
    if not args.lyrics_file and args.vocal != "instrumental":
        fail(3, "args", "--lyrics-file required unless --vocal instrumental")

    lyrics = ""
    if args.lyrics_file:
        with open(args.lyrics_file, encoding="utf-8") as f:
            lyrics = f.read().strip()

    ok, info, step, msg = preflight(args.host)
    if not ok:
        fail(2 if step == "server_check" else 7 if step == "queue" else 8, step, msg, **info)

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
    safe_title = re.sub(r"[^A-Za-z0-9]+", "_", args.title).strip("_") or "song"
    tags, bpm, keyscale = resolve_style(args)
    wf = build_workflow(tags, lyrics, seed, bpm, args.duration,
                        args.timesignature, args.language, keyscale, safe_title)

    # submit
    try:
        resp = http_json(args.host + "/prompt", {"prompt": wf})
        prompt_id = resp["prompt_id"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:2000]
        fail(3, "submit", f"ComfyUI rejected the workflow (HTTP {e.code}): {body}")
    except Exception as e:
        fail(3, "submit", str(e))

    print(f"[gen-music] tags={tags!r} bpm={bpm} key={keyscale!r} seed={seed}; "
          f"submitted {prompt_id}, waiting up to {args.timeout}s ...", file=sys.stderr)

    # wait
    deadline = time.time() + args.timeout
    entry = None
    while time.time() < deadline:
        time.sleep(3)
        try:
            hist = http_json(f"{args.host}/history/{prompt_id}")
        except Exception:
            continue
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("status_str") in ("success", "error"):
            break
    if not entry:
        fail(4, "wait", f"timeout after {args.timeout}s", prompt_id=prompt_id)

    st = entry.get("status", {})
    if st.get("status_str") == "error":
        err = "execution error"
        for m in st.get("messages", []):
            if m[0] == "execution_error":
                err = (f"{m[1].get('node_type')} (node {m[1].get('node_id')}): "
                       f"{m[1].get('exception_message')}")
        fail(5, "execute", err, prompt_id=prompt_id)

    # download audio outputs
    os.makedirs(args.output_dir, exist_ok=True)
    files = []
    for node_id, node_out in (entry.get("outputs") or {}).items():
        for o in node_out.get("audio", []):
            q = urllib.parse.urlencode({"filename": o["filename"],
                                        "subfolder": o.get("subfolder", ""),
                                        "type": o.get("type", "output")})
            with urllib.request.urlopen(f"{args.host}/view?{q}", timeout=120) as r:
                data = r.read()
            fname = f"{safe_title}_{seed}.mp3"
            path = os.path.join(args.output_dir, fname).replace("\\", "/")
            with open(path, "wb") as f:
                f.write(data)
            files.append(path)
    if not files:
        fail(6, "download", "no audio outputs found in history", prompt_id=prompt_id)

    # reply is EXACTLY the MEDIA line — nothing else. Extra text after the path
    # breaks the embed on small echo models; metadata lives in JSON fields.
    print(json.dumps({"status": "success", "file": files, "seed": seed,
                      "prompt_id": prompt_id, "duration_s": args.duration,
                      "bpm": bpm, "keyscale": keyscale, "tags": tags,
                      "media_line": f'MEDIA:"{files[0]}"',
                      "reply": f'MEDIA:"{files[0]}"'}))


if __name__ == "__main__":
    main()