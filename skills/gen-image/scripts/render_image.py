#!/usr/bin/env python3
"""Render an image with Flux on a local ComfyUI server.

Supports BOTH model layouts automatically:
  A) a checkpoint in models/checkpoints/      -> CheckpointLoaderSimple graph
  B) a bare UNET in models/diffusion_models/  -> UNETLoader + DualCLIPLoader + VAELoader
     (this is the layout used by flux1-dev.safetensors on this machine)

Submits the workflow, waits, downloads the PNG. Stdlib only.

Example:
  python render_image.py --prompt "professional photo of ..." \
    --ratio 16:9 --title "Winter Coat" --seed -1 --output-dir ./images

Prints one JSON object on stdout:
  {"status": "success", "file": ["...png"], "seed": N, "prompt_id": "...",
   "media_line": "MEDIA:...", "reply": "..."}
Exit codes: 0 ok, 2 server down, 3 submit rejected, 4 timeout, 5 execution error, 6 no output.

Note: Flux dev runs at cfg 1.0 with FluxGuidance as the real knob; the
negative prompt is kept for completeness but has little effect.
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

DEFAULT_NEGATIVE = ("blurry, low quality, worst quality, jpeg artifacts, watermark, "
                    "text, logo, signature, deformed, disfigured, bad anatomy, "
                    "extra fingers, mutated hands, lowres, ugly, duplicate, cropped")
RATIOS = {"1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344),
          "4:3": (1152, 864), "3:4": (864, 1152)}

# Preferred files for the bare-UNET layout (layout B)
PREFERRED_UNET = "flux1-dev.safetensors"
PREFERRED_CLIP_L = "clip_l.safetensors"
PREFERRED_T5 = "t5xxl_fp16.safetensors"
PREFERRED_VAE = "ae.safetensors"


def object_info_choices(host, node, field, timeout=15):
    """Return the list of selectable values for a node input, or [] on failure."""
    try:
        info = http_json(f"{host}/object_info/{node}", timeout=timeout)
        spec = info[node]["input"]
        for section in ("required", "optional"):
            if field in spec.get(section, {}):
                val = spec[section][field][0]
                return list(val) if isinstance(val, list) else []
    except Exception:
        pass
    return []


def find_existing(choices, preferred, contains=None):
    """Pick `preferred` if present, else the first entry containing `contains`."""
    if preferred in choices:
        return preferred
    if contains:
        for c in choices:
            if contains.lower() in c.lower():
                return c
    return None


def build_checkpoint_graph(prompt, negative, width, height, seed, prefix,
                           ckpt, steps, cfg, sampler, scheduler,
                           use_text_negative=False):
    """Layout A: a single all-in-one checkpoint (e.g. flux1-schnell-fp8).

    At cfg 1.0 the official template zeroes the negative instead of encoding
    it (a text negative has no effect at cfg 1.0). Pass use_text_negative=True
    (cfg > 1.0 only) to encode the negative text for real.
    """
    wf = {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": int(width), "height": int(height), "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                         "sampler_name": sampler, "scheduler": scheduler,
                         "denoise": 1.0, "model": ["4", 0],
                         "positive": ["6", 0], "negative": ["14", 0],
                         "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }
    if use_text_negative:
        wf["7"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": negative, "clip": ["4", 1]}}
        wf["3"]["inputs"]["negative"] = ["7", 0]
    else:
        wf["14"] = {"class_type": "ConditioningZeroOut",
                    "inputs": {"conditioning": ["6", 0]}}
    return wf


def build_unet_graph(prompt, negative, width, height, seed, prefix,
                     unet, clip_l, clip_t5, vae, weight_dtype, guidance,
                     steps, cfg, sampler, scheduler, use_text_negative=False):
    """Layout B: bare diffusion model + separate text encoders + VAE.

    Mirrors the official ComfyUI `flux_dev_checkpoint_example.json` template:
    the negative is `ConditioningZeroOut` of the positive encode (at cfg 1.0 a
    text negative has no mathematical effect, so zeroing is what the template
    does and it saves a full T5 encode). FluxGuidance is kept as an explicit
    node so guidance stays adjustable — without it Flux falls back to 3.5.

    Pass use_text_negative=True (cfg > 1.0 only) to encode the negative text.
    """
    wf = {
        "10": {"class_type": "UNETLoader",
               "inputs": {"unet_name": unet, "weight_dtype": weight_dtype}},
        "11": {"class_type": "DualCLIPLoader",
               "inputs": {"clip_name1": clip_l, "clip_name2": clip_t5,
                          "type": "flux", "device": "default"}},
        "12": {"class_type": "VAELoader",
               "inputs": {"vae_name": vae}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["11", 0]}},
        "13": {"class_type": "FluxGuidance",
               "inputs": {"conditioning": ["6", 0], "guidance": float(guidance)}},
        "5": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": int(width), "height": int(height), "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                         "sampler_name": sampler, "scheduler": scheduler,
                         "denoise": 1.0, "model": ["10", 0],
                         "positive": ["13", 0], "negative": ["14", 0],
                         "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["12", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }
    if use_text_negative:
        wf["7"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": negative, "clip": ["11", 0]}}
        wf["3"]["inputs"]["negative"] = ["7", 0]
    else:
        wf["14"] = {"class_type": "ConditioningZeroOut",
                    "inputs": {"conditioning": ["6", 0]}}
    return wf


def http_json(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fail(code, step, msg, **extra):
    out = {"status": "error", "step": step, "error": msg}
    out.update(extra)
    print(json.dumps(out))
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(description="Render an image via Flux Schnell on local ComfyUI")
    ap.add_argument("--prompt", required=True, help="Positive prompt (English). If --style is given, the style row + quality tail are appended automatically.")
    ap.add_argument("--style", default=None,
                    choices=["realistic", "anime", "cartoon", "watercolor",
                             "3d character", "hand drawing"],
                    help="Appends the skill's style enhancement row + quality tail to the prompt")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE,
                    help="Negative prompt (little effect at cfg 1.0)")
    ap.add_argument("--ratio", default="1:1", choices=sorted(RATIOS))
    ap.add_argument("--width", type=int, default=None, help="Override width (multiple of 16)")
    ap.add_argument("--height", type=int, default=None, help="Override height (multiple of 16)")
    ap.add_argument("--title", default="image", help="Title (used for output filename)")
    ap.add_argument("--style-label", default=None, help="Style label for the reply line (defaults to --style)")
    ap.add_argument("--seed", type=int, default=-1, help="-1 = random")
    ap.add_argument("--host", default="http://127.0.0.1:8188")
    ap.add_argument("--output-dir", default="./images")
    ap.add_argument("--timeout", type=int, default=300, help="Max seconds to wait")
    # model selection (everything is auto-detected by default)
    ap.add_argument("--model", default=None,
                    help="Force a model name (checkpoint OR bare unet file name)")
    ap.add_argument("--clip-l", default=None, help="CLIP-L text encoder file (layout B)")
    ap.add_argument("--clip-t5", default=None, help="T5 text encoder file (layout B)")
    ap.add_argument("--vae", default=None, help="VAE file (layout B)")
    ap.add_argument("--weight-dtype", default="default",
                    choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    help="UNETLoader weight dtype (fp8_e4m3fn roughly halves VRAM)")
    ap.add_argument("--guidance", type=float, default=3.5, help="FluxGuidance value (layout B)")
    ap.add_argument("--steps", type=int, default=0,
                    help="Sampler steps (0 = auto: 20 for dev, 4 for schnell)")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--negative-mode", default="auto", choices=["auto", "zero", "text"],
                    help="auto: zero the negative at cfg<=1.0 (matches the official "
                         "Flux template), encode it as text at cfg>1.0")
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    args = ap.parse_args()

    # --style composes the final prompt mechanically (style row + quality tail).
    # Keeps the model free of the style table; it just passes user words.
    STYLE_ROWS = {
        "realistic": "professional photograph, shot on 85mm lens, f/1.4, soft natural lighting, shallow depth of field, detailed skin texture, photorealistic, 8k uhd",
        "anime": "high quality anime illustration, clean crisp lineart, cel shading, vibrant colors, expressive eyes, detailed background, official key visual",
        "cartoon": "cartoon style illustration, bold clean outlines, flat vibrant colors, playful exaggerated proportions, expressive character design",
        "watercolor": "watercolor painting, soft color washes, wet-on-wet bleeding edges, paper texture, delicate brush strokes, pastel palette",
        "3d character": "3d character render, pixar style, subsurface scattering, soft studio lighting, detailed textures, octane render",
        "hand drawing": "hand-drawn pencil sketch, graphite texture, crosshatching shading, sketchbook paper grain, artistic line work",
    }
    prompt = args.prompt
    if args.style:
        prompt = prompt.strip().rstrip(",") + ", " + STYLE_ROWS[args.style] + \
            ", masterpiece quality, highly detailed, sharp focus"
    style_label = args.style_label or args.style or "image"
    print(f"[gen-image] prompt: {prompt}", file=sys.stderr)

    width, height = RATIOS[args.ratio]
    if args.width:
        width = args.width
    if args.height:
        height = args.height
    if width % 16 or height % 16:
        fail(3, "validate", f"width/height must be multiples of 16, got {width}x{height}")

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**63 - 1)
    safe_title = re.sub(r"[^A-Za-z0-9]+", "_", args.title).strip("_") or "image"

    # 0. server check
    try:
        http_json(args.host + "/system_stats", timeout=10)
    except Exception as e:
        fail(2, "server_check",
             f"ComfyUI not reachable at {args.host}: {e}. Ask the user to start ComfyUI first.")

    # 0b. detect which model layout this ComfyUI install has
    ckpts = object_info_choices(args.host, "CheckpointLoaderSimple", "ckpt_name")
    unets = object_info_choices(args.host, "UNETLoader", "unet_name")

    model = args.model
    layout = None
    if model and model in ckpts:
        layout = "checkpoint"
    elif model and model in unets:
        layout = "unet"
    elif model:
        fail(3, "model", f"model '{model}' is in neither checkpoints nor diffusion_models. "
                         f"checkpoints={ckpts} unets={unets}")
    else:
        flux_ckpt = find_existing(ckpts, "", contains="flux")
        flux_unet = find_existing(unets, PREFERRED_UNET, contains="flux")
        if flux_ckpt:
            layout, model = "checkpoint", flux_ckpt
        elif flux_unet:
            layout, model = "unet", flux_unet
        else:
            fail(3, "model", f"no Flux model found. checkpoints={ckpts} unets={unets}")

    is_dev = "dev" in model.lower()
    steps = args.steps or (4 if not is_dev else 20)
    weight_dtype = args.weight_dtype
    if layout == "unet" and is_dev and weight_dtype == "default":
        # unquantized dev weights are ~24 GB; fp8 halves that with no visible quality loss
        weight_dtype = "fp8_e4m3fn"

    use_text_negative = (args.negative_mode == "text"
                         or (args.negative_mode == "auto" and args.cfg > 1.0))

    if layout == "checkpoint":
        wf = build_checkpoint_graph(args.prompt, args.negative, width, height, seed,
                                    safe_title, model, steps, args.cfg,
                                    args.sampler, args.scheduler,
                                    use_text_negative=use_text_negative)
    else:
        encoders = object_info_choices(args.host, "DualCLIPLoader", "clip_name1")
        vaes = object_info_choices(args.host, "VAELoader", "vae_name")
        clip_l = args.clip_l or find_existing(encoders, PREFERRED_CLIP_L, contains="clip_l")
        clip_t5 = args.clip_t5 or find_existing(encoders, PREFERRED_T5, contains="t5xxl")
        vae = args.vae or find_existing(vaes, PREFERRED_VAE, contains="ae")
        missing = [n for n, v in (("clip_l", clip_l), ("t5xxl", clip_t5), ("vae", vae)) if not v]
        if missing:
            fail(3, "model", f"bare-UNET layout needs {missing} but none found. "
                             f"encoders={encoders} vaes={vaes}")
        wf = build_unet_graph(args.prompt, args.negative, width, height, seed,
                              safe_title, model, clip_l, clip_t5, vae,
                              weight_dtype, args.guidance, steps, args.cfg,
                              args.sampler, args.scheduler,
                              use_text_negative=use_text_negative)

    # 1. submit
    try:
        resp = http_json(args.host + "/prompt", {"prompt": wf})
        prompt_id = resp["prompt_id"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:2000]
        fail(3, "submit", f"ComfyUI rejected the workflow (HTTP {e.code}): {body}")
    except Exception as e:
        fail(3, "submit", str(e))

    print(f"[gen-image] {model} ({layout}, {steps} steps, dtype={weight_dtype}) "
          f"prompt_id={prompt_id}, waiting up to {args.timeout}s ...", file=sys.stderr)

    # 2. wait
    deadline = time.time() + args.timeout
    entry = None
    while time.time() < deadline:
        time.sleep(2)
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

    # 3. download image outputs
    os.makedirs(args.output_dir, exist_ok=True)
    files = []
    for node_id, node_out in (entry.get("outputs") or {}).items():
        for o in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": o["filename"],
                                        "subfolder": o.get("subfolder", ""),
                                        "type": o.get("type", "output")})
            with urllib.request.urlopen(f"{args.host}/view?{q}", timeout=120) as r:
                data = r.read()
            fname = f"{safe_title}_{seed}.png"
            path = os.path.join(args.output_dir, fname).replace("\\", "/")
            with open(path, "wb") as f:
                f.write(data)
            files.append(path)
    if not files:
        fail(6, "download", "no image outputs found in history", prompt_id=prompt_id)

    # The reply is EXACTLY the MEDIA line — nothing else. Appending an info
    # line (style/seed/size) breaks the embed on small echo models: they copy
    # the extra text into the middle of the path or wrap the block. Metadata
    # stays in the JSON fields, never in the reply text.
    print(json.dumps({"status": "success", "file": files, "seed": seed,
                      "prompt_id": prompt_id, "size": f"{width}x{height}",
                      "model": model, "layout": layout, "steps": steps,
                      "style_label": style_label,
                      "media_line": f'MEDIA:"{files[0]}"',
                      "reply": f'MEDIA:"{files[0]}"'}))


if __name__ == "__main__":
    main()
