#!/usr/bin/env python3
"""Edit an image (instruction-following) with Flux.2 Klein on a local ComfyUI.

Uploads the source image to ComfyUI, builds the official Flux.2 Klein
image-edit graph (UNETLoader fp8 + CLIPLoader type=flux2 + VAELoader,
ReferenceLatent conditioning, Flux2Scheduler 4 steps, cfg 1.0), submits it,
waits, downloads the edited PNG. Stdlib only.

Example:
  python edit_image.py --image ./photo.jpg --prompt "Change the bag color to blue." \
    --title edit --seed -1 --output-dir ./images

Prints one JSON object on stdout:
  {"status": "success", "file": ["..."], "seed": N, "prompt_id": "...",
   "media_line": "MEDIA:\"...\"", "reply": "MEDIA:\"...\""}
Exit codes: 0 ok, 2 server down, 3 submit rejected, 4 timeout, 5 execution error, 6 no output.

Graph rules (from the official template) — do NOT deviate:
  - cfg stays 1.0; the positive branch goes through ReferenceLatent, the
    negative branch is ConditioningZeroOut(positive) -> ReferenceLatent.
  - Flux2Scheduler steps = 4 (distilled model); sampler euler.
  - EmptyFlux2LatentImage w/h come from GetImageSize of the scaled input,
    so the output keeps the input's aspect ratio.
"""
import argparse
import json
import mimetypes
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PREFERRED_UNET = "flux-2-klein-4b-fp8.safetensors"
PREFERRED_CLIP = "qwen_3_4b.safetensors"
PREFERRED_VAE = "flux2-vae.safetensors"
CLIP_TYPE = "flux2"


def http_json(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def object_info_choices(host, node, field, timeout=15):
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
    if preferred in choices:
        return preferred
    if contains:
        for c in choices:
            if contains.lower() in c.lower():
                return c
    return None


def fail(code, step, msg, **extra):
    out = {"status": "error", "step": step, "error": msg}
    out.update(extra)
    print(json.dumps(out))
    sys.exit(code)


def upload_image(host, path, timeout=180):
    """POST the image to ComfyUI's /upload/image, return the stored filename."""
    # Prefix the stored name so an upload never clobbers a file the user
    # already has in ComfyUI's input/ dir (overwrite=true targets same name).
    ext = os.path.splitext(path)[1].lower() or ".png"
    filename = f"hermes_edit_{uuid.uuid4().hex[:10]}{ext}"
    ctype = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        payload = f.read()
    boundary = "----hermes" + uuid.uuid4().hex
    body = b""
    for name, value in (("overwrite", "true"), ("type", "input")):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
             f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
    body += payload + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(host + "/upload/image", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    sub = resp.get("subfolder") or ""
    return f"{sub}/{resp['name']}" if sub else resp["name"]


def build_edit_graph(instruction, image_names, unet, clip, vae, mp, steps, seed):
    """Official Flux.2 Klein image-edit graph (see module docstring).

    image_names[0] is the edit target: its size drives the output latent.
    Any further entries are extra references (e.g. a style/background source),
    chained through additional ReferenceLatent nodes — the node appends to the
    reference_latents list, so chaining == multi-reference conditioning.
    """
    wf = {
        "70": {"class_type": "UNETLoader",
               "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "71": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": clip, "type": CLIP_TYPE, "device": "default"}},
        "72": {"class_type": "VAELoader",
               "inputs": {"vae_name": vae}},
        "76": {"class_type": "LoadImage",
               "inputs": {"image": image_names[0]}},
        "80": {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"upscale_method": "nearest-exact", "megapixels": float(mp),
                          "resolution_steps": 1, "image": ["76", 0]}},
        "99": {"class_type": "GetImageSize",
               "inputs": {"image": ["80", 0]}},
        "74": {"class_type": "CLIPTextEncode",
               "inputs": {"text": instruction, "clip": ["71", 0]}},
        "122": {"class_type": "VAEEncode",
                "inputs": {"pixels": ["80", 0], "vae": ["72", 0]}},
        "82": {"class_type": "ConditioningZeroOut",
               "inputs": {"conditioning": ["74", 0]}},
        "66": {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": ["99", 0], "height": ["99", 1], "batch_size": 1}},
        "62": {"class_type": "Flux2Scheduler",
               "inputs": {"steps": int(steps), "width": ["99", 0], "height": ["99", 1]}},
        "61": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": "euler"}},
        "73": {"class_type": "RandomNoise",
               "inputs": {"noise_seed": int(seed)}},
        "64": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["73", 0], "guider": ["63", 0],
                          "sampler": ["61", 0], "sigmas": ["62", 0],
                          "latent_image": ["66", 0]}},
        "65": {"class_type": "VAEDecode",
               "inputs": {"samples": ["64", 0], "vae": ["72", 0]}},
        "9":  {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "Flux2-Klein", "images": ["65", 0]}},
    }
    positive, negative = "74", "82"
    for i, name in enumerate(image_names):
        load, scale = 76 + i * 10, 80 + i * 10
        vaeenc, ref_pos, ref_neg = 300 + i, 200 + i * 2, 201 + i * 2
        if i:  # image_names[0] already wired as the base pair above
            wf[str(load)] = {"class_type": "LoadImage", "inputs": {"image": name}}
            wf[str(scale)] = {"class_type": "ImageScaleToTotalPixels",
                              "inputs": {"upscale_method": "nearest-exact",
                                         "megapixels": float(mp),
                                         "resolution_steps": 1, "image": [str(load), 0]}}
            wf[str(vaeenc)] = {"class_type": "VAEEncode",
                               "inputs": {"pixels": [str(scale), 0], "vae": ["72", 0]}}
        else:
            vaeenc = 122
        wf[str(ref_pos)] = {"class_type": "ReferenceLatent",
                            "inputs": {"conditioning": [positive, 0],
                                       "latent": [str(vaeenc), 0]}}
        wf[str(ref_neg)] = {"class_type": "ReferenceLatent",
                            "inputs": {"conditioning": [negative, 0],
                                       "latent": [str(vaeenc), 0]}}
        positive, negative = str(ref_pos), str(ref_neg)
    wf["63"] = {"class_type": "CFGGuider",
                "inputs": {"cfg": 1.0, "model": ["70", 0],
                           "positive": [positive, 0], "negative": [negative, 0]}}
    return wf


def main():
    ap = argparse.ArgumentParser(description="Instruction-edit an image with Flux.2 Klein (local ComfyUI)")
    ap.add_argument("--image", required=True, nargs="+",
                    help="Edit target, then optional extra reference images "
                         "(path or http(s) URL). First = the photo to edit; "
                         "extras supply style/background the prompt can name.")
    ap.add_argument("--prompt", required=True,
                    help="Edit instruction, English, imperative (e.g. 'Change the bag color to blue.')")
    ap.add_argument("--title", default="edit", help="Title (used for output filename)")
    ap.add_argument("--seed", type=int, default=-1, help="-1 = random")
    ap.add_argument("--megapixels", type=float, default=1.0,
                    help="Resize the input to this many MP before editing (aspect kept)")
    ap.add_argument("--steps", type=int, default=4, help="Flux2Scheduler steps (distilled: 4)")
    ap.add_argument("--host", default="http://127.0.0.1:8188")
    ap.add_argument("--output-dir", default="./images")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--model", default=None, help="Force the diffusion model file name")
    ap.add_argument("--clip", default=None, help="Force the text encoder file name")
    ap.add_argument("--vae", default=None, help="Force the VAE file name")
    args = ap.parse_args()

    instruction = args.prompt.strip()
    if not instruction.endswith((".", "!", "?")):
        instruction += "."
    print(f"[edit-image] instruction: {instruction}", file=sys.stderr)

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**63 - 1)
    safe_title = re.sub(r"[^A-Za-z0-9]+", "_", args.title).strip("_") or "edit"

    try:
        http_json(args.host + "/system_stats", timeout=10)
    except Exception as e:
        fail(2, "server_check", f"ComfyUI not reachable at {args.host}: {e}")

    # resolve model files against what this install actually has
    unets = object_info_choices(args.host, "UNETLoader", "unet_name")
    clips = object_info_choices(args.host, "CLIPLoader", "clip_name")
    vaes = object_info_choices(args.host, "VAELoader", "vae_name")
    unet = args.model or find_existing(unets, PREFERRED_UNET, contains="klein") \
        or next((u for u in unets if "flux-2" in u.lower()), None)
    clip = args.clip or find_existing(clips, PREFERRED_CLIP, contains="qwen_3_4b") \
        or next((c for c in clips if "qwen_3" in c.lower()), None)
    vae = args.vae or find_existing(vaes, PREFERRED_VAE, contains="flux2") \
        or next((v for v in vaes if "flux2" in v.lower()), None)
    missing = [n for n, v in (("unet", unet), ("clip", clip), ("vae", vae)) if not v]
    if missing:
        fail(3, "model", f"cannot resolve Flux.2 {missing}. unets={unets} clips={clips} vaes={vaes}")

    # source + optional extra references: URLs down to temp files, then upload
    tmp_paths, image_names = [], []
    for src in args.image:
        tmp_path = None
        if src.startswith(("http://", "https://")):
            tmp_path = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)) or ".",
                                    f".edit_src_{uuid.uuid4().hex[:8]}.img")
            try:
                with urllib.request.urlopen(src, timeout=60) as r, open(tmp_path, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                fail(3, "download_source", f"cannot download source image: {e}")
            tmp_paths.append(tmp_path)
            src = tmp_path
        if not os.path.isfile(src):
            fail(3, "source", f"source image not found: {src}")
        try:
            image_names.append(upload_image(args.host, src))
        except urllib.error.HTTPError as e:
            fail(3, "upload", f"upload failed (HTTP {e.code}): {e.read().decode(errors='replace')[:300]}")
        except Exception as e:
            fail(3, "upload", f"upload failed: {e}")
    for tmp_path in tmp_paths:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    wf = build_edit_graph(instruction, image_names, unet, clip, vae,
                          args.megapixels, args.steps, seed)
    try:
        resp = http_json(args.host + "/prompt", {"prompt": wf})
        prompt_id = resp["prompt_id"]
    except urllib.error.HTTPError as e:
        fail(3, "submit", f"ComfyUI rejected the workflow (HTTP {e.code}): "
                          f"{e.read().decode(errors='replace')[:2000]}")
    except Exception as e:
        fail(3, "submit", str(e))

    print(f"[edit-image] {unet} + {clip} + {vae} ({args.steps} steps, {args.megapixels}MP) "
          f"prompt_id={prompt_id}, waiting up to {args.timeout}s ...", file=sys.stderr)

    deadline = time.time() + args.timeout
    entry = None
    while time.time() < deadline:
        time.sleep(2)
        try:
            entry = http_json(f"{args.host}/history/{prompt_id}").get(prompt_id)
        except Exception:
            continue
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

    os.makedirs(args.output_dir, exist_ok=True)
    files = []
    for node_out in (entry.get("outputs") or {}).values():
        for o in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": o["filename"],
                                        "subfolder": o.get("subfolder", ""),
                                        "type": o.get("type", "output")})
            with urllib.request.urlopen(f"{args.host}/view?{q}", timeout=120) as r:
                data = r.read()
            path = os.path.join(args.output_dir, f"{safe_title}_{seed}.png").replace("\\", "/")
            with open(path, "wb") as f:
                f.write(data)
            files.append(path)
    if not files:
        fail(6, "download", "no image outputs found in history", prompt_id=prompt_id)

    line = f'MEDIA:"{files[0]}"'
    print(json.dumps({"status": "success", "file": files, "seed": seed,
                      "prompt_id": prompt_id, "model": unet, "clip": clip, "vae": vae,
                      "steps": args.steps, "megapixels": args.megapixels,
                      "source": image_names if len(image_names) > 1 else image_names[0],
                      "media_line": line, "reply": line}))


if __name__ == "__main__":
    main()
