#!/usr/bin/env python3
"""Verify the three ComfyUI skills in this pack, offline. Stdlib only.

  python verify_pack.py            # offline checks only
  python verify_pack.py --help

Offline (no ComfyUI, no GPU):
  - each skill has SKILL.md with valid frontmatter (name + description)
  - every support file the SKILL.md references actually exists
  - every script compiles and imports cleanly
  - gen-edit-image's graph + upload self-check

With ComfyUI up (`--live`), additionally:
  - the server answers /system_stats
  - the models each skill needs appear in /object_info (file-name presence)

Exit 0 = everything checked passed.
"""
import argparse
import importlib.util
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.join(HERE, "skills")
HOST = "http://127.0.0.1:8188"

# What each skill must find on the ComfyUI server (any one name per group).
REQUIRED_MODELS = {
    "gen-image": {
        "UNETLoader": ["flux"],
        "DualCLIPLoader": ["clip_l", "t5xxl"],
        "VAELoader": ["ae", "flux"],
    },
    "gen-music": {
        "UNETLoader": ["acestep"],
        "DualCLIPLoader": ["qwen_0.6b_ace15", "qwen_4b_ace15"],
        "VAELoader": ["ace_1.5_vae"],
    },
    "gen-edit-image": {
        "UNETLoader": ["klein", "flux-2"],
        "CLIPLoader": ["qwen_3"],
        "VAELoader": ["flux2"],
    },
}

failures = []


def check(ok, label, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def frontmatter(path):
    text = open(path, encoding="utf-8").read()
    if not text.startswith("---"):
        return None, text
    m = re.search(r"\n---\s*\n", text[3:])
    if not m:
        return None, text
    import yaml
    return yaml.safe_load(text[3 : m.start() + 3]) or {}, text


def referenced_files(skill_dir, text):
    """Support files the SKILL.md explicitly names (same rule the hub uses)."""
    refs = re.findall(
        r"(?:references|templates|scripts|assets)/[^\s)`\"'<>]+", text
    )
    return sorted({r.rstrip(".,;)") for r in refs if "*" not in r})


def http_json(url, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def choices(node, field):
    info = http_json(f"{HOST}/object_info/{node}")
    spec = info[node]["input"]
    for section in ("required", "optional"):
        if field in spec.get(section, {}):
            val = spec[section][field][0]
            return list(val) if isinstance(val, list) else []
    return []


def offline():
    print("== offline checks")
    names = sorted(
        d for d in os.listdir(SKILLS)
        if os.path.isdir(os.path.join(SKILLS, d))
    )
    check(names == ["gen-edit-image", "gen-image", "gen-music"],
          "exactly the three expected skill folders", ", ".join(names))

    for name in names:
        sdir = os.path.join(SKILLS, name)
        skill_md = os.path.join(sdir, "SKILL.md")
        print(f"-- {name}")
        fm, text = frontmatter(skill_md)
        check(bool(fm and fm.get("name") == name), "frontmatter parses, name matches")
        check(bool(fm and fm.get("description")), "frontmatter has description")

        refs = referenced_files(sdir, text)
        missing = [r for r in refs if not os.path.exists(os.path.join(sdir, r))]
        check(not missing, f"all {len(refs)} referenced support files exist",
              "missing: " + ", ".join(missing) if missing else "")

        for root, _dirs, files in os.walk(sdir):
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(root, f)
                rel = os.path.relpath(p, sdir)
                try:
                    with tempfile.TemporaryDirectory() as td:
                        py_compile.compile(p, doraise=True,
                                           cfile=os.path.join(td, "out.pyc"))
                    check(True, f"{rel} compiles")
                except py_compile.PyCompileError as e:
                    check(False, f"{rel} compiles", str(e)[:120])

    print("-- gen-edit-image self-check")
    sc = os.path.join(SKILLS, "gen-edit-image", "scripts", "selfcheck.py")
    r = subprocess.run([sys.executable, sc], capture_output=True, text=True)
    check(r.returncode == 0, "selfcheck.py exits 0",
          (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")


def live():
    print("\n== live ComfyUI checks")
    try:
        st = http_json(f"{HOST}/system_stats", timeout=8)
    except Exception as e:
        check(False, f"ComfyUI reachable at {HOST}", str(e)[:80])
        return
    dev = (st.get("devices") or [{}])[0]
    check(True, f"ComfyUI reachable ({st.get('system', {}).get('comfyui_version')})")
    print(f"        VRAM free ~{round(dev.get('vram_free', 0) / 1048576)} MB")

    for name, groups in REQUIRED_MODELS.items():
        print(f"-- {name} models")
        for node, needles in groups.items():
            if node == "DualCLIPLoader":
                found = choices(node, "clip_name1")
            elif node == "CLIPLoader":
                found = choices(node, "clip_name")
            elif node == "UNETLoader":
                found = choices(node, "unet_name")
            else:
                found = choices(node, "vae_name")
            hit = [c for c in found if any(n in c.lower() for n in needles)]
            check(bool(hit), f"{node} has {'/'.join(needles)}",
                  ", ".join(hit) if hit else f"have: {found}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify the ComfyUI skills pack")
    ap.add_argument("--live", action="store_true",
                    help="also check the running ComfyUI server + its model files")
    args = ap.parse_args()
    offline()
    if args.live:
        live()
    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)
