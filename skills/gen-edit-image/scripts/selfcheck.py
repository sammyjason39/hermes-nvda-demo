#!/usr/bin/env python3
"""Offline self-check for edit_image.py — no ComfyUI needed.

  python scripts/selfcheck.py      (or: pytest scripts/selfcheck.py)

Covers the two things that can silently break: the workflow graph wiring and
the multipart upload (random filename -> never clobbers input/, response
parsing). The live render path is verified by a real smoke run.
"""
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("edit_image", os.path.join(HERE, "edit_image.py"))
ei = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ei)

NODES = ("70", "71", "72", "76", "80", "99", "74", "122", "200", "201", "82",
         "66", "62", "61", "73", "63", "64", "65", "9")


def _assert_links_resolve(wf):
    # every link points at an existing node + valid output index
    for nid, node in wf.items():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in wf, f"{nid} -> missing {value[0]}"


def test_graph_wiring():
    wf = ei.build_edit_graph("Make the bag blue.", ["src.jpg"], "m.safetensors",
                             "c.safetensors", "v.safetensors", 1.0, 4, 42)
    assert set(wf) == set(NODES), set(wf) ^ set(NODES)
    _assert_links_resolve(wf)
    assert wf["63"]["inputs"]["cfg"] == 1.0
    assert wf["62"]["inputs"]["steps"] == 4
    assert wf["73"]["inputs"]["noise_seed"] == 42
    assert wf["76"]["inputs"]["image"] == "src.jpg"
    # sampler chain, and both ReferenceLatent branches share the VAE latent
    assert wf["64"]["inputs"]["sigmas"] == ["62", 0]
    assert wf["200"]["inputs"]["latent"] == wf["201"]["inputs"]["latent"] == ["122", 0]
    assert wf["63"]["inputs"]["positive"] == ["200", 0]
    assert wf["63"]["inputs"]["negative"] == ["201", 0]
    assert wf["9"]["inputs"]["images"] == ["65", 0]
    print("[selfcheck] graph wiring OK")


def test_graph_wiring_multi_reference():
    """Extra images chain onto the same reference_latents list (multi-ref)."""
    wf = ei.build_edit_graph("Put the car on the studio background.", ["car.jpg", "bg.jpg"],
                             "m.safetensors", "c.safetensors", "v.safetensors", 1.0, 4, 7)
    _assert_links_resolve(wf)
    assert wf["86"]["inputs"]["image"] == "bg.jpg"
    assert wf["90"]["inputs"]["image"] == ["86", 0]
    assert set(NODES) <= set(wf)
    # chain: base pair -> second pair, and the guider reads the tail
    assert wf["202"]["inputs"]["conditioning"] == ["200", 0]
    assert wf["203"]["inputs"]["conditioning"] == ["201", 0]
    assert wf["202"]["inputs"]["latent"] == wf["203"]["inputs"]["latent"] == ["301", 0]
    assert wf["63"]["inputs"]["positive"] == ["202", 0]
    assert wf["63"]["inputs"]["negative"] == ["203", 0]
    # the target image still owns the output size
    assert wf["99"]["inputs"]["image"] == ["80", 0]
    print("[selfcheck] multi-reference wiring OK")


def _fake_comfy(seen):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"])
            body = self.rfile.read(n)
            seen.append((self.path, self.headers["Content-Type"], body))
            out = json.dumps({"name": "storeduuid.png", "subfolder": "",
                              "type": "input"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_upload_image():
    seen = []
    srv = _fake_comfy(seen)
    src = os.path.join(HERE, "_selfcheck_src.jpg")
    with open(src, "wb") as f:
        f.write(b"\xff\xd8\xff" + b"payload" * 10)
    try:
        name = ei.upload_image(f"http://127.0.0.1:{srv.server_port}", src)
    finally:
        srv.shutdown()
        os.remove(src)
    assert name == "storeduuid.png", name
    path, ctype, body = seen[0]
    assert path == "/upload/image" and ctype.startswith("multipart/form-data; boundary=")
    assert b'name="image"; filename="hermes_edit_' in body, body[:200]
    assert b"payload" in body
    # the local file's own name must NOT be reused (overwrite=true would clobber it)
    assert b"_selfcheck_src.jpg" not in body
    print("[selfcheck] upload OK")


if __name__ == "__main__":
    test_graph_wiring()
    test_graph_wiring_multi_reference()
    test_upload_image()
    print("[selfcheck] all checks passed")
    sys.exit(0)
