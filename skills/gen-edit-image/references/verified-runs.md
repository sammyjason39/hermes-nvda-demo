# Gen Edit Image — verified runs & mechanics

Evidence base for this skill. Every "works" claim in SKILL.md traces to a row
here from a real run, not an expectation.

## Multi-reference (verified)

`ReferenceLatent` does `conditioning_set_values(..., {"reference_latents": [...]}, append=True)`
(`comfy_extras/nodes_edit_model.py`), so chaining the node appends refs rather
than replacing them. The graph chains a positive and a negative `ReferenceLatent`
per image and feeds the tail into `CFGGuider`. Flux2 default ref method is
`offset` (`comfy/ldm/flux/model.py`), which places each reference as its own
block — no `FluxKontextMultiReferenceLatentMethod` needed.

Verified run: target = `Downloads\aventador-Lamb-scaled.jpg` (white Huracán),
reference = `Downloads\ca8fb51b...webp` (blue studio car shot).

| Instruction | Result (vision check) |
|---|---|
| "Replace the background with the smooth blue studio background from the reference image." | Studio-blue background placed, **but the car body turned pastel blue** — the reference's dominant colour bled onto the subject. |
| "Keep the car body pearl white and unchanged. Only replace the road and trees background with the plain blue studio background from the second image." | Background blue studio gradient, body pearl white with black splitter/wing/mirrors and red rims intact. Correct. |

Lesson: with a second reference image, name the subject's preserved attributes
explicitly; a bare "use the reference's background" instruction lets the
model copy the reference's colour too.

## Verified renders (source: `demo/ComfyUI/input/images.jpg`, 4 steps, 1 MP)

| Instruction | Seed | Result (vision check on the OUTPUT) |
|---|---|---|
| Change the bag color to blue. | 42 | "Success" JSON, but source subject wears a strapless top — **no bag exists**. Model recolored the top region instead. |
| Change the top she is wearing to a bright red color. | 7 | Top red, hair/face/pose/background intact. |
| Change the background to a sunny beach with ocean behind her. | 11 | Beach + ocean + sky, subject preserved. |
| Change the top she is wearing to a bright green color. | 3 | Top vivid green; noted softening/over-saturation vs source (1 MP downscale). |
| Change the car body color to glossy black. (Lamborghini, uploaded by user) | random | Body glossy black; badge, Y-LEDs, wing, wheels, motion blur all intact. |

Timing after models are warm: **~10–25 s per edit** (4 steps, 1 MP). First run
after a ComfyUI start additionally pays the model load (UNET 4 B fp8 ≈ 4 GB +
Qwen3-4B encoder ≈ 8 GB).

Lesson from row 1: a plausible-looking output is NOT proof the edit worked.
Always vision-check the result and ask "did the named object actually change?".

## Where the source-image path comes from (verified)

The user drops an image into the chat; no path needs to be typed. Flow:

1. Desktop calls gateway `image.attach` (local path) or `image.attach_bytes`
   (remote: bytes are base64'd up, then staged).
2. The path is queued in `session["attached_images"]`
   (`tui_gateway/server.py`).
3. On `prompt.submit`, `_enrich_with_attached_images()`
   (`tui_gateway/server.py`, ~line 4771) runs `vision_analyze_tool` on each
   path and PREPENDS to the user message:

   ```
   [The user attached an image:
   <vision description>]
   [You can examine it with vision_analyze using image_url: C:\...\file.jpg]
   ```

   That `image_url:` value is the real local path → pass it to `--image`.

So no extra input plumbing is needed in the script: by the time the model sees
the turn, the path is already in context. Verified end-to-end with a
`Downloads\aventador-Lamb-scaled.jpg` upload.

## Upload mechanics

`upload_image()` POSTs multipart to `/upload/image` (stdlib `urllib`, no
requests dependency) with fields `overwrite=true`, `type=input`.

The stored filename is **always** `hermes_edit_<10 hex><ext>` — never the
source's own basename. Reason: `overwrite=true` plus a same-name file already
sitting in ComfyUI's `input/` dir would silently clobber the user's file. A
random name makes that impossible. ComfyUI accumulates these hashed copies;
they are safe to delete any time.

## Self-check without ComfyUI

`scripts/selfcheck.py` (plain `python selfcheck.py`, or pytest if available)
asserts:

- all graph nodes present, every `[node_id, index]` link resolves to a node
  that exists;
- cfg 1.0, steps 4, seed propagated, both `ReferenceLatent` branches share the
  same VAE latent, sampler chain `62→64`, `65→9`;
- multi-reference: a 2-image build adds a second LoadImage/Scale/VAEEncode
  triplet, chains the second `ReferenceLatent` pair onto the first, keeps the
  edit target as the size source, and points the guider at the chain tail;
- multipart upload against a throwaway local HTTP server: right path, right
  content type, hashed filename, payload intact, response parsed.

Run it after editing `edit_image.py`, then do one real render — the self-check
cannot catch model-side problems (wrong instruction semantics, VRAM limits).

## Environment note

Scripts run from the Hermes skill dir; from git-bash always pass
`C:/Users/...` style paths (an MSYS `/c/Users/...` argv can arrive as
`C:\c\Users\...`). Quote every path — the Windows user dir has a space.
