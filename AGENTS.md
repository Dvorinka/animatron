# animatron

Batch sprite-sheet slicer + animation viewer for AI-generated character sheets
("<Char> - NN label.png", frames packed in rows at roughly constant pitch,
RGBA transparent background).

## Pipeline

Two ways in:

1. **In-browser (easiest)**: open `index.html`, click "open folder" or drag a
   folder of sheets onto the page. `slicer.js` ports the full slicing
   algorithm to JS - frames are masked canvases in memory, identical output
   to the Python pipeline. Works from `file://`, no server needed.
2. **Batch (Python)**: for the full 120-character run + persistent frame
   PNGs:

```bash
# one-time setup (PEP 668 box - venv required)
python3 -m venv .venv && .venv/bin/pip install pillow numpy scipy

# slice every sheet under <dir> -> sprites.js manifest + frames/ PNGs
.venv/bin/python tools/slice.py <sheet-dir> sprites.js

# view
python3 -m http.server 8777   # then open http://localhost:8777/
```

`index.html` also works via `file://` (manifest is a script include, frames are
relative `<img>` paths). Source PNGs are never modified.

## Output format

`tools/slice.py` writes one transparent PNG per frame under
`frames/<safe-sheet-name>/rN/fNN.png` containing ONLY the pixels owned by that
frame (component-masked - foreign pixels can never appear in the crop), plus
`PAD` px of transparent margin in every direction. `sprites.js` is
`window.SPRITES = {characters:[{name, clips:[{name, frames}]}]}` where each
frame is `[file, ax, ay]` - PNG path and the anchor point in crop-relative
pixels. The viewer pins the anchor to a fixed screen point.

## How slicing works (tools/slice.py)

1. Rows split on >=12px fully-empty horizontal bands.
2. Frame pitch per row via autocorrelation of per-column ALPHA MASS
   (binary occupancy goes flat when sprites touch edge-to-edge; alpha mass
   stays periodic). Smallest strong peak wins so 2*P harmonics don't
   masquerade as pitch. No peak -> single-frame row.
3. DP segmentation: cut candidates are centers of every empty column run
   (free) plus pitch-grid points (penalized); cost = (width-pitch)^2, so
   cuts snap to real gaps and only slice content when no gap exists.
4. Pitch re-estimated as the median cell width, DP rerun (kills drift).
5. `heal_cuts`: a cut inside a contiguous atom moves to the atom edge with
   less material - the straddling sprite lands whole in one frame.
6. `adopt_orphans`: a small atom isolated from its own cell's bulk joins
   the neighboring cell it sits closer to (thrown blades, sparks).
7. Connected components are labelled on the WHOLE sheet (not per band) and
   assigned in two levels: component -> band by majority mass (strays fully
   inside the empty gap between rows go to the nearest cell bbox - a banner
   leaf poking above its row still lands in the banner's frame), then
   component -> cell by majority column mass inside that band.
8. Fused blobs: a component wider than 1.2*pitch means sprites physically
   touch. Instead of a straight cut at the cell boundary, each internal
   boundary is moved to the blob's THINNEST column within +-0.2*pitch -
   the neck. A blade crossing a boundary detaches at its narrowest point
   instead of being sliced mid-face.
9. Anchor X = least-squares FITTED cell center for the clip (the pitch grid
   is the ground truth of pose placement; per-frame body detection jitters
   and causes visible sliding). Anchor Y = lowest owned row restricted to
   the body blob's column span (the ground contact).
10. Masked crop exported with PAD transparent margin (extends past row-band
    edges - the mask already excludes everything foreign).

## Interpolation (tools/tween.py)

```bash
.venv/bin/pip install opencv-python-headless
.venv/bin/python tools/tween.py sprites.js --steps 1   # 1 in-between = 2x fps
```

DIS optical flow between consecutive frames; both frames are warped half-way
toward each other on an anchor-aligned canvas and alpha-blended. Interpolated
PNGs land in `frames/_tween/<clip>/` and are inserted into the manifest -
the viewer needs no changes (same `[file, ax, ay]` entries). Rerun
`tools/slice.py` to regenerate the manifest without tweens. Quality: clean
for subtle motion (idle breathing), mildly ghosty on large fast swings -
judge per character.

## Viewer (index.html)

- Each clip preloads its frame PNGs; playback draws the masked PNG with
  `imageSmoothingEnabled=false`, anchor pinned to (center, 85% height).
- Zoom is capped per-clip at `fit` = largest scale where every frame's
  full extent stays on canvas - heads/weapons can never be clipped.
- Filmstrip thumbs paint on image `load` events (not `onload` - draw()
  would overwrite the handler for the current frame).

## Tuning knobs (top of slice.py)

- `PITCH_LO`/`PITCH_HI`: pitch search range (150-900px suits ~350-450px cells)
- `ROW_GAP`: raise if rows bleed together
- `MIN_W`/`MIN_H`: noise floor for stray-pixel cells
- `ORPHAN_GAP`: min isolation gap for edge-fragment adoption
- `HARD` (in `dp_cells`): penalty for cutting through content
- `PAD`: transparent margin around exported frames
- `FEET_BAND`: fraction of frame height treated as the ground-contact band

## Debug

Inspect the exported PNGs directly - each `frames/**/fNN.png` is the exact
masked frame. For overlay debugging of cell boundaries:

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0,'tools')
from slice import bands, segment_row
from PIL import Image, ImageDraw
import numpy as np
im = Image.open("sheet.png").convert("RGBA")
a = np.asarray(im)[...,3] > 10
d = ImageDraw.Draw(im)
for y0,y1 in bands(a.any(axis=1), 12):
    m = a[y0:y1]
    for x0,x1 in segment_row(m.any(axis=0), np.asarray(im)[...,3][y0:y1].sum(0)):
        d.rectangle([x0,y0,x1,y1], outline=(255,0,0,255), width=2)
im.save("overlay.png")
EOF
```

## Known limits

- Sprites that overlap with zero empty pixels between them stay one fused
  component; it is split at the cell boundary (unavoidable ambiguity).
- Authored body motion inside a pose (a lunge whose feet genuinely shift)
  is preserved as pose - the ground-contact point is what stays pinned.
- Very irregular pitch inside one row can drift; re-check filmstrip per sheet.
