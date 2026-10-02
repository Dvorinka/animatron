#!/usr/bin/env python3
"""Slice character sprite sheets into animation clips.

Input : a directory of PNG sheets ("<Char> - NN label.png"), frames packed in
        rows. Frames may nearly touch (<4px gaps) or scatter into fragments
        (dissolve effects >200px apart internally).

Method: split each row into column fragments (gap>=FRAG_GAP), merge fragments
        separated by <MERGE_GAP when the combined width stays under MAX_W,
        then subdivide oversized blocks at estimated pitch snapped to the
        nearest empty column.

Output: sprites.js -> window.SPRITES = {characters: [...]}, plus masked
        per-frame PNG crops in frames/ (only owned components rendered -
        foreign pixels can never appear inside a frame). Sheets are never
        modified.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ALPHA_MIN = 10   # alpha > this counts as occupied
ROW_GAP = 12     # empty px between rows
PITCH_LO = 150   # autocorrelation lag range for frame pitch
PITCH_HI = 900
MIN_W = 12       # discard slices narrower than this
MIN_H = 12
ORPHAN_GAP = 10  # min gap isolating a small atom for neighbor adoption
PAD = 12         # transparent padding around each exported frame
FEET_BAND = 0.25 # bottom fraction of body blob used for the anchor

# ponytail: clip names are neutral ("label · r2"), semantics left to user


def bands(occupied: np.ndarray, min_gap: int):
    """[start,end) occupied ranges split by >=min_gap empty columns/rows."""
    idx = np.flatnonzero(occupied)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > min_gap)
    starts = np.concatenate(([idx[0]], idx[cuts + 1]))
    ends = np.concatenate((idx[cuts], [idx[-1]]))
    return [(int(s), int(e) + 1) for s, e in zip(starts, ends)]


def empty_runs(col_occ: np.ndarray, lo: int, hi: int):
    """Empty (center,length) runs inside [lo,hi)."""
    occ = col_occ[lo:hi]
    idx = np.flatnonzero(~occ)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    out = []
    for s, e in zip(np.concatenate(([0], cuts + 1)),
                    np.concatenate((cuts, [len(idx) - 1]))):
        out.append((lo + (int(idx[s]) + int(idx[e])) // 2,
                    int(idx[e]) - int(idx[s]) + 1))
    return out


def est_pitch(alpha_col: np.ndarray) -> float:
    """Frame pitch via autocorrelation of the per-column alpha mass.

    Binary occupancy goes flat on rows where sprites touch edge to edge;
    alpha mass stays periodic. Returns the smallest strong peak so a
    2*P harmonic cannot masquerade as the pitch.
    """
    v = alpha_col.astype(np.float64)
    v -= v.mean()
    ac = np.correlate(v, v, "full")[len(v) - 1:]
    hi = min(PITCH_HI, len(v) // 2)
    if hi <= PITCH_LO or ac[0] <= 0:
        return 400.0
    seg = ac[PITCH_LO:hi] / ac[0]
    peaks = [PITCH_LO + i for i in range(1, len(seg) - 1)
             if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1] and seg[i] > 0.2]
    if not peaks:
        return float(len(v))  # no periodicity -> single frame
    best = max(seg[p - PITCH_LO] for p in peaks)
    return float(min(p for p in peaks if seg[p - PITCH_LO] >= 0.5 * best))


def dp_cells(cand, free, x1, p):
    """Min-cost segmentation over candidate cuts -> list of (a,b) cells."""
    wmin, wmax = 0.45 * p, 1.9 * p
    n = len(cand)
    HARD = (0.30 * p) ** 2          # penalty for cutting through sprite content
    INF = float("inf")
    dp = [INF] * n
    prev = [-1] * n
    dp[0] = 0.0
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            w = cand[j] - cand[i]
            if w > wmax:
                break
            if w < wmin or dp[i] == INF:
                continue
            cost = (w - p) ** 2
            if cand[j] not in free and cand[j] != x1:
                cost += HARD
            if dp[i] + cost < dp[j]:
                dp[j] = dp[i] + cost
                prev[j] = i
    cuts = []
    j = n - 1
    while j > 0:
        cuts.append(cand[j])
        j = prev[j] if prev[j] >= 0 else 0
    cuts.append(cand[0])
    cuts.reverse()
    return [(a, b) for a, b in zip(cuts, cuts[1:])]


def adopt_orphans(cells, atoms, p):
    """Move small edge atoms into the neighboring cell they sit closer to.

    A detached part (a thrown blade, spark) separated by a gap from its own
    cell's bulk but close to the next cell's body belongs to that frame.
    """
    cells = [list(c) for c in cells]
    ai = 0  # atoms sorted by position; index boundary pairs lazily
    for k in range(len(cells) - 1):
        c = (cells[k][1] + cells[k + 1][0]) // 2
        while ai + 1 < len(atoms) and atoms[ai + 1][0] <= c:
            ai += 1
        a, b = atoms[ai], atoms[ai + 1] if ai + 1 < len(atoms) else None
        if a is None or b is None or a[1] > c or b[0] < c:
            continue
        wa, wb = a[1] - a[0], b[1] - b[0]
        pa = atoms[ai - 1] if ai > 0 else None
        nb = atoms[ai + 2] if ai + 2 < len(atoms) else None
        gap_ab = b[0] - a[1]
        # left orphan: small atom at end of cell k, far from its own bulk,
        # near the next cell's body -> adopt right
        if (wa < 0.22 * p and wb > 0.30 * p and pa is not None
                and a[0] - pa[1] > ORPHAN_GAP and a[0] - pa[1] > gap_ab
                and a[0] - cells[k][0] >= 0.45 * p):
            cells[k][1] = a[0]
            cells[k + 1][0] = a[0]
        # right orphan: symmetric, adopt left
        elif (wb < 0.22 * p and wa > 0.30 * p and nb is not None
              and nb[0] - b[1] > ORPHAN_GAP and nb[0] - b[1] > gap_ab
              and cells[k + 1][1] - b[1] >= 0.45 * p):
            cells[k][1] = b[1]
            cells[k + 1][0] = b[1]
    return [(a, b) for a, b in cells]


def heal_cuts(cells, atoms, p):
    """Resolve boundaries that slice through content.

    If an atom straddles a cut, the cut sits inside a sprite (no empty
    column existed). Move the boundary to the atom edge on the side with
    less material, so one frame keeps the whole piece.
    """
    cells = [list(c) for c in cells]
    ai = 0
    for k in range(len(cells) - 1):
        c = cells[k][1]
        while ai < len(atoms) and atoms[ai][1] <= c:
            ai += 1
        if ai >= len(atoms):
            break
        a = atoms[ai]
        if not (a[0] < c < a[1]):
            continue                    # boundary already in an empty gap
        if a[1] - a[0] > 1.9 * p:
            continue                    # fused beyond repair, keep grid cut
        left, right = c - a[0], a[1] - c
        if left <= right:
            cells[k][1] = cells[k + 1][0] = a[0]
        else:
            cells[k][1] = cells[k + 1][0] = a[1]
    return [(a, b) for a, b in cells]


def segment_row(col_occ: np.ndarray, alpha_col: np.ndarray):
    """Column ranges of logical frames in one row.

    Frames sit on a roughly constant pitch. DP over candidate cuts: centers
    of empty column runs are free, pitch-grid cuts are penalized, cost
    rewards uniform widths. Pitch is re-estimated from the first solution
    (autocorrelation error drifts otherwise), then a second DP runs and
    detached edge atoms get adopted by the neighbor they sit closer to.
    """
    occ = np.flatnonzero(col_occ)
    if occ.size == 0:
        return []
    x0, x1 = int(occ[0]), int(occ[-1]) + 1
    p = est_pitch(alpha_col)
    runs = empty_runs(col_occ, x0, x1)
    free = {c for c, _ in runs}

    cells = []
    for _ in range(2):  # estimate pitch, segment, refine pitch, segment again
        grid = [round(x0 + k * p) for k in range(1, int((x1 - x0) / p) + 1)]
        cand = sorted(set([x0, x1] + list(free) + grid))
        cand = [c for c in cand if x0 <= c <= x1]
        cells = dp_cells(cand, free, x1, p)
        if len(cells) < 3:
            break
        widths = sorted(b - a for a, b in cells)
        p = widths[len(widths) // 2]  # median cell width ~= true pitch
    if len(cells) > 1:
        atoms = bands(col_occ, 1)       # contiguous occupied column runs
        cells = heal_cuts(cells, atoms, p)
        cells = adopt_orphans(cells, atoms, p)
    return cells


def parse_name(stem: str):
    """'Char - 10 attack 1 - Name' -> (char, label)."""
    parts = [p.strip() for p in stem.split(" - ")]
    if len(parts) >= 2:
        label = re.sub(r"^\d+\s*", "", " - ".join(parts[1:]))
        return parts[0], label
    return stem, "sheet"


def slice_sheet(path: Path, outdir: Path, root: Path):
    im = Image.open(path).convert("RGBA")
    px = np.asarray(im)
    al = px[..., 3]
    a = al > ALPHA_MIN
    H, W = a.shape
    rbands = bands(a.any(axis=1), ROW_GAP)
    labels, ncomp = ndimage.label(a, structure=np.ones((3, 3)))
    comp_mass = np.bincount(labels.ravel()); comp_mass[0] = 0

    # component bboxes + per-column pixel counts (bbox slices)
    cobjs = ndimage.find_objects(labels)
    bbox = np.zeros((ncomp + 1, 4))           # y0, x0, y1, x1
    for i, sl in enumerate(cobjs):
        if sl is not None:
            bbox[i + 1] = sl[0].start, sl[1].start, sl[0].stop, sl[1].stop

    # cells per band
    cellinfo = []                             # (ri, ci, cx0, cx1, y0, y1, p)
    for ri, (y0, y1) in enumerate(rbands):
        cells = segment_row(a[y0:y1].any(axis=0), al[y0:y1].sum(axis=0))
        if not cells:
            continue
        p = float(np.median([c[1] - c[0] for c in cells]))
        for ci, (cx0, cx1) in enumerate(cells):
            cellinfo.append((ri, ci, cx0, cx1, y0, y1, p))

    # component -> band: majority mass inside band rows; a component lying
    # entirely in a gap goes to the nearest cell bbox (banner fragments that
    # poke above their own row, dust drifting between rows)
    band_of = np.full(H, -1)
    for ri, (y0, y1) in enumerate(rbands):
        band_of[y0:y1] = ri
    band_pix = np.repeat(band_of, W)
    bandmat = np.zeros((ncomp + 1, len(rbands) + 1), dtype=np.int64)
    np.add.at(bandmat, (labels.ravel(), band_pix + 1), 1)
    mass_in_band = bandmat[:, 1:]
    comp_band = mass_in_band.argmax(axis=1)
    in_gap = mass_in_band.max(axis=1) == 0

    def nearest_cell(cid):
        cy0, cx0, cy1, cx1 = bbox[cid]
        best, bd = -1, np.inf
        for gi, (_, _, ex0, ex1, ey0, ey1, _) in enumerate(cellinfo):
            dx = max(ex0 - cx1, cx0 - ex1, 0)
            dy = max(ey0 - cy1, cy0 - ey1, 0)
            d = dx * dx + dy * dy
            if d < bd:
                bd, best = d, gi
        return best

    # component -> cell (global index into cellinfo). Whole-component when
    # <= 1.2 pitches wide; wider = fused sprites -> split at the blob's
    # thinnest columns near each boundary (a neck cut leaves no hard seam).
    owner = np.full(ncomp + 1, -1)
    fused_own = np.full(a.shape, -1, dtype=np.int32)
    band_cells = {}
    for gi, (ri, ci, cx0, cx1, y0, y1, p) in enumerate(cellinfo):
        band_cells.setdefault(ri, []).append(gi)

    for cid in range(1, ncomp + 1):
        if comp_mass[cid] == 0:
            continue
        if in_gap[cid]:
            owner[cid] = nearest_cell(cid)
            continue
        ri = comp_band[cid]
        gis = band_cells[ri]
        cy0, cx0, cy1, cx1 = bbox[cid]
        hit = [(gi, cellinfo[gi]) for gi in gis
               if cellinfo[gi][2] < cx1 and cellinfo[gi][3] > cx0]
        if not hit:
            owner[cid] = nearest_cell(cid)
            continue
        sl = cobjs[cid - 1]
        cmass = (labels[sl] == cid).sum(axis=0)
        cols = np.arange(int(cx0), int(cx1))
        p = hit[0][1][6]
        if len(hit) == 1 or cx1 - cx0 <= 1.2 * p:
            # majority mass across the cells it overlaps
            best, bv = hit[0][0], -1
            for gi, (_, _, ex0, ex1, *_r) in hit:
                v = cmass[(cols >= ex0) & (cols < ex1)].sum()
                if v > bv:
                    bv, best = v, gi
            owner[cid] = best
            continue
        # fused blob: neck-cut at the thinnest column near each boundary
        hit.sort(key=lambda t: t[1][2])
        gmap = {cellinfo[g][1]: g for g in gis}
        lo_i, hi_i = hit[0][1][1], hit[-1][1][1]
        necks = []
        for k in range(lo_i, hi_i):
            b = cellinfo[gmap[k]][3]      # right edge of cell k = boundary
            w = int(0.2 * p)
            lo = max(int(cx0), b - w)
            hi = min(int(cx1), b + w)
            cut = lo + int(np.argmin(cmass[lo - int(cx0):hi - int(cx0)])) \
                if hi > lo else b
            necks.append(cut)
        gidx = np.searchsorted(necks, cols, side="right")
        comp_sel = labels[sl] == cid
        pix_own = np.take([gi for gi, _ in hit], gidx)[None, :]
        fused_view = fused_own[sl]
        fused_view[comp_sel] = np.broadcast_to(pix_own,
                                               comp_sel.shape)[comp_sel]
    clips = []
    for gi, (ri, ci, cx0, cx1, y0, y1, p) in enumerate(cellinfo):
        mine = np.flatnonzero(owner == gi)
        fm = np.isin(labels, mine) | (fused_own == gi)
        if not fm.any():
            continue
        xs = np.flatnonzero(fm.any(axis=0))
        ys = np.flatnonzero(fm.any(axis=1))
        fx0, fx1 = int(xs[0]), int(xs[-1]) + 1
        fy0, fy1 = int(ys[0]), int(ys[-1]) + 1
        if fx1 - fx0 < MIN_W or fy1 - fy0 < MIN_H:
            continue
        # anchor X = ideal grid position for this cell, computed AFTER all
        # cells are collected (a per-frame body estimate jitters - feet,
        # debris and pose shifts move the detected point; the pitch grid is
        # the ground truth of where the animator placed each pose).
        # anchor Y = lowest owned row restricted to the body column span -
        # that is the ground contact (soles), which stays consistent.
        fl, _ = ndimage.label(fm, structure=np.ones((3, 3)))
        fmass = np.bincount(fl.ravel()); fmass[0] = 0
        if fmass.size > 1 and fmass.max() >= 0.3 * fm.sum():
            body = fl == fmass.argmax()
            bc = np.flatnonzero(body.any(axis=0))
            m = int(0.15 * p)
            gx0 = max(0, int(bc[0]) - m)
            gx1 = min(W, int(bc[-1]) + m)
        else:
            gx0, gx1 = 0, W
        g = fm.copy()
        g[:, :gx0] = False
        g[:, gx1:] = False
        band_h = max(24, int((fy1 - fy0) * FEET_BAND))
        g[:max(0, fy1 - band_h)] = False
        if g.any():
            ay = float(np.flatnonzero(g.any(axis=1))[-1] + 1)
        else:
            ay = float(fy1)
        # export the masked frame: only owned components, nothing else.
        # Crop extends PAD px into the sheet - the mask already excludes
        # everything foreign, so neighbors can never bleed in.
        cy0, cy1c = max(0, fy0 - PAD), min(H, fy1 + PAD)
        dx0, dx1 = max(0, fx0 - PAD), min(W, fx1 + PAD)
        sub_m = fm[cy0:cy1c, dx0:dx1]
        out = np.zeros((cy1c - cy0, dx1 - dx0, 4), dtype=np.uint8)
        out[sub_m] = px[cy0:cy1c, dx0:dx1][sub_m]
        rel = outdir / f"r{ri + 1}" / f"f{ci + 1:02d}.png"
        rel.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(rel)
        clips.append((rel.relative_to(root).as_posix(),
                      ay - cy0, ri, gi, dx0))
    rows = {}
    for ri in sorted({f[2] for f in clips}):
        fs = [f for f in clips if f[2] == ri]
        # anchor X = fitted grid center: a least-squares line through the
        # cell centers is immune to the per-boundary nudges that healed or
        # orphaned cells introduce, so the pin point cannot jitter
        cis = np.array([cellinfo[f[3]][1] for f in fs])
        cents = np.array([(cellinfo[f[3]][2] + cellinfo[f[3]][3]) / 2
                          for f in fs])
        if len(fs) > 1:
            b_, a_ = np.polyfit(cis, cents, 1)
        else:
            b_, a_ = 0.0, float(cents[0])
        rows[ri] = [[f[0], round(a_ + b_ * cellinfo[f[3]][1] - f[4], 1),
                     round(f[1], 1)]
                    for f in fs]
    return [(ri, fs) for ri, fs in sorted(rows.items())]


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "sprites.js").resolve()
    root = out.parent  # frame paths are stored relative to the manifest
    frames_root = root / "frames"
    chars = {}
    for png in sorted(src.rglob("*.png")):
        name, label = parse_name(png.stem)
        safe = re.sub(r"[^\w-]+", "_", png.stem)
        rows = slice_sheet(png, frames_root / safe, root)
        ch = chars.setdefault(name, {"name": name, "clips": []})
        for ri, frames in rows:
            # frame = [file, ax, ay]: masked PNG + anchor point (feet,
            # crop-relative) the renderer pins to a fixed screen spot
            ch["clips"].append({
                "name": f"{label} · r{ri + 1}" if len(rows) > 1 else label,
                "frames": [[f[0], round(f[1], 1), round(f[2], 1)]
                           for f in frames],
            })
        print(f"{png.name}: {sum(len(f) for _, f in rows)} frames, "
              f"{len(rows)} rows "
              f"({', '.join(str(len(f)) for _, f in rows)})")
    data = {"characters": [chars[k] for k in sorted(chars)]}
    out.write_text("window.SPRITES = " + json.dumps(data) + ";\n")
    print(f"-> {out} ({sum(len(c['clips']) for c in data['characters'])} clips)")


if __name__ == "__main__":
    main()
