/* slicer.js - in-browser port of tools/slice.py.
 * Drop a folder of "<Char> - NN label.png" sheets onto the viewer; frames are
 * segmented, component-owned, masked and anchored entirely in JS - no server,
 * no Python. Produces the same clip/frame structure as sprites.js, except each
 * frame carries a <canvas> instead of a file path.
 */
"use strict";

const SL_ALPHA_MIN = 10, SL_ROW_GAP = 12, SL_PITCH_LO = 150,
      SL_PITCH_HI = 900, SL_MIN_W = 12, SL_MIN_H = 12,
      SL_ORPHAN_GAP = 10, SL_PAD = 12, SL_FEET_BAND = 0.25;

/* ---------- geometry helpers ---------- */

function slBands(occ, minGap) {                 // occupied -> [start,end)
  const idx = [];
  for (let i = 0; i < occ.length; i++) if (occ[i]) idx.push(i);
  if (!idx.length) return [];
  const out = []; let s = idx[0], prev = idx[0];
  for (let j = 1; j < idx.length; j++) {
    if (idx[j] - prev > minGap) { out.push([s, prev + 1]); s = idx[j]; }
    prev = idx[j];
  }
  out.push([s, prev + 1]);
  return out;
}

function slEmptyRuns(occ, lo, hi) {             // [[center,len], ...]
  const runs = []; let s = -1;
  for (let i = lo; i < hi; i++) {
    if (!occ[i]) { if (s < 0) s = i; }
    else if (s >= 0) { runs.push([(s + i - 1) >> 1, i - s]); s = -1; }
  }
  if (s >= 0) runs.push([(s + hi - 1) >> 1, hi - s]);
  return runs;
}

function slEstPitch(v) {                        // autocorrelation pitch
  const n = v.length;
  const hi = Math.min(SL_PITCH_HI, n >> 1);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += v[i];
  mean /= n;
  const d = new Float64Array(n);
  let ac0 = 0;
  for (let i = 0; i < n; i++) { d[i] = v[i] - mean; ac0 += d[i] * d[i]; }
  if (hi <= SL_PITCH_LO || ac0 <= 0) return 400;
  const S = new Float64Array(hi);
  for (let l = SL_PITCH_LO; l < hi; l++) {
    let s = 0;
    for (let i = 0; i + l < n; i++) s += d[i] * d[i + l];
    S[l] = s / ac0;
  }
  const peaks = [];
  for (let l = SL_PITCH_LO + 1; l < hi - 1; l++)
    if (S[l] >= S[l - 1] && S[l] > S[l + 1] && S[l] > 0.2) peaks.push(l);
  if (!peaks.length) return n;                  // no periodicity = one frame
  let best = 0;
  for (const p of peaks) best = Math.max(best, S[p]);
  for (const p of peaks) if (S[p] >= 0.5 * best) return p;
}

function slDpCells(cand, free, x1, p) {
  const wmin = 0.45 * p, wmax = 1.9 * p, HARD = (0.30 * p) ** 2;
  const n = cand.length;
  const dp = new Float64Array(n).fill(Infinity), prev = new Int32Array(n).fill(-1);
  dp[0] = 0;
  for (let j = 1; j < n; j++) {
    for (let i = j - 1; i >= 0; i--) {
      const w = cand[j] - cand[i];
      if (w > wmax) break;
      if (w < wmin || dp[i] === Infinity) continue;
      let cost = (w - p) * (w - p);
      if (!free.has(cand[j]) && cand[j] !== x1) cost += HARD;
      if (dp[i] + cost < dp[j]) { dp[j] = dp[i] + cost; prev[j] = i; }
    }
  }
  const cuts = [];
  for (let j = n - 1; j > 0;) { cuts.push(cand[j]); j = prev[j] >= 0 ? prev[j] : 0; }
  cuts.push(cand[0]); cuts.reverse();
  const cells = [];
  for (let i = 0; i + 1 < cuts.length; i++) cells.push([cuts[i], cuts[i + 1]]);
  return cells;
}

function slHealCuts(cells, atoms, p) {
  cells = cells.map(c => c.slice());
  let ai = 0;
  for (let k = 0; k < cells.length - 1; k++) {
    const c = cells[k][1];
    while (ai < atoms.length && atoms[ai][1] <= c) ai++;
    if (ai >= atoms.length) break;
    const a = atoms[ai];
    if (!(a[0] < c && c < a[1])) continue;
    if (a[1] - a[0] > 1.9 * p) continue;
    const edge = (c - a[0] <= a[1] - c) ? a[0] : a[1];
    cells[k][1] = cells[k + 1][0] = edge;
  }
  return cells;
}

function slAdoptOrphans(cells, atoms, p) {
  cells = cells.map(c => c.slice());
  let ai = 0;
  for (let k = 0; k < cells.length - 1; k++) {
    const c = (cells[k][1] + cells[k + 1][0]) >> 1;
    while (ai + 1 < atoms.length && atoms[ai + 1][0] <= c) ai++;
    const a = atoms[ai], b = atoms[ai + 1];
    if (!a || !b || a[1] > c || b[0] < c) continue;
    const wa = a[1] - a[0], wb = b[1] - b[0];
    const pa = atoms[ai - 1], nb = atoms[ai + 2];
    const gap = b[0] - a[1];
    if (wa < 0.22 * p && wb > 0.30 * p && pa && a[0] - pa[1] > SL_ORPHAN_GAP
        && a[0] - pa[1] > gap && a[0] - cells[k][0] >= 0.45 * p) {
      cells[k][1] = cells[k + 1][0] = a[0];
    } else if (wb < 0.22 * p && wa > 0.30 * p && nb && nb[0] - b[1] > SL_ORPHAN_GAP
               && nb[0] - b[1] > gap && cells[k + 1][1] - b[1] >= 0.45 * p) {
      cells[k][1] = cells[k + 1][0] = b[1];
    }
  }
  return cells;
}

function slSegmentRow(colOcc, alphaCol) {
  const occ = [];
  for (let i = 0; i < colOcc.length; i++) if (colOcc[i]) occ.push(i);
  if (!occ.length) return [];
  const x0 = occ[0], x1 = occ[occ.length - 1] + 1;
  let p = slEstPitch(alphaCol);
  const free = new Set();
  for (const [c] of slEmptyRuns(colOcc, x0, x1)) free.add(c);
  let cells = [];
  for (let pass = 0; pass < 2; pass++) {
    const grid = [];
    for (let k = 1; k <= Math.floor((x1 - x0) / p); k++)
      grid.push(Math.round(x0 + k * p));
    const cand = [...new Set([x0, x1, ...free, ...grid])]
      .filter(c => c >= x0 && c <= x1).sort((a, b) => a - b);
    cells = slDpCells(cand, free, x1, p);
    if (cells.length < 3) break;
    const ws = cells.map(c => c[1] - c[0]).sort((a, b) => a - b);
    p = ws[ws.length >> 1];
  }
  if (cells.length > 1) {
    const atoms = slBands(colOcc, 1);
    cells = slHealCuts(cells, atoms, p);
    cells = slAdoptOrphans(cells, atoms, p);
  }
  return cells;
}

/* ---------- connected components (8-connectivity flood fill) ---------- */

function slLabel(mask, W, H) {
  const lab = new Int32Array(W * H);
  const stack = new Int32Array(W * H);
  const mass = [0], bbox = [null];
  let n = 0;
  for (let i = 0; i < W * H; i++) {
    if (!mask[i] || lab[i]) continue;
    n++; let sp = 0;
    stack[sp++] = i; lab[i] = n;
    let m = 0, x0 = W, x1 = 0, y0 = H, y1 = 0;
    while (sp) {
      const px = stack[--sp]; m++;
      const y = (px / W) | 0, x = px - y * W;
      if (x < x0) x0 = x; if (x + 1 > x1) x1 = x + 1;
      if (y < y0) y0 = y; if (y + 1 > y1) y1 = y + 1;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const q = ny * W + nx;
        if (mask[q] && !lab[q]) { lab[q] = n; stack[sp++] = q; }
      }
    }
    mass[n] = m; bbox[n] = [y0, x0, y1, x1];
  }
  return { lab, ncomp: n, mass, bbox };
}

/* ---------- sheet -> clips of canvases ---------- */

function slSliceSheet(px, W, H) {
  const al = new Uint8Array(W * H), mask = new Uint8Array(W * H);
  const rowOcc = new Uint8Array(H);
  for (let i = 0, j = 0; i < px.length; i += 4, j++) {
    const a = px[i + 3];
    if (a > SL_ALPHA_MIN) { mask[j] = 1; rowOcc[(j / W) | 0] = 1; }
    al[j] = a;
  }
  const rbands = slBands(rowOcc, SL_ROW_GAP);
  const { lab, ncomp, mass, bbox } = slLabel(mask, W, H);

  // cells per band
  const cellinfo = [];                          // {ri,ci,cx0,cx1,y0,y1,p}
  for (let ri = 0; ri < rbands.length; ri++) {
    const [y0, y1] = rbands[ri];
    const colOcc = new Uint8Array(W);
    const alphaCol = new Float64Array(W);
    for (let y = y0; y < y1; y++)
      for (let x = 0; x < W; x++) {
        const q = y * W + x;
        if (mask[q]) { colOcc[x] = 1; alphaCol[x] += al[q]; }
      }
    const cells = slSegmentRow(colOcc, alphaCol);
    if (!cells.length) continue;
    const ws = cells.map(c => c[1] - c[0]).sort((a, b) => a - b);
    const p = ws[ws.length >> 1];
    cells.forEach((c, ci) =>
      cellinfo.push({ ri, ci, cx0: c[0], cx1: c[1], y0, y1, p }));
  }
  const bandCells = {};
  cellinfo.forEach((c, gi) => (bandCells[c.ri] = bandCells[c.ri] || []).push(gi));

  // component -> band
  const bandOf = new Int32Array(H).fill(-1);
  rbands.forEach(([y0, y1], ri) => bandOf.fill(ri, y0, y1));
  const bandMass = new Uint32Array((ncomp + 1) * (rbands.length + 1));
  for (let y = 0; y < H; y++) {
    const b = bandOf[y];
    for (let x = 0; x < W; x++) {
      const cid = lab[y * W + x];
      if (cid) bandMass[cid * (rbands.length + 1) + b + 1]++;
    }
  }
  const compBand = new Int32Array(ncomp + 1), inGap = new Uint8Array(ncomp + 1);
  for (let cid = 1; cid <= ncomp; cid++) {
    let best = 0, bv = 0;
    for (let ri = 0; ri < rbands.length; ri++) {
      const v = bandMass[cid * (rbands.length + 1) + ri + 1];
      if (v > bv) { bv = v; best = ri; }
    }
    compBand[cid] = best;
    inGap[cid] = bv === 0;
  }
  const nearestCell = cid => {
    const [cy0, cx0, cy1, cx1] = bbox[cid];
    let best = -1, bd = Infinity;
    cellinfo.forEach((c, gi) => {
      const dx = Math.max(c.cx0 - cx1, cx0 - c.cx1, 0);
      const dy = Math.max(c.y0 - cy1, cy0 - c.y1, 0);
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = gi; }
    });
    return best;
  };

  // component -> cell ownership
  const owner = new Int32Array(ncomp + 1).fill(-1);
  const fusedOwn = new Int32Array(W * H).fill(-1);
  const cellMass = new Float64Array(cellinfo.length);
  const cellComps = cellinfo.map(() => []);
  for (let cid = 1; cid <= ncomp; cid++) {
    if (!mass[cid]) continue;
    if (inGap[cid]) { owner[cid] = nearestCell(cid); continue; }
    const ri = compBand[cid], gis = bandCells[ri] || [];
    const [cy0, cx0, cy1, cx1] = bbox[cid];
    const hit = gis.filter(gi => {
      const c = cellinfo[gi];
      return c.cx0 < cx1 && c.cx1 > cx0;
    });
    if (!hit.length) { owner[cid] = nearestCell(cid); continue; }
    // per-column mass inside the comp bbox
    const cmass = new Float64Array(cx1 - cx0);
    for (let y = cy0; y < cy1; y++)
      for (let x = cx0; x < cx1; x++)
        if (lab[y * W + x] === cid) cmass[x - cx0]++;
    const p = cellinfo[hit[0]].p;
    if (hit.length === 1 || cx1 - cx0 <= 1.2 * p) {
      let best = hit[0], bv = -1;
      for (const gi of hit) {
        const c = cellinfo[gi];
        let v = 0;
        for (let x = Math.max(cx0, c.cx0); x < Math.min(cx1, c.cx1); x++)
          v += cmass[x - cx0];
        if (v > bv) { bv = v; best = gi; }
      }
      owner[cid] = best;
      continue;
    }
    // fused blob: neck cuts
    hit.sort((a, b) => cellinfo[a].cx0 - cellinfo[b].cx0);
    const gmap = {};
    for (const gi of gis) gmap[cellinfo[gi].ci] = gi;
    const loI = cellinfo[hit[0]].ci, hiI = cellinfo[hit[hit.length - 1]].ci;
    const necks = [];
    for (let k = loI; k < hiI; k++) {
      const b = cellinfo[gmap[k]].cx1;
      const w = Math.floor(0.2 * p);
      const lo = Math.max(cx0, b - w), hi2 = Math.min(cx1, b + w);
      let cut = b;
      if (hi2 > lo) {
        let mv = Infinity;
        for (let x = lo; x < hi2; x++)
          if (cmass[x - cx0] < mv) { mv = cmass[x - cx0]; cut = x; }
      }
      necks.push(cut);
    }
    for (let y = cy0; y < cy1; y++)
      for (let x = cx0; x < cx1; x++) {
        if (lab[y * W + x] !== cid) continue;
        let k = 0;
        while (k < necks.length && x >= necks[k]) k++;
        fusedOwn[y * W + x] = hit[k < hit.length ? k : hit.length - 1];
      }
  }

  // per-pixel owner: fused splits override component ownership
  const pixOwner = new Int32Array(W * H).fill(-1);
  for (let i = 0; i < W * H; i++) {
    const cid = lab[i];
    if (!cid) continue;
    pixOwner[i] = fusedOwn[i] >= 0 ? fusedOwn[i] : owner[cid];
  }

  // per-cell bbox + mass bookkeeping
  const cbox = cellinfo.map(() => [H, W, 0, 0]);  // y0,x0,y1,x1
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const gi = pixOwner[y * W + x];
      if (gi < 0) continue;
      cellMass[gi]++;
      const b = cbox[gi];
      if (y < b[0]) b[0] = y; if (y + 1 > b[2]) b[2] = y + 1;
      if (x < b[1]) b[1] = x; if (x + 1 > b[3]) b[3] = x + 1;
    }
  for (let cid = 1; cid <= ncomp; cid++)
    if (owner[cid] >= 0) cellComps[owner[cid]].push(cid);

  const rows = {};
  cellinfo.forEach((c, gi) => {
    const [fy0, fx0, fy1, fx1] = cbox[gi];
    if (fy1 <= fy0 || fx1 - fx0 < SL_MIN_W || fy1 - fy0 < SL_MIN_H) return;
    // body blob = largest owned component
    let bodyCid = 0, bm = 0;
    for (const cid of cellComps[gi])
      if (mass[cid] > bm) { bm = mass[cid]; bodyCid = cid; }
    let gx0 = 0, gx1 = W;
    if (bodyCid && bm >= 0.3 * cellMass[gi]) {
      const mrg = Math.floor(0.15 * c.p);
      gx0 = Math.max(0, bbox[bodyCid][1] - mrg);
      gx1 = Math.min(W, bbox[bodyCid][3] + mrg);
    }
    // anchor Y = lowest owned row inside the body column span
    const bandH = Math.max(24, Math.floor((fy1 - fy0) * SL_FEET_BAND));
    let ay = fy1;
    for (let y = Math.max(fy0, fy1 - bandH); y < fy1; y++)
      for (let x = gx0; x < gx1; x++)
        if (pixOwner[y * W + x] === gi && y + 1 > ay) ay = y + 1;
    // masked crop + PAD
    const cy0 = Math.max(0, fy0 - SL_PAD), cy1c = Math.min(H, fy1 + SL_PAD);
    const dx0 = Math.max(0, fx0 - SL_PAD), dx1 = Math.min(W, fx1 + SL_PAD);
    const out = new ImageData(dx1 - dx0, cy1c - cy0);
    for (let y = cy0; y < cy1c; y++)
      for (let x = dx0; x < dx1; x++) {
        if (pixOwner[y * W + x] !== gi) continue;
        const s = (y * W + x) * 4, d = ((y - cy0) * (dx1 - dx0) + x - dx0) * 4;
        out.data[d] = px[s]; out.data[d + 1] = px[s + 1];
        out.data[d + 2] = px[s + 2]; out.data[d + 3] = px[s + 3];
      }
    const cnv = document.createElement("canvas");
    cnv.width = dx1 - dx0; cnv.height = cy1c - cy0;
    cnv.getContext("2d").putImageData(out, 0, 0);
    (rows[c.ri] = rows[c.ri] || []).push({
      cnv, cy0, dx0, ay, ci: c.ci, cc: (c.cx0 + c.cx1) / 2 });
  });

  // fitted grid anchor X per row
  const clips = [];
  for (const ri of Object.keys(rows).sort((a, b) => a - b)) {
    const fs = rows[ri];
    let a_ = 0, b_ = 0;
    if (fs.length > 1) {
      const n = fs.length;
      let mi = 0, mc = 0;
      for (const f of fs) { mi += f.ci; mc += f.cc; }
      mi /= n; mc /= n;
      let num = 0, den = 0;
      for (const f of fs) {
        num += (f.ci - mi) * (f.cc - mc);
        den += (f.ci - mi) * (f.ci - mi);
      }
      b_ = den ? num / den : 0; a_ = mc - b_ * mi;
    } else {
      a_ = fs[0].cc;
    }
    clips.push(fs.map(f => [f.cnv, +(a_ + b_ * f.ci - f.dx0).toFixed(1),
                            +(f.ay - f.cy0).toFixed(1)]));
  }
  return clips;
}

function slParseName(stem) {
  const parts = stem.split(" - ").map(s => s.trim());
  if (parts.length >= 2)
    return [parts[0], parts.slice(1).join(" - ").replace(/^\d+\s*/, "")];
  return [stem, "sheet"];
}
