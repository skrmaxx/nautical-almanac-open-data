"""
Port approaches from AIS traffic, for Port to port's "follow the traffic" option: for each port, the ways ships take
from the harbour out to the lane network (the dredged channel, the fairway through the islands), as a small tree of
lane pieces from the port to the lane points round it.

Usage: python -I tools/ais/port_approaches.py <lanes.bin refined by refine_lanes.py> <ports.tsv> <rank_005.u8>
         <meta_005.json> <world_land_10m.bin> <sea_hazards.bin> <out lanes.bin> <work folder (refine_lanes.py's)>
         [--reach 45] [--workers 6]

For each port (not one inside a canal or strait, nor past 72 deg): the cheapest ways on the traffic grid from the
port's cell (sea cells costing 1 + A (1 - rank)^G, so ships' own tracks are cheapest; land within 5 NM of the port,
where the harbour position is ashore on the map, at 20) towards the nearest lane point in each 30-degree sector
between 8 and --reach NM away, each followed only to its offing: the first cell 3 NM from land, or 12 NM along. Only ways that bend
(a fairway between islands) or whose straight line crosses land past the port's own are kept.
Simplified to straight pieces that stay at sea (within the port's 5 NM, also over its land); ways out ending within
3 NM of each other kept once. A port whose ways out are all straight lines gets none.

Writes a copy of the lanes file with, after its other sections:
  int 0x41505232 ("APR2"); int ports; per port: int lat * 1e5, int lon * 1e5, byte ways;
  per way: byte points; per point (the port itself left out): int lat * 1e5, int lon * 1e5.
Needs numpy, scikit-image, Pillow (and refine_lanes.py beside it).
"""
import math
import os
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np
from skimage.graph import MCP_Geometric

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lane_traffic import read_lanes, nm, Grid  # noqa: E402
from refine_lanes import (A, G, arg, land_fraction, hazard_cells, passage_keep_out, disc)  # noqa: E402

PORT_LAND_NM = 5.0
OFFING_NM = 3.0
MAX_WAY_NM = 12.0


def sections(path):
    """The base network's points and pieces, and how many points the later sections add."""
    b = open(path, "rb").read()
    _, keys, ll, e = read_lanes(path)
    o = 8 + sum(2 + len(k.encode()) for k in keys) + 4 + 8 * len(ll) + 4 + 5 * len(e)
    extra = 0
    while o + 8 <= len(b):
        magic, n = struct.unpack_from(">ii", b, o); o += 8
        if magic == 0x54524631:
            o += n
        elif magic == 0x41495331:
            extra += n; o += 8 * n
            (m,) = struct.unpack_from(">i", b, o); o += 4 + 9 * m
        else:
            raise SystemExit("lanes file has a section %x already: start from refine_lanes.py's output" % magic)
    return b, np.array(ll), e, len(ll) + extra


W = {}


def init(state):
    W.update(state)


def approach(port):
    """The tree for one port: (points as (lat, lon), pieces as (i, j) into the points or ('L', lane point)), or None."""
    name, la, lo = port
    lat0, cell, cols, rows, lon0 = W["lat0"], W["cell"], W["cols"], W["rows"], W["lon0"]
    reach = W["reach"]
    ru = int((lat0 - la) / cell); cu = int((lo - lon0) / cell)
    k = math.cos(math.radians(la))
    hr = int((reach + 5) / 60 / cell); hc = int(hr / max(k, 0.25))
    r0, r1, c0, c1 = ru - hr, ru + hr + 1, cu - hc, cu + hc + 1
    if r0 < 0 or r1 > rows or c0 < 0 or c1 > cols:
        return None
    sub = W["cost"][r0:r1, c0:c1].copy()
    # The port's own land: the harbour position is often ashore on the map.
    pr = int(PORT_LAND_NM / 60 / cell); pc = int(pr / max(k, 0.25))
    yy, xx = np.ogrid[-pr:pr + 1, -pc:pc + 1]
    near = (yy ** 2 + (xx * k) ** 2) <= pr ** 2
    blk = sub[ru - r0 - pr:ru - r0 + pr + 1, cu - c0 - pc:cu - c0 + pc + 1]
    blk[(blk < 0) & near] = 20.0
    sub[ru - r0, cu - c0] = 1.0
    mcp = MCP_Geometric(sub, sampling=(1.0, k))
    cum, _ = mcp.find_costs([(ru - r0, cu - c0)])
    ll = W["ll"]
    # The nearest lane point in each 30-degree sector, 8 to reach NM away, that the grid reaches.
    best = {}
    for v in W["bins_near"](la, lo):
        lv, ov = ll[v]
        d = nm(la, lo, lv, ov)
        if d < 8 or d > reach:
            continue
        rv = int((lat0 - lv) / cell) - r0; cv = int((ov - lon0) / cell) - c0
        if not (1 <= rv < sub.shape[0] - 1 and 1 <= cv < sub.shape[1] - 1):
            continue
        b = cum[rv - 1:rv + 2, cv - 1:cv + 2]
        if not np.isfinite(b.min()):
            continue
        brg = math.degrees(math.atan2((ov - lo) * k, lv - la)) % 360
        sec = int(brg // 30)
        if sec not in best or d < best[sec][0]:
            i = int(np.argmin(b)); best[sec] = (d, v, (rv - 1 + i // 3, cv - 1 + i % 3))
    if not best:
        return None
    # Each way out, from the port to its offing: the first cell OFFING_NM or more from land (by the map), or MAX_WAY_NM
    # along the way, whichever comes first. Past it the router's own joining takes over.
    from scipy.ndimage import distance_transform_edt
    sea = W["sea"]
    win_sea = sea[r0:r1, c0:c1]
    dland = distance_transform_edt(win_sea, sampling=(1.0, k)) * cell * 60
    branches = []
    for d, v, end in best.values():
        path = mcp.traceback(end)
        run_nm = 0.0; off = None
        for i in range(1, len(path)):
            (y0, x0), (y1, x1) = path[i - 1], path[i]
            run_nm += math.hypot(y1 - y0, (x1 - x0) * k) * cell * 60
            if (dland[y1, x1] >= OFFING_NM and run_nm >= 2.0) or run_nm >= MAX_WAY_NM:
                off = i
                break
        if off is None:
            continue
        cells = [(r + r0, c + c0) for r, c in path[:off + 1]]
        keep = simplify_run(cells, sea, (ru, cu), pr, pc)
        pts = [(la, lo)] + [(lat0 - (cells[i][0] + 0.5) * cell, lon0 + (cells[i][1] + 0.5) * cell) for i in keep[1:]]
        length = sum(nm(*p, *q) for p, q in zip(pts, pts[1:]))
        straight = nm(*pts[0], *pts[-1])
        # Only a way the router would not take by itself: one that bends (a fairway between islands) or whose straight
        # line crosses land past the port's own.
        bends = length > straight * 1.05 + 0.5
        if not bends and not chord_blocked(cells[0], cells[-1], sea, (ru, cu), pr, pc):
            continue
        branches.append((length, straight, pts))
    # One way per offing (ways ending within 3 NM of each other are the same way out); none where every way out is
    # straight (an open roadstead: the router's straight join is right there).
    branches.sort()
    out = []
    for length, straight, pts in branches:
        if any(nm(*pts[-1], *o[-1]) < 3.0 for o in out):
            continue
        out.append(pts)
    if not out:
        return None
    return name, la, lo, out


def chord_blocked(a, b, sea, port, pr, pc):
    m = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 2
    for t in np.linspace(0, 1, m):
        y = int(round(a[0] + (b[0] - a[0]) * t)); x = int(round(a[1] + (b[1] - a[1]) * t))
        if not sea[y, x] and not (abs(y - port[0]) <= pr and abs(x - port[1]) <= pc):
            return True
    return False


def simplify_run(cells, sea, port, pr, pc):
    """Douglas-Peucker over a run of cells (tolerance one cell), a chord kept only where it stays at sea (or on the
    port's own land)."""
    pts = np.array(cells, float)

    def ok(y, x):
        y, x = int(round(y)), int(round(x))
        return sea[y, x] or (abs(y - port[0]) <= pr and abs(x - port[1]) <= pc)

    def clear(i, j):
        a, b = pts[i], pts[j]
        m = int(max(abs(b - a))) * 2 + 2
        return all(ok(*(a + (b - a) * t)) for t in np.linspace(0, 1, m))

    out = [0]

    def rec(i, j):
        if j <= i + 1:
            return
        a, b = pts[i], pts[j]
        d = b - a
        nrm = math.hypot(*d) or 1
        dev = np.abs((pts[i + 1:j, 0] - a[0]) * d[1] - (pts[i + 1:j, 1] - a[1]) * d[0]) / nrm
        k = int(np.argmax(dev)) + i + 1
        if dev.max() > 1.0 or not clear(i, j):
            rec(i, k); out.append(k); rec(k, j)

    rec(0, len(pts) - 1)
    out.append(len(pts) - 1)
    return sorted(set(out))


def bins_near_factory(bins, reach):
    def f(la, lo):
        dl = reach / 60 + 0.1; dlo = dl / max(0.2, math.cos(math.radians(la)))
        out = []
        for y in range(math.floor(la - dl), math.floor(la + dl) + 1):
            for x in range(math.floor(lo - dlo), math.floor(lo + dlo) + 1):
                out += bins.get((y, x), [])
        return out
    return f


def run(port):
    if "bins_near" not in W:
        W["bins_near"] = bins_near_factory(W["bins"], W["reach"])
    return approach(port)


def main():
    lanes_path, ports_path, rank_path, meta_path, land_path, haz_path, out_path, work = sys.argv[1:9]
    reach = arg("--reach", 45.0); workers = arg("--workers", 6, int)
    t0 = time.time()
    raw, ll, e, first = sections(lanes_path)
    grid = Grid(rank_path, meta_path, 0)
    s_raw = grid.raw.astype(np.float32) / 255
    cache = os.path.join(work, "landfrac.npy")
    lf = np.load(cache) if os.path.exists(cache) else land_fraction(land_path, grid)
    haz = hazard_cells(haz_path, grid)
    keep_out = passage_keep_out(grid, ll, e, ring=15.0)
    sea = (lf < 0.5) & ~haz
    cost = np.where(sea & ~keep_out, 1 + A * (1 - s_raw) ** G, -1).astype(np.float32)
    print("grid ready, %.0f s" % (time.time() - t0), flush=True)
    ports = []
    for line in open(ports_path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        try:
            la, lo = float(f[8]), float(f[9])
        except (IndexError, ValueError):
            continue
        if abs(la) > 72:
            continue
        r = int((grid.lat0 - la) / grid.cell); c = int(((lo - grid.lon0) % 360) / grid.cell)
        if not (0 <= r < grid.rows) or keep_out[r, c]:
            continue
        ports.append((f[1], la, lo))
    bins = {}
    for i, (la, lo) in enumerate(ll):
        bins.setdefault((math.floor(la), math.floor(lo)), []).append(i)
    state = dict(lat0=grid.lat0, cell=grid.cell, rows=grid.rows, cols=grid.cols, lon0=grid.lon0, cost=cost, ll=ll,
                 sea=sea, rank=grid.raw, reach=reach, bins=bins)
    print("ports", len(ports), flush=True)
    if workers > 1:
        with Pool(workers, initializer=init, initargs=(state,)) as pool:
            res = pool.map(run, ports, chunksize=8)
    else:
        init(state); res = [run(p) for p in ports]
    res = [r for r in res if r]
    out = bytearray(raw)
    out += struct.pack(">ii", 0x41505232, len(res))
    nb = npts = 0
    gj = {"type": "FeatureCollection", "features": []}
    for name, la, lo, branches in res:
        out += struct.pack(">iiB", int(round(la * 1e5)), int(round(lo * 1e5)), len(branches))
        for pts in branches:
            out += struct.pack(">B", len(pts) - 1)
            for a, b in pts[1:]:
                out += struct.pack(">ii", int(round(a * 1e5)), int(round(b * 1e5)))
            nb += 1; npts += len(pts) - 1
            gj["features"].append({"type": "Feature", "properties": {"port": name},
                                   "geometry": {"type": "LineString", "coordinates": [[b, a] for a, b in pts]}})
    open(out_path, "wb").write(out)
    import json
    json.dump(gj, open(os.path.join(work, "approaches.geojson"), "w"))
    print("wrote %s (%d bytes): %d ports, %d ways out, %d points; %.0f s" % (
        out_path, len(out), len(res), nb, npts, time.time() - t0))


if __name__ == "__main__":
    main()
