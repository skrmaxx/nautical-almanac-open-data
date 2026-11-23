"""
Refines the MARNET lane network (assets/lanes.bin) with AIS traffic density, offline, for Port to port's "follow the
traffic" option. Nothing here is needed on the phone: the app gets one byte per lane piece and the few lanes added.

Usage: python -I tools/ais/refine_lanes.py <lanes.bin> <rank_005.u8> <meta_005.json> <world_land_10m.bin>
         <sea_hazards.bin> <out lanes.bin> <work folder> [--rounds 2] [--window 400] [--workers 6]

What it does:
1. How busy each lane piece is (lane_traffic.py): the mean, at points 1 NM apart, of the busiest grid cell within
   about 6 NM. Stored as one byte per piece.
2. Lanes MARNET lacks: from lane points about 0.5 deg apart, the cheapest way on the traffic grid (sea cells with
   traffic at or above --min-rank, each step costing 1 + A (1 - rank)^G) to every lane point within --window NM is set
   against the cheapest way along the lanes (each piece costing its length (1 + A (1 - busy)^G)). Where the grid's way
   is much cheaper (less than --ratio of the lanes', and --gain NM or more saved), it is a lane ships use and MARNET
   does not have, where the straight line between the two crosses land (a clear one the router takes anyway): its
   cells, simplified to straight pieces that stay at sea, are added as lane pieces between the two
   lane points. The biggest savings are taken first, one per corridor; --rounds repeats this on the network with the
   lanes added.
   An added lane keeps out of the canals and straits (30 NM round any piece MARNET or build_lanes.py marks as one, the
   passages' own circles and the gates), so closing a passage still closes it; and of the polar seas past 72 deg.
3. Writes a copy of lanes.bin with, after its pieces (older readers ignore both):
     int 0x54524631 ("TRF1"); int pieces; per piece: byte busy * 255;
     int 0x41495331 ("AIS1"); int new points; per point: int lat * 1e5, int lon * 1e5 (numbered after the file's own);
       int new pieces; per piece: int a, int b (point numbers), byte busy * 255.
   The app uses the added pieces only with the traffic option on.
4. In the work folder: report.json (each lane added: ends, miles, the saving), added.geojson, and previews.

Needs numpy, scipy, scikit-image, Pillow.
"""
import json
import math
import os
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.graph import MCP_Geometric

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "routes"))
from lane_traffic import read_lanes, nm, Grid, along  # noqa: E402
from build_lanes import GATES, crosses  # noqa: E402

# Passage.inside circles from SeaRoutes.kt: (lat, lon, NM).
PASSAGE_CIRCLES = [(30.55, 32.33, 35.0), (54.3, 9.65, 12.0), (1.3, 103.6, 25.0), (36.0, -5.55, 20.0), (51.05, 1.55, 20.0),
                   (40.75, 28.2, 90.0), (55.85, 12.75, 20.0), (-53.2, -70.9, 40.0)]

A, G = 3.0, 2.0


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


# ---------------------------------------------------------------- land

def land_fraction(path, grid, fine=5):
    """Share of each 0.05 deg cell that is land (Natural Earth 10m tiles), by drawing the rings at 0.01 deg."""
    b = open(path, "rb").read()
    o = 0
    ver, ntiles = struct.unpack_from(">ii", b, o); o += 8
    assert ver == 2
    frac = np.zeros((grid.rows, grid.cols), np.float32)
    fc = grid.cell / fine
    for _ in range(ntiles):
        tx, ty, nr = struct.unpack_from(">hhi", b, o); o += 8
        x0, y0 = -180 + 10 * tx, -90 + 10 * ty
        rings = []
        for _ in range(nr):
            (n,) = struct.unpack_from(">i", b, o); o += 4
            v = np.frombuffer(b, ">u2", 2 * n, o).reshape(n, 2).astype(np.float64) * 10 / 65535; o += 4 * n
            rings.append(np.column_stack([x0 + v[:, 0], y0 + v[:, 1]]))
        # The window of grid cells over this tile, drawn at fine resolution.
        r0 = max(0, int(math.floor((grid.lat0 - (y0 + 10)) / grid.cell))); r1 = min(grid.rows, int(math.ceil((grid.lat0 - y0) / grid.cell)))
        c0 = max(0, int(math.floor((x0 - grid.lon0) / grid.cell))); c1 = min(grid.cols, int(math.ceil((x0 + 10 - grid.lon0) / grid.cell)))
        if r1 <= r0 or c1 <= c0:
            continue
        wlat0 = grid.lat0 - r0 * grid.cell; wlon0 = grid.lon0 + c0 * grid.cell
        img = Image.new("L", ((c1 - c0) * fine, (r1 - r0) * fine), 0)
        d = ImageDraw.Draw(img)
        for r in rings:
            if len(r) < 3:
                continue
            d.polygon([((x - wlon0) / fc, (wlat0 - y) / fc) for x, y in r], fill=1)
        a = np.asarray(img, np.float32).reshape(r1 - r0, fine, c1 - c0, fine).mean(axis=(1, 3))
        # Pixels outside the tile are left 0 by the clipped rings, so tiles add up.
        frac[r0:r1, c0:c1] += a
    return np.clip(frac, 0, 1)


def hazard_cells(path, grid):
    b = open(path, "rb").read()
    o = 0
    magic, n = struct.unpack_from(">ii", b, o); o += 8
    assert magic == 0x48415A32
    out = np.zeros((grid.rows, grid.cols), bool)
    for _ in range(n):
        (k,) = struct.unpack_from(">h", b, o); o += 2
        la, lo = struct.unpack_from(">ii", b, o); o += 8
        pts = [(la / 1e4, lo / 1e4)]
        for _ in range(k - 1):
            dla, dlo = struct.unpack_from(">hh", b, o); o += 4
            la += dla; lo += dlo
            pts.append((la / 1e4, lo / 1e4))
        for (a1, o1), (a2, o2) in zip(pts, pts[1:] or pts):
            m = max(2, int(max(abs(a2 - a1), abs(o2 - o1)) / 0.01) + 1)
            for t in np.linspace(0, 1, m):
                mark(out, grid, a1 + (a2 - a1) * t, o1 + (o2 - o1) * t)
    return out


def mark(arr, grid, lat, lon, val=True):
    r = int((grid.lat0 - lat) / grid.cell); c = int(((lon - grid.lon0) % 360) / grid.cell)
    if 0 <= r < grid.rows and 0 <= c < grid.cols:
        arr[r, c] = val


def disc(arr, grid, lat, lon, radius_nm):
    rr = int(math.ceil(radius_nm / 60 / grid.cell))
    r = int((grid.lat0 - lat) / grid.cell); c = int(((lon - grid.lon0) % 360) / grid.cell)
    k = max(0.05, math.cos(math.radians(lat)))
    cc = int(math.ceil(rr / k))
    for dr in range(-rr, rr + 1):
        for dc in range(-cc, cc + 1):
            if (dr ** 2 + (dc * k) ** 2) <= rr ** 2 and 0 <= r + dr < grid.rows:
                arr[r + dr, (c + dc) % grid.cols] = True


def passage_keep_out(grid, ll, e, ring=30.0):
    """Cells a lane added from traffic keeps out of: within ring NM of any piece marked as a canal or strait, the
    passages' own circles, and 10 NM round the gates, so closing a passage still closes it."""
    keep_out = np.zeros((grid.rows, grid.cols), bool)
    for a, b, p in e:
        if p:
            la, lo, _ = along(ll[a, 0], ll[a, 1], ll[b, 0], ll[b, 1], step=10.0)
            for x, y in zip(la, lo):
                disc(keep_out, grid, x, y, ring)
            disc(keep_out, grid, ll[a, 0], ll[a, 1], ring); disc(keep_out, grid, ll[b, 0], ll[b, 1], ring)
    for la, lo, r in PASSAGE_CIRCLES:
        disc(keep_out, grid, la, lo, r)
    for _, ((a1, o1), (a2, o2)) in GATES:
        for t in np.linspace(0, 1, 50):
            disc(keep_out, grid, a1 + (a2 - a1) * t, o1 + (o2 - o1) * t, 10.0)
    return keep_out


# ---------------------------------------------------------------- the search, per source (in worker processes)

W = {}


def init(state):
    W.update(state)


def search(u):
    grid_lat0, cell, rows, cols, lon0 = W["lat0"], W["cell"], W["rows"], W["cols"], W["lon0"]
    cost, ll, window = W["cost"], W["ll"], W["window"]
    la, lo = ll[u]
    ru = int((grid_lat0 - la) / cell); cu = int((lo - lon0) / cell)
    k = math.cos(math.radians(la))
    hr = int(window / 60 / cell); hc = int(hr / max(k, 0.25))
    r0, r1, c0, c1 = ru - hr, ru + hr + 1, cu - hc, cu + hc + 1
    if r0 < 0 or r1 > rows or c0 < 0 or c1 > cols:
        return None  # near the edge of the grid or the 180th meridian: left as it is
    sub = cost[r0:r1, c0:c1].copy()
    sub[ru - r0, cu - c0] = 1.0
    mcp = MCP_Geometric(sub, sampling=(1.0, k))
    cum, _ = mcp.find_costs([(ru - r0, cu - c0)])
    step_nm = cell * 60
    lane = W["lane"]
    cand = W["near"]
    best = None
    for v in cand:
        lv, ov = ll[v]
        d = nm(la, lo, lv, ov)
        if d < 40 or d > window * 0.95:
            continue
        rv = int((grid_lat0 - lv) / cell) - r0; cv = int((ov - lon0) / cell) - c0
        if not (1 <= rv < sub.shape[0] - 1 and 1 <= cv < sub.shape[1] - 1):
            continue
        blk = cum[rv - 1:rv + 2, cv - 1:cv + 2]
        g = float(blk.min()) * step_nm + step_nm
        if not np.isfinite(g):
            continue
        m = lane[v]
        if not np.isfinite(m):
            m = 1e9
        gain = m - g
        if g < W["ratio"] * m and gain >= W["gain"] and (best is None or gain > best[0]) and blocked(ru, cu, rv + r0, cv + c0):
            i = int(np.argmin(blk)); best = (gain, v, g, m, (rv - 1 + i // 3, cv - 1 + i % 3))
    if best is None:
        return None
    gain, v, g, m, end = best
    path = mcp.traceback(end)
    return dict(u=int(u), v=int(v), gain=gain, grid=g, lanes=m if m < 1e8 else None,
                cells=[(int(r + r0), int(c + c0)) for r, c in path])


# ---------------------------------------------------------------- main

def main():
    lanes_path, rank_path, meta_path, land_path, haz_path, out_path, work = sys.argv[1:8]
    rounds = arg("--rounds", 2, int); window = arg("--window", 400.0); workers = arg("--workers", 6, int)
    min_rank = arg("--min-rank", 0.4); ratio = arg("--ratio", 0.75); gain = arg("--gain", 30.0)
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    _, keys, ll, e = read_lanes(lanes_path)
    ll = np.array(ll)
    grid = Grid(rank_path, meta_path, 2)
    s_raw = grid.raw.astype(np.float32) / 255
    # 1. How busy each piece is.
    busy = np.zeros(len(e), np.float32); elen = np.zeros(len(e))
    for k, (a, b, p) in enumerate(e):
        la, lo, d = along(ll[a, 0], ll[a, 1], ll[b, 0], ll[b, 1])
        busy[k] = np.mean([grid.at(x, y) for x, y in zip(la, lo)]); elen[k] = d
    print("pieces scored, %.0f s" % (time.time() - t0))
    # 2. The grid: sea, traffic, away from passages and the polar seas.
    cache = os.path.join(work, "landfrac.npy")
    if os.path.exists(cache):
        lf = np.load(cache)
    else:
        lf = land_fraction(land_path, grid); np.save(cache, lf)
    haz = hazard_cells(haz_path, grid)
    print("land and hazards, %.0f s; land cells %.1f%%" % (time.time() - t0, (lf >= 0.5).mean() * 100))
    keep_out = passage_keep_out(grid, ll, e)
    lat_c = grid.lat0 - (np.arange(grid.rows) + 0.5) * grid.cell
    polar = np.abs(lat_c) > 72
    sea = (lf < 0.5) & ~haz
    ok = sea & (s_raw >= min_rank) & ~keep_out & ~polar[:, None]
    cost = np.where(ok, 1 + A * (1 - s_raw) ** G, -1).astype(np.float32)
    print("grid ready, %.0f s; passable %.1f%% of cells" % (time.time() - t0, ok.mean() * 100))

    # Sources: one lane point per 0.5 deg cell, away from passages.
    src = {}
    for i, (la, lo) in enumerate(ll):
        if abs(la) > 72:
            continue
        r = int((grid.lat0 - la) / grid.cell); c = int(((lo - grid.lon0) % 360) / grid.cell)
        if keep_out[r, c]:
            continue
        key = (math.floor(la * 2), math.floor(lo * 2))
        cen = ((key[0] + 0.5) / 2, (key[1] + 0.5) / 2)
        dd = (la - cen[0]) ** 2 + (lo - cen[1]) ** 2
        if key not in src or dd < src[key][0]:
            src[key] = (dd, i)
    sources = sorted(i for _, i in src.values())
    print("sources", len(sources))

    # Points binned by 1 deg, for "lane points within the window".
    bins = {}
    for i, (la, lo) in enumerate(ll):
        bins.setdefault((math.floor(la), math.floor(lo)), []).append(i)

    new_pts = []      # (lat, lon)
    new_edges = []    # (a, b, busy) with a, b in the joint numbering
    added = []
    n0 = len(ll)
    for rnd in range(rounds):
        # The lanes' graph with what is added so far, each piece costing length * (1 + A (1 - busy)^G).
        allll = np.vstack([ll, np.array(new_pts).reshape(-1, 2)]) if new_pts else ll
        ea = list(e["a"].astype(int)) + [x[0] for x in new_edges]
        eb = list(e["b"].astype(int)) + [x[1] for x in new_edges]
        w = list(elen * (1 + A * (1 - busy) ** G)) + [
            nm(*allll[x[0]], *allll[x[1]]) * (1 + A * (1 - x[2]) ** G) for x in new_edges]
        n = len(allll)
        gm = csr_matrix((w + w, (ea + eb, eb + ea)), shape=(n, n))
        state = dict(lat0=grid.lat0, cell=grid.cell, rows=grid.rows, cols=grid.cols, lon0=grid.lon0, cost=cost, ll=ll, sea=sea,
                     window=window, ratio=ratio, gain=gain)
        results = []
        # Lane costs from each source, in batches (dijkstra over many sources at once is fast in scipy); the grid's
        # search in worker processes, which get the grid once.
        lim = window * (1 + A) * 2
        pool = Pool(workers, initializer=init, initargs=(state,)) if workers > 1 else None
        if pool is None:
            init(state)
        for b0 in range(0, len(sources), 200):
            batch = sources[b0:b0 + 200]
            dist = dijkstra(gm, directed=False, indices=batch, limit=lim)[:, :n0]
            tasks = []
            for j, u in enumerate(batch):
                la, lo = ll[u]
                dl = window / 60; dlo = dl / max(0.2, math.cos(math.radians(la)))
                c = []
                for y in range(math.floor(la - dl), math.floor(la + dl) + 1):
                    for x in range(math.floor(lo - dlo), math.floor(lo + dlo) + 1):
                        c += bins.get((y, x), [])
                tasks.append((u, c, dist[j]))
            res = pool.map(search_with, tasks, chunksize=4) if pool else [search_with(t) for t in tasks]
            results += [r for r in res if r]
            print("round %d: %d/%d sources, %d candidates, %.0f s" % (rnd + 1, min(b0 + 200, len(sources)), len(sources),
                                                                     len(results), time.time() - t0), flush=True)
        if pool:
            pool.close(); pool.join()
        # The biggest savings first, one per corridor.
        results.sort(key=lambda r: -r["gain"])
        taken = np.zeros((grid.rows, grid.cols), bool)
        got = 0
        for r in results:
            cells = r["cells"]
            if sum(taken[c] for c in cells) > 0.3 * len(cells):
                continue
            pts = simplify(cells, sea & ~keep_out, grid)
            way = [tuple(ll[r["u"]])] + [p for p in pts[1:-1]] + [tuple(ll[r["v"]])]
            if any(crosses(p, q, g1, g2) for p, q in zip(way, way[1:]) for _, (g1, g2) in GATES):
                continue
            for (y, x) in cells:
                taken[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = True
            ids = [r["u"]]
            for p in way[1:-1]:
                new_pts.append(p); ids.append(n0 + len(new_pts) - 1)
            ids.append(r["v"])
            sb = float(np.mean([s_raw[c] for c in cells]))
            for a_, b_ in zip(ids, ids[1:]):
                new_edges.append((a_, b_, sb))
            miles = sum(nm(*p, *q) for p, q in zip(way, way[1:]))
            added.append(dict(round=rnd + 1, frm=[round(x, 4) for x in ll[r["u"]]], to=[round(x, 4) for x in ll[r["v"]]],
                              miles=round(miles, 1), grid_cost=round(r["grid"], 1),
                              lanes_cost=None if r["lanes"] is None else round(r["lanes"], 1), gain=round(r["gain"], 1),
                              busy=round(sb, 3), way=[[round(a, 4), round(b, 4)] for a, b in way]))
            got += 1
        print("round %d: %d lanes added" % (rnd + 1, got))
        if got == 0:
            break

    # 3. The file.
    src_bytes = open(lanes_path, "rb").read()
    out = bytearray(src_bytes)
    out += struct.pack(">ii", 0x54524631, len(e)) + np.clip(np.round(busy * 255), 0, 255).astype(np.uint8).tobytes()
    out += struct.pack(">ii", 0x41495331, len(new_pts))
    for la, lo in new_pts:
        out += struct.pack(">ii", int(round(la * 1e5)), int(round(lo * 1e5)))
    out += struct.pack(">i", len(new_edges))
    for a_, b_, sb in new_edges:
        out += struct.pack(">iiB", a_, b_, int(round(sb * 255)))
    open(out_path, "wb").write(out)
    report = dict(params=dict(A=A, G=G, min_rank=min_rank, ratio=ratio, gain=gain, window=window, rounds=rounds),
                  pieces=len(e), new_points=len(new_pts), new_pieces=len(new_edges), added=added)
    json.dump(report, open(os.path.join(work, "report.json"), "w"), indent=1)
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {k: v for k, v in a.items() if k != "way"},
         "geometry": {"type": "LineString", "coordinates": [[b, a_] for a_, b in a["way"]]}} for a in added]}
    json.dump(gj, open(os.path.join(work, "added.geojson"), "w"))
    print("wrote %s (%d bytes): %d lanes added, %d points, %d pieces; %.0f s" % (
        out_path, len(out), len(added), len(new_pts), len(new_edges), time.time() - t0))


def blocked(r1, c1, r2, c2):
    """Whether the straight line between two cells crosses land: only then is a lane needed, as the router already
    takes a straight leg wherever it is clear."""
    sea = W["sea"]
    m = max(abs(r2 - r1), abs(c2 - c1)) * 2 + 2
    # Land within 15 NM of an end (a lane point in a harbour, behind a breakwater) does not count.
    end = 15.0 / (W["cell"] * 60) / max(1.0, math.hypot(r2 - r1, c2 - c1))
    for t in np.linspace(0, 1, m):
        if end < t < 1 - end and not sea[int(round(r1 + (r2 - r1) * t)), int(round(c1 + (c2 - c1) * t))]:
            return True
    return False


def search_with(args):
    u, W["near"], W["lane"] = args
    return search(u)


def simplify(cells, sea, grid, tol=1.0):
    """Cell path to points (lat, lon): Douglas-Peucker in cells, a chord kept only where it stays on sea cells."""
    pts = np.array(cells, float)

    def clear(i, j):
        a, b = pts[i], pts[j]
        m = int(max(abs(b - a))) * 2 + 2
        for t in np.linspace(0, 1, m):
            y, x = a + (b - a) * t
            if not sea[int(round(y)), int(round(x)) % grid.cols]:
                return False
        return True

    def rec(i, j, out):
        if j <= i + 1:
            return
        a, b = pts[i], pts[j]
        d = b - a
        nrm = math.hypot(*d) or 1
        dev = np.abs((pts[i + 1:j, 0] - a[0]) * d[1] - (pts[i + 1:j, 1] - a[1]) * d[0]) / nrm
        k = int(np.argmax(dev)) + i + 1
        if dev.max() > tol or not clear(i, j):
            rec(i, k, out); out.append(k); rec(k, j, out)

    keep = [0]
    rec(0, len(pts) - 1, keep)
    keep.append(len(pts) - 1)
    keep = sorted(set(keep))
    return [(grid.lat0 - (pts[k, 0] + 0.5) * grid.cell, grid.lon0 + (pts[k, 1] + 0.5) * grid.cell) for k in keep]


if __name__ == "__main__":
    main()
