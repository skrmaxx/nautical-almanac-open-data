"""
Builds the traffic-density dataset for Port to port from a summed World Bank / IMF layer (aggregate_wb.py's output).

Usage: python -I tools/ais/build_density.py <sum_005.npy> <out folder> [--lon0 -180.0128112750 --lat0 85.0001479370]

Writes, for 0.05 deg and 0.1 deg cells:
- density_<res>.u8: one byte per cell, 0 = no position seen, 1..255 = log1p(count) scaled so 255 is the 99.9th
  percentile of the cells with traffic (ports and anchorages, all far above it, are clipped there);
- rank_<res>.u8: one byte per cell, 0 = none, 1..255 = how busy the cell is among the cells with traffic: the mean of
  its percentile worldwide and its percentile within its 10 x 10 deg block, so a lane in a region with thin satellite
  reception still ranks as a lane there. This is the s of the routing cost step * (1 + a * (1 - s) ** g);
- the same two deflated (.u8.z), and the sizes deflated as fixed 1 x 1 deg tiles, for the app's asset budget;
- meta.json: grid origin, cell size, rows, columns, the scaling, source and licence;
- previews: world and close-ups of known passages (PNG, log density).
Needs numpy.
"""
import json
import os
import struct
import sys
import zlib

import numpy as np

SOURCE = ("World Bank / IMF Global Shipping Traffic Density, Commercial layer (Cerdeiro, Komaromi, Liu and Saeed "
          "2020), AIS Jan 2015 - Feb 2021, https://datacatalog.worldbank.org/search/dataset/0037580, CC BY 4.0")

PASSAGES = {  # name: (lat_n, lat_s, lon_w, lon_e)
    "dover": (52.5, 49.5, -2.0, 4.0), "gibraltar": (37.5, 34.5, -8.0, -3.0), "malacca_singapore": (6.0, 0.5, 98.0, 105.5),
    "suez_red_sea": (32.0, 11.0, 32.0, 45.0), "panama": (10.5, 7.0, -81.5, -78.5), "cape_good_hope": (-30.0, -38.0, 14.0, 24.0),
    "north_atlantic": (55.0, 35.0, -60.0, -5.0), "japan_korea": (40.0, 30.0, 125.0, 142.0),
}


def png(gray, path):
    raw = b"".join(b"\x00" + gray[y].tobytes() for y in range(gray.shape[0]))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", gray.shape[1], gray.shape[0], 8, 0, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def percentile_rank(v):
    """Each value's percentile (0..1] among the values given (ties share the higher rank)."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, np.float64)
    ranks[order] = np.arange(1, v.size + 1)
    # Equal values: the highest rank of their group.
    sv = v[order]
    last = np.r_[np.nonzero(np.diff(sv))[0], sv.size - 1]
    grp = np.searchsorted(last, np.arange(sv.size))
    ranks[order] = last[grp] + 1
    return ranks / v.size


def products(s, cell, lon0, lat0, out, tag):
    rows, cols = s.shape
    has = s > 0
    lv = np.log1p(s)
    top = np.percentile(lv[has], 99.9)
    dens = np.zeros(s.shape, np.uint8)
    dens[has] = np.clip(np.round(lv[has] / top * 255), 1, 255)
    # Rank, from counts smoothed over 3 x 3 cells (open-ocean traffic is single scattered positions; summed with its
    # neighbours a lane is continuous): the mean of the worldwide percentile and the percentile after a smooth regional
    # shift, so a lane where reception is thin (a satellite gap) still ranks as a lane there. The regional reference is
    # the 90th percentile in 5 x 5 deg blocks, interpolated between block centres (no seams at block edges).
    p = np.pad(s, 1)
    sm = sum(p[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    hs = sm > 0
    ls = np.log1p(sm)
    block = int(round(5 / cell))
    nbr, nbc = (rows + block - 1) // block, (cols + block - 1) // block
    gref = np.percentile(ls[hs], 90)
    ref = np.full((nbr, nbc), gref)
    for i in range(nbr):
        for j in range(nbc):
            b = ls[i * block:(i + 1) * block, j * block:(j + 1) * block]
            m = hs[i * block:(i + 1) * block, j * block:(j + 1) * block]
            if m.sum() >= 100:
                ref[i, j] = np.percentile(b[m], 90)
    # Bilinear between block centres: along the rows, then the columns.
    rc = (np.arange(nbr) + 0.5) * block; cc = (np.arange(nbc) + 0.5) * block
    mid = np.stack([np.interp(np.arange(rows), rc, ref[:, j]) for j in range(nbc)], axis=1)
    field = np.stack([np.interp(np.arange(cols), cc, mid[r]) for r in range(rows)], axis=0)
    shifted = ls - field + gref
    rank = np.zeros(s.shape, np.float64)
    rank[hs] = (percentile_rank(ls[hs]) + percentile_rank(shifted[hs])) / 2
    blend = np.zeros(s.shape, np.uint8)
    blend[hs] = np.clip(np.round(rank[hs] * 255), 1, 255)
    # The same in 16 levels (0 = none, 1..15), the size an asset would have.
    blend16 = np.zeros(s.shape, np.uint8)
    blend16[hs] = np.clip(np.ceil(rank[hs] * 15), 1, 15)
    sizes = {}
    for name, a in (("density", dens), ("rank", blend), ("rank16", blend16)):
        a.tofile(os.path.join(out, "%s_%s.u8" % (name, tag)))
        z = zlib.compress(a.tobytes(), 9)
        open(os.path.join(out, "%s_%s.u8.z" % (name, tag)), "wb").write(z)
        tiled = 0; t = int(round(1 / cell))
        for r0 in range(0, rows, t):
            for c0 in range(0, cols, t):
                b = a[r0:r0 + t, c0:c0 + t]
                if b.any(): tiled += len(zlib.compress(b.tobytes(), 9)) + 8
        sizes[name] = {"raw": a.size, "deflated": len(z), "tiles_1deg_deflated": tiled}
    meta = {"source": SOURCE, "cell_deg": cell, "rows": rows, "cols": cols, "lat_top": lat0, "lon_left": lon0,
            "row_order": "north to south", "density": "log1p(count) / log1p(p99.9) * 255, 0 = none",
            "rank": "counts summed over 3x3 cells; mean of the worldwide percentile and the percentile after a smooth "
                    "regional shift (5-deg-block p90, interpolated), * 255, 0 = none; rank16 the same in 1..15",
            "cells_with_traffic": int(has.sum()), "sizes": sizes}
    json.dump(meta, open(os.path.join(out, "meta_%s.json" % tag), "w"), indent=1)
    print(tag, json.dumps(sizes))
    return blend


def main():
    src, out = sys.argv[1], sys.argv[2]
    lon0 = float(sys.argv[sys.argv.index("--lon0") + 1]) if "--lon0" in sys.argv else -180.0128112750
    lat0 = float(sys.argv[sys.argv.index("--lat0") + 1]) if "--lat0" in sys.argv else 85.0001479370
    os.makedirs(out, exist_ok=True)
    s05 = np.load(src)
    rows, cols = s05.shape
    d05 = products(s05, 0.05, lon0, lat0, out, "005")  # the rank layer, for the previews
    h, w = rows // 2 * 2, cols // 2 * 2
    s10 = s05[:h, :w].reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    products(s10, 0.1, lon0, lat0, out, "010")
    prev = os.path.join(out, "preview"); os.makedirs(prev, exist_ok=True)
    png(d05[::4, ::4], os.path.join(prev, "world_02deg.png"))
    for name, (n, s_, w_, e) in PASSAGES.items():
        r0, r1 = int((lat0 - n) / 0.05), int((lat0 - s_) / 0.05)
        c0, c1 = int((w_ - lon0) / 0.05), int((e - lon0) / 0.05)
        crop = d05[r0:r1, c0:c1]
        k = max(1, 700 // crop.shape[1])  # enlarged, each cell a square of k pixels
        png(np.repeat(np.repeat(crop, k, axis=0), k, axis=1), os.path.join(prev, name + ".png"))
    print("previews in", prev)


if __name__ == "__main__":
    main()
