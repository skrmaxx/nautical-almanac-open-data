"""
How busy each MARNET lane piece in assets/lanes.bin is, from the traffic rank grid (build_density.py's rank_005.u8).

Usage: python -I tools/ais/lane_traffic.py <lanes.bin> <rank_005.u8> <meta_005.json> <out.npy> [--reach 2] [--write <lanes out.bin>]

For each piece: points every 1 NM along it; at each, the busiest cell within --reach cells (0.05 deg each, so 2 is
about 6 NM: MARNET's lines are drawn by hand and lie off the traffic by a few miles); the piece's figure is the mean
of those, 0..1. Writes one float per piece, in the file's order, and prints how the lanes' length spreads over it.
With --write, also a copy of lanes.bin with the figures added at its end, which older readers ignore:
  int 0x54524631 ("TRF1"); int pieces; per piece: byte figure * 255 (unsigned).
Needs numpy.
"""
import json
import math
import struct
import sys

import numpy as np


def read_lanes(path):
    b = open(path, "rb").read()
    o = 0
    magic, npass = struct.unpack_from(">ii", b, o); o += 8
    keys = []
    for _ in range(npass):
        (n,) = struct.unpack_from(">h", b, o); o += 2
        keys.append(b[o:o + n].decode()); o += n
    (n,) = struct.unpack_from(">i", b, o); o += 4
    ll = np.frombuffer(b, ">i4", 2 * n, o).reshape(n, 2) / 1e5; o += 8 * n
    (m,) = struct.unpack_from(">i", b, o); o += 4
    e = np.frombuffer(b, np.dtype([("a", ">u2"), ("b", ">u2"), ("p", "i1")]), m, o)
    return magic, keys, ll, e


def nm(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * math.asin(min(1, math.sqrt(d))) * 3437.74677


def max_filter(a, r):
    out = a.copy()
    p = np.pad(a, r)
    rows, cols = a.shape
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            np.maximum(out, p[r + dr:r + dr + rows, r + dc:r + dc + cols], out=out)
    return out


class Grid:
    def __init__(self, rank_path, meta_path, reach):
        m = json.load(open(meta_path))
        self.rows, self.cols, self.cell = m["rows"], m["cols"], m["cell_deg"]
        self.lat0, self.lon0 = m["lat_top"], m["lon_left"]
        r = np.fromfile(rank_path, np.uint8).reshape(self.rows, self.cols)
        self.raw = r
        self.near = max_filter(r, reach) if reach > 0 else r

    def at(self, lat, lon, layer=None):
        a = self.near if layer is None else layer
        lon = (lon - self.lon0) % 360 + self.lon0
        r = int((self.lat0 - lat) / self.cell); c = int((lon - self.lon0) / self.cell)
        if not (0 <= r < self.rows): return 0
        return a[r, min(c, self.cols - 1)] / 255.0


def along(lat1, lon1, lat2, lon2, step=1.0):
    """Points every step NM on the straight piece (in latitude and longitude, as the pieces are short)."""
    if lon2 - lon1 > 180: lon2 -= 360
    if lon2 - lon1 < -180: lon2 += 360
    d = nm(lat1, lon1, lat2, lon2)
    k = max(1, int(math.ceil(d / step)))
    t = (np.arange(k) + 0.5) / k
    return lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t, d


def main():
    lanes, rank, meta, out = sys.argv[1:5]
    reach = int(sys.argv[sys.argv.index("--reach") + 1]) if "--reach" in sys.argv else 2
    _, keys, ll, e = read_lanes(lanes)
    g = Grid(rank, meta, reach)
    score = np.zeros(len(e), np.float32); length = np.zeros(len(e))
    for k, (a, b, p) in enumerate(e):
        la, lo, d = along(ll[a, 0], ll[a, 1], ll[b, 0], ll[b, 1])
        score[k] = np.mean([g.at(x, y) for x, y in zip(la, lo)])
        length[k] = d
    np.save(out, score)
    if "--write" in sys.argv:
        dst = sys.argv[sys.argv.index("--write") + 1]
        src = open(lanes, "rb").read()
        q = np.clip(np.round(score * 255), 0, 255).astype(np.uint8)
        open(dst, "wb").write(src + struct.pack(">ii", 0x54524631, len(e)) + q.tobytes())
        print("wrote", dst)
    tot = length.sum()
    print("pieces %d, %.0f NM" % (len(e), tot))
    for lo_, hi in ((0, 0.001), (0.001, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = (score >= lo_) & (score < hi)
        print("score %.2f-%.2f: %5.1f%% of length, %d pieces" % (lo_, hi, length[sel].sum() / tot * 100, sel.sum()))


if __name__ == "__main__":
    main()
