"""
Packs Eurostat's MARNET sea-lane network (EUPL-1.2, https://github.com/eurostat/searoute) into assets/lanes.bin
for the app's Port to port tool.

Usage: python -I tools/routes/build_lanes.py <marnet_plus_10km.gpkg> <out lanes.bin>

What it changes in the network (the EUPL asks that changes be stated; they are all here):
- coordinates rounded to 1e-5 degree, the line at 180 degrees joined, longitudes past 180 brought back to -180..180;
- each line split into its straight pieces, repeated pieces dropped, the few points not joined to the rest dropped;
- the passages MARNET tags (pass = suez, panama, ...) kept, and more tagged where a piece crosses a gate below
  (straits MARNET does not tag).

File (big-endian, as java.nio.ByteBuffer reads it):
  int 0x4C414E31 ("LAN1"); int passages; per passage: short length, UTF-8 key;
  int nodes; per node: int lat * 1e5, int lon * 1e5;
  int edges; per edge: char a, char b (node numbers), byte passage (0 none, else 1 + its number in the list).
"""
import math
import sqlite3
import struct
import sys

# MARNET's own tags, under our keys.
MARNET_TAGS = {
    "suez": "suez", "panama": "panama", "kiel": "kiel", "corinth": "corinth", "malacca": "malacca",
    "gibraltar": "gibraltar", "dover": "dover", "babelmandeb": "babelmandeb", "magellan": "magellan",
    "bering": "bering", "northwest": "northwest", "northeast": "northeast",
}

# Gates for the straits MARNET leaves untagged: (key, [(lat, lon), (lat, lon)]), both ends on land, so every lane
# through the strait crosses the gate.
GATES = [
    ("hormuz", [(26.20, 56.30), (27.15, 56.30)]),
    ("sunda", [(-6.20, 106.00), (-5.70, 105.60)]),
    ("lombok", [(-8.45, 115.45), (-8.45, 116.15)]),
    ("torres", [(-11.00, 142.60), (-9.00, 142.60)]),
    ("turkish", [(41.10, 28.95), (41.10, 29.15)]),      # Bosphorus
    ("turkish", [(40.35, 26.45), (40.22, 26.60)]),      # Dardanelles, between Eceabat and Lapseki
    ("sound", [(56.00, 12.45), (56.10, 12.85)]),        # Oresund at Helsingor
    ("belts", [(55.33, 10.70), (55.36, 11.25)]),        # Great Belt at the bridge
    ("belts", [(55.50, 9.65), (55.52, 9.85)]),          # Little Belt at Middelfart
    ("taiwan", [(24.10, 120.75), (25.45, 119.00)]),
    ("messina", [(38.20, 15.45), (38.22, 15.70)]),
    ("bonifacio", [(41.45, 9.20), (41.18, 9.20)]),
]

PASSAGES = list(dict.fromkeys(list(MARNET_TAGS.values()) + [g[0] for g in GATES]))


def read_lines(path):
    c = sqlite3.connect(path)
    for g, p in c.execute("select geometry, pass from type"):
        flags = g[3]
        env = (flags >> 1) & 7
        w = g[8 + [0, 32, 48, 48, 64][env]:]
        bo = "<" if w[0] == 1 else ">"
        t = struct.unpack(bo + "I", w[1:5])[0]
        if t != 2:
            raise ValueError("not a line: %d" % t)
        n = struct.unpack(bo + "I", w[5:9])[0]
        yield [struct.unpack(bo + "dd", w[9 + 16 * i:25 + 16 * i]) for i in range(n)], p


def key(lon, lat):
    lon = round(lon, 5)
    lat = round(lat, 5)
    if lon >= 180:
        lon = round(lon - 360, 5)
    if lon < -180:
        lon = round(lon + 360, 5)
    if lon == -180.0:
        lon = 180.0
    return lat, lon


def crosses(a, b, c, d):
    """Whether segment a-b crosses c-d, as (lat, lon) in a plane, longitudes taken near a's."""
    def unwrap(p, near):
        lon = p[1]
        while lon - near > 180: lon -= 360
        while lon - near < -180: lon += 360
        return p[0], lon
    b = unwrap(b, a[1]); c = unwrap(c, a[1]); d = unwrap(d, a[1])

    def orient(p, q, r):
        return (q[1] - p[1]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[1] - p[1])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def main(src, out):
    nodes = {}
    edges = {}
    for pts, tag in read_lines(src):
        ks = [key(x, y) for x, y in pts]
        for a, b in zip(ks, ks[1:]):
            if a == b:
                continue
            for k in (a, b):
                nodes.setdefault(k, len(nodes))
            ia, ib = nodes[a], nodes[b]
            e = (min(ia, ib), max(ia, ib))
            p = MARNET_TAGS.get(tag) if tag else None
            if p is None and tag:
                raise ValueError("unknown MARNET tag " + tag)
            if e not in edges or (p and not edges[e]):
                edges[e] = p
    # Keep the largest connected part only.
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    from collections import Counter
    big = Counter(find(i) for i in range(len(nodes))).most_common(1)[0][0]
    pos = [None] * len(nodes)
    for k, i in nodes.items():
        pos[i] = k
    keep = [i for i in range(len(nodes)) if find(i) == big]
    renum = {old: new for new, old in enumerate(keep)}
    print("nodes %d of %d kept" % (len(keep), len(nodes)))
    # Gates.
    cut = Counter()
    for (a, b), p in list(edges.items()):
        if p or a not in renum:
            continue
        for n, (gk, (g1, g2)) in enumerate(GATES):
            if crosses(pos[a], pos[b], g1, g2):
                edges[(a, b)] = gk
                cut[n] += 1
                break
    for n, (gk, _) in enumerate(GATES):
        print("gate %-10s %d cuts %d pieces" % (gk, n, cut[n]))
        if cut[n] == 0:
            raise SystemExit("gate %s (%d) cuts nothing" % (gk, n))
    kept_edges = [(renum[a], renum[b], p) for (a, b), p in sorted(edges.items()) if a in renum]
    if len(keep) > 65535:
        raise SystemExit("too many nodes for 16-bit numbers")
    buf = bytearray()
    buf += struct.pack(">ii", 0x4C414E31, len(PASSAGES))
    for p in PASSAGES:
        b = p.encode("utf-8")
        buf += struct.pack(">h", len(b)) + b
    buf += struct.pack(">i", len(keep))
    for old in keep:
        lat, lon = pos[old]
        buf += struct.pack(">ii", int(round(lat * 1e5)), int(round(lon * 1e5)))
    buf += struct.pack(">i", len(kept_edges))
    for a, b, p in kept_edges:
        buf += struct.pack(">HHb", a, b, 0 if p is None else 1 + PASSAGES.index(p))
    with open(out, "wb") as f:
        f.write(buf)
    tags = Counter(p for _, _, p in kept_edges if p)
    print("edges %d, tagged %s" % (len(kept_edges), dict(tags)))
    print("wrote %s, %d bytes" % (out, len(buf)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
