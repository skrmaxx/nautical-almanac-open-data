"""
Writes lanes.bin (the app's packed sea-lane network) out as GeoJSON: one LineString per straight piece, with the
passage it belongs to (suez, dover, hormuz, ...) where it has one. This is the network in an open, editable form.

Where the file has the layers made from AIS ship traffic (refine_lanes.py, port_approaches.py), they are written too:
each MARNET piece gets `busy` (0..1, how busy the water along it is); the lanes added where ships go and MARNET has
none are pieces with `kind` "added"; the ports' ways out to sea are LineStrings with `kind` "port_way".

Usage: python lanes_to_geojson.py lanes.bin lanes.geojson
"""
import json
import struct
import sys


def main(src, out):
    b = open(src, "rb").read()
    p = 0

    def take(fmt):
        nonlocal p
        v = struct.unpack_from(fmt, b, p)
        p += struct.calcsize(fmt)
        return v

    magic, = take(">i")
    if magic != 0x4C414E31:
        raise ValueError("not a LAN1 file")
    passages = []
    for _ in range(take(">i")[0]):
        n, = take(">h")
        passages.append(b[p:p + n].decode("utf-8"))
        p += n
    nodes = [(lat / 1e5, lon / 1e5) for lat, lon in (take(">ii") for _ in range(take(">i")[0]))]
    features = []
    for _ in range(take(">i")[0]):
        a, c, k = take(">HHb")
        (la1, lo1), (la2, lo2) = nodes[a], nodes[c]
        features.append({"type": "Feature", "properties": {"passage": passages[k - 1] if k else None},
                         "geometry": {"type": "LineString", "coordinates": [[lo1, la1], [lo2, la2]]}})
    base = len(features)
    added = ways = 0
    # The AIS layers, each a tagged section after the network.
    while p + 8 <= len(b):
        tag, n = take(">ii")
        if tag == 0x54524631:      # TRF1: how busy each MARNET piece is
            for i in range(n):
                features[i]["properties"]["busy"] = round(b[p + i] / 255, 3)
            p += n
        elif tag == 0x41495331:    # AIS1: lanes added from traffic
            first = len(nodes)
            nodes += [(lat / 1e5, lon / 1e5) for lat, lon in (take(">ii") for _ in range(n))]
            for _ in range(take(">i")[0]):
                a, c, busy = take(">iiB")
                (la1, lo1), (la2, lo2) = nodes[a], nodes[c]
                features.append({"type": "Feature", "properties": {"kind": "added", "busy": round(busy / 255, 3)},
                                 "geometry": {"type": "LineString", "coordinates": [[lo1, la1], [lo2, la2]]}})
                added += 1
        elif tag == 0x41505232:    # APR2: ports' ways out to sea
            for _ in range(n):
                la, lo, k = take(">iiB")
                for _ in range(k):
                    m, = take(">B")
                    pts = [(la / 1e5, lo / 1e5)] + [(x / 1e5, y / 1e5) for x, y in (take(">ii") for _ in range(m))]
                    features.append({"type": "Feature", "properties": {"kind": "port_way"},
                                     "geometry": {"type": "LineString", "coordinates": [[o, a_] for a_, o in pts]}})
                    ways += 1
        else:
            raise ValueError("unknown section %08x" % tag)
    json.dump({"type": "FeatureCollection", "features": features}, open(out, "w"), separators=(",", ":"))
    print("%d nodes, %d pieces, %d passages; %d added pieces, %d port ways" % (len(nodes), base, len(passages), added, ways))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
