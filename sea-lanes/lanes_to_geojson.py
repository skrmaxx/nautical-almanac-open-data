"""
Writes lanes.bin (the app's packed sea-lane network) out as GeoJSON: one LineString per straight piece, with the
passage it belongs to (suez, dover, hormuz, ...) where it has one. This is the network in an open, editable form.

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
    json.dump({"type": "FeatureCollection", "features": features}, open(out, "w"), separators=(",", ":"))
    print("%d nodes, %d pieces, %d passages" % (len(nodes), len(features), len(passages)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
