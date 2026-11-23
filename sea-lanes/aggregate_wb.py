"""
Sums a World Bank / IMF Global Shipping Traffic Density layer (a zipped BigTIFF, 0.005 deg, int32 hourly AIS position
counts, Jan 2015 - Feb 2021; CC BY 4.0) to a coarser grid, straight from the zip (the 9.8 GB TIFF is never unpacked).

Usage: python -I tools/ais/aggregate_wb.py <layer.zip> <factor> <out.npy>
  factor 10 -> 0.05 deg cells, 20 -> 0.1 deg. Output: float64 array [rows, cols], row 0 at the TIFF's top (85 N),
  column 0 at its left (-180.0128). The TIFF's tiles are not stored in file order, so they are read by the
  TileOffsets table (tag 324), in order of offset.

Source: https://datacatalog.worldbank.org/search/dataset/0037580 (Cerdeiro, Komaromi, Liu and Saeed, 2020).
Needs numpy.
"""
import struct
import sys
import zipfile

import numpy as np

NODATA = 2147483647


def read_ifd(f, size):
    """The BigTIFF's first IFD as {tag: (type, count, value-or-offset bytes)}; the IFD is at the end of the file."""
    f.seek(0)
    head = f.read(16)
    assert head[:4] == b"II+\x00", "not a little-endian BigTIFF"
    ifd = struct.unpack("<Q", head[8:16])[0]
    base = max(0, ifd - 8_000_000)  # the offset arrays are written just before the IFD
    f.seek(base)
    tail = f.read()
    n = struct.unpack("<Q", tail[ifd - base:ifd - base + 8])[0]
    tags = {}
    for i in range(n):
        p = ifd - base + 8 + 20 * i
        tag, typ, cnt = struct.unpack("<HHQ", tail[p:p + 12])
        tags[tag] = (typ, cnt, tail[p + 12:p + 20])
    return tags, tail, base


def main():
    zp, factor, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    z = zipfile.ZipFile(zp)
    name = [i for i in z.infolist() if i.filename.lower().endswith(".tif")][0]
    f = z.open(name)
    tags, tail, base = read_ifd(f, name.file_size)
    W = struct.unpack("<Q", tags[256][2])[0] if tags[256][0] == 16 else struct.unpack("<I", tags[256][2][:4])[0]
    H = struct.unpack("<Q", tags[257][2])[0] if tags[257][0] == 16 else struct.unpack("<I", tags[257][2][:4])[0]
    T = struct.unpack("<H", tags[322][2][:2])[0]
    assert struct.unpack("<H", tags[258][2][:2])[0] == 32 and struct.unpack("<H", tags[339][2][:2])[0] == 2, "not int32"
    typ, cnt, val = tags[324]
    off = struct.unpack("<Q", val)[0]
    offs = np.frombuffer(tail[off - base:off - base + 8 * cnt], "<u8").astype(np.int64)
    tx = (W + T - 1) // T
    cw, ch = (W + factor - 1) // factor, (H + factor - 1) // factor
    sums = np.zeros(ch * cw, np.float64)
    f = z.open(name)
    pos = 0
    yy, xx = np.mgrid[0:T, 0:T]
    for k, i in enumerate(np.argsort(offs, kind="stable")):
        o = int(offs[i])
        if o == 0:
            continue
        while pos < o:
            pos += len(f.read(min(o - pos, 1 << 24)))
        buf = f.read(T * T * 4)
        pos += len(buf)
        t = np.frombuffer(buf, "<i4").reshape(T, T).astype(np.int64)
        ty, txi = divmod(int(i), tx)
        gy, gx = ty * T + yy, txi * T + xx
        ok = (gy < H) & (gx < W) & (t != NODATA) & (t > 0)
        if ok.any():
            idx = (gy[ok] // factor) * cw + gx[ok] // factor
            np.add.at(sums, idx, t[ok])
        if k % 20000 == 0:
            print("tile", k, "/", len(offs), flush=True)
    sums = sums.reshape(ch, cw)
    np.save(out, sums)
    print("grid", cw, "x", ch, "cells > 0:", int((sums > 0).sum()), "max", sums.max())


if __name__ == "__main__":
    main()
