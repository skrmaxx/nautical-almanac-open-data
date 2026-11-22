# Sea-lane network

This is the sea-lane network the Nautical Almanac app's Port to port tool routes on. It is published here as the
European Union Public Licence v1.2 asks for a work derived from an EUPL-licensed one.

## Origin

The network is MARNET (`marnet_plus_10km`), the maritime network of Eurostat's SeaRoute:
https://github.com/eurostat/searoute. It is © European Union and licensed under EUPL-1.2.

MARNET is based on the Oak Ridge National Laboratory Global Shipping Lane Network (2000). Eurostat enriched it with
lines around European coasts, based on AIS data.

## What was changed

- Coordinates rounded to 0.00001 degree.
- The network joined across the 180th meridian, with longitudes brought back to −180…180.
- Each line split into its straight pieces. Repeated pieces were dropped, and so were the few points not joined to
  the rest.
- MARNET's own passage tags kept: Suez, Panama, Kiel, Corinth, Malacca, Gibraltar, Dover, Bab-el-Mandeb, Magellan,
  Bering, Northwest, Northeast.
- More straits tagged where a piece crosses a line drawn across them: Hormuz, Sunda, Lombok, Torres, the Bosphorus
  and Dardanelles, the Sound, the Great and Little Belts, Taiwan, Messina and Bonifacio.
- Packed into a binary file.

Nothing else was changed. `build_lanes.py` makes every change listed here.

## Files

| File | What it is |
|---|---|
| `lanes.bin` | The network exactly as the app carries it. The file format is described at the top of `build_lanes.py`. |
| `lanes.geojson` | The same network as GeoJSON: one LineString per straight piece, with its `passage` where it has one. |
| `build_lanes.py` | Makes `lanes.bin` from MARNET's `marnet_plus_10km.gpkg`. |
| `lanes_to_geojson.py` | Writes `lanes.bin` out as `lanes.geojson`. |
| `LICENCE` | The EUPL-1.2 text. |

## Rebuild

```
python build_lanes.py marnet_plus_10km.gpkg lanes.bin
python lanes_to_geojson.py lanes.bin lanes.geojson
```

Python 3 is enough; no other packages are needed.

## Licence

EUPL-1.2 (see `LICENCE`), as for MARNET itself.

**Not for navigation:** this is a planning network of common shipping lanes. Plan passages on official charts.
