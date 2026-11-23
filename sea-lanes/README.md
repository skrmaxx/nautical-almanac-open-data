# Sea-lane network

This is the sea-lane network the Nautical Almanac app's Port to port tool routes on. It is published here as the
European Union Public Licence v1.2 asks for a work derived from an EUPL-licensed one.

## Origin

The network is MARNET (`marnet_plus_10km`), the maritime network of Eurostat's SeaRoute:
https://github.com/eurostat/searoute. It is © European Union and licensed under EUPL-1.2.

MARNET is based on the Oak Ridge National Laboratory Global Shipping Lane Network (2000). Eurostat enriched it with
lines around European coasts, based on AIS data.

The layers added from ship traffic (below) are made from the World Bank / IMF *Global Shipping Traffic Density*,
Commercial layer (Cerdeiro, Komaromi, Liu and Saeed, 2020; AIS January 2015 to February 2021),
https://datacatalog.worldbank.org/search/dataset/0037580, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). It was processed: summed to 0.05° cells and turned into ranks of how
busy each cell is.

## What was changed

In MARNET itself:

- Coordinates rounded to 0.00001 degree.
- The network joined across the 180th meridian, with longitudes brought back to −180…180.
- Each line split into its straight pieces. Repeated pieces were dropped, and so were the few points not joined to
  the rest.
- MARNET's own passage tags kept: Suez, Panama, Kiel, Corinth, Malacca, Gibraltar, Dover, Bab-el-Mandeb, Magellan,
  Bering, Northwest, Northeast.
- More straits tagged where a piece crosses a line drawn across them: Hormuz, Sunda, Lombok, Torres, the Bosphorus
  and Dardanelles, the Sound, the Great and Little Belts, Taiwan, Messina and Bonifacio.
- Packed into a binary file.

Added after the network, from ship traffic (the app uses them only when "Follow ship traffic" is on):

- How busy each piece is: the mean, at points 1 NM apart, of the busiest cell within about 6 NM.
- 545 lanes MARNET lacks (3,777 pieces): where the cheapest way over water ships use is much shorter than MARNET's
  way round, and the straight line between the two crosses land. They keep out of the canals and straits tagged
  above, so closing a passage in the app still closes it.
- 484 ways out to sea from 411 ports: the channel ships use from the port to an offing 3 NM off the land or 12 NM
  along, where that way bends or its straight line crosses land.

Nothing else was changed. The scripts here make every change listed.

## Files

| File | What it is |
|---|---|
| `lanes.bin` | The network exactly as the app carries it. The format is described at the top of `build_lanes.py` (the network), `refine_lanes.py` (busy and added lanes) and `port_approaches.py` (ways out). |
| `lanes.geojson` | The same as GeoJSON: one LineString per straight piece, with its `passage` where it has one and `busy`; the added pieces with `kind` "added"; the ports' ways out with `kind` "port_way". |
| `build_lanes.py` | Makes the network from MARNET's `marnet_plus_10km.gpkg`. |
| `aggregate_wb.py`, `build_density.py` | Sum the World Bank / IMF traffic layer to 0.05° cells and rank them (`rank_005.u8`). |
| `lane_traffic.py`, `refine_lanes.py` | How busy each piece is, and the added lanes. |
| `port_approaches.py` | The ports' ways out (it reads the app's `ports.tsv`, from the NGA World Port Index, public domain). |
| `lanes_to_geojson.py` | Writes `lanes.bin` out as `lanes.geojson`. |
| `LICENCE` | The EUPL-1.2 text. |

## Rebuild

```
python build_lanes.py marnet_plus_10km.gpkg lanes_marnet.bin
python aggregate_wb.py shipdensity_commercial_.zip 10 wbc05_sum.npy
python build_density.py wbc05_sum.npy grid
python refine_lanes.py lanes_marnet.bin grid/rank_005.u8 grid/meta_005.json world_land_10m.bin sea_hazards.bin lanes_ais.bin work
python port_approaches.py lanes_ais.bin ports.tsv grid/rank_005.u8 grid/meta_005.json world_land_10m.bin sea_hazards.bin lanes.bin work
python lanes_to_geojson.py lanes.bin lanes.geojson
```

`build_lanes.py` and `lanes_to_geojson.py` need Python 3 alone. The traffic steps also need numpy, scipy,
scikit-image and Pillow. `world_land_10m.bin` and `sea_hazards.bin` are the app's own packed Natural Earth coast,
reefs and small islands (public domain).

## Licence

EUPL-1.2 (see `LICENCE`), as for MARNET itself. The ship traffic data it was refined with: CC BY 4.0, World Bank / IMF.

**Not for navigation:** this is a planning network of common shipping lanes. Plan passages on official charts.
