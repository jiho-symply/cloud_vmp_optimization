from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NyisoZone:
    site_id: str
    nyiso_zone: str
    proxy_city: str
    lat: float
    lon: float


NYISO_ZONES = [
    NyisoZone("WEST", "WEST", "Buffalo", 42.8864, -78.8784),
    NyisoZone("GENESE", "GENESE", "Rochester", 43.1566, -77.6088),
    NyisoZone("CENTRL", "CENTRL", "Syracuse", 43.0481, -76.1474),
    NyisoZone("NORTH", "NORTH", "Watertown", 43.9748, -75.9108),
    NyisoZone("MHKVL", "MHK VL", "Utica", 43.1009, -75.2327),
    NyisoZone("CAPITL", "CAPITL", "Albany", 42.6526, -73.7562),
    NyisoZone("HUDVL", "HUD VL", "Poughkeepsie", 41.7004, -73.9210),
    NyisoZone("MILLWD", "MILLWD", "White Plains", 41.0340, -73.7629),
    NyisoZone("DUNWOD", "DUNWOD", "Yonkers", 40.9312, -73.8988),
    NyisoZone("NYC", "N.Y.C.", "New York City", 40.7128, -74.0060),
    NyisoZone("LONGIL", "LONGIL", "Hicksville", 40.7684, -73.5251),
]


def zone_from_mapping(value: dict) -> NyisoZone:
    return NyisoZone(
        site_id=str(value["site_id"]),
        nyiso_zone=str(value["nyiso_zone"]),
        proxy_city=str(value["proxy_city"]),
        lat=float(value["lat"]),
        lon=float(value["lon"]),
    )


def zone_lookup(zones: list[NyisoZone]) -> dict[str, NyisoZone]:
    return {zone.nyiso_zone: zone for zone in zones}
