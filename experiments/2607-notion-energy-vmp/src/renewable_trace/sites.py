from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    site_id: str
    state: str
    lat: float
    lon: float

    @property
    def point_wkt(self) -> str:
        return f"POINT({self.lon} {self.lat})"


def site_from_mapping(value: dict) -> Site:
    return Site(
        site_id=str(value["site_id"]),
        state=str(value["state"]),
        lat=float(value["lat"]),
        lon=float(value["lon"]),
    )
