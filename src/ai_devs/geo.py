"""Geographic utility functions.

Reusable across tasks that need distance calculations on Earth's surface.
"""

import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth (Haversine formula).

    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees).
        lat2, lon2: Latitude and longitude of point 2 (degrees).

    Returns:
        Distance in kilometres.
    """
    R = 6371.0  # Earth's mean radius in km

    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def find_nearest_facility(
    person_coords: list[tuple[float, float]],
    facilities: dict[str, dict],
    lat_key: str = "latitude",
    lon_key: str = "longitude",
    active_key: str = None,
) -> dict | None:
    """Find the facility nearest to any of a person's observed coordinates.

    Args:
        person_coords: List of (lat, lon) tuples for observed locations.
        facilities: Dict mapping facility_id -> info dict.
        lat_key: Key in info dict for latitude.
        lon_key: Key in info dict for longitude.
        active_key: If set, skip facilities where info[active_key] is falsy.

    Returns:
        Dict with facility_id, distance_km, person_lat, person_lon,
        facility_lat, facility_lon — or None if no valid facility found.
    """
    best = None
    for facility_id, info in facilities.items():
        if active_key and not info.get(active_key):
            continue
        f_lat, f_lon = info.get(lat_key), info.get(lon_key)
        if f_lat is None or f_lon is None:
            continue
        for p_lat, p_lon in person_coords:
            dist = haversine_distance(p_lat, p_lon, f_lat, f_lon)
            if best is None or dist < best["distance_km"]:
                best = {
                    "facility_id": facility_id,
                    "distance_km": round(dist, 2),
                    "person_lat": p_lat,
                    "person_lon": p_lon,
                    "facility_lat": f_lat,
                    "facility_lon": f_lon,
                }
    return best


# Known Polish city coordinates (lat, lon) — extend as needed.
POLISH_CITY_COORDS: dict[str, tuple[float, float]] = {
    "Białystok": (53.1325, 23.1688),
    "Bielsko-Biała": (49.8224, 19.0446),
    "Bydgoszcz": (53.1235, 18.0084),
    "Chełm": (51.1431, 23.4716),
    "Chelmno": (53.3493, 18.4260),
    "Chełmno": (53.3493, 18.4260),
    "Częstochowa": (50.8118, 19.1203),
    "Elbląg": (54.1522, 19.4048),
    "Gdańsk": (54.3520, 18.6466),
    "Gdynia": (54.5189, 18.5305),
    "Gliwice": (50.2945, 18.6714),
    "Grudziądz": (53.4837, 18.7536),
    "Katowice": (50.2649, 19.0238),
    "Kielce": (50.8661, 20.6286),
    "Kraków": (50.0647, 19.9450),
    "Lublin": (51.2465, 22.5684),
    "Łódź": (51.7592, 19.4560),
    "Olsztyn": (53.7784, 20.4801),
    "Opole": (50.6751, 17.9213),
    "Piotrków Trybunalski": (51.4053, 19.7033),
    "Płock": (52.5463, 19.7065),
    "Poznań": (52.4064, 16.9252),
    "Radom": (51.4027, 21.1471),
    "Rzeszów": (50.0412, 21.9991),
    "Sopot": (54.4416, 18.5601),
    "Szczecin": (53.4285, 14.5528),
    "Tczew": (54.0927, 18.7975),
    "Toruń": (53.0138, 18.5984),
    "Warszawa": (52.2297, 21.0122),
    "Wrocław": (51.1079, 17.0385),
    "Zabrze": (50.3249, 18.7857),
    "Żarnowiec": (54.7571, 18.0942),
}
