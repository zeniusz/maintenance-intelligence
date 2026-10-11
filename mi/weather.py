"""Open-Meteo Historical Weather API client with JSON cache and synthetic fallback."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"


def _synthetic(f, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Seasonal + daily sine temperature, humidity anti-correlated. Colder at higher latitude."""
    rng = np.random.default_rng(int(f.facility_id[1:]))
    mean = 25 - 0.4 * f.latitude
    doy, hour = ts.dayofyear.to_numpy(), ts.hour.to_numpy()
    noise = np.convolve(rng.normal(0, 1, len(ts) + 23), np.ones(24) / 4, "valid")
    temp = mean + 10 * np.sin(2 * np.pi * (doy - 110) / 365) + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + noise
    hum = np.clip(75 - 1.2 * (temp - mean) + rng.normal(0, 5, len(ts)), 25, 100)
    return pd.DataFrame({"timestamp": ts, "ambient_temp_c": temp.round(1), "humidity_pct": hum.round(0),
                         "weather_source": "synthetic"})


def _fetch(f, ts: pd.DatetimeIndex, cache: Path) -> pd.DataFrame | None:
    """Return real hourly weather, or None on any failure (caller falls back)."""
    start, end = ts[0].date(), ts[-1].date()
    path = cache / f"{f.facility_id}_{start}_{end}.json"
    try:
        cached = path.exists()
        if cached:
            raw = json.loads(path.read_text())
        else:
            r = requests.get(URL, timeout=15, params=dict(
                latitude=f.latitude, longitude=f.longitude, start_date=str(start), end_date=str(end),
                hourly="temperature_2m,relative_humidity_2m", timezone="UTC"))
            r.raise_for_status()
            raw = r.json()
        h = raw["hourly"]
        df = pd.DataFrame({"timestamp": pd.to_datetime(h["time"]), "ambient_temp_c": h["temperature_2m"],
                           "humidity_pct": h["relative_humidity_2m"]}).set_index("timestamp").reindex(ts)
        if len(h["time"]) < 0.9 * len(ts) or df.isna().all().any():
            raise ValueError("incomplete response")
        if not cached:
            cache.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw))
        df = df.interpolate(limit_direction="both").rename_axis("timestamp").reset_index()
        return df.assign(weather_source="open-meteo")
    except (requests.RequestException, KeyError, ValueError, OSError) as e:
        print(f"[weather] {f.facility_id}: using synthetic fallback ({e})")
        return None


def get_current(facilities: pd.DataFrame) -> pd.DataFrame:
    """Live conditions from the Open-Meteo forecast API (one call). Empty frame on any failure."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params=dict(
            latitude=",".join(map(str, facilities.latitude)), longitude=",".join(map(str, facilities.longitude)),
            current="temperature_2m,relative_humidity_2m,wind_speed_10m", timezone="UTC"))
        r.raise_for_status()
        cur = pd.DataFrame([j["current"] for j in r.json()])
        return cur.assign(facility_id=facilities.facility_id.to_numpy(), time=pd.to_datetime(cur.time))
    except (requests.RequestException, KeyError, ValueError, TypeError) as e:
        print(f"[weather] live conditions unavailable ({e})")
        return pd.DataFrame()


def get_weather(facilities: pd.DataFrame, ts: pd.DatetimeIndex, cache: Path, offline: bool = False) -> pd.DataFrame:
    """Hourly weather for every facility; real data where available, synthetic otherwise."""
    frames = []
    for f in facilities.itertuples():
        df = None if offline else _fetch(f, ts, cache)
        frames.append((_synthetic(f, ts) if df is None else df).assign(facility_id=f.facility_id))
    return pd.concat(frames, ignore_index=True)
