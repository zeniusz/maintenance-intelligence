"""Reproducible synthetic data: 100 fictional assets, 5 fictional facilities, 6 months of hourly readings."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .weather import get_weather

DATA = Path(__file__).resolve().parents[1] / "data"
SEED, START, HOURS = 42, "2025-01-01", 181 * 24  # 1 Jan - 30 Jun 2025
SENSORS = ["temperature_c", "vibration_mm_s", "pressure_bar", "power_kw"]

FACILITIES = pd.DataFrame({
    "facility_id": [f"F{i}" for i in range(1, 6)],
    "name": ["Alpenrand Refinery", "Danube Gas Hub", "Baltic Terminal", "Carpathian Compression Station",
             "Nordsee Storage Depot"],
    "country": ["Austria", "Hungary", "Poland", "Romania", "Germany"],
    "latitude": [48.21, 47.50, 54.35, 45.80, 53.55],
    "longitude": [16.37, 19.04, 18.65, 24.15, 8.58]})

# Nominal (temperature, vibration, pressure, power) per asset type
TYPES = {"Centrifugal Pump": (65, 2.5, 12, 90), "Compressor": (95, 4.0, 35, 450),
         "Electric Motor": (70, 2.0, 3, 200), "Heat Exchanger Fan": (55, 1.5, 2, 30),
         "Gas Turbine": (140, 5.0, 20, 800)}

# Injected fault types: sensor -> (mode, amount). "mul": x*(1+amount*ramp), "add": x+amount*ramp
FAULTS = {"bearing_wear": {"vibration_mm_s": ("mul", 1.5), "temperature_c": ("add", 6)},
          "overheating": {"temperature_c": ("add", 25), "power_kw": ("mul", 0.1)},
          "pressure_loss": {"pressure_bar": ("mul", -0.4), "power_kw": ("mul", -0.15)},
          "power_surge": {"power_kw": ("mul", 0.5), "vibration_mm_s": ("mul", 0.2)}}


def generate(out: Path = DATA, offline: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ts = pd.date_range(START, periods=HOURS, freq="h")
    A, T = 100, HOURS
    weather = get_weather(FACILITIES, ts, out / "cache", offline)
    assets = pd.DataFrame({
        "asset_id": [f"A{i:03d}" for i in range(1, A + 1)],
        "facility_id": np.repeat(FACILITIES.facility_id.to_numpy(), A // 5),
        "asset_type": rng.choice(list(TYPES), A),
        "criticality": rng.integers(1, 6, A),
        "install_year": rng.integers(2005, 2023, A)})
    base = np.array([TYPES[t] for t in assets.asset_type]) * rng.normal(1, 0.08, (A, 4))
    amb = (weather.pivot(index="facility_id", columns="timestamp", values="ambient_temp_c")
           .reindex(index=assets.facility_id, columns=ts).to_numpy())

    # Fault episodes; ~60% of severe ones end in a failure followed by downtime
    eps = []
    for a in range(A):
        for _ in range(rng.poisson(2)):
            dur, sev = int(rng.integers(8, 49)), rng.uniform(0.6, 1.4)
            eps.append(dict(a=a, s=int(rng.integers(0, T - dur - 72)), dur=dur, sev=sev, type=str(rng.choice(list(FAULTS))),
                            fail=bool(sev > 1.0 and rng.random() < 0.6), down=int(rng.integers(8, 73))))

    # Normal behaviour: daily load cycle, duty cycle (20% of assets are standby units), slow drift on 15% of assets
    load = 1 + 0.12 * np.sin(2 * np.pi * (ts.hour.to_numpy() - 8) / 24) + rng.normal(0, 0.03, (A, T))
    duty = np.where(rng.random(A) < 0.2, 0.6, 0.97)
    run = rng.random((A, T // 12 + 1))[:, np.arange(T) // 12] < duty[:, None]
    for e in eps:
        run[e["a"], e["s"]:e["s"] + e["dur"]] = True
        if e["fail"]:
            run[e["a"], e["s"] + e["dur"]:e["s"] + e["dur"] + e["down"]] = False
    drift = np.linspace(0, 1, T) * (rng.random(A) < 0.15)[:, None]
    n = lambda sd: rng.normal(0, sd, (A, T))
    b = [base[:, [i]] for i in range(4)]
    temp = b[0] + 0.3 * (amb - 10) + 25 * (load - 1) + 4 * drift + n(0.8)
    temp = np.where(run, temp, amb + 0.15 * (temp - amb))
    vib = np.where(run, b[1] * (1 + 1.5 * (load - 1) + 0.15 * drift) * (1 + n(0.05)), 0.1 * b[1] * (1 + n(0.1)))
    pres = np.where(run, b[2] * (1 + 0.5 * (load - 1) + n(0.02)), 0.05 * b[2] * (1 + n(0.1)))
    power = np.where(run, b[3] * load * (1 + n(0.03)), 0.05 * b[3] * (1 + n(0.1)))
    X = np.stack([temp, vib, pres, power])

    for e in eps:
        sl, ramp = slice(e["s"], e["s"] + e["dur"]), np.linspace(0.3, 1, e["dur"]) * e["sev"]
        for s, (mode, amt) in FAULTS[e["type"]].items():
            i = SENSORS.index(s)
            X[i, e["a"], sl] = X[i, e["a"], sl] * (1 + amt * ramp) if mode == "mul" else X[i, e["a"], sl] + amt * ramp
    X[1:] = np.clip(X[1:], 0, None)

    # Data quality problems: random NaNs, multi-hour sensor dropouts, -999 sentinel codes on temperature
    X[rng.random(X.shape) < 0.015] = np.nan
    for _ in range(15):
        a, s = int(rng.integers(A)), int(rng.integers(0, T - 12))
        X[:, a, s:s + int(rng.integers(3, 13))] = np.nan
    X[0][rng.random((A, T)) < 0.0005] = -999

    readings = pd.DataFrame({
        "asset_id": np.repeat(assets.asset_id.to_numpy(), T), "timestamp": np.tile(ts, A),
        **{s: X[i].ravel().round(2) for i, s in enumerate(SENSORS)},
        "operating_hours": (rng.integers(5000, 60000, (A, 1)) + np.cumsum(run, axis=1)).ravel()})

    episodes = pd.DataFrame([dict(asset_id=assets.asset_id[e["a"]], start=ts[e["s"]], end=ts[e["s"] + e["dur"] - 1],
                                  anomaly_type=e["type"], severity=round(e["sev"], 2), led_to_failure=e["fail"])
                             for e in eps])
    rows = []
    for a, row in assets.iterrows():  # preventive work from 300 days before the window until its end
        for d in sorted(rng.integers(-300, T // 24, rng.integers(2, 5))):
            rows.append((row.asset_id, ts[0] + pd.Timedelta(days=int(d)), "preventive", int(rng.integers(2, 9)),
                         int(rng.uniform(0.5, 1.5) * 1500), "Scheduled inspection and lubrication"))
    for e in eps:
        if e["fail"]:
            c = assets.criticality[e["a"]]
            rows.append((assets.asset_id[e["a"]], ts[e["s"] + e["dur"] - 1], "corrective", e["down"],
                         int(rng.uniform(0.5, 1.5) * 8000 * c), f"Failure after {e['type']}"))
    maint = pd.DataFrame(rows, columns=["asset_id", "date", "type", "downtime_hours", "cost_eur", "description"])
    maint = maint.sort_values(["asset_id", "date"]).reset_index(drop=True).rename_axis("event_id").reset_index()

    assets.insert(1, "asset_name", assets.asset_type + " " + assets.asset_id)
    for name, df in dict(assets=assets, facilities=FACILITIES, sensor_readings=readings, weather=weather,
                         maintenance_history=maint, anomaly_episodes=episodes).items():
        df.to_csv(out / f"{name}.csv", index=False)
    print(f"Wrote {len(readings):,} readings, {len(episodes)} fault episodes to {out}")


if __name__ == "__main__":
    generate(offline="--offline" in sys.argv)
