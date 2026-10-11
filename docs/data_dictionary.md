# Data dictionary

All data is **synthetic** (seed 42) except `weather.csv`, which holds Open-Meteo reanalysis values for the fictional facilities' coordinates (`weather_source` says which). Period: 2025-01-01 00:00 to 2025-06-30 23:00 UTC, hourly (4,344 h × 100 assets = 434,400 rows).

| File | Grain | Columns |
|---|---|---|
| `facilities.csv` | facility | `facility_id`, `name`, `country`, `latitude`, `longitude` (fictional sites, real-world coordinates for weather) |
| `assets.csv` | asset | `asset_id`, `asset_name`, `facility_id`, `asset_type` (5 types), `criticality` (1 low – 5 high business impact), `install_year` |
| `sensor_readings.csv` | asset × hour | `asset_id`, `timestamp`, `temperature_c`, `vibration_mm_s` (RMS velocity), `pressure_bar`, `power_kw`, `operating_hours` (cumulative running hours) |
| `weather.csv` | facility × hour | `facility_id`, `timestamp`, `ambient_temp_c`, `humidity_pct`, `weather_source` (`open-meteo` / `synthetic`) |
| `maintenance_history.csv` | event | `event_id`, `asset_id`, `date`, `type` (`preventive` / `corrective`), `downtime_hours`, `cost_eur`, `description`. Covers 300 days before the window; corrective rows are failures |
| `anomaly_episodes.csv` | injected fault | `asset_id`, `start`, `end` (inclusive), `anomaly_type`, `severity`, `led_to_failure`. **Evaluation labels only: never model inputs** |

`sensor_readings.csv` is git-ignored (24 MB) and regenerated identically by `python -m mi.data_gen`.

## Synthetic assumptions
- Normal behaviour: per-asset nominal values by type (±8%), daily load cycle (±12%, peak 14:00), sensor noise, temperature coupled to ambient (+0.3 °C per °C). 15% of assets drift slowly. 20% are standby units (≈60% duty); idle hours read near zero and `operating_hours` stops increasing.
- Faults (≈187 episodes, ≈1.2% of hours, 8–48 h each): `bearing_wear`, `overheating`, `pressure_loss`, `power_surge`, ramping from 30% to 100% of severity. About 60% of severe episodes end in a failure with 8–72 h downtime and a corrective maintenance record.
- Data quality: 1.5% random missing values, 15 multi-hour sensor dropouts, and `-999` sentinel codes on ~0.05% of temperature readings.
- Fault signatures are a simplification: real failures are messier, so the metrics here show the method works, not what accuracy real assets would give.
