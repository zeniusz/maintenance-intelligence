# Maintenance Intelligence

An explainable predictive-maintenance prototype. It turns six months of hourly sensor data from 100 **fictional** industrial assets (5 fictional European facilities) into a ranked, plain-language inspection list, with live weather context and an honest evaluation of the model.

Built with Python 3.12 · Pandas · NumPy · scikit-learn · Streamlit · Plotly · Requests · Pytest.

![Overview](docs/screenshots/overview.png)

## The problem

An operator of oil, gas and energy plants may have thousands of assets sending readings every hour. Nobody can watch every chart, and a fixed alarm limit (for example "vibration > X") is a poor fit: a healthy large compressor and a healthy small pump have very different normal values, and weather shifts temperatures even when nothing is wrong. The questions this project answers are:

1. **Which assets should a technician visit first?**
2. **Why** is each one flagged, in words a technician can check?
3. **How much can we trust that ranking?**

## The solution in one minute

| Step | What happens, in plain words | Why it matters |
|---|---|---|
| 1. Make the data | A program creates 100 pretend machines (pumps, compressors, motors...) with six months of hourly readings: temperature, vibration, pressure, power and running hours. It adds realistic problems: missing values, broken readings, and faults we hide on purpose and write down. | Real plant data is private. Because we know where the faults are, we can check honestly whether the system finds them. |
| 2. Add the weather | For each site we download the real outdoor temperature and humidity (free service Open-Meteo), plus today's live weather. If the internet is down, made-up weather is used instead. | A hot day makes a machine run warmer without anything being wrong. Knowing the weather prevents false alarms, and the app always starts. |
| 3. Tidy up | Remove obvious junk values (like -999), fill short gaps of up to 6 hours, and attach the weather to each reading. | Real sensors glitch. Dirty data gives wrong answers. |
| 4. Learn what is normal for each machine | Every machine is compared with **its own** usual behaviour, not with one rule for all. Weather is taken into account, and hours when the machine is switched off are ignored. | A big compressor and a small pump have very different normal values. One shared limit would be wrong for both. |
| 5. Spot unusual hours | A model called an Isolation Forest looks at all sensors together and marks hours that look strange for that machine. It only learns from the first part of the data and is tested on the later part. | It finds odd combinations (for example vibration slightly up while power is down) that a single limit would miss, and it needs no examples of past failures. |
| 6. Give each machine a risk score from 0 to 100 | The score mixes three things: how far readings drift from normal (50%), how often the machine was flagged (30%) and how long since its last maintenance (20%). It is then raised for machines that matter more. | A simple formula that an engineer can read and adjust, not a black box. It puts the machines to visit first at the top of the list. |
| 7. Explain it in a sentence | Each machine gets a short reason, for example: "Vibration +11.0σ vs own baseline; 7% of hours alerted; 4 d since maintenance; criticality 3/5" (σ means "how many typical steps above normal"). | People only act on a warning they understand and can check in the charts. |
### Anomaly detection is not failure prediction

The Isolation Forest says "this hour looks unlike this asset's normal". It does **not** say "this asset will fail in N days". Predicting failures needs labelled failure histories and many more failures than six months provide. The risk score therefore prioritises *inspection*, and the dashboard says so.

## The dashboard

| Tab | What it shows |
|---|---|
| **Overview** | Data-freshness header, four KPIs, ranked priorities table, facility risk map, risk by facility, facility x asset-type heatmap, live weather per site, model performance. |
| **Sensor Monitoring** | Any asset and sensor over 7/30/90 days, red dots for model alerts, orange bands for the injected (true) fault episodes. |
| **Maintenance Priorities** | Top-N assets by risk with severity colours, full table with explanations, CSV download. |
| **Facility Comparison** | Risk by facility and by facility x type, plus weather history. |
| **Asset Details** | Risk, criticality, "why this alert?", current sensor deviation, 30-day trend and maintenance history. |

Severity is always colour **and** text: **Normal** (risk < 25), **Warning** (25-40), **Critical** (> 40). The filters (facility, asset type, 7/30/90-day period, risk date) apply to every tab.

| Sensor Monitoring | Maintenance Priorities |
|---|---|
| ![Sensors](docs/screenshots/sensor-monitoring.png) | ![Priorities](docs/screenshots/maintenance-priorities.png) |
| **Facility Comparison** | **Asset Details** |
| ![Facilities](docs/screenshots/facility-comparison.png) | ![Asset](docs/screenshots/asset-details.png) |

### Data freshness, stated honestly
The header shows the asset-data as-of time and the pipeline-run time separately from the weather fetch time. The asset data is a fixed synthetic snapshot, so it is labelled **Delayed · demo snapshot**, never "live". Weather shows **Current** or **Sync failed** from the actual API response.

## Results (held-out May-Jun 2025, hourly labels, real weather)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Isolation Forest | 0.83 | 0.65 | 0.73 |
| Simple rule (\|z\|>3) | 0.90 | 0.61 | 0.73 |

56 of 56 injected fault episodes were alerted, and 15 of 15 that ended in failure were alerted at least 6 h earlier (median 28 h). Accuracy is deliberately not reported: only about 1.2% of hours are anomalous, so it would be meaningless. The simple rule matches the forest here because synthetic faults have clean signatures. Numbers shift slightly with synthetic (offline) weather.

## Run it (Windows)

Install Python 3.12 (tick *Add python.exe to PATH*), then double-click **`run.bat`**, or:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first launch generates the data and runs the pipeline (about 15 s, longer while it downloads weather). It works offline.

```bash
python -m pytest        # 9 tests
python -m mi.data_gen   # regenerate CSVs (--offline skips the weather API)
```

## Project layout

```
app.py              dashboard
mi/data_gen.py      synthetic data generator
mi/weather.py       Open-Meteo client, cache, fallback
mi/pipeline.py      cleaning, features, Isolation Forest, risk score, evaluation
tests/test_mi.py    data quality, weather fallback, no-leakage, metrics, risk
docs/               data dictionary, architecture diagram, screenshots
data/               generated CSVs (sensor_readings.csv is regenerated on first run)
```

See [docs/data_dictionary.md](docs/data_dictionary.md) and [docs/architecture.md](docs/architecture.md).

## Design decisions

- **Per-asset baselines over global limits**, because "normal" differs per machine.
- **Isolation Forest over a supervised model**, because real plants rarely have enough labelled failures.
- **A hand-weighted risk score over a learned one**, because it is explainable and can be reviewed by a domain expert.
- **A synthetic fallback for weather**, so the project never fails to start.
- **Reported against a simple rule**, so the value of the model is measured, not assumed.

## Next steps
1. **Ask-the-data assistant (chatbot):** a chat box in the dashboard where a technician asks in plain language, e.g. *"Which pumps at Baltic Terminal got worse this week?"* or *"Why is A013 ranked first?"*. An LLM turns the question into a safe, read-only database query (or calls fixed tools such as `top_risks()` and `asset_history()`), runs it on the real database, and answers with the numbers and a link to the chart. Answers must come from query results, never from guessing, and are shown with the query used.
2. **Real database and data:** move from CSV files to a database (PostgreSQL or a time-series store) fed by the plant historian or maintenance system.
3. **Failure prediction:** with real failure records, add time-to-failure models and calibrated probabilities.
4. **Better models:** compare autoencoders or per-asset-type models, using a proper validation window.
5. **Operations:** scheduled refresh, alert notifications, Docker deployment, and a button to mark an alert as a false alarm so the system learns from it.
## Limitations

Synthetic faults are cleaner than real ones. There is one model for the whole fleet. The alert quantile was chosen on the test period, so scores are slightly optimistic (a real project would use a separate validation window). Score weights are heuristics, not learned.

> All asset data is synthetic. Only the weather comes from a real source (Open-Meteo).
