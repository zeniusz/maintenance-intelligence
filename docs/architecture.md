# Architecture

```mermaid
flowchart LR
    API[Open-Meteo Historical API] -->|requests, 15 s timeout| W[mi/weather.py<br/>JSON cache + synthetic fallback]
    W --> G[mi/data_gen.py<br/>seeded generator]
    G --> CSV[(data/*.csv)]
    CSV --> C[pipeline.clean<br/>sentinels, gaps, weather join]
    C --> F[pipeline.fit_score<br/>baselines + Isolation Forest<br/>fit on training period only]
    F --> R[pipeline.risk_table<br/>explainable 0-100 score]
    F --> E[pipeline.evaluate<br/>held-out P / R / F1]
    EP[(anomaly_episodes.csv<br/>labels)] -.evaluation only.-> E
    R --> UI[app.py Streamlit + Plotly<br/>5 sections]
    E --> UI
```

| Module | Responsibility |
|---|---|
| `mi/weather.py` | Fetch + cache Open-Meteo hourly temperature/humidity; any failure → synthetic weather for that facility |
| `mi/data_gen.py` | Assets, readings, faults, maintenance, labels → CSV |
| `mi/pipeline.py` | Cleaning, features, model, risk score, evaluation (`run()` ties them together) |
| `app.py` | Dashboard only: no business logic |

**Leakage control:** time split at day 120. Baselines, ambient slope, Isolation Forest and the alert threshold use pre-split rows only; metrics use post-split rows only; labels are joined after scoring. `tests/test_mi.py::test_no_leakage_from_test_period` corrupts the test period and asserts nothing fitted changes.
