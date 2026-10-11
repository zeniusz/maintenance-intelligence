import numpy as np
import pandas as pd
import pytest
import requests

from mi import data_gen, pipeline as P, weather


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    data_gen.generate(d, offline=True)
    return d


@pytest.fixture(scope="module")
def result(data_dir):
    return P.run(data_dir)


def test_data_shape_quality_and_labels(data_dir):
    r = pd.read_csv(data_dir / "sensor_readings.csv")
    assert r.asset_id.nunique() == 100 and len(r) == 100 * data_gen.HOURS
    assert r[data_gen.SENSORS].isna().any().any() and (r.temperature_c == -999).any()
    assert pd.read_csv(data_dir / "facilities.csv").shape[0] == 5
    assert len(pd.read_csv(data_dir / "anomaly_episodes.csv")) > 50


def test_generation_is_reproducible(data_dir, tmp_path):
    data_gen.generate(tmp_path, offline=True)
    pd.testing.assert_frame_equal(pd.read_csv(data_dir / "sensor_readings.csv"), pd.read_csv(tmp_path / "sensor_readings.csv"))


def test_weather_falls_back_when_api_fails(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(weather.requests, "get", boom)
    ts = pd.date_range("2025-01-01", periods=48, freq="h")
    w = weather.get_weather(data_gen.FACILITIES, ts, tmp_path)
    assert len(w) == 5 * 48 and (w.weather_source == "synthetic").all() and w.ambient_temp_c.notna().all()


def test_weather_parses_and_caches_api_response(monkeypatch, tmp_path):
    ts = pd.date_range("2025-01-01", periods=48, freq="h")
    payload = {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in ts], "temperature_2m": [5.0] * 48,
                          "relative_humidity_2m": [80] * 48}}

    class Resp:
        def raise_for_status(self): pass
        def json(self): return payload
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: Resp())
    w = weather.get_weather(data_gen.FACILITIES, ts, tmp_path)
    assert (w.weather_source == "open-meteo").all() and len(list(tmp_path.glob("*.json"))) == 5


def test_live_weather_parses_and_fails_gracefully(monkeypatch):
    cur = {"time": "2025-01-01T10:00", "temperature_2m": 3.2, "relative_humidity_2m": 70, "wind_speed_10m": 9.0}

    class Resp:
        def raise_for_status(self): pass
        def json(self): return [{"current": cur}] * 5
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: Resp())
    assert len(weather.get_current(data_gen.FACILITIES)) == 5
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
    assert weather.get_current(data_gen.FACILITIES).empty


def test_no_leakage_from_test_period(result):
    r, split = result["readings"], result["split"]
    r2 = r.copy()
    r2.loc[r2.timestamp >= split, data_gen.SENSORS] *= 3  # wreck the test period
    a, thr_a = P.fit_score(r, split)
    b, thr_b = P.fit_score(r2, split)
    assert thr_a == thr_b
    pd.testing.assert_series_equal(a.anomaly_score[a.timestamp < split], b.anomaly_score[b.timestamp < split])


def test_detector_beats_chance_and_metrics_valid(result):
    m = result["metrics"]
    assert ((m >= 0) & (m <= 1)).all().all()
    assert m.loc["Isolation Forest", "recall"] > 0.5 and m.loc["Isolation Forest", "f1"] > 0.5
    assert result["readings"].is_anomaly.mean() < 0.05  # imbalanced: accuracy would be meaningless


def test_risk_table_ranked_bounded_and_explained(result):
    t = P.risk_table(result["readings"], result["assets"], result["maint"], result["readings"].timestamp.max())
    assert len(t) == 100 and t.risk.is_monotonic_decreasing and t.risk.between(0, 100).all()
    assert t.explanation.str.contains("σ").all()


def test_criticality_scales_risk(result):
    r, mt, a = result["readings"], result["maint"], result["assets"].copy()
    as_of = r.timestamp.max()
    lo = P.risk_table(r, a.assign(criticality=1), mt, as_of).set_index("asset_id").risk
    hi = P.risk_table(r, a.assign(criticality=5), mt, as_of).set_index("asset_id").risk.reindex(lo.index)
    assert (hi >= lo).all() and (hi > lo).any()
