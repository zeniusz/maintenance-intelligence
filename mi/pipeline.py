"""Clean -> features -> Isolation Forest anomaly detection -> explainable risk score -> evaluation.

Leakage control: every statistic (baselines, ambient slope, model, alert threshold) is fitted only on rows
before `split`. Evaluation uses only rows at/after `split`. Labels are never used as features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from .data_gen import DATA, SENSORS, generate

# Relevance weight per sensor in the risk score (vibration is the strongest early indicator)
WEIGHTS = dict(zip(SENSORS, [0.85, 1.0, 0.7, 0.7]))
ZC, RC = [f"z_{s}" for s in SENSORS], [f"zr_{s}" for s in SENSORS]  # hourly z-score, 6h rolling mean z-score
WINDOW, TRAIN_DAYS = 168, 120


def load(data=DATA) -> dict:
    if not (data / "sensor_readings.csv").exists():
        generate(data)
    rd = lambda n, **k: pd.read_csv(data / f"{n}.csv", **k)
    return dict(readings=rd("sensor_readings", parse_dates=["timestamp"]), assets=rd("assets"),
                facilities=rd("facilities"), weather=rd("weather", parse_dates=["timestamp"]),
                maint=rd("maintenance_history", parse_dates=["date"]),
                episodes=rd("anomaly_episodes", parse_dates=["start", "end"]))


def clean(r: pd.DataFrame, assets: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Drop sentinels, interpolate short gaps (<=6h), join weather, flag running/valid hours."""
    r = r.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    r[SENSORS] = r[SENSORS].mask(r[SENSORS] < -50)  # -999 sentinel codes
    r[SENSORS] = r.groupby("asset_id")[SENSORS].transform(lambda s: s.interpolate(limit=6))
    r["facility_id"] = r.asset_id.map(assets.set_index("asset_id").facility_id)
    r = r.merge(weather, on=["facility_id", "timestamp"], how="left")
    r["running"] = r.groupby("asset_id").operating_hours.diff().fillna(1) > 0
    r["valid"] = r[SENSORS + ["ambient_temp_c"]].notna().all(axis=1)
    return r


def label(r: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Expand fault episodes into hourly ground truth (evaluation only; never a model input)."""
    ids, T, t0, hour = sorted(r.asset_id.unique()), r.timestamp.nunique(), r.timestamp.min(), pd.Timedelta("1h")
    y = np.zeros(len(r), bool)
    for e in episodes.itertuples():
        i = ids.index(e.asset_id) * T + int((e.start - t0) / hour)
        y[i:i + int((e.end - e.start) / hour) + 1] = True
    return r.assign(is_anomaly=y)


def fit_score(r: pd.DataFrame, split: pd.Timestamp, seed: int = 42):
    """Per-asset robust z-scores + Isolation Forest, all fitted on the training period only."""
    r = r.copy()
    tr = (r.timestamp < split) & r.running & r.valid
    slope = pd.Series({a: np.polyfit(g.ambient_temp_c, g.temperature_c, 1)[0]
                       for a, g in r[tr].groupby("asset_id")})
    X = r[SENSORS].copy()
    X["temperature_c"] -= r.asset_id.map(slope) * r.ambient_temp_c  # remove weather effect
    ids = r.asset_id[tr]
    med = X[tr].groupby(ids).median()
    mad = ((X[tr] - med.loc[ids].to_numpy()).abs().groupby(ids).median() * 1.4826).clip(lower=1e-3)
    Z = ((X - med.loc[r.asset_id].to_numpy()) / mad.loc[r.asset_id].to_numpy()).where(r.running, 0)
    r[ZC] = Z.to_numpy()
    r[RC] = r.groupby("asset_id")[ZC].rolling(6, min_periods=3).mean().reset_index(level=0, drop=True).to_numpy()
    F = ZC + RC
    ok = r.valid & r[F].notna().all(axis=1)
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(r.loc[tr & ok, F])
    r["anomaly_score"] = np.nan
    r.loc[ok, "anomaly_score"] = -model.score_samples(r.loc[ok, F])  # higher = more anomalous
    thr = r.loc[tr & ok, "anomaly_score"].quantile(0.99)  # alert on top 1% of training scores (best P/R balance in a sweep of .985-.995)
    r["alert"] = (r.anomaly_score > thr) & r.running
    return r, thr


def risk_table(r, assets, maint, as_of) -> pd.DataFrame:
    """Explainable 0-100 risk per asset over the 7 days before `as_of`.

    risk = 100 * (0.5*deviation + 0.3*alert_share + 0.2*maintenance) * criticality_multiplier
    deviation   = worst sensor's 95th-pct weighted |6h z| / 5 (capped at 1)
    alert_share = share of hours flagged by the model, saturating at 10%
    maintenance = 0.7*days since last work/180 + 0.3*any failure in last 90 days
    """
    as_of = pd.Timestamp(as_of)
    w = r[(r.timestamp <= as_of) & (r.timestamp > as_of - pd.Timedelta(hours=WINDOW))]
    sev = (w[RC].abs() / 5).clip(0, 1).fillna(0) * list(WEIGHTS.values())
    sev.columns = SENSORS
    q = sev.groupby(w.asset_id).quantile(0.95)
    driver = q.idxmax(axis=1)
    peak = {a: g[f"zr_{driver[a]}"].pipe(lambda s: s.loc[s.abs().fillna(0).idxmax()])
            for a, g in w.groupby("asset_id")}
    out = pd.DataFrame({"deviation": q.max(axis=1), "alert_share": w.groupby("asset_id").alert.mean(),
                        "driver": driver, "peak_z": pd.Series(peak).fillna(0)})
    last = maint[maint.date <= as_of].groupby("asset_id").date.max().reindex(out.index)
    fails = maint[(maint.type == "corrective") & (maint.date <= as_of) & (maint.date > as_of - pd.Timedelta(days=90))]
    out["days_since_maint"] = (as_of - last).dt.days.fillna(365)
    out["recent_failures"] = fails.groupby("asset_id").size().reindex(out.index).fillna(0)
    m = 0.7 * (out.days_since_maint / 180).clip(0, 1) + 0.3 * out.recent_failures.clip(0, 1)
    out = out.join(assets.set_index("asset_id"))
    mult = (0.6 + 0.1 * out.criticality) / 1.1
    out["risk"] = (100 * (0.5 * out.deviation + 0.3 * (out.alert_share / 0.1).clip(0, 1) + 0.2 * m) * mult).round(1)
    out["level"] = pd.cut(out.risk, [-1, 25, 40, 101], labels=["Low", "Medium", "High"])
    out["explanation"] = out.apply(lambda x: (
        f"{x.driver.split('_')[0].title()} {x.peak_z:+.1f}σ vs own baseline; {x.alert_share:.0%} of hours alerted; "
        f"{x.days_since_maint:.0f} d since maintenance; criticality {x.criticality}/5"), axis=1)
    return out.sort_values("risk", ascending=False).rename_axis("asset_id").reset_index()


def evaluate(r: pd.DataFrame, episodes: pd.DataFrame, split) -> dict:
    """Row-level P/R/F1 on the held-out period + event-level detection and early-warning lead time."""
    te = r[r.timestamp >= split]
    preds = {"Isolation Forest": te.alert, "Baseline rule (|z|>3)": (te[RC].abs().max(axis=1) > 3) & te.running}
    metrics = pd.DataFrame({k: dict(precision=precision_score(te.is_anomaly, p, zero_division=0),
                                    recall=recall_score(te.is_anomaly, p, zero_division=0),
                                    f1=f1_score(te.is_anomaly, p, zero_division=0)) for k, p in preds.items()}).T
    alerts = r[r.alert & (r.timestamp >= split)][["asset_id", "timestamp"]]
    ev = episodes[episodes.start >= split].copy()
    first = [alerts[(alerts.asset_id == e.asset_id) & alerts.timestamp.between(e.start, e.end)].timestamp.min()
             for e in ev.itertuples()]
    ev["lead_h"] = (ev.end - pd.Series(first, index=ev.index)).dt.total_seconds() / 3600  # hours before episode end
    fl = ev[ev.led_to_failure]
    return dict(metrics=metrics, events=dict(
        episodes=len(ev), detected=int(ev.lead_h.notna().sum()),
        failures=len(fl), failures_warned_6h=int((fl.lead_h >= 6).sum()), median_lead_h=float(fl.lead_h.median())))


def run(data=DATA) -> dict:
    d = load(data)
    r = label(clean(d.pop("readings"), d["assets"], d["weather"]), d["episodes"])
    split = r.timestamp.min() + pd.Timedelta(days=TRAIN_DAYS)
    r, thr = fit_score(r, split)
    return dict(d, readings=r, split=split, threshold=thr, **evaluate(r, d["episodes"], split))
