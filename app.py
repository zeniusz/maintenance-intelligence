"""Maintenance Intelligence dashboard. Run: streamlit run app.py"""
import itertools

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

from mi import pipeline as P
from mi.data_gen import SENSORS
from mi.weather import get_current

_n = itertools.count()  # unique keys for styled card containers
st.set_page_config("Maintenance Intelligence", "🛠️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, button, input { font-family: 'Inter', sans-serif; }
.block-container { padding: 3rem 2rem 2rem; max-width: 1600px; }
h1 { font-size: 1.5rem !important; font-weight: 600 !important; padding: 0 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; padding: 0 0 .4rem !important; }
[class*="st-key-card"] { background: #111b2e; border: 1px solid #1f2d44; border-radius: 5px; padding: .8rem 1rem; }
[data-testid="stMetricLabel"] p { font-size: .75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight: 600; font-size: 1.9rem; }
[data-testid="stMetricValue"] > div { white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.2; }
button[data-baseweb="tab"][aria-selected="true"] { font-weight: 600; color: #2dd4bf; }
[data-baseweb="tab-highlight"] { background: #2dd4bf; height: 2px; }
[data-testid="stDataFrame"] { border: 1px solid #1f2d44; border-radius: 5px; overflow: hidden; }
[data-testid="stPlotlyChart"] { background: #111b2e; border: 1px solid #1f2d44; border-radius: 5px; padding: .5rem; }
[class*="st-key-card"] [data-testid="stPlotlyChart"] { background: transparent; border: 0; padding: 0; }
.fresh { display: flex; gap: 1.5rem; justify-content: flex-end; font-size: .78rem; color: #94a3b8; line-height: 1.35; }
.fresh b { color: #e5edf5; font-weight: 500; }
.pill { white-space: nowrap; padding: 1px 8px; border-radius: 99px; font-size: .7rem; font-weight: 600; margin-right: .4rem; }
.ok { background: #0f4c47; color: #5eead4; } .warn { background: #5a3d0a; color: #fbbf24; } .bad { background: #5a1414; color: #f87171; }
@media (max-width: 640px) { .fresh { flex-direction: column; gap: .5rem; justify-content: flex-start; } .block-container { padding: 2.6rem 1rem 2rem; } }
</style>""", unsafe_allow_html=True)
pio.templates["mi"] = pio.templates["plotly_dark"]
pio.templates["mi"].layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Inter",
                                  margin=dict(l=10, r=30, t=40, b=10), title_font_size=14, hovermode="x unified")
pio.templates["mi"].layout.xaxis.update(showgrid=False)
pio.templates["mi"].layout.yaxis.update(gridcolor="#233047", zeroline=False)
px.defaults.template = "mi"
px.defaults.color_continuous_scale = "Teal"


@st.cache_resource(show_spinner="Preparing synthetic data and running the AI pipeline (first run only)...")
def load():
    return P.run(), pd.Timestamp.now("UTC")  # timestamp = when the pipeline last completed successfully


@st.cache_data(ttl=900, show_spinner=False)
def live(_fac):  # refreshed at most every 15 min, or via the Refresh button
    return get_current(_fac), pd.Timestamp.now("UTC")


D, synced = load()
R, assets, fac = D["readings"], D["assets"], D["facilities"]
COLORS = {"Low": "#14b8a6", "Medium": "#f59e0b", "High": "#ef4444"}  # teal / amber / red, always with a text label
STATUS = {"Low": "Normal", "Medium": "Warning", "High": "Critical"}
TAG = {"Low": ":green[Normal]", "Medium": ":orange[Warning]", "High": ":red[Critical]"}
LO, HI = (25 - 15) / 30, (40 - 15) / 30  # Warning / Critical thresholds on the 15-45 colour range
SCALE = [(0, COLORS["Low"]), (LO, COLORS["Low"]), (LO, COLORS["Medium"]), (HI, COLORS["Medium"]), (HI, COLORS["High"]), (1, COLORS["High"])]  # hard bands, not a blended gradient
LAST = R.timestamp.max()
H = 330  # standard chart height
LBL = {"temperature_c": "Temperature (°C)", "vibration_mm_s": "Vibration (mm/s)", "pressure_bar": "Pressure (bar)", "power_kw": "Power (kW)"}
DRV = {k: v.split(" (")[0] for k, v in LBL.items()}


@st.cache_data
def risks(as_of):
    return P.risk_table(R, assets, D["maint"], as_of).join(fac.set_index("facility_id").name.rename("facility"), on="facility_id")


def sensor_fig(a, sensors, start, end, truth=True):
    d = R[(R.asset_id == a) & R.timestamp.between(start, end)]
    fig = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True, subplot_titles=[LBL[s] for s in sensors], vertical_spacing=0.08)
    ep = D["episodes"][(D["episodes"].asset_id == a) & (D["episodes"].end >= start) & (D["episodes"].start <= end)]
    for i, s in enumerate(sensors, 1):
        fig.add_scatter(x=d.timestamp, y=d[s], mode="lines", line=dict(width=1, color="#94a3b8"), name=s, showlegend=False, row=i, col=1)
        al = d[d.alert]
        fig.add_scatter(x=al.timestamp, y=al[s], mode="markers", marker=dict(color=COLORS["High"], size=5),
                        name="Model alert", showlegend=i == 1, row=i, col=1)
        for e in ep.itertuples() if truth else []:
            fig.add_vrect(x0=e.start, x1=e.end, fillcolor="orange", opacity=0.2, line_width=0, row=i, col=1)
    return fig.update_layout(height=max(300, 230 * len(sensors)), margin=dict(t=30, b=10), legend_orientation="h")


TINT = {"Critical": "background-color: rgba(239,68,68,.22); color: #fecaca", "Warning": "background-color: rgba(245,158,11,.2); color: #fde68a"}


def tint(df, cols):
    """Colour severity cells (text label stays, so colour is never the only cue)."""
    return df.style.map(lambda v: TINT.get(v, ""), subset=cols)


def reset():
    st.session_state.update(fsel=[], tsel=[], period=30, asof=LAST.date())


# ---- Header: title left, data freshness right
live_df, fetched = live(fac)
now = pd.Timestamp.now("UTC")
data_state = ("ok", "Current") if now - LAST.tz_localize("UTC") < pd.Timedelta(hours=2) else ("warn", "Delayed · demo snapshot")
wx_state = ("ok", "Current") if not live_df.empty else ("bad", "Sync failed")
h = st.columns([2.2, 3.2, 0.9], vertical_alignment="center")
h[0].title("Maintenance Intelligence")
h[1].markdown(
    f"""<div class="fresh"><div><span class="pill {data_state[0]}">{data_state[1]}</span><br>Asset data as of <b>{LAST:%d %b %Y %H:%M}</b><br>
    Pipeline run <b>{synced:%d %b %H:%M UTC}</b></div>
    <div><span class="pill {wx_state[0]}">{wx_state[1]}</span><br>Weather (Open-Meteo, separate source)<br>
    {'fetched <b>' + fetched.strftime('%d %b %H:%M UTC') + '</b>' if not live_df.empty else 'live feed unavailable'}</div></div>""",
    unsafe_allow_html=True)
with h[2]:
    if st.button("Refresh weather", width="stretch", help="Re-fetches live weather. Asset data is a fixed synthetic snapshot."):
        live.clear()
        st.rerun()
    with st.popover("About the data", width="stretch"):
        st.markdown("**Asset data is fully synthetic** (100 fictional assets, 5 fictional facilities).  \n"
                    f"**Weather:** live conditions from the Open-Meteo forecast API; model history from the Open-Meteo archive "
                    f"({'/'.join(sorted(D['weather'].weather_source.unique()))}).  \n"
                    "**Alerts** = hours flagged by the Isolation Forest. **Critical** = risk score > 40 (Warning 25-40, Normal < 25).")

# ---- Filters: compact, collapsible
ss = st.session_state
n_active = bool(ss.get("fsel")) + bool(ss.get("tsel")) + (ss.get("period", 30) != 30) + (ss.get("asof", LAST.date()) != LAST.date())
with st.expander(f"Filters · {n_active} active" if n_active else "Filters · none active", icon=":material/tune:"):
    c = st.columns([2, 2, 1.4, 1.4, 0.8], vertical_alignment="bottom")
    # empty selection = no filter
    fsel = c[0].multiselect("Facilities", list(fac.name), key="fsel", placeholder="All facilities (type to search)") or list(fac.name)
    tsel = c[1].multiselect("Asset types", sorted(assets.asset_type.unique()), key="tsel",
                            placeholder="All asset types (type to search)") or sorted(assets.asset_type.unique())
    period = c[2].segmented_control("Analysis period", [7, 30, 90], default=30, format_func=lambda d: f"{d} d", key="period") or 30
    day = c[3].date_input("Risk assessment date", LAST.date(), min_value=(R.timestamp.min() + pd.Timedelta(days=8)).date(),
                          max_value=LAST.date(), key="asof", help="Risk is scored on the 7 days before this date. Default: latest data.")
    c[4].button("Reset all", on_click=reset, width="stretch")

as_of = min(pd.Timestamp(day) + pd.Timedelta(hours=23), LAST)
T = risks(as_of)
T = T[T.facility.isin(fsel) & T.asset_type.isin(tsel)]
if T.empty:
    st.warning("No assets match the filters.")
    st.stop()
T = T.assign(status=T.level.map(STATUS), driver_label=T.driver.map(DRV))
n_alerts = int(R.alert[R.asset_id.isin(T.asset_id) & (R.timestamp > as_of - pd.Timedelta(days=period)) & (R.timestamp <= as_of)].sum())
n_crit = int((T.level == "High").sum())

tabs = st.tabs(["Overview", "Sensor Monitoring", "Maintenance Priorities", "Facility Comparison", "Asset Details"])

with tabs[0]:
    for col, (lbl, val, sub) in zip(st.columns(4), [
            ("Assets monitored", len(T), f"Across {T.facility.nunique()} facilities"),
            ("Critical assets", n_crit, f"{n_crit / len(T):.0%} of assets · risk > 40"),
            ("Average risk", f"{T.risk.mean():.1f}", "Scale 0-100 · as of risk date"),
            ("Model alerts", f"{n_alerts:,}", f"Alert-hours, last {period} days")]):
        with col.container(border=True, key=f"card_{lbl[:5]}"):
            st.metric(lbl, val)
            st.caption(sub)

    with st.container(border=True, key="card_prio"):
        st.subheader("Maintenance priorities")
        top = T.head(10).reset_index(drop=True)
        sel = st.dataframe(tint(top.assign(rank=top.index + 1)[["rank", "asset_id", "asset_name", "facility", "driver_label", "peak_z", "risk", "status"]], ["status"]),
                           hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="prio", column_config={
            "rank": st.column_config.NumberColumn("#", width="small"), "asset_id": "Asset ID", "asset_name": "Equipment", "facility": "Facility",
            "driver_label": "Main driver", "peak_z": st.column_config.NumberColumn("Peak deviation (σ)", format="%+.1f"),
            "risk": st.column_config.NumberColumn("Risk score (0-100)", format="%.0f"), "status": "Severity"})
        if sel.selection.rows:
            ss["d"] = top.asset_id[sel.selection.rows[0]]
            st.caption(f"**{ss['d']}**: {top.set_index('asset_id').explanation[ss['d']]}. Open the Asset Details tab for the full view.")
        else:
            st.caption("Ranked by risk score. Select a row to see why it was flagged.")

    c1, c2, c3 = st.columns([1.3, 1, 1.1])
    with c1.container(border=True, key="card_map"):
        st.subheader("Facility risk map")
        g = T.groupby(["facility", "facility_id"]).agg(avg_risk=("risk", "mean"), critical=("level", lambda x: (x == "High").sum()),
                                                        assets=("asset_id", "count")).reset_index().merge(fac, on="facility_id")
        lv = live_df.set_index("facility_id") if not live_df.empty else None
        g["now"] = [f"{lv.temperature_2m[f]:.1f} °C, {lv.relative_humidity_2m[f]:.0f}% RH, wind {lv.wind_speed_10m[f]:.0f} km/h"
                    if lv is not None else "live weather unavailable" for f in g.facility_id]
        st.plotly_chart(px.scatter_map(g, lat="latitude", lon="longitude", size="assets", color="avg_risk", zoom=3.3, center=dict(lat=50, lon=15),
                                       color_continuous_scale=SCALE, range_color=(15, 45), hover_name="facility",
                                       hover_data={"avg_risk": ":.1f", "critical": True, "assets": True, "now": True, "latitude": False, "longitude": False},
                                       size_max=26, map_style="carto-darkmatter",
                                       labels={"avg_risk": "Avg risk", "critical": "Critical assets", "assets": "Assets", "now": "Weather now"})
                        .update_layout(height=H, margin=dict(l=0, r=0, t=0, b=0)), width="stretch")
    with c2.container(border=True, key="card_fac"):
        st.subheader("Average risk by facility")
        st.plotly_chart(px.bar(g.sort_values("avg_risk"), x="avg_risk", y=g.sort_values("avg_risk").facility.str.split().str[0], orientation="h", text_auto=".1f", color="avg_risk",
                               color_continuous_scale=SCALE, range_color=(15, 45), labels={"avg_risk": "Average risk", "facility": ""})
                        .update_layout(height=H, coloraxis_showscale=False, yaxis_title=None, margin=dict(l=0, r=10, t=0, b=0)), width="stretch")
    with c3.container(border=True, key="card_type"):
        st.subheader("Risk by facility × asset type")
        hm = T.pivot_table(index="facility", columns="asset_type", values="risk", aggfunc="mean").round(0).rename(index=lambda s: s.split()[0], columns=lambda s: s.split()[-1])
        st.plotly_chart(px.imshow(hm, text_auto=".0f", color_continuous_scale=SCALE, zmin=15, zmax=45, aspect="auto",
                                  labels={"color": "Avg risk", "x": "", "y": ""})
                        .update_xaxes(side="top").update_layout(height=H, coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0)), width="stretch")

    with st.container(border=True, key="card_live"):
        st.subheader("Live conditions at facilities")
        if live_df.empty:
            st.info("Live weather feed unavailable right now. Historical/synthetic weather is still used by the model.")
        else:
            for col, f in zip(st.columns(5), fac.itertuples()):
                x = live_df[live_df.facility_id == f.facility_id].iloc[0]
                col.metric(f.name, f"{x.temperature_2m:.1f} °C")
                col.caption(f"{x.relative_humidity_2m:.0f}% RH · wind {x.wind_speed_10m:.0f} km/h")
            st.caption("Open-Meteo forecast API, current conditions. Context only; the model trains on the historical archive.")

    with st.expander("Model performance (separate from operational filters)"):
        m, ev = D["metrics"], D["events"]
        st.caption(f"Evaluated on held-out data after {D['split']:%d %b %Y}; baselines, model and threshold were fitted before that date only.")
        st.dataframe(m.style.format("{:.2f}"), width="stretch")
        st.markdown(f"**Event level:** {ev['detected']}/{ev['episodes']} injected fault episodes detected; "
                    f"{ev['failures_warned_6h']}/{ev['failures']} failures alerted ≥6 h earlier (median {ev['median_lead_h']:.0f} h).")
        st.info("**Anomaly detection ≠ failure prediction.** The Isolation Forest flags readings unlike an asset's own normal behaviour; "
                "it does not estimate failure probability. The risk score blends anomalies with criticality and maintenance state to prioritise inspections.")

with tabs[1]:
    c = st.columns([2, 1, 1], vertical_alignment="bottom")
    a = c[0].selectbox("Asset", T.asset_id, format_func=lambda x: f"{x} · {T.set_index('asset_id').asset_name[x]}", key="m")
    s = c[1].selectbox("Sensor", ["All"] + SENSORS)
    truth = c[2].toggle("Show injected truth", False, help="Orange bands = synthetic fault episodes (evaluation labels)")
    st.plotly_chart(sensor_fig(a, SENSORS if s == "All" else [s], as_of - pd.Timedelta(days=period), as_of, truth), width="stretch")
    st.caption("Red dots = Isolation Forest alerts. Idle (standby) hours show as low readings.")

with tabs[2]:
    n = int(st.columns([1, 4])[0].number_input("Show top N assets", 1, len(T), min(15, len(T)), help=f"Shows the N highest-risk assets after filtering ({len(T)} match). Type a number or use +/-."))
    top = T.head(n)
    cols = ["asset_id", "asset_type", "facility", "criticality", "driver_label", "peak_z", "risk", "status", "explanation"]
    st.dataframe(tint(top[cols].reset_index(drop=True), ["status"]), width="stretch", hide_index=True, height=min(740, 38 + 35 * n), column_config={
        "asset_id": "Asset", "asset_type": "Type", "facility": "Facility", "criticality": "Criticality", "status": "Status",
        "driver_label": "Main driver", "peak_z": st.column_config.NumberColumn("Peak deviation (σ)", format="%+.1f"),
        "explanation": st.column_config.TextColumn("Why", width="large"),
        "risk": st.column_config.NumberColumn("Risk (0-100)", format="%.1f")})
    st.download_button("Download ranking (CSV)", top[cols].to_csv(index=False), "maintenance_priorities.csv")
    st.caption("risk = 100 × (0.5·sensor deviation + 0.3·alert share + 0.2·maintenance overdue) × criticality multiplier (0.7–1.0)")

with tabs[3]:
    g = T.groupby("facility").agg(avg_risk=("risk", "mean"), critical=("level", lambda x: (x == "High").sum()),
                                  warning=("level", lambda x: (x == "Medium").sum()), alert_share=("alert_share", "mean"), assets=("asset_id", "count")).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(g.sort_values("avg_risk"), x="avg_risk", y="facility", orientation="h", text_auto=".1f", color="avg_risk", color_continuous_scale=SCALE, range_color=(15, 45),
                          title="Average risk by facility", labels={"avg_risk": "Average risk (0-100)", "facility": ""}).update_traces(textfont_color="#0b1220").update_layout(height=H, coloraxis_showscale=False), width="stretch")
    b.plotly_chart(px.imshow(T.pivot_table(index="facility", columns="asset_type", values="risk", aggfunc="mean").round(1), text_auto=True, color_continuous_scale=SCALE, zmin=15, zmax=45,
                             title="Average risk: facility × equipment type", labels={"color": "Avg risk", "x": "", "y": ""}, aspect="auto").update_traces(textfont_color="#0b1220").update_layout(height=H, margin=dict(l=10, r=30, t=40, b=90)), width="stretch")
    st.dataframe(g.round(2), width="stretch", hide_index=True, column_config={
        "facility": "Facility", "avg_risk": st.column_config.NumberColumn("Average risk", format="%.1f"), "critical": "Critical", "warning": "Warning",
        "alert_share": st.column_config.NumberColumn("Share of hours alerted", format="percent"), "assets": "Assets"})
    w = D["weather"].merge(fac[["facility_id", "name"]], on="facility_id")
    w = w[w.name.isin(fsel)].groupby(["name", pd.Grouper(key="timestamp", freq="D")])[["ambient_temp_c", "humidity_pct"]].mean().reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.line(w, x="timestamp", y="ambient_temp_c", color="name", title="Daily ambient temperature (°C, historical)", labels={"timestamp": "", "ambient_temp_c": "°C", "name": ""}).update_layout(legend=dict(orientation="h", y=-0.2)), width="stretch")
    b.plotly_chart(px.line(w, x="timestamp", y="humidity_pct", color="name", title="Daily relative humidity (%, historical)", labels={"timestamp": "", "humidity_pct": "%", "name": ""}).update_layout(legend=dict(orientation="h", y=-0.2)), width="stretch")
with tabs[4]:
    a = st.selectbox("Asset", T.asset_id, format_func=lambda x: f"{x} · {T.set_index('asset_id').asset_name[x]}", key="d")
    row = T.set_index("asset_id").loc[a]
    with st.container(border=True, key=f"card_{next(_n)}"):
        c = st.columns([1, 2.4, 1, 1])
        c[0].metric("Risk score", f"{row.risk:.1f}")
        c[0].markdown(TAG[row.level])
        c[1].metric("Facility", row.facility)
        c[2].metric("Criticality", f"{row.criticality}/5")
        c[3].metric("Installed", int(row.install_year))
    st.markdown(f"**Diagnostic findings, 7 days to {as_of:%d %b %Y}.** {row.explanation}. The chart below shows the reading right now, which can be lower than the peak.")
    now = R[(R.asset_id == a) & (R.timestamp == as_of)]
    if not now.empty:
        z = now[P.RC].iloc[0].rename(lambda x: DRV[x[3:]]).rename("Deviation (σ)").reset_index()
        z["Level"] = z["Deviation (σ)"].abs().map(lambda v: "Outside ±3σ" if v > 3 else "Within normal range")
        st.plotly_chart(px.bar(z, x="Deviation (σ)", y="index", orientation="h", color="Level", color_discrete_map={"Outside ±3σ": COLORS["High"], "Within normal range": "#64748b"},
                               title=f"Deviation from own baseline right now (6 h mean to {as_of:%d %b %Y %H:%M}); red = outside ±3σ")
                        .add_vline(x=0, line_color="#94a3b8").add_vline(x=3, line_dash="dash", line_color="#475569").add_vline(x=-3, line_dash="dash", line_color="#475569")
                        .update_layout(height=250, yaxis_title=None, showlegend=False), width="stretch")
    st.plotly_chart(sensor_fig(a, SENSORS, as_of - pd.Timedelta(days=30), as_of), width="stretch", key="detail_trend")
    st.subheader("Maintenance history")
    mh = D["maint"][D["maint"].asset_id == a].sort_values("date", ascending=False).drop(columns=["asset_id"])
    st.dataframe(mh.rename(columns=lambda x: x.replace("_eur", " (EUR)").replace("_", " ").capitalize()), width="stretch", hide_index=True)
