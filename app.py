"""Maintenance Intelligence dashboard. Run: streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

from mi import pipeline as P
from mi.data_gen import SENSORS
from mi.weather import get_current

st.set_page_config("Maintenance Intelligence", ":material/monitoring:", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container { padding: 2.4rem 2rem 2rem; max-width: 1600px; } @media (max-width: 768px) { .block-container { padding: 2.2rem .75rem 2rem; } }</style>", unsafe_allow_html=True)  # no native setting for page padding
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


ss = st.session_state
try:
    D, synced = load()
except Exception as e:  # first-run pipeline failure: clear message instead of a stack trace
    st.error("Maintenance data could not be prepared, so no risk results are available. Reload the page to try again. If it keeps failing, the data files may be missing or damaged.")
    st.caption(f"Technical detail: {type(e).__name__}: {e}")
    st.stop()
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


def sensor_fig(a, sensors, start, end, truth=False):
    d = R[(R.asset_id == a) & R.timestamp.between(start, end)]
    fig = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True, subplot_titles=[LBL[s] for s in sensors], vertical_spacing=0.08)
    ep = D["episodes"][(D["episodes"].asset_id == a) & (D["episodes"].end >= start) & (D["episodes"].start <= end)]
    for i, s in enumerate(sensors, 1):
        fig.add_scatter(x=d.timestamp, y=d[s], mode="lines", line=dict(width=1, color="#94a3b8"), name=s, showlegend=False, row=i, col=1)
        al = d[d.alert]
        fig.add_scatter(x=al.timestamp, y=al[s], mode="markers", marker=dict(color=COLORS["High"], size=4),
                        name="Model alert", showlegend=i == 1, row=i, col=1)
        for e in ep.itertuples() if truth else []:
            fig.add_vrect(x0=e.start, x1=e.end, fillcolor="orange", opacity=0.2, line_width=0, row=i, col=1)
    return fig.update_layout(height=max(300, 230 * len(sensors)), margin=dict(t=30, b=10), legend_orientation="h")


TINT = {"Critical": "background-color: rgba(239,68,68,.22); color: #fecaca", "Warning": "background-color: rgba(245,158,11,.2); color: #fde68a"}


def tint(df, cols):
    """Colour severity cells (text label stays, so colour is never the only cue) and the 7-day change (red up, teal down)."""
    sty = df.style.map(lambda v: TINT.get(v, ""), subset=cols)
    if "change" in df.columns:
        sty = sty.map(lambda v: "" if pd.isna(v) or v == 0 else ("color: #fca5a5" if v > 0 else "color: #5eead4"), subset=["change"])
    return sty


def show(fig, **kw):
    """Plotly chart with an accessible name taken from the chart title."""
    st.plotly_chart(fig, **{"width": "stretch", "alt": fig.layout.title.text or "Time series chart", **kw})


def table(data, **kw):
    st.dataframe(data, **{"width": "stretch", "alt": "Data table", **kw})


def go(asset):
    ss["d"] = ss["m"] = asset  # one selected asset for both the monitoring and details tabs
    ss["tab"] = "Asset details"


def sync_asset(src):
    """Keep the Sensor monitoring and Asset details pickers on the same asset."""
    ss["m" if src == "d" else "d"] = ss[src]


def open_asset(key):
    """Row selected in a priorities table: jump to that asset in the Asset details tab."""
    rows = ss[key].selection.rows
    if rows:
        go(ss[f"{key}_ids"][rows[0]])

def reset():
    st.session_state.update(fsel=[], tsel=[], period="30", asof=LAST.date())


# ---- Header: title left, data freshness right
live_df, fetched = live(fac)
now = pd.Timestamp.now("UTC")
data_ok = now - LAST.tz_localize("UTC") < pd.Timedelta(hours=2)
h = st.columns([2, 1.4, 1.4, 0.8], vertical_alignment="center")
h[0].title("Maintenance Intelligence")
with h[1]:
    st.badge("Live data" if data_ok else "Historical simulation · synthetic data", color="green" if data_ok else "blue")
    st.caption(f"Asset data as of **{LAST:%d %b %Y %H:%M}**")
with h[2]:
    st.badge("Live weather" if not live_df.empty else "Sync failed", color="green" if not live_df.empty else "red")
    st.caption("Weather (Open-Meteo), context only" + ("" if not live_df.empty else "  \nlive feed unavailable"))
with h[3]:
    with st.popover("About · refresh", width="stretch"):
        if st.button("Refresh weather", width="stretch", help="Re-fetches live weather. Asset data is a fixed synthetic snapshot."):
            live.clear()
            st.rerun()
        st.markdown(f"**Pipeline last ran** {synced:%d %b %Y %H:%M UTC}. **Weather fetched** " + (f"{fetched:%d %b %Y %H:%M UTC}" if not live_df.empty else "n/a (feed unavailable)") + ".  \n"
                    "**Asset data is fully synthetic** (100 fictional assets, 5 fictional facilities).  \n"
                    f"**Weather:** live conditions from the Open-Meteo forecast API; model history from the Open-Meteo archive "
                    f"({'/'.join(sorted(D['weather'].weather_source.unique()))}).  \n"
                    "**Alerts** = hours flagged by the Isolation Forest. **Critical** = risk above 40, **Warning** = above 25 up to 40, **Normal** = 25 or less.  \n"
                    "**What the score is:** it ranks assets for inspection from anomaly detection, criticality and maintenance history. It is not a failure prediction and has not been validated on real plant data.  \n"
                    "**Weather** is shown for context only; the risk results come from the historical simulation.  \n"
                    "**Sharing:** the page address holds your sites, filters, tab and selected asset. Copy it to share this exact view.")

# ---- Shareable view: filters and asset live in the URL (?site=..&type=..&period=7&date=2025-06-20&asset=A061)
qp = st.query_params
if "seeded" not in ss:
    ss.seeded = True
    sites, types = qp.get_all("site"), qp.get_all("type")
    ss.fsel, ss.tsel = [v for v in sites if v in set(fac.name)], [v for v in types if v in set(assets.asset_type)]
    ss.stale_link = len(ss.fsel) < len(sites) or len(ss.tsel) < len(types) or (qp.get("period") not in (None, "7", "30", "90"))
    if qp.get("period") in ("7", "30", "90"):
        ss.period = qp["period"]
    try:
        ss.asof = max(min(pd.Timestamp(qp["date"]).date(), LAST.date()), (R.timestamp.min() + pd.Timedelta(days=8)).date())
    except (KeyError, ValueError):
        pass

# ---- Sites are always visible (empty = all sites; the line under the controls says exactly what is in view)
fsel = st.multiselect("Sites", list(fac.name), key="fsel", placeholder="All sites (select one or more to review them together)") or list(fac.name)
if ss.pop("stale_link", False):
    st.info("Some saved selections in this link are no longer available. The view has been updated.")
n_active = bool(ss.get("tsel")) + (ss.get("period", "30") != "30") + (ss.get("asof", LAST.date()) != LAST.date())
with st.expander(f"More filters · {n_active} active" if n_active else "More filters · none active", icon=":material/tune:"):
    c = st.columns([2, 1.4, 1.4, 0.8], vertical_alignment="bottom")
    tsel = c[0].multiselect("Asset types", sorted(assets.asset_type.unique()), key="tsel",
                            placeholder="All asset types (type to search)") or sorted(assets.asset_type.unique())
    period = int((c[1].segmented_control("Chart and alert window", ["7", "30", "90"], default="30", format_func=lambda d: f"{d} d", key="period", help="Sets the window for charts and the alert count. Risk itself is always scored on the last 7 days.") or "30"))
    day = c[2].date_input("Risk assessment date", LAST.date(), min_value=(R.timestamp.min() + pd.Timedelta(days=8)).date(),
                          max_value=LAST.date(), key="asof", help="Risk is always scored on the 7 days before this date. Default: latest data.")
    c[3].button("Reset all", on_click=reset, width="stretch")
as_of = min(pd.Timestamp(day) + pd.Timedelta(hours=23), LAST)
T = risks(as_of)
T = T[T.facility.isin(fsel) & T.asset_type.isin(tsel)]
if T.empty:
    st.info("No assets match these filters. Select other sites or clear the filters.")
    st.button("Clear filters", on_click=reset)
    st.stop()
T = T.assign(status=T.level.map(STATUS), driver_label=T.driver.map(DRV))
if "seeded_asset" not in ss:
    ss.seeded_asset = True
    if qp.get("asset") in set(T.asset_id):
        ss.d = ss.m = qp["asset"]
    elif qp.get("asset"):
        ss.asset_notice = True  # shared link points to an asset outside the current selection
for k in ("d", "m"):
    if ss.get(k) not in set(T.asset_id):
        ss.pop(k, None)  # asset left the current selection: fall back to the highest-risk one
n_alerts = int(R.alert[R.asset_id.isin(T.asset_id) & (R.timestamp > as_of - pd.Timedelta(days=period)) & (R.timestamp <= as_of)].sum())
n_crit = int((T.level == "High").sum())
WEEK = pd.Timedelta(days=7)
prev = risks(as_of - WEEK).set_index("asset_id") if as_of - WEEK >= R.timestamp.min() + pd.Timedelta(days=8) else None  # risk 7 days earlier, same assets
T = T.assign(change=T.risk - T.asset_id.map(prev.risk) if prev is not None else float("nan"))
n_crit_prev = int((prev.loc[prev.index.isin(T.asset_id), "level"] == "High").sum()) if prev is not None else None
avg_prev = prev.loc[prev.index.isin(T.asset_id), "risk"].mean() if prev is not None else None

_by_id = T.set_index("asset_id")


def fmt_asset(x):
    """Dropdown label: shows risk and severity so the assets that need attention stand out (list is sorted by risk, highest first)."""
    return f"{x} · {_by_id.asset_type[x]} · {_by_id.facility[x].split()[0]} · risk {_by_id.risk[x]:.0f} ({_by_id.status[x]})"


sites_txt = "All sites" if T.facility.nunique() == len(fac) else ", ".join(sorted(T.facility.unique()))
st.caption(f"**In view:** {sites_txt} · {len(T)} assets · assessed {as_of:%d %b %Y}")
if ss.pop("asset_notice", False):
    st.info("The asset in this link is not in the current selection, so the highest-risk asset is shown instead.")
tabs = st.tabs(["Overview", "Sensor monitoring", "Maintenance priorities", "Facility comparison", "Asset details"], key="tab", bind="query-params")

with tabs[0]:
    st.subheader("Inspect next")
    flagged = T[T.level != "Low"].head(3)  # only assets that actually need attention
    if flagged.empty:
        st.success("No assets are flagged in this view: every asset is Normal (risk 25 or less).")
    for col, r in zip(st.columns(3), flagged.itertuples()):
        with col.container(border=True, gap="xsmall"):
            st.badge(STATUS[r.level], color={"High": "red", "Medium": "orange", "Low": "green"}[r.level])
            st.markdown(f"**{r.asset_id} · {r.asset_type}**")
            st.caption(r.facility)
            st.metric("Risk score", f"{r.risk:.1f} / 100", None if pd.isna(r.change) else f"{r.change:+.1f} pts vs 7 d earlier", delta_color="inverse")
            st.divider()
            left, right = st.columns(2)
            left.caption("Main driver")
            left.markdown(f"**{r.driver_label} {r.peak_z:+.1f}σ**")
            right.caption("Hours alerted (last 7 d)")
            right.markdown(f"**{r.alert_share:.0%}**")
            st.button("Investigate", key=f"go_{r.asset_id}", on_click=go, args=(r.asset_id,), icon=":material/arrow_forward:", width="stretch")

    for col, (lbl, val, sub, delta) in zip(st.columns(4), [
            ("Assets assessed", len(T), f"Across {T.facility.nunique()} facilities", None),
            ("Critical assets", n_crit, f"{n_crit / len(T):.0%} of assets · risk > 40", None if n_crit_prev is None else ("No change vs 7 d earlier" if n_crit_prev == n_crit else f"{n_crit - n_crit_prev:+d} vs 7 d earlier")),
            ("Average risk", f"{T.risk.mean():.1f}", "Scale 0-100", None if avg_prev is None else f"{T.risk.mean() - avg_prev:+.1f} vs 7 d earlier"),
            ("Model alerts", f"{n_alerts:,}", f"Alert-hours, last {period} days", None)]):
        with col.container(border=True, height=185):
            st.metric(lbl, val, delta, delta_color="off" if delta and delta.startswith("No change") else "inverse", delta_arrow="off" if delta and delta.startswith("No change") else "auto")
            st.caption(sub)
    with st.container(border=True):
        st.subheader("Maintenance priorities")
        top = T.head(10).reset_index(drop=True)
        ss["prio_ids"] = list(top.asset_id)
        table(tint(top.assign(rank=top.index + 1)[["rank", "asset_id", "asset_type", "facility", "risk", "change", "status", "driver_label", "peak_z"]], ["status"]),
                           hide_index=True, width="stretch", on_select=lambda: open_asset("prio"), selection_mode="single-row", key="prio", alt="Top 10 maintenance priorities, ranked by risk score. Select a row to open the asset.", column_config={
            "rank": st.column_config.NumberColumn("#", width="small"), "asset_id": st.column_config.TextColumn("Asset", width="small"), "asset_type": "Equipment", "facility": st.column_config.TextColumn("Facility", width="large"),
            "risk": st.column_config.NumberColumn("Risk score", format="%.1f"), "change": st.column_config.NumberColumn("7-day change (pts)", format="%+.1f"), "status": "Severity",
            "driver_label": "Main driver", "peak_z": st.column_config.NumberColumn("Peak deviation (σ)", format="%+.1f")})
        st.caption("Ranked by risk score. Select a row to open that asset in Asset details.")

    if prev is None:
        st.caption("Seven-day comparison is unavailable for this date: there is not enough earlier data.")

    c1, c2, c3 = st.columns([1.3, 1, 1.1])
    with c1.container(border=True):
        st.subheader("Facility risk map")
        g = T.groupby(["facility", "facility_id"]).agg(avg_risk=("risk", "mean"), critical=("level", lambda x: (x == "High").sum()),
                                                        assets=("asset_id", "count")).reset_index().merge(fac, on="facility_id")
        lv = live_df.set_index("facility_id") if not live_df.empty else None
        g["now"] = [f"{lv.temperature_2m[f]:.1f} °C, {lv.relative_humidity_2m[f]:.0f}% RH, wind {lv.wind_speed_10m[f]:.0f} km/h"
                    if lv is not None and f in lv.index else "live weather unavailable" for f in g.facility_id]
        show(px.scatter_map(g, lat="latitude", lon="longitude", size="assets", color="avg_risk", zoom=3.3, center=dict(lat=50, lon=15),
                                       color_continuous_scale=SCALE, range_color=(15, 45), hover_name="facility",
                                       hover_data={"avg_risk": ":.1f", "critical": True, "assets": True, "now": True, "latitude": False, "longitude": False},
                                       size_max=26, map_style="carto-darkmatter",
                                       labels={"avg_risk": "Avg risk", "critical": "Critical assets", "assets": "Assets", "now": "Weather now"})
                        .update_layout(height=H, margin=dict(l=0, r=0, t=0, b=0)), width="stretch")
    with c2.container(border=True):
        st.subheader("Assets needing attention")
        nc = T.assign(short=T.facility.str.split().str[0]).groupby(["short", "status"]).size().reset_index(name="assets")
        nc = nc[nc.status != "Normal"]
        if nc.empty:
            st.caption("No critical or warning assets in this view.")
        else:
            show(px.bar(nc, x="assets", y="short", color="status", orientation="h", color_discrete_map={"Critical": COLORS["High"], "Warning": COLORS["Medium"]},
                        category_orders={"status": ["Critical", "Warning"]}, labels={"assets": "Assets", "short": "", "status": ""})
                 .update_xaxes(dtick=1).update_layout(height=H, margin=dict(l=0, r=10, t=0, b=0), legend=dict(orientation="h", y=1.08)), alt="Number of critical and warning assets per facility")
    with c3.container(border=True):
        st.subheader("Risk by facility × asset type")
        hm = T.pivot_table(index="facility", columns="asset_type", values="risk", aggfunc="mean").round(0).rename(index=lambda s: s.split()[0], columns=lambda s: s.split()[-1])
        show(px.imshow(hm, text_auto=".0f", color_continuous_scale=SCALE, zmin=15, zmax=45, aspect="auto",
                                  labels={"color": "Avg risk", "x": "", "y": ""})
                        .update_xaxes(side="top").update_layout(height=H, coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0)), width="stretch")
        st.caption("Cell colour = average risk: teal 25 or below, amber above 25 to 40, red above 40.")

    with st.container(border=True):
        st.subheader("Live conditions at facilities")
        if live_df.empty:
            st.info("Live weather feed unavailable right now. Historical/synthetic weather is still used by the model.")
        else:
            for col, f in zip(st.columns(5), fac.itertuples()):
                row_w = live_df[live_df.facility_id == f.facility_id]
                if row_w.empty:
                    col.metric(f.name, "n/a")
                    col.caption("Weather unavailable for this site")
                    continue
                x = row_w.iloc[0]
                col.metric(f.name, f"{x.temperature_2m:.1f} °C")
                col.caption(f"{x.relative_humidity_2m:.0f}% RH · wind {x.wind_speed_10m:.0f} km/h")
            st.caption("Open-Meteo forecast API, current conditions. Context only; the model trains on the historical archive.")

    with st.expander("Model performance (separate from operational filters)"):
        m, ev = D["metrics"], D["events"]
        st.caption(f"Evaluated on held-out data after {D['split']:%d %b %Y}; baselines, model and threshold were fitted before that date only.")
        table(m.style.format("{:.2f}"), alt="Precision, recall and F1 of the Isolation Forest and a simple rule on held-out data", width="stretch")
        st.markdown(f"**Event level:** {ev['detected']}/{ev['episodes']} injected fault episodes detected; "
                    f"{ev['failures_warned_6h']}/{ev['failures']} failures alerted ≥6 h earlier (median {ev['median_lead_h']:.0f} h).")
        st.info("**Anomaly detection ≠ failure prediction.** The Isolation Forest flags readings unlike an asset's own normal behaviour; "
                "it does not estimate failure probability. The risk score blends anomalies with criticality and maintenance state to prioritise inspections.")

with tabs[1]:
    c = st.columns([2, 1, 1], vertical_alignment="bottom")
    a = c[0].selectbox("Asset", list(T.asset_id), format_func=fmt_asset, key="m", on_change=sync_asset, args=("m",), help="Type to search. The list follows the Facilities and Asset types filters at the top.")
    main_sensor = T.set_index("asset_id").driver[a]  # open on the sensor driving this asset's risk, not four tall charts
    s = c[1].selectbox("Sensor", ["All"] + SENSORS, index=1 + SENSORS.index(main_sensor), format_func=lambda x: x if x == "All" else LBL[x])
    truth = c[2].toggle("Show simulated fault periods", False, help="Orange bands mark the faults injected into the synthetic data. They are evaluation labels, not model output.")
    win = R[(R.asset_id == a) & R.timestamp.between(as_of - pd.Timedelta(days=period), as_of)]
    no_data = win.empty or win[SENSORS].isna().all(axis=None)
    if no_data:
        st.warning("No sensor readings are available for this asset and period, so its condition cannot be assessed here.")
    elif not win.running.any():
        st.info("This asset was on standby during this period. Low readings do not necessarily indicate a fault.")
    st.caption(f"{(as_of - pd.Timedelta(days=period)):%d %b} to {as_of:%d %b %Y}. Red dots are hours when the model flagged the asset as a whole (all four sensors together), so they appear on every chart, even where that sensor looks normal.")
    if not no_data:
        show(sensor_fig(a, SENSORS if s == "All" else [s], as_of - pd.Timedelta(days=period), as_of, truth), alt=f"Sensor readings for {a} over the last {period} days with model alerts")
    st.caption("Standby hours read low.")
    ss["ovl"] = [x for x in ss.get("ovl", []) if x in set(T.asset_id) and x != a]
    also = st.multiselect("Overlay other assets on one sensor (optional)", [x for x in T.asset_id if x != a], max_selections=4, format_func=fmt_asset, key="ovl",
                          placeholder="Select up to 4 assets, then choose a single sensor above", help="Type to search. Options are sorted by risk, highest first.")
    if also and s == "All":
        st.info("Choose a single sensor above to overlay the selected assets on one chart.")
    elif also:
        ov = R[R.asset_id.isin([a] + also) & R.timestamp.between(as_of - pd.Timedelta(days=period), as_of) & R.running]
        ov = ov.assign(Asset=ov.asset_id).set_index("timestamp").groupby("Asset")[s].resample("6h").mean().reset_index()
        show(px.line(ov, x="timestamp", y=s, color="Asset", labels={"timestamp": "", s: LBL[s]}, title=f"{LBL[s]}: selected assets, 6-hour mean while running").update_layout(height=H, legend=dict(orientation="h", y=-0.2, title=None)),
             alt=f"{LBL[s]} for the selected assets over the last {period} days")

with tabs[2]:
    n = int(st.columns([1, 4])[0].number_input("Show top N assets", 1, len(T), min(15, len(T)), help=f"Shows the N highest-risk assets after filtering ({len(T)} match). Type a number or use +/-."))
    top = T.head(n).reset_index(drop=True).assign(rank=lambda d: d.index + 1)
    cols = ["rank", "asset_id", "asset_type", "facility", "risk", "change", "status", "driver_label", "peak_z"]
    ss["prio2_ids"] = list(top.asset_id)
    table(tint(top[cols], ["status"]), width="stretch", hide_index=True, height=min(740, 38 + 35 * n), on_select=lambda: open_asset("prio2"), selection_mode="single-row", key="prio2", alt="Maintenance priorities with driver, deviation, risk and severity. Select a row to open the asset.", column_config={
        "rank": st.column_config.NumberColumn("#", width="small"), "asset_id": st.column_config.TextColumn("Asset", width="small"), "asset_type": "Equipment", "facility": st.column_config.TextColumn("Facility", width="large"), "status": "Status",
        "driver_label": "Main driver", "peak_z": st.column_config.NumberColumn("Peak deviation (σ)", format="%+.1f"), "change": st.column_config.NumberColumn("7-day change (pts)", format="%+.1f"),
        "explanation": st.column_config.TextColumn("Why", width="large"),
        "risk": st.column_config.NumberColumn("Risk (0-100)", format="%.1f")})
    with st.expander("Why assets received these scores", icon=":material/notes:"):
        st.markdown("\n".join(f"- **{r.asset_id}** ({r.status}): {r.explanation}" for r in top.itertuples()))
    st.download_button("Download ranking (CSV)", top[cols + ["criticality", "explanation"]].to_csv(index=False), f"maintenance_priorities_{as_of:%Y-%m-%d}.csv")
    st.caption("Select a row to open that asset in Asset details.")
    with st.expander("How the risk score is calculated", icon=":material/calculate:"):
        st.markdown("Risk = 100 × (0.5 × sensor deviation + 0.3 × share of hours alerted + 0.2 × time since maintenance) × criticality multiplier (about 0.64 to 1.0). "
                    "It ranks assets for inspection; it is not a failure prediction.")

with tabs[3]:
    g = T.groupby("facility").agg(avg_risk=("risk", "mean"), critical=("level", lambda x: (x == "High").sum()),
                                  warning=("level", lambda x: (x == "Medium").sum()), alert_share=("alert_share", "mean"), assets=("asset_id", "count")).reset_index()
    a, b = st.columns(2)
    with a.container(border=True):
        st.subheader("Assets needing attention by facility")
        nc = T.groupby(["facility", "status"]).size().reset_index(name="assets")
        nc = nc[nc.status != "Normal"]
        if nc.empty:
            st.caption("No critical or warning assets in this view.")
        else:
            show(px.bar(nc, x="assets", y="facility", color="status", orientation="h", text_auto=True,
                        color_discrete_map={"Critical": COLORS["High"], "Warning": COLORS["Medium"]}, category_orders={"status": ["Critical", "Warning"]}, labels={"assets": "Assets", "facility": "", "status": ""})
                 .update_traces(textfont_color="#0b1220").update_xaxes(dtick=1).update_layout(height=H, margin=dict(l=0, r=10, t=0, b=0), legend=dict(orientation="h", y=1.08)), alt="Number of critical and warning assets per facility")
    with b.container(border=True):
        st.subheader("Average risk by facility and equipment type")
        pv = T.pivot_table(index="facility", columns="asset_type", values="risk", aggfunc="mean").round(1)
        show(px.imshow(pv.rename(index=lambda s: s.split()[0], columns=lambda s: s.replace(" ", "<br>", 1)), text_auto=True, color_continuous_scale=SCALE, zmin=15, zmax=45,
                       labels={"color": "Avg risk", "x": "", "y": ""}, aspect="auto").update_traces(textfont_color="#0b1220")
             .update_xaxes(side="top", tickangle=0).update_layout(height=H, coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0)), alt="Average risk for each facility and equipment type")
        st.caption("Cell colour = average risk of the assets in that cell: teal 25 or below, amber above 25 to 40, red above 40.")
    table(g.sort_values(["critical", "avg_risk"], ascending=False).round(2), alt="Facility summary: average risk, critical and warning assets, share of hours alerted", hide_index=True, column_config={
        "facility": "Facility", "avg_risk": st.column_config.NumberColumn("Average risk", format="%.1f"), "critical": "Critical", "warning": "Warning",
        "alert_share": st.column_config.NumberColumn("Share of hours alerted", format="percent"), "assets": "Assets"})
    st.download_button("Download facility summary (CSV)", g.sort_values(["critical", "avg_risk"], ascending=False).round(2).to_csv(index=False), f"facility_summary_{as_of:%Y-%m-%d}.csv")
    w = D["weather"].merge(fac[["facility_id", "name"]], on="facility_id")
    w = w[w.name.isin(fsel)].groupby(["name", pd.Grouper(key="timestamp", freq="D")])[["ambient_temp_c", "humidity_pct"]].mean().reset_index()
    a, b = st.columns(2)
    for col, y, title, unit in [(a, "ambient_temp_c", "Daily ambient temperature (°C)", "°C"), (b, "humidity_pct", "Daily relative humidity (%)", "%")]:
        with col.container(border=True):
            st.subheader(title)
            show(px.line(w, x="timestamp", y=y, color="name", labels={"timestamp": "", y: unit, "name": ""}).update_layout(height=H, margin=dict(l=0, r=10, t=10, b=0), legend=dict(orientation="h", y=-0.15)),
                 alt=f"{title}, historical, per facility")
with tabs[4]:
    a = st.selectbox("Asset", list(T.asset_id), format_func=fmt_asset, key="d", on_change=sync_asset, args=("d",), help="Type to search. The list follows the Facilities and Asset types filters at the top.")
    row = T.set_index("asset_id").loc[a]
    with st.container(border=True):
        st.markdown(f"**{row.asset_name}** · {row.facility}")
        c = st.columns(3)
        c[0].metric("Risk score", f"{row.risk:.1f}", None if pd.isna(row.change) else f"{row.change:+.1f} vs 7 d earlier", delta_color="inverse")
        c[0].markdown(TAG[row.level])
        c[1].metric("Criticality", f"{row.criticality}/5")
        c[2].metric("Installed", int(row.install_year))
    st.markdown(f"**{'Why flagged' if row.level != 'Low' else 'Reading summary (not flagged)'}.** {row.explanation}.")
    ss["cmp"] = [x for x in ss.get("cmp", []) if x in set(T.asset_id) and x != a]
    others = st.multiselect("Compare with other assets (optional)", [x for x in T.asset_id if x != a], max_selections=4, format_func=fmt_asset, key="cmp",
                            placeholder="Select up to 4 assets, for example the next highest risks", help="Type to search. Options are sorted by risk, highest first.")
    if others:
        cmp = T.set_index("asset_id").loc[[a] + others].reset_index()
        table(tint(cmp[["asset_id", "asset_type", "facility", "risk", "change", "status", "driver_label", "peak_z", "days_since_maint"]], ["status"]), hide_index=True,
              alt="Comparison of the selected assets: risk, 7-day change, severity, main driver and days since maintenance", column_config={
                  "asset_id": "Asset", "asset_type": "Equipment", "facility": st.column_config.TextColumn("Facility", width="large"), "risk": st.column_config.NumberColumn("Risk score", format="%.1f"),
                  "change": st.column_config.NumberColumn("7-day change (pts)", format="%+.1f"), "status": "Severity", "driver_label": "Main driver",
                  "peak_z": st.column_config.NumberColumn("Peak deviation (σ)", format="%+.1f"), "days_since_maint": st.column_config.NumberColumn("Days since maintenance", format="%d")})
    w7 = R[(R.asset_id == a) & (R.timestamp > as_of - WEEK) & (R.timestamp <= as_of)]
    if not w7.empty:
        peak = w7[P.RC].apply(lambda s: s.loc[s.abs().idxmax()] if s.notna().any() else float("nan"))
        last = w7[P.RC].iloc[-1]
        z = pd.DataFrame({"Sensor": [DRV[c[3:]] for c in P.RC] * 2, "Deviation (σ)": list(peak) + list(last),
                          "Reading": ["Peak, last 7 days"] * 4 + [f"Latest (6 h mean to {as_of:%d %b %H:%M})"] * 4})
        show(px.bar(z, x="Deviation (σ)", y="Sensor", color="Reading", orientation="h", barmode="group",
                    color_discrete_map={"Peak, last 7 days": "#e5edf5", f"Latest (6 h mean to {as_of:%d %b %H:%M})": "#64748b"},
                    title="Deviation from own baseline: 7-day peak vs latest reading (dashed lines = ±3σ)")
             .add_vline(x=0, line_color="#94a3b8").add_vline(x=3, line_dash="dash", line_color="#475569").add_vline(x=-3, line_dash="dash", line_color="#475569")
             .update_layout(height=300, yaxis_title=None, legend=dict(orientation="h", y=-0.2, title=None)), alt="Peak versus latest deviation from baseline for each sensor")
        worst = last.abs().idxmax()
        st.markdown(f"**Latest reading** (6 h mean to {as_of:%d %b %Y %H:%M}): largest deviation is **{DRV[worst[3:]]} {last[worst]:+.1f}σ**, "
                    f"{'outside' if abs(last[worst]) > 3 else 'within'} the ±3σ normal range. **7-day peak:** {DRV[peak.abs().idxmax()[3:]]} {peak[peak.abs().idxmax()]:+.1f}σ.")
        st.caption("A high peak with a low latest value means the condition has eased since; both high means it is still ongoing. \"Latest\" is the end of the assessed period, not today.")
    st.caption("Sensor trend: always the last 30 days (the chart window setting applies to Sensor monitoring).")
    show(sensor_fig(a, SENSORS, as_of - pd.Timedelta(days=30), as_of), width="stretch", key="detail_trend")
    st.subheader("Maintenance history")
    mh = D["maint"][D["maint"].asset_id == a].sort_values("date", ascending=False).drop(columns=["asset_id"])
    table(mh.rename(columns=lambda x: x.replace("_eur", " (EUR)").replace("_", " ").capitalize()), alt="Maintenance history of the selected asset", hide_index=True)

# ---- write the current view back to the URL so it can be shared
for key, val in {"site": ss.get("fsel"), "type": ss.get("tsel"), "period": ss.get("period") if ss.get("period") != "30" else None,
                 "date": str(day) if day != LAST.date() else None, "asset": ss.get("d") if ss.get("tab") in ("Asset details", "Sensor monitoring") else None}.items():
    if val:
        qp[key] = val
    elif key in qp:
        del qp[key]
