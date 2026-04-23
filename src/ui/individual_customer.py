"""
Individual Customer Tab — TwinVal POC
7 sub-tabs: Portfolio Overview · My Properties · Valuation Timeline ·
            Sensors & Health · Valuations & Reports · Exchange & Trading ·
            Account & Settings

All data is simulated.  No real sensor feeds.
"""

import hashlib
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PROPERTIES = {
    "villa_damansara": {
        "name": "Villa Damansara",
        "short": "Villa Damansara",
        "type": "Residential Bungalow",
        "built_year": 2008,
        "land_area_sqft": 4_200,
        "builtup_sqft": 3_800,
        "land_value": 850_000,
        "structure_value": 650_000,
        "SHF": 0.88, "ESF": 0.91, "USS": 0.14, "PDP": 0.93, "CI": 0.87,
        "sensors_active": 8,
        "baseline_market_value": 1_450_000,   # independent market appraisal
        "sensors": [
            {"name": "Temperature (Ground Fl)",    "reading": "28.4 °C",   "category": "Environmental", "status": "OK",      "last_tx": "2s ago",  "battery": 94},
            {"name": "Humidity",                   "reading": "67 %",      "category": "Environmental", "status": "OK",      "last_tx": "2s ago",  "battery": 88},
            {"name": "Structural Vibration (Beam)","reading": "0.12 mm/s", "category": "Structural",    "status": "OK",      "last_tx": "5s ago",  "battery": 76},
            {"name": "Foundation Strain Gauge",    "reading": "142 μɛ",    "category": "Structural",    "status": "Warning", "last_tx": "8s ago",  "battery": 61},
            {"name": "Occupancy Counter",          "reading": "4 persons", "category": "Usage",         "status": "OK",      "last_tx": "2s ago",  "battery": 95},
            {"name": "Electrical Load",            "reading": "4.2 kW",    "category": "Usage",         "status": "OK",      "last_tx": "3s ago",  "battery": 82},
            {"name": "Roof Moisture",              "reading": "12.1 %",    "category": "Safety",        "status": "OK",      "last_tx": "12s ago", "battery": 71},
            {"name": "Air Quality (CO₂)",          "reading": "612 ppm",   "category": "Environmental", "status": "OK",      "last_tx": "4s ago",  "battery": 90},
        ],
        "anomalies": [
            {"timestamp": "2026-04-22 14:33", "sensor": "Foundation Strain Gauge",    "event": "Reading exceeded upper threshold (142 μɛ > 130 μɛ)", "severity": "Warning"},
            {"timestamp": "2026-04-21 08:12", "sensor": "Structural Vibration (Beam)","event": "Brief spike to 0.28 mm/s — returned to normal",        "severity": "Info"},
            {"timestamp": "2026-04-18 19:45", "sensor": "Humidity",                   "event": "Above 80 % for > 2 h — possible condensation",          "severity": "Warning"},
            {"timestamp": "2026-04-10 11:20", "sensor": "Roof Moisture",              "event": "Elevated after rainfall — normal recovery pattern",      "severity": "Info"},
        ],
        "maintenance_events": [
            {"day_offset": -72, "label": "Roof inspection"},
            {"day_offset": -38, "label": "Sensor calibration"},
            {"day_offset": -15, "label": "Foundation check"},
        ],
    },
    "shophouse_georgetown": {
        "name": "Shophouse Georgetown",
        "short": "Shophouse GT",
        "type": "Commercial Shophouse",
        "built_year": 1975,
        "land_area_sqft": 1_800,
        "builtup_sqft": 3_600,
        "land_value": 1_200_000,
        "structure_value": 480_000,
        "SHF": 0.72, "ESF": 0.78, "USS": 0.31, "PDP": 0.71, "CI": 0.74,
        "sensors_active": 6,
        "baseline_market_value": 1_550_000,
        "sensors": [
            {"name": "Footfall Counter",          "reading": "127 /hr",    "category": "Usage",         "status": "OK",      "last_tx": "3s ago",  "battery": 88},
            {"name": "HVAC Load",                 "reading": "8.7 kW",     "category": "Usage",         "status": "Warning", "last_tx": "15s ago", "battery": 44},
            {"name": "Temperature (Shopfront)",   "reading": "31.2 °C",    "category": "Environmental", "status": "OK",      "last_tx": "3s ago",  "battery": 72},
            {"name": "Electrical Load",           "reading": "11.4 kW",    "category": "Usage",         "status": "OK",      "last_tx": "4s ago",  "battery": 65},
            {"name": "Air Quality (PM2.5)",       "reading": "38 μg/m³",   "category": "Environmental", "status": "Warning", "last_tx": "22s ago", "battery": 38},
            {"name": "Structural Vibration (Wall)","reading": "0.31 mm/s", "category": "Structural",    "status": "OK",      "last_tx": "6s ago",  "battery": 80},
        ],
        "anomalies": [
            {"timestamp": "2026-04-23 07:15", "sensor": "Air Quality (PM2.5)", "event": "PM2.5 at 38 μg/m³ — approaching advisory threshold",      "severity": "Warning"},
            {"timestamp": "2026-04-22 13:40", "sensor": "HVAC Load",           "event": "Abnormal load pattern — maintenance recommended",           "severity": "Warning"},
            {"timestamp": "2026-04-20 09:05", "sensor": "HVAC Load",           "event": "HVAC cycling inefficiency detected",                        "severity": "Warning"},
            {"timestamp": "2026-04-15 16:30", "sensor": "Structural Vibration (Wall)", "event": "Elevated vibration during nearby roadworks",        "severity": "Info"},
            {"timestamp": "2026-04-08 12:00", "sensor": "Electrical Load",     "event": "Peak load 14.2 kW — exceeded normal operating range",      "severity": "Warning"},
        ],
        "maintenance_events": [
            {"day_offset": -80, "label": "HVAC service"},
            {"day_offset": -43, "label": "Electrical audit"},
            {"day_offset": -20, "label": "Structural survey"},
        ],
    },
}

PROP_KEYS   = list(PROPERTIES.keys())
PROP_NAMES  = [PROPERTIES[k]["name"] for k in PROP_KEYS]

# ─────────────────────────────────────────────────────────────────────────────
# FORMULA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def health_factor(p: dict) -> float:
    return p["SHF"] * p["ESF"] * (1 - p["USS"]) * p["PDP"] * p["CI"]


def rtpmv(p: dict) -> float:
    return p["land_value"] + p["structure_value"] * health_factor(p)


def trading_status(hf: float, ci: float) -> str:
    if hf >= 0.75 and ci >= 0.80:
        return "ACTIVE"
    if hf >= 0.55 or ci >= 0.65:
        return "RESTRICTED"
    return "HALTED"


def spread_pct(hf: float, ci: float) -> float:
    """Wider spread when CI or health is lower."""
    return max(0.01, 0.02 + (1 - ci) * 0.06 + (1 - hf) * 0.04)


def bid_ask(rtpmv_val: float, hf: float, ci: float):
    sp = spread_pct(hf, ci)
    return rtpmv_val * (1 - sp / 2), rtpmv_val * (1 + sp / 2), sp * 100


def rm(v: float) -> str:
    """Format as Malaysian Ringgit."""
    return f"RM {v:,.0f}"


def pct_arrow(v: float) -> str:
    arrow = "▲" if v >= 0 else "▼"
    color = "green" if v >= 0 else "red"
    return f'<span style="color:{color}">{arrow} {abs(v):.2f}%</span>'


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATED TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

def _rtpmv_series(prop_key: str, days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic 90-day RTPMV series with noise + event dips."""
    p   = PROPERTIES[prop_key]
    rng = np.random.default_rng(seed + PROP_KEYS.index(prop_key))
    hf  = health_factor(p)
    base = rtpmv(p)

    today     = datetime.now().date()
    dates     = [today - timedelta(days=days - i - 1) for i in range(days)]
    noise     = rng.normal(0, base * 0.003, days).cumsum()
    vals      = base + noise

    # Apply maintenance event dips (slight dip then recovery)
    for ev in p["maintenance_events"]:
        idx = days + ev["day_offset"]
        if 0 <= idx < days:
            dip = base * 0.008
            for j in range(idx, min(idx + 5, days)):
                vals[j] -= dip * (1 - (j - idx) / 5)

    # Land / structure split
    land_vals   = [p["land_value"]] * days
    struct_vals = [v - p["land_value"] for v in vals]

    return pd.DataFrame({
        "date":      dates,
        "rtpmv":     vals,
        "land":      land_vals,
        "structure": struct_vals,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATED VALUATION REPORTS (hash chain)
# ─────────────────────────────────────────────────────────────────────────────

def _valuation_reports(prop_key: str) -> list[dict]:
    p   = PROPERTIES[prop_key]
    hf  = health_factor(p)
    rv  = rtpmv(p)
    today = datetime.now()

    records = []
    prev_hash = "0" * 64
    rng = random.Random(42 + PROP_KEYS.index(prop_key))
    for i in range(14):
        ts    = today - timedelta(days=13 - i, hours=rng.randint(0, 23))
        noise = rng.uniform(-0.01, 0.01)
        val   = rv * (1 + noise * (14 - i) / 14)
        blob  = f"{prop_key}|{ts.isoformat()}|{val:.2f}|{prev_hash}"
        h     = hashlib.sha256(blob.encode()).hexdigest()
        records.append({
            "Report ID":     f"TVR-{2026040000 + i + PROP_KEYS.index(prop_key) * 14:08d}",
            "Property":      p["short"],
            "Date":          ts.strftime("%Y-%m-%d %H:%M"),
            "RTPMV":         rm(val),
            "Hash (12)":     h[:12],
            "Chain Status":  "VALID",
            "Indicators":    f"SHF {p['SHF']} · ESF {p['ESF']} · USS {p['USS']} · PDP {p['PDP']} · CI {p['CI']}",
            "Trading":       trading_status(hf, p["CI"]),
        })
        prev_hash = h
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

def _init_ic_state():
    if "ic_selected_prop" not in st.session_state:
        st.session_state["ic_selected_prop"] = PROP_KEYS[0]
    if "ic_timeline_range" not in st.session_state:
        st.session_state["ic_timeline_range"] = "90d"
    if "ic_show_land" not in st.session_state:
        st.session_state["ic_show_land"] = True
    if "ic_show_struct" not in st.session_state:
        st.session_state["ic_show_struct"] = True
    if "ic_audit_mode" not in st.session_state:
        st.session_state["ic_audit_mode"] = False


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hf_colour(hf: float) -> str:
    if hf >= 0.80:
        return "#2e7d32"
    if hf >= 0.60:
        return "#e65100"
    return "#c62828"


def _status_badge(status: str) -> str:
    colours = {"ACTIVE": "#2e7d32", "RESTRICTED": "#e65100", "HALTED": "#c62828"}
    icons   = {"ACTIVE": "🟢", "RESTRICTED": "🟡", "HALTED": "🔴"}
    c = colours.get(status, "#666")
    icon = icons.get(status, "⚪")
    return (
        f'<span style="background:{c};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.78rem;font-weight:600;">'
        f'{icon} {status}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 1: PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def _render_portfolio_overview():
    all_hf   = {k: health_factor(PROPERTIES[k]) for k in PROP_KEYS}
    all_rv   = {k: rtpmv(PROPERTIES[k]) for k in PROP_KEYS}
    total_rv = sum(all_rv.values())
    avg_hf   = sum(all_hf[k] * all_rv[k] for k in PROP_KEYS) / total_rv
    tot_sens = sum(PROPERTIES[k]["sensors_active"] for k in PROP_KEYS)

    # ── Top-level metrics ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Portfolio Value", rm(total_rv))
    with c2:
        st.metric("Portfolio Health Score", f"{avg_hf:.2f}",
                  help="Weighted average Health Factor across all properties")
    with c3:
        st.metric("Properties Monitored", len(PROP_KEYS))
    with c4:
        st.metric("Active Sensors", tot_sens)

    st.markdown("---")

    # ── Alert summary ────────────────────────────────────────────────────────
    warn_count = sum(
        sum(1 for s in PROPERTIES[k]["sensors"] if s["status"] == "Warning")
        for k in PROP_KEYS
    )
    anom_count = sum(
        sum(1 for a in PROPERTIES[k]["anomalies"] if a["severity"] == "Warning")
        for k in PROP_KEYS
    )
    ac1, ac2, ac3 = st.columns(3)
    ac1.error(f"⚠️  Sensor Warnings: **{warn_count}**")
    ac2.warning(f"📋 Anomaly Alerts: **{anom_count}**")
    ac3.success("✅ Info Events: **3**")

    st.markdown("---")

    # ── Portfolio RTPMV trend (30 days, stacked) ─────────────────────────────
    st.markdown("#### Portfolio Value — Last 30 Days")
    fig = go.Figure()
    colours = {"villa_damansara": "#1565c0", "shophouse_georgetown": "#2e7d32"}
    for k in PROP_KEYS:
        df = _rtpmv_series(k, days=90).tail(30)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rtpmv"],
            mode="lines", name=PROPERTIES[k]["short"],
            line=dict(color=colours[k], width=2),
            fill="tozeroy",
            fillcolor=colours[k].replace(")", ",0.08)").replace("rgb", "rgba")
            if colours[k].startswith("rgb") else colours[k] + "14",
        ))
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
        yaxis_title="RTPMV (RM)", legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Property Cards")

    # ── Property cards ───────────────────────────────────────────────────────
    cols = st.columns(len(PROP_KEYS))
    for i, k in enumerate(PROP_KEYS):
        p      = PROPERTIES[k]
        hf     = all_hf[k]
        rv     = all_rv[k]
        status = trading_status(hf, p["CI"])
        colour = _hf_colour(hf)
        with cols[i]:
            st.markdown(f"""
            <div style="background:#f8f9fa;border-radius:8px;padding:1rem;
                        border:1px solid #e0e0e0;border-top:3px solid {colour};">
              <div style="font-size:0.75rem;color:#666;margin-bottom:4px;">
                {p['type']} · Built {p['built_year']}
              </div>
              <div style="font-weight:700;font-size:1.05rem;margin-bottom:8px;">
                {p['name']}
              </div>
              <div style="font-size:1.3rem;font-weight:700;color:#0d47a1;">
                {rm(rv)}
              </div>
              <div style="margin:6px 0;">
                <span style="color:{colour};font-weight:600;">
                  Health: {hf:.2f}
                </span>
              </div>
              <div>{_status_badge(status)}</div>
              <div style="font-size:0.75rem;color:#666;margin-top:6px;">
                {p['sensors_active']} sensors active
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 2: MY PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

def _render_my_properties():
    selected = st.session_state["ic_selected_prop"]
    p        = PROPERTIES[selected]
    hf       = health_factor(p)
    rv       = rtpmv(p)
    chron_age = datetime.now().year - p["built_year"]
    eff_age   = chron_age * (1 - hf * 0.4)   # effective age declines with health

    st.markdown(f"#### {p['name']}")

    # ── Property metadata ────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Type",           p["type"])
    m2.metric("Built Year",     str(p["built_year"]))
    m3.metric("Land Area",      f"{p['land_area_sqft']:,} sqft")
    m4.metric("Built-up Area",  f"{p['builtup_sqft']:,} sqft")

    st.markdown("---")

    # ── RTPMV breakdown ──────────────────────────────────────────────────────
    v1, v2, v3 = st.columns(3)
    with v1:
        st.metric("Land Value",      rm(p["land_value"]))
    with v2:
        structure_contribution = p["structure_value"] * hf
        st.metric("Structure Value (adj.)", rm(structure_contribution),
                  help="Structure Value × Health Factor")
    with v3:
        delta = rv - p["baseline_market_value"]
        delta_pct = delta / p["baseline_market_value"] * 100
        st.metric("Current RTPMV", rm(rv),
                  delta=f"{delta_pct:+.1f}% vs baseline",
                  delta_color="normal")

    st.markdown(f"""
    <div style="background:#e8f5e9;border-left:4px solid #2e7d32;
                padding:0.5rem 1rem;border-radius:4px;margin:0.5rem 0;
                font-size:0.85rem;color:#1b5e20;">
      <strong>Formula:</strong>
      RTPMV = Land Value + Structure Value × Health Factor<br/>
      = {rm(p['land_value'])} + {rm(p['structure_value'])} × {hf:.4f}
      = <strong>{rm(rv)}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Radar chart — 5 indicators ───────────────────────────────────────────
    col_radar, col_ages = st.columns(2)

    with col_radar:
        st.markdown("**Health Factor Breakdown**")
        cats   = ["SHF", "ESF", "1−USS", "PDP", "CI"]
        values = [p["SHF"], p["ESF"], 1 - p["USS"], p["PDP"], p["CI"]]
        fig    = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=cats + [cats[0]],
            fill="toself",
            line_color="#1565c0",
            fillcolor="rgba(21,101,192,0.15)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1], showticklabels=True,
                                       tickfont=dict(size=9))),
            height=300, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#f8f9fa",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Health Factor", f"{hf:.4f}")

    with col_ages:
        st.markdown("**Age Analysis**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Chronological Age", "Effective Age"],
            y=[chron_age, eff_age],
            marker_color=["#1565c0", _hf_colour(hf)],
            text=[f"{chron_age} yrs", f"{eff_age:.1f} yrs"],
            textposition="outside",
        ))
        fig2.update_layout(
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
            yaxis_title="Years", showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.metric("Sensors Active", p["sensors_active"])
        last_ts = (datetime.now() - timedelta(seconds=2)).strftime("%H:%M:%S")
        st.caption(f"Last sensor data: {last_ts} (live)")


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 3: VALUATION TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

def _render_valuation_timeline():
    selected = st.session_state["ic_selected_prop"]
    p        = PROPERTIES[selected]

    # Controls
    cc1, cc2, cc3, cc4 = st.columns([2, 1, 1, 1])
    with cc1:
        rng_choice = st.radio("Date range", ["30d", "90d", "All"], horizontal=True,
                              key="ic_timeline_range")
    with cc2:
        show_land   = st.checkbox("Land component",      value=True, key="ic_show_land")
    with cc3:
        show_struct = st.checkbox("Structure component", value=True, key="ic_show_struct")

    days_map = {"30d": 30, "90d": 90, "All": 90}
    df = _rtpmv_series(selected).tail(days_map[rng_choice])

    fig = go.Figure()

    # Main RTPMV line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rtpmv"],
        mode="lines", name="RTPMV",
        line=dict(color="#0d47a1", width=2.5),
    ))

    if show_land:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["land"],
            mode="lines", name="Land Value",
            line=dict(color="#2e7d32", width=1.5, dash="dot"),
        ))

    if show_struct:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["structure"],
            mode="lines", name="Structure (adj.)",
            line=dict(color="#e65100", width=1.5, dash="dot"),
        ))

    # Maintenance event markers
    today = datetime.now().date()
    for ev in p["maintenance_events"]:
        ev_date = today + timedelta(days=ev["day_offset"])
        if df["date"].min() <= ev_date <= df["date"].max():
            fig.add_vline(x=str(ev_date), line_dash="dash",
                          line_color="#6a1b9a", line_width=1,
                          annotation_text=ev["label"],
                          annotation_position="top left",
                          annotation_font_size=10)

    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
        yaxis_title="Value (RM)", xaxis_title="Date",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Key metrics table ────────────────────────────────────────────────────
    st.markdown("#### Key Valuation Metrics")
    rv_vals  = df["rtpmv"].values
    chg_30   = (rv_vals[-1] - rv_vals[max(0, len(rv_vals)-30)]) / rv_vals[max(0, len(rv_vals)-30)] * 100
    vol_idx  = np.std(rv_vals) / np.mean(rv_vals) * 100

    km1, km2, km3, km4, km5 = st.columns(5)
    km1.metric("Current RTPMV",  rm(rv_vals[-1]))
    km2.metric("Highest RTPMV",  rm(rv_vals.max()))
    km3.metric("Lowest RTPMV",   rm(rv_vals.min()))
    km4.metric("30-Day Change",  f"{chg_30:+.2f}%",
               delta_color="normal" if chg_30 >= 0 else "inverse")
    km5.metric("Volatility Index", f"{vol_idx:.2f}%",
               help="Coefficient of variation over selected period")


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 4: SENSORS & HEALTH
# ─────────────────────────────────────────────────────────────────────────────

def _render_sensors_health():
    selected = st.session_state["ic_selected_prop"]
    p        = PROPERTIES[selected]
    ci       = p["CI"]

    # ── CI Gauge ─────────────────────────────────────────────────────────────
    col_gauge, col_cats = st.columns([1, 2])

    with col_gauge:
        st.markdown("**Confidence Index (CI) Gauge**")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ci,
            number={"font": {"size": 36}, "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar":  {"color": _hf_colour(ci)},
                "steps": [
                    {"range": [0, 0.60], "color": "#ffebee"},
                    {"range": [0.60, 0.80], "color": "#fff8e1"},
                    {"range": [0.80, 1],    "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "#0d47a1", "width": 3},
                    "thickness": 0.75, "value": ci,
                },
            },
            title={"text": "CI", "font": {"size": 14}},
        ))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=0),
                          paper_bgcolor="#f8f9fa")
        st.plotly_chart(fig, use_container_width=True)

    with col_cats:
        st.markdown("**Sensor Category Summary**")
        cats  = ["Structural", "Environmental", "Usage", "Safety"]
        for cat in cats:
            cat_sens = [s for s in p["sensors"] if s["category"] == cat]
            warn     = sum(1 for s in cat_sens if s["status"] == "Warning")
            ok_count = len(cat_sens) - warn
            icon     = "⚠️" if warn else "✅"
            st.markdown(
                f"{icon} **{cat}** — {ok_count} OK"
                + (f", {warn} Warning" if warn else ""),
            )

    st.markdown("---")

    # ── Sensor table ─────────────────────────────────────────────────────────
    st.markdown("#### Live Sensor Feed")
    sensor_df = pd.DataFrame(p["sensors"])
    sensor_df["Status"] = sensor_df["status"].apply(
        lambda s: "⚠️ " + s if s == "Warning" else ("❌ " + s if s == "Offline" else "✅ " + s)
    )
    sensor_df["Battery"] = sensor_df["battery"].apply(lambda b: f"{b}%")

    display_cols = ["name", "category", "reading", "Status", "last_tx", "Battery"]
    col_map = {
        "name": "Sensor", "category": "Category", "reading": "Reading",
        "last_tx": "Last TX",
    }
    st.dataframe(
        sensor_df[display_cols].rename(columns=col_map),
        use_container_width=True, hide_index=True,
    )

    # ── Anomaly log ──────────────────────────────────────────────────────────
    st.markdown("#### Anomaly Log")
    anom_rows = []
    for a in p["anomalies"]:
        icon = "⚠️" if a["severity"] == "Warning" else "ℹ️"
        anom_rows.append({
            "Time":     a["timestamp"],
            "Sensor":   a["sensor"],
            "Event":    a["event"],
            "Severity": f'{icon} {a["severity"]}',
        })
    st.dataframe(pd.DataFrame(anom_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 5: VALUATIONS & REPORTS
# ─────────────────────────────────────────────────────────────────────────────

def _render_valuations_reports():
    selected = st.session_state["ic_selected_prop"]
    p        = PROPERTIES[selected]
    reports  = _valuation_reports(selected)

    # ── Chain integrity banner ───────────────────────────────────────────────
    st.success("🔒 Chain Integrity: All 14 records verified — hash chain intact")

    # ── Audit mode toggle ────────────────────────────────────────────────────
    audit = st.toggle("Audit Mode — show frozen indicator snapshot per report",
                      value=st.session_state["ic_audit_mode"],
                      key="ic_audit_mode")

    st.markdown(f"#### Valuation Reports — {p['name']}")

    # ── Reports table ────────────────────────────────────────────────────────
    cols_base   = ["Report ID", "Date", "RTPMV", "Hash (12)", "Chain Status", "Trading"]
    cols_audit  = cols_base + ["Indicators"]
    display     = cols_audit if audit else cols_base

    df = pd.DataFrame(reports)[display]
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Download placeholders ────────────────────────────────────────────────
    st.markdown("#### Report Downloads")
    dl_cols = st.columns(min(4, len(reports)))
    for i, rep in enumerate(reports[:4]):
        with dl_cols[i]:
            st.button(f"📄 {rep['Report ID']}", key=f"dl_{rep['Report ID']}",
                      use_container_width=True)
    st.info("📥 PDF download available in production deployment. "
            "Reports are cryptographically sealed at generation time.")

    # ── Valuation token summary ──────────────────────────────────────────────
    st.markdown("#### Valuation Token Summary (Latest)")
    latest = reports[-1]
    t1, t2, t3 = st.columns(3)
    t1.metric("Token ID",       latest["Report ID"])
    t2.metric("Property",       latest["Property"])
    t3.metric("RTPMV at Issue", latest["RTPMV"])

    st.markdown(f"""
    <div style="background:#f8f9fa;border:1px solid #e0e0e0;border-radius:6px;
                padding:0.8rem;font-family:monospace;font-size:0.78rem;color:#333;">
      <strong>Hash:</strong> {latest['Hash (12)']}…<br/>
      <strong>Issued:</strong> {latest['Date']}<br/>
      <strong>Indicators:</strong> {latest['Indicators']}<br/>
      <strong>Trading Status at Issuance:</strong> {latest['Trading']}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 6: EXCHANGE & TRADING
# ─────────────────────────────────────────────────────────────────────────────

def _render_exchange_trading():
    st.markdown("#### Exchange & Trading — All Properties")

    # ── Per-property exchange cards ──────────────────────────────────────────
    cols = st.columns(len(PROP_KEYS))
    all_circuit_breaks = []

    for i, k in enumerate(PROP_KEYS):
        p      = PROPERTIES[k]
        hf     = health_factor(p)
        rv     = rtpmv(p)
        bid, ask, sp_pct = bid_ask(rv, hf, p["CI"])
        status = trading_status(hf, p["CI"])
        colour = _hf_colour(hf)

        # Simulate price volatility (random walk seed)
        rng   = np.random.default_rng(99 + PROP_KEYS.index(k))
        vol24 = float(rng.uniform(2.1, 6.8))
        circuit_breaker = vol24 > 5.0
        if circuit_breaker:
            all_circuit_breaks.append(p["name"])

        with cols[i]:
            st.markdown(f"**{p['name']}**")
            st.markdown(_status_badge(status), unsafe_allow_html=True)
            st.metric("RTPMV", rm(rv))

            ex1, ex2 = st.columns(2)
            ex1.metric("Bid",  rm(bid),  help="Bid = RTPMV × (1 − spread/2)")
            ex2.metric("Ask",  rm(ask),  help="Ask = RTPMV × (1 + spread/2)")

            st.markdown(f"**Spread:** {sp_pct:.1f}%  ·  "
                        f"**Health:** {hf:.2f}  ·  **CI:** {p['CI']:.2f}")

            vol_col = "#c62828" if vol24 > 5 else "#2e7d32"
            st.markdown(
                f'<span style="color:{vol_col};font-size:0.82rem;">'
                f'24h Volatility: {vol24:.1f}%'
                + (" ⚡ CIRCUIT BREAKER" if circuit_breaker else "")
                + "</span>",
                unsafe_allow_html=True,
            )

    # ── Circuit breaker warning ──────────────────────────────────────────────
    if all_circuit_breaks:
        st.error(
            f"⚡ Circuit breaker triggered for: **{', '.join(all_circuit_breaks)}** "
            "— 24h price volatility exceeded 5%. Trading temporarily restricted."
        )

    st.markdown("---")

    # ── Exchange parameter history (last 10 updates) ─────────────────────────
    st.markdown("#### Exchange Parameter History — Last 10 Updates")
    selected = st.session_state["ic_selected_prop"]
    p   = PROPERTIES[selected]
    hf  = health_factor(p)
    rv  = rtpmv(p)

    hist_rows = []
    rng2 = np.random.default_rng(55 + PROP_KEYS.index(selected))
    for j in range(10):
        ts_offset = timedelta(hours=j * 2 + rng2.integers(0, 2))
        ts = datetime.now() - ts_offset
        noise_rv = rv * (1 + rng2.uniform(-0.005, 0.005))
        noise_hf = hf * (1 + rng2.uniform(-0.01, 0.01))
        b, a, sp = bid_ask(noise_rv, noise_hf, p["CI"])
        hist_rows.append({
            "Time":    ts.strftime("%Y-%m-%d %H:%M"),
            "RTPMV":   rm(noise_rv),
            "Bid":     rm(b),
            "Ask":     rm(a),
            "Spread":  f"{sp:.1f}%",
            "HF":      f"{noise_hf:.4f}",
            "Status":  trading_status(noise_hf, p["CI"]),
        })

    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUB-TAB 7: ACCOUNT & SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

def _render_account_settings():
    # ── Customer profile ─────────────────────────────────────────────────────
    st.markdown("#### Customer Profile")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Name",          "Muqtadar Musawir Quraishi")
    p2.metric("Subscription",  "Professional")
    p3.metric("Properties",    str(len(PROP_KEYS)))
    p4.metric("Account Since", "January 2026")

    st.markdown("---")

    # ── Sensor installation status ───────────────────────────────────────────
    st.markdown("#### Sensor Installation Status")
    for k in PROP_KEYS:
        p = PROPERTIES[k]
        total_cats = 4   # Structural, Environmental, Usage, Safety
        inst_cats  = sum(
            1 for cat in ["Structural", "Environmental", "Usage", "Safety"]
            if any(s["category"] == cat for s in p["sensors"])
        )
        pct = inst_cats / total_cats
        st.markdown(f"**{p['name']}** — {p['sensors_active']} sensors active "
                    f"across {inst_cats}/{total_cats} categories")
        st.progress(pct)

    st.markdown("---")

    # ── Settings columns ─────────────────────────────────────────────────────
    col_notif, col_share = st.columns(2)

    with col_notif:
        st.markdown("#### Notification Preferences")
        st.toggle("📧 Email alerts",            value=True,  key="notif_email")
        st.toggle("📱 SMS alerts",              value=False, key="notif_sms")
        st.toggle("🔧 Maintenance reminders",   value=True,  key="notif_maint")
        st.toggle("📊 Weekly valuation digest", value=True,  key="notif_digest")

    with col_share:
        st.markdown("#### Data Sharing Settings")
        st.toggle("🏦 Share with lender",   value=False, key="share_lender")
        st.toggle("🛡️ Share with insurer",  value=False, key="share_insurer")
        st.toggle("🏛️ Share with JPPH",     value=False, key="share_jpph")
        st.info("Data shared only via encrypted, permissioned channels. "
                "You control access at all times.")

    st.markdown("---")

    # ── Subscription management ──────────────────────────────────────────────
    st.markdown("#### Subscription Management")
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        st.markdown("""
        **Professional Plan — RM 499/month**
        - Up to 3 properties · Full sensor suite · Weekly reports · Lender sharing · Exchange listing
        """)
    with sc2:
        st.button("Manage Subscription", use_container_width=True)
    st.info("📋 Subscription management available in production. "
            "Contact support@twinval.com during the pilot period.")


# ─────────────────────────────────────────────────────────────────────────────
# PROPERTY SELECTOR (shared across sub-tabs)
# ─────────────────────────────────────────────────────────────────────────────

def _render_property_selector():
    """Compact inline selector — persists across sub-tabs via session_state."""
    col_sel, col_info = st.columns([2, 5])
    with col_sel:
        choice = st.selectbox(
            "Selected property",
            options=PROP_KEYS,
            format_func=lambda k: PROPERTIES[k]["name"],
            key="ic_selected_prop",
            label_visibility="collapsed",
        )
    with col_info:
        p   = PROPERTIES[choice]
        hf  = health_factor(p)
        rv  = rtpmv(p)
        status = trading_status(hf, p["CI"])
        st.markdown(
            f"📍 **{p['name']}** &nbsp;·&nbsp; "
            f"{rm(rv)} &nbsp;·&nbsp; "
            f"HF {hf:.2f} &nbsp;·&nbsp; "
            + _status_badge(status),
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_individual_customer_tab():
    """Top-level entry called from app.py — renders the full Individual Customer tab."""
    _init_ic_state()

    # ── Demo Mode banner ─────────────────────────────────────────────────────
    st.warning(
        "⚠️ **Demo Mode** — All sensor readings and valuations are simulated "
        "for illustration purposes. No real sensor data is used.",
        icon="⚠️",
    )

    st.markdown("### 👤 Individual Customer Dashboard")

    # ── Sub-tabs ─────────────────────────────────────────────────────────────
    (
        tab_overview,
        tab_props,
        tab_timeline,
        tab_sensors,
        tab_reports,
        tab_exchange,
        tab_account,
    ) = st.tabs([
        "🏠 Portfolio Overview",
        "🏘️ My Properties",
        "📈 Valuation Timeline",
        "📡 Sensors & Health",
        "📋 Valuations & Reports",
        "⚖️ Exchange & Trading",
        "⚙️ Account & Settings",
    ])

    with tab_overview:
        _render_portfolio_overview()

    with tab_props:
        _render_property_selector()
        st.markdown("---")
        _render_my_properties()

    with tab_timeline:
        _render_property_selector()
        st.markdown("---")
        _render_valuation_timeline()

    with tab_sensors:
        _render_property_selector()
        st.markdown("---")
        _render_sensors_health()

    with tab_reports:
        _render_property_selector()
        st.markdown("---")
        _render_valuations_reports()

    with tab_exchange:
        _render_exchange_trading()

    with tab_account:
        _render_account_settings()
