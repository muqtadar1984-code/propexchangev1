"""
PATENT ELEMENT E/E1: Indicator Transformation Curve Visualiser

Renders each indicator's exact mathematical curve with a live dot showing
where the current sensor value maps on the transformation.

Layout: 3 charts on row 1 (SHF, ESF, USS) + 2 charts on row 2 (PDP, CI)
Each chart is tall, labelled clearly, with shaded fill and a prominent dot.
"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config.buildings import ASHRAE_BUILDINGS

CURVE_RES = 300

# Colour per indicator — used for line, fill, and dot
COLORS = {
    "SHF": "#2e7d32",   # green  (structural)
    "ESF": "#1565c0",   # blue   (environment)
    "USS": "#e65100",   # orange (usage)
    "PDP": "#6a1b9a",   # purple (aging)
    "CI":  "#00838f",   # teal   (confidence)
}


# ── Chart builder ─────────────────────────────────────────────────────────────
def _make_fig(title: str, subtitle: str, xs, ys,
              dot_x: float, dot_y: float,
              color: str,
              x_label: str, y_label: str,
              height: int = 300,
              x_range=None, y_range=None,
              extra_shapes=None) -> go.Figure:
    """
    Build a Plotly figure with:
      • Shaded area under the curve
      • Prominent live dot with crosshair
      • Clean readable layout matching Streamlit's default light theme
    """
    hex_color = color
    # Convert hex to rgba for fill (Plotly requires rgba, not 8-digit hex)
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    fill_color = f"rgba({r},{g},{b},0.13)"

    fig = go.Figure()

    # Shaded area under curve
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        fill="tozeroy",
        fillcolor=fill_color,
        line=dict(color=color, width=2.5),
        mode="lines",
        hovertemplate=f"{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<extra></extra>",
        name="curve",
    ))

    # Vertical crosshair down to x-axis
    fig.add_shape(
        type="line",
        x0=dot_x, x1=dot_x, y0=0, y1=dot_y,
        line=dict(color=color, width=1.5, dash="dot"),
    )
    # Horizontal crosshair to y-axis
    fig.add_shape(
        type="line",
        x0=xs[0], x1=dot_x, y0=dot_y, y1=dot_y,
        line=dict(color=color, width=1.5, dash="dot"),
    )

    # Any extra reference shapes (zone shading, floor lines, etc.)
    for shape in (extra_shapes or []):
        fig.add_shape(**shape)

    # Live dot — large and clear
    fig.add_trace(go.Scatter(
        x=[dot_x], y=[dot_y],
        mode="markers+text",
        marker=dict(color=color, size=14, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"  {dot_y:.4f}"],
        textposition="middle right",
        textfont=dict(size=12, color=color),
        hovertemplate=(
            f"<b>{x_label}:</b> {dot_x:.4f}<br>"
            f"<b>{y_label}:</b> {dot_y:.4f}<extra></extra>"
        ),
        name="live value",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            font=dict(size=13),
            x=0.0, xanchor="left",
        ),
        height=height,
        showlegend=False,
        margin=dict(l=55, r=20, t=60, b=50),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=11)),
            range=x_range,
            gridcolor="#f0f0f0",
            zerolinecolor="#e0e0e0",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=11)),
            range=y_range,
            gridcolor="#f0f0f0",
            zerolinecolor="#e0e0e0",
            tickfont=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        transition=dict(duration=400, easing="cubic-in-out"),
        uirevision="static",   # tells Plotly to preserve zoom/pan on data updates
    )

    return fig


# ── Individual curve builders ─────────────────────────────────────────────────
def fig_shf(s: float, shf: float, height=300) -> go.Figure:
    xs = np.linspace(0, 1, CURVE_RES)
    ys = 1 - xs ** 2
    return _make_fig(
        title="SHF — Structural Health Factor",
        subtitle="y = 1 − s²   |   s = 0.6·vibration + 0.4·strain",
        xs=xs, ys=ys, dot_x=s, dot_y=shf,
        color=COLORS["SHF"],
        x_label="Structural stress  s", y_label="SHF score",
        height=height,
        x_range=[0, 1], y_range=[0, 1.05],
    )


def fig_esf(e: float, esf: float,
            tau1=0.40, tau2=0.80, epsilon=0.55,
            height=300) -> go.Figure:
    def _esf(x):
        if x < tau1:   return 1.0
        if x <= tau2:  return 1.0 - ((x - tau1) / (tau2 - tau1)) * (1.0 - epsilon)
        return epsilon * 0.8

    xs = np.linspace(0, 1, CURVE_RES)
    ys = np.array([_esf(x) for x in xs])

    zones = [
        dict(type="rect", x0=0,    x1=tau1, y0=0, y1=1.05,
             fillcolor="rgba(76,175,80,0.07)", line_width=0, layer="below"),
        dict(type="rect", x0=tau1, x1=tau2, y0=0, y1=1.05,
             fillcolor="rgba(255,152,0,0.07)", line_width=0, layer="below"),
        dict(type="rect", x0=tau2, x1=1,    y0=0, y1=1.05,
             fillcolor="rgba(244,67,54,0.07)", line_width=0, layer="below"),
    ]
    return _make_fig(
        title="ESF — Environmental Stability Factor",
        subtitle="Piecewise: safe (< 0.4) → linear ramp → floor (> 0.8)",
        xs=xs, ys=ys, dot_x=e, dot_y=esf,
        color=COLORS["ESF"],
        x_label="Environmental stress  e", y_label="ESF score",
        height=height,
        x_range=[0, 1], y_range=[0, 1.05],
        extra_shapes=zones,
    )


def fig_uss(u: float, uss: float,
            dead_zone=0.60, gamma=2.5,
            height=300) -> go.Figure:
    def _uss(x):
        if x <= dead_zone: return 1.0
        excess = (x - dead_zone) / (1.0 - dead_zone)
        return 1.0 - excess ** gamma

    xs = np.linspace(0, 1, CURVE_RES)
    ys = np.array([_uss(x) for x in xs])

    zones = [
        dict(type="rect", x0=0, x1=dead_zone, y0=0, y1=1.05,
             fillcolor="rgba(76,175,80,0.07)", line_width=0, layer="below"),
        dict(type="line", x0=dead_zone, x1=dead_zone, y0=0, y1=1.05,
             line=dict(color="#bdbdbd", width=1.5, dash="dash")),
    ]
    return _make_fig(
        title="USS — Usage Stress Score",
        subtitle=f"Dead zone ≤ {dead_zone}  |  excess^{gamma} penalty beyond",
        xs=xs, ys=ys, dot_x=u, dot_y=uss,
        color=COLORS["USS"],
        x_label="Usage intensity  u", y_label="USS score",
        height=height,
        x_range=[0, 1], y_range=[0, 1.05],
        extra_shapes=zones,
    )


def fig_pdp(p: float, pdp: float,
            alpha=8.0, theta=0.65, floor=0.40,
            height=300) -> go.Figure:
    def _pdp(x):
        try:
            raw = 1.0 / (1.0 + math.exp(alpha * (x - theta)))
        except OverflowError:
            raw = 0.0
        return floor + (1.0 - floor) * raw

    xs = np.linspace(0, 1, CURVE_RES)
    ys = np.array([_pdp(x) for x in xs])

    floor_line = [dict(type="line", x0=0, x1=1, y0=floor, y1=floor,
                       line=dict(color="#bdbdbd", width=1.5, dash="dash"))]
    return _make_fig(
        title="PDP — Predictive Deterioration Penalty",
        subtitle=f"Sigmoid aging  |  floor = {floor}  |  midpoint θ = {theta}",
        xs=xs, ys=ys, dot_x=p, dot_y=pdp,
        color=COLORS["PDP"],
        x_label="Effective age / max life  p", y_label="PDP score",
        height=height,
        x_range=[0, 1], y_range=[0, 1.05],
        extra_shapes=floor_line,
    )


def fig_ci(sigma: float, ci: float,
           lambda_c=3.0,
           height=300) -> go.Figure:
    xs = np.linspace(0, 1.2, CURVE_RES)
    ys = np.exp(-lambda_c * xs)
    return _make_fig(
        title="CI — Confidence Index",
        subtitle=f"y = exp(−λσ)   |   λ = {lambda_c}   |   derived from sensor history std dev",
        xs=xs, ys=ys, dot_x=sigma, dot_y=ci,
        color=COLORS["CI"],
        x_label="Sensor std deviation  σ", y_label="CI score",
        height=height,
        x_range=[0, 1.2], y_range=[0, 1.05],
    )


# ── Slot setup (call ONCE before the while loop) ──────────────────────────────
def setup_curve_slots(selected_buildings: list) -> dict:
    """
    Create the static layout skeleton for all indicator curves and return
    a dict of st.empty() slots that can be updated in-place each cycle.

    Call this ONCE before the while loop.  The returned dict has the shape:
        { bkey: { "kpi": slot, "shf": slot, "esf": slot,
                  "uss": slot, "pdp": slot, "ci": slot, "caption": slot } }
    """
    slots = {}
    for bkey in selected_buildings:
        bname = ASHRAE_BUILDINGS[bkey]["name"]

        st.markdown(
            f"#### 📈 {bname} — Indicator Transformation Curves  *(Patent E/E1)*"
        )

        kpi_slot = st.empty()          # metrics strip

        st.markdown("---")

        # Row 1: SHF | ESF | USS
        r1c1, r1c2, r1c3 = st.columns(3)
        shf_slot = r1c1.empty()
        esf_slot = r1c2.empty()
        uss_slot = r1c3.empty()

        # Row 2: PDP | CI | spacer
        r2c1, r2c2, _ = st.columns(3)
        pdp_slot = r2c1.empty()
        ci_slot  = r2c2.empty()

        caption_slot = st.empty()
        st.divider()

        slots[bkey] = {
            "kpi":     kpi_slot,
            "shf":     shf_slot,
            "esf":     esf_slot,
            "uss":     uss_slot,
            "pdp":     pdp_slot,
            "ci":      ci_slot,
            "caption": caption_slot,
        }

    return slots


# ── Live updater (call each cycle inside the while loop) ─────────────────────
def render_indicator_curves_live(data_logs: dict, selected_buildings: list,
                                 curve_slots: dict) -> None:
    """
    Push updated figures into the pre-created st.empty() slots.
    No keys → no DuplicateElementKey errors.
    Plotly.react() diff-updates the charts → smooth dot movement.
    """
    MAX_LIFE = 120
    CHART_H  = 320

    for bkey in selected_buildings:
        logs = data_logs.get(bkey, [])
        if not logs or bkey not in curve_slots:
            continue

        cycle = logs[-1]
        slots = curve_slots[bkey]
        ps    = cycle["processed_sensors"]
        ind   = cycle["indicators"]
        pdp_d = cycle["valuation"]["pdp_detail"]

        # Compute inputs
        s_shf  = float(np.clip(0.6 * ps["vibration"] + 0.4 * ps["strain"], 0, 1))
        e_esf  = float(np.clip(0.4 * ps["moisture"]  + 0.3 * ps["temperature"]
                               + 0.3 * ps["air_quality"], 0, 1))
        u_uss  = float(np.clip(0.5 * ps["occupancy"] + 0.5 * ps["electrical_load"], 0, 1))
        p_pdp  = float(np.clip(pdp_d.get("effective_age", 0) / MAX_LIFE, 0, 1))
        ci_val = float(ind["CI"])
        sigma_ci = float(np.clip(-math.log(max(ci_val, 1e-9)) / 3.0, 0, 1.2))
        health = cycle["valuation"]["health_factor"]

        # KPI strip
        with slots["kpi"].container():
            kpi_cols = st.columns(6)
            for col, (label, val, color) in zip(kpi_cols, [
                ("SHF", ind["SHF"],  COLORS["SHF"]),
                ("ESF", ind["ESF"],  COLORS["ESF"]),
                ("USS", ind["USS"],  COLORS["USS"]),
                ("PDP", ind["PDP"],  COLORS["PDP"]),
                ("CI",  ci_val,      COLORS["CI"]),
            ]):
                col.metric(label, f"{val:.4f}")
            kpi_cols[5].metric("Health ×", f"{health:.4f}")

        # Stable keys scoped to each building — safe here because every
        # plotly_chart call targets its own pre-created st.empty() slot,
        # so Streamlit sees an in-place update, never a duplicate.
        cfg = {"displayModeBar": False}
        slots["shf"].plotly_chart(fig_shf(s_shf, ind["SHF"], height=CHART_H),
                                  use_container_width=True, config=cfg,
                                  key=f"{bkey}_shf")
        slots["esf"].plotly_chart(fig_esf(e_esf, ind["ESF"], height=CHART_H),
                                  use_container_width=True, config=cfg,
                                  key=f"{bkey}_esf")
        slots["uss"].plotly_chart(fig_uss(u_uss, ind["USS"], height=CHART_H),
                                  use_container_width=True, config=cfg,
                                  key=f"{bkey}_uss")
        slots["pdp"].plotly_chart(fig_pdp(p_pdp, ind["PDP"], height=CHART_H),
                                  use_container_width=True, config=cfg,
                                  key=f"{bkey}_pdp")
        slots["ci"].plotly_chart(fig_ci(sigma_ci, ci_val, height=CHART_H),
                                 use_container_width=True, config=cfg,
                                 key=f"{bkey}_ci")

        # Caption
        with slots["caption"].container():
            st.caption(
                f"Health = SHF × ESF × USS × PDP × CI "
                f"= {ind['SHF']:.4f} × {ind['ESF']:.4f} × {ind['USS']:.4f} "
                f"× {ind['PDP']:.4f} × {ci_val:.4f} = **{health:.4f}**"
            )


# ── Public renderer (used when engine is STOPPED — no while loop) ─────────────
def render_indicator_curves(data_logs: dict, selected_buildings: list,
                            cycle_id: int = 0) -> None:
    """
    Static one-shot render for the stopped-engine view.
    Uses cycle_id in keys so re-runs never collide.
    """
    latest = {
        bk: logs[-1]
        for bk, logs in data_logs.items()
        if logs and bk in selected_buildings
    }
    if not latest:
        st.info("▶️ Start the engine to see live indicator curve positions.")
        return

    MAX_LIFE = 120

    for bkey, cycle in latest.items():
        bname = ASHRAE_BUILDINGS[bkey]["name"]
        ps    = cycle["processed_sensors"]
        ind   = cycle["indicators"]
        pdp_d = cycle["valuation"]["pdp_detail"]

        # Compute combined inputs
        s_shf    = float(np.clip(0.6 * ps["vibration"] + 0.4 * ps["strain"], 0, 1))
        e_esf    = float(np.clip(0.4 * ps["moisture"]  + 0.3 * ps["temperature"]
                                  + 0.3 * ps["air_quality"], 0, 1))
        u_uss    = float(np.clip(0.5 * ps["occupancy"] + 0.5 * ps["electrical_load"], 0, 1))
        p_pdp    = float(np.clip(pdp_d.get("effective_age", 0) / MAX_LIFE, 0, 1))
        ci_val   = float(ind["CI"])
        sigma_ci = float(np.clip(-math.log(max(ci_val, 1e-9)) / 3.0, 0, 1.2))
        health   = cycle["valuation"]["health_factor"]

        # ── Header + KPI strip ───────────────────────────────────────────────
        st.markdown(
            f"#### 📈 {bname} — Indicator Transformation Curves  *(Patent E/E1)*"
        )

        kpi_cols = st.columns(6)
        for col, (label, val, color) in zip(kpi_cols, [
            ("SHF", ind["SHF"],  COLORS["SHF"]),
            ("ESF", ind["ESF"],  COLORS["ESF"]),
            ("USS", ind["USS"],  COLORS["USS"]),
            ("PDP", ind["PDP"],  COLORS["PDP"]),
            ("CI",  ci_val,      COLORS["CI"]),
        ]):
            col.metric(label, f"{val:.4f}")
        kpi_cols[5].metric("Health ×", f"{health:.4f}")

        st.markdown("---")

        CHART_H = 320

        # ── Row 1: SHF | ESF | USS ───────────────────────────────────────────
        # Stable keys are safe here because render_indicator_curves() is called
        # once per st.rerun() — keys are seen exactly once per script execution.
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(fig_shf(s_shf, ind["SHF"], height=CHART_H),
            use_container_width=True, config={"displayModeBar": False},
            key=f"{bkey}_shf")
        c2.plotly_chart(fig_esf(e_esf, ind["ESF"], height=CHART_H),
            use_container_width=True, config={"displayModeBar": False},
            key=f"{bkey}_esf")
        c3.plotly_chart(fig_uss(u_uss, ind["USS"], height=CHART_H),
            use_container_width=True, config={"displayModeBar": False},
            key=f"{bkey}_uss")

        # ── Row 2: PDP | CI | (spacer) ──────────────────────────────────────
        c4, c5, _ = st.columns(3)
        c4.plotly_chart(fig_pdp(p_pdp, ind["PDP"], height=CHART_H),
            use_container_width=True, config={"displayModeBar": False},
            key=f"{bkey}_pdp")
        c5.plotly_chart(fig_ci(sigma_ci, ci_val, height=CHART_H),
            use_container_width=True, config={"displayModeBar": False},
            key=f"{bkey}_ci")

        # ── Computation chain caption ────────────────────────────────────────
        st.caption(
            f"Health = SHF × ESF × USS × PDP × CI "
            f"= {ind['SHF']:.4f} × {ind['ESF']:.4f} × {ind['USS']:.4f} "
            f"× {ind['PDP']:.4f} × {ci_val:.4f} = **{health:.4f}**"
        )
        st.divider()
