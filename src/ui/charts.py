"""
Sensor and indicator time-series charts.
"""

import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.buildings import ASHRAE_BUILDINGS


def render_sensor_charts(data_logs):
    """Render sensor and indicator time series — only selected buildings."""
    st.markdown(
        '<div class="exchange-header">📈 Technical Indicators & Sensor Feed</div>',
        unsafe_allow_html=True,
    )

    selected = st.session_state.selected_buildings
    building_keys = [k for k in selected if k in data_logs and len(data_logs[k]) > 0]

    if not building_keys:
        st.info("No data yet for selected buildings.")
        return

    tabs = st.tabs([ASHRAE_BUILDINGS[k]["name"][:20] for k in building_keys])

    for tab, bkey in zip(tabs, building_keys):
        with tab:
            logs = data_logs[bkey]
            if len(logs) < 2:
                st.info("Collecting data...")
                continue

            recent = logs[-60:]

            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'SHF (Structural)', 'ESF (Environment)',
                    'USS (Usage)', 'PDP (Deterioration)',
                    'CI (Confidence)', 'RTPMV ($)',
                ),
            )

            x = list(range(len(recent)))

            for col_idx, (key, color) in enumerate([
                ("SHF", "#1565c0"), ("ESF", "#2e7d32"), ("USS", "#e65100"),
                ("PDP", "#6a1b9a"), ("CI", "#00838f"),
            ], 1):
                row = 1 if col_idx <= 3 else 2
                col = col_idx if col_idx <= 3 else col_idx - 3
                vals = [d["indicators"][key] for d in recent]
                fig.add_trace(go.Scatter(
                    x=x, y=vals, mode='lines',
                    line=dict(color=color, width=2), name=key,
                ), row=row, col=col)

            rtpmv_vals = [d["valuation"]["rtpmv"] for d in recent]
            fig.add_trace(go.Scatter(
                x=x, y=rtpmv_vals, mode='lines',
                line=dict(color='#d32f2f', width=2.5), name='RTPMV',
            ), row=2, col=3)

            fig.update_layout(height=380, showlegend=False,
                              margin=dict(l=30, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"ind_{bkey}_{time.time()}")

            # ASHRAE meter + weather + PDP detail + hash chain
            if recent:
                latest = recent[-1]
                wcol1, wcol2, wcol3, wcol4 = st.columns(4)
                with wcol1:
                    st.caption("📊 ASHRAE Meters")
                    for mk, mv in latest["ashrae_meters"].items():
                        st.text(f"  {mk}: {mv:.1f} kWh")
                with wcol2:
                    st.caption("🌡️ Weather (Kuala Lumpur)")
                    w = latest["weather"]
                    st.text(f"  Air: {w['air_temperature_c']:.1f}°C")
                    st.text(f"  Humidity: {w['humidity_pct']:.0f}%")
                    st.text(f"  Wind: {w['wind_speed_ms']:.1f} m/s")
                    st.text(f"  Precip: {w['precip_depth_mm']:.1f} mm")
                with wcol3:
                    st.caption("🏗️ PDP Aging Model")
                    pd_d = latest["valuation"].get("pdp_detail", {})
                    if pd_d:
                        st.text(f"  Actual Age: {pd_d['chronological_age']} yrs")
                        st.text(f"  Effective Age: {pd_d['effective_age']} yrs")
                        st.text(f"  Maintenance: {pd_d['maintenance_factor']:.2f}")
                        st.text(f"  Age Reduction: {pd_d['age_reduction_pct']}%")
                        st.text(f"  PDP Floor: {pd_d['floor']}")
                with wcol4:
                    st.caption("🔗 Hash Chain (G1)")
                    v = latest["verification"]
                    st.text(f"  Block #{v['block_index']}")
                    st.text(f"  Hash: {v['block_hash']}")
                    st.text(f"  Prev: {v['prev_hash']}")
                    st.text(f"  Chain: {v['chain_length']} blocks")
