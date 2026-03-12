"""
REAL-TIME DIGITAL TWIN-BASED PROPERTY VALUATION & EXCHANGE SYSTEM
Version 2.0 — ASHRAE Live Data Edition  (build: 2026-03-12)

Data Source: ASHRAE Great Energy Predictor III (Building Data Genome Project 2)
Region: Kuala Lumpur & Selangor, Malaysia (3.14°N, 101.69°E)
Buildings: 5 Malaysian commercial properties across KL, Shah Alam & Subang Jaya

PATENT KEY ELEMENTS IMPLEMENTED:
- Element A:    Real-time digital twin-based property valuation system
- Element B/B1: Multi-modal IoT sensor data (ASHRAE GEPIII hourly meters)
- Element C/C1: Data preprocessing (Savitzky-Golay denoising, z-score, alignment)
- Element D/D1: Continuously updated digital twin model
- Element E/E1: Technical indicators (SHF, ESF, USS, PDP, Confidence Index)
- Element F/F1/F2: RTPMV valuation engine — multiplicative multi-factor
- Element F2.1:  Government valuation baseline with manual override
- Element G/G1:  Hash-chain verification — each record links to previous hash
- Element H/H1:  Multi-property exchange with live order book
- Element H1.1:  Sensor-driven market controls & circuit breakers

Copyright (c) 2025 — Digital Twin Valuation System
Patent Pending (India + PCT Filing)
"""

import time

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.ui.session import init_session_state
from src.ui.sidebar import render_sidebar
from src.ui.building_info import render_building_info
from src.ui.exchange_view import render_exchange_view
from src.ui.charts import render_sensor_charts
from src.ui.hash_chain_tab import render_hash_chain_tab
from src.ui.valuation_table import render_valuation_table
from src.ui.indicator_curves import render_indicator_curves

REIT_FRONTEND_URL = "http://localhost:5173"

# ── Page config — must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Digital Twin Property Exchange",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Real-Time Digital Twin Property Valuation & Exchange System v2.0"
    }
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: bold; color: #0d47a1;
        text-align: center; padding: 0.8rem;
        background: linear-gradient(135deg, #e3f2fd 0%, #fff 50%, #e8f5e9 100%);
        border-radius: 10px; margin-bottom: 1.5rem;
        border: 1px solid #bbdefb;
    }
    .exchange-header {
        font-size: 1.1rem; font-weight: 600; color: #1b5e20;
        background: #e8f5e9; padding: 0.4rem 0.8rem; border-radius: 5px;
        border-left: 4px solid #2e7d32; margin: 0.5rem 0;
    }
    .bid-price { color: #2e7d32; font-weight: bold; }
    .ask-price { color: #c62828; font-weight: bold; }
    .hash-chain { font-family: monospace; font-size: 0.7rem; color: #666; }
    .building-card {
        background: #f8f9fa; border-radius: 8px; padding: 0.8rem;
        border: 1px solid #e0e0e0; margin-bottom: 0.5rem;
    }
    .govt-val { color: #0d47a1; font-weight: bold; font-size: 1.1rem; }
    .circuit-breaker {
        background: #ffebee; color: #b71c1c; padding: 0.3rem 0.6rem;
        border-radius: 4px; font-weight: bold; text-align: center;
    }
    div[data-testid="stMetric"] label { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


def main():
    init_session_state()

    st.markdown(
        '<div class="main-header">'
        '🏢 Digital Twin Property Valuation & Exchange System'
        '<br/><small style="font-size:0.5em; color:#666;">'
        'ASHRAE GEPIII Profiles · Kuala Lumpur, Malaysia · Patent Pending'
        '</small></div>',
        unsafe_allow_html=True,
    )

    render_sidebar()

    # ── Top-level tabs — REIT tab rendered BEFORE any while loop so the iframe
    #    is already in the DOM when the engine loop starts in tab_twin.
    tab_reit, tab_twin = st.tabs(["📊 REIT Dashboard", "🔬 Digital Twin Engine"])

    # ── Tab 1: REIT Dashboard (iframe embed) ─────────────────────────────────
    with tab_reit:
        hdr_col, btn_col = st.columns([5, 1])
        with hdr_col:
            st.markdown("### 📊 TwinVal REIT Intelligence")
            st.caption(
                "Live React dashboard — start the Vite dev server first: "
                "`cd frontend && npm install && npm run dev`"
            )
        with btn_col:
            st.link_button(
                "Full Screen ↗",
                REIT_FRONTEND_URL,
                use_container_width=True,
                help="Open the REIT dashboard in a new browser tab",
            )

        components.iframe(REIT_FRONTEND_URL, height=900, scrolling=True)

        st.info(
            "💡 If the dashboard is blank, make sure the Vite server is running on "
            f"port 5173. Command: `cd \"{__file__.replace('app.py', 'frontend')}\" && npm run dev`"
        )

    # ── Tab 2: Digital Twin Engine (existing logic) ───────────────────────────
    with tab_twin:

        if st.session_state.running:
            # ── Process one data cycle ────────────────────────────────────────
            for bkey in st.session_state.selected_buildings:
                orch = st.session_state.orchestrators[bkey]
                overrides = None
                event = st.session_state.scenario_event
                if event and event["building"] == bkey:
                    overrides = event["overrides"]

                cycle = orch.process_cycle(overrides)
                st.session_state.data_logs[bkey].append(cycle)

                st.session_state.exchange.update_property(
                    bkey,
                    cycle["valuation"]["rtpmv"],
                    cycle["indicators"]["CI"],
                    cycle["valuation"]["health_factor"],
                    cycle["timestamp"],
                )

            # ── Render UI ─────────────────────────────────────────────────────
            # Each st.rerun() is a fresh script execution so keys are seen
            # exactly once per run — stable keys work, no DuplicateElementKey.
            render_building_info()
            render_exchange_view(st.session_state.exchange.get_exchange_summary())
            render_indicator_curves(
                st.session_state.data_logs,
                st.session_state.selected_buildings,
            )
            render_hash_chain_tab()
            render_valuation_table()

            time.sleep(st.session_state.cycle_speed)
            st.rerun()

        else:
            render_building_info()

            has_data = any(len(v) > 0 for v in st.session_state.data_logs.values())

            if has_data:
                render_exchange_view(st.session_state.exchange.get_exchange_summary())
                render_indicator_curves(
                    st.session_state.data_logs,
                    st.session_state.selected_buildings,
                )

                with st.expander("🔗 Hash-Chain Audit Trail", expanded=False):
                    render_hash_chain_tab()

                render_valuation_table()

                st.markdown("---")
                all_data = []
                for bkey in st.session_state.selected_buildings:
                    logs = st.session_state.data_logs.get(bkey, [])
                    for d in logs:
                        flat = {
                            "building": bkey,
                            "timestamp": d["timestamp"],
                            "rtpmv": d["valuation"]["rtpmv"],
                            "health": d["valuation"]["health_factor"],
                            **{f"ind_{k}": v for k, v in d["indicators"].items()},
                            **{f"sensor_{k}": v for k, v in d["raw_sensors"].items()},
                            **{f"meter_{k}": v for k, v in d["ashrae_meters"].items()},
                            "block_hash": d["verification"]["block_hash"],
                        }
                        all_data.append(flat)

                if all_data:
                    csv = pd.DataFrame(all_data).to_csv(index=False)
                    st.download_button(
                        "📥 Export All Data (CSV)",
                        csv, "digital_twin_exchange_data.csv", "text/csv",
                    )
            else:
                st.info("👆 Click **START** in the sidebar to begin the valuation engine and live exchange")
                st.markdown("---")
                st.markdown("""
                **What this system demonstrates:**

                1. **ASHRAE Real Data** — Sensor readings derived from GEPIII statistical distributions (1,448 buildings, 20M+ hourly data points)
                2. **Government Valuation Baseline** — Each building starts at county-assessed value; override manually in sidebar
                3. **Multi-Property Exchange** — Live order book with bid/ask bands that react to sensor events
                4. **Hash-Chain Audit** — Every valuation record cryptographically linked to the previous one
                5. **Scenario Injection** — Trigger pipe bursts, HVAC failures, hurricanes and watch the exchange respond

                *All 5 buildings are in Greater Kuala Lumpur — same tropical weather profile, different usage patterns and property types.*
                """)


if __name__ == "__main__":
    main()
