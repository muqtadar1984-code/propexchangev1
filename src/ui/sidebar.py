"""
Sidebar controls.
"""

import os
import streamlit as st

from config.buildings import ASHRAE_BUILDINGS

_REIT_URL = os.environ.get("VITE_REIT_URL", "http://localhost:5173").strip()


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.markdown("## ⚙️ System Controls")
        st.markdown("---")

        # ── REIT Dashboard Link ───────────────────────────────────────────────
        st.link_button(
            "📊 Open REIT Dashboard ↗",
            _REIT_URL,
            use_container_width=True,
            help=f"Opens the TwinVal REIT Intelligence dashboard at {_REIT_URL}",
        )
        if "localhost" in _REIT_URL:
            st.caption("🖥️ Run `npm run dev` in /frontend to activate")
        else:
            st.caption(f"🌐 Cloud dashboard: `{_REIT_URL}`")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ START", use_container_width=True):
                st.session_state.running = True
        with col2:
            if st.button("⏹️ STOP", use_container_width=True):
                st.session_state.running = False

        if st.button("🔄 RESET SYSTEM", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.session_state.cycle_speed = st.slider(
            "Cycle Speed (sec)", 0.3, 3.0, 0.8, 0.1
        )

        st.markdown("---")
        st.markdown("### 🏢 Building Selection")
        st.caption("Choose 1 or more buildings to display and process.")
        selected = st.multiselect(
            "Active Buildings",
            options=list(ASHRAE_BUILDINGS.keys()),
            default=st.session_state.selected_buildings,
            format_func=lambda x: f"{ASHRAE_BUILDINGS[x]['name']} ({ASHRAE_BUILDINGS[x]['primary_use']})",
            key="building_selector",
        )
        if selected:
            st.session_state.selected_buildings = selected
        else:
            st.session_state.selected_buildings = list(ASHRAE_BUILDINGS.keys())[:1]
            st.warning("At least one building must be selected.")

        st.markdown("---")
        st.markdown("### 💰 Market Value Override")
        st.caption("Government valuation is default. Override replaces it for RTPMV.")

        for bkey in st.session_state.selected_buildings:
            bdata = ASHRAE_BUILDINGS[bkey]
            govt_val = bdata["govt_valuation"]
            override = st.number_input(
                f"{bdata['name'][:20]}",
                min_value=1_000_000,
                max_value=500_000_000,
                value=st.session_state.manual_overrides.get(bkey, govt_val),
                step=500_000,
                key=f"mv_{bkey}",
                help=f"Govt: ${govt_val:,.0f}"
            )
            if override != govt_val:
                st.session_state.manual_overrides[bkey] = override
                st.session_state.orchestrators[bkey].valuation.update_mv(override)
            elif bkey in st.session_state.manual_overrides:
                del st.session_state.manual_overrides[bkey]
                st.session_state.orchestrators[bkey].valuation.update_mv(govt_val)

        st.markdown("---")
        st.markdown("### 🌪️ Scenario Events")
        st.caption("Inject real-world events to see exchange impact.")
        event = st.selectbox("Trigger Event", [
            "None",
            "🌊 Pipe Burst (moisture spike)",
            "🌡️ HVAC Failure (temp spike)",
            "⚡ Power Surge (electrical spike)",
            "🏗️ Structural Alert (vibration spike)",
            "🌀 Storm Warning (all sensors)",
        ], key="event_select")

        event_building = st.selectbox(
            "Affected Building",
            st.session_state.selected_buildings,
            format_func=lambda x: ASHRAE_BUILDINGS[x]["name"],
            key="event_building",
        )

        if event != "None":
            overrides = {
                "🌊 Pipe Burst (moisture spike)": {"moisture": 0.95, "air_quality": 0.7},
                "🌡️ HVAC Failure (temp spike)": {"temperature": 0.9, "electrical_load": 0.1},
                "⚡ Power Surge (electrical spike)": {"electrical_load": 0.98, "vibration": 0.6},
                "🏗️ Structural Alert (vibration spike)": {"vibration": 0.85, "strain": 0.7},
                "🌀 Storm Warning (all sensors)": {
                    "vibration": 0.8, "strain": 0.65, "moisture": 0.9,
                    "temperature": 0.7, "electrical_load": 0.3,
                },
            }
            st.session_state.scenario_event = {
                "building": event_building,
                "overrides": overrides.get(event, {}),
            }
        else:
            st.session_state.scenario_event = None

        st.markdown("---")
        st.markdown("### 📋 Patent Elements")
        st.markdown("""
        ✅ **B/B1**: ASHRAE IoT Sensors
        ✅ **C/C1**: Savitzky-Golay + Z-Score
        ✅ **D/D1**: Digital Twin
        ✅ **E/E1**: SHF·ESF·USS·PDP·CI
        ✅ **F/F2**: RTPMV Engine
        ✅ **F2.1**: Govt Valuation + Override
        ✅ **G/G1**: Hash-Chain Audit
        ✅ **H/H1.1**: Exchange + Controls
        """)

        st.markdown("---")
        st.caption("Data: ASHRAE GEPIII Profiles · Kuala Lumpur, Malaysia")
