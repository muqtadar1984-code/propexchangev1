"""
Streamlit session state initialisation.
"""

import streamlit as st

from config.buildings import ASHRAE_BUILDINGS
from src.core.orchestrator import BuildingOrchestrator
from src.exchange.exchange import PropertyExchange


def init_session_state():
    """Initialize session state on first load."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.running = False
        st.session_state.exchange = PropertyExchange()
        st.session_state.orchestrators = {}
        st.session_state.data_logs = {}
        st.session_state.manual_overrides = {}
        st.session_state.scenario_event = None
        st.session_state.cycle_speed = 0.8
        st.session_state.selected_buildings = list(ASHRAE_BUILDINGS.keys())[:1]

        for bkey, bdata in ASHRAE_BUILDINGS.items():
            orch = BuildingOrchestrator(bkey)
            st.session_state.orchestrators[bkey] = orch
            st.session_state.data_logs[bkey] = []
            st.session_state.exchange.register_property(
                bkey, bdata["govt_valuation"], bdata
            )
