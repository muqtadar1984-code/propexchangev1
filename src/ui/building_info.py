"""
Building information panel.
"""

from datetime import datetime

import streamlit as st

from config.buildings import ASHRAE_BUILDINGS


def render_building_info():
    """Render building information panel with coordinates — only selected buildings."""
    selected = st.session_state.selected_buildings
    st.markdown(
        '<div class="exchange-header">🏢 ASHRAE Building Portfolio — Kuala Lumpur & Selangor, Malaysia</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(selected))
    for i, bkey in enumerate(selected):
        bdata = ASHRAE_BUILDINGS[bkey]
        with cols[i]:
            orch = st.session_state.orchestrators[bkey]
            is_override = bkey in st.session_state.manual_overrides
            mv_label = (
                f"Override: ${orch.valuation.base_mv:,.0f}"
                if is_override
                else f"Govt: ${bdata['govt_valuation']:,.0f}"
            )

            st.markdown(f"""
            <div class="building-card">
                <strong>{bdata['name']}</strong><br/>
                <small>📍 {bdata['latitude']:.4f}°N, {abs(bdata['longitude']):.4f}°W</small><br/>
                <small>🏷️ {bdata['primary_use']} · {bdata['floor_count']} floors · {bdata['square_feet']:,} sqft</small><br/>
                <small>🔨 Built {bdata['year_built']} · Age: {datetime.now().year - bdata['year_built']} yrs</small><br/>
                <small>🌡️ Climate: {bdata['climate_zone']}</small><br/>
                <small>📊 Meters: {', '.join(bdata['meters'].keys())}</small><br/>
                <span class="govt-val">{mv_label}</span>
            </div>
            """, unsafe_allow_html=True)
