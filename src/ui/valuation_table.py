"""
Valuation history table.
"""

import pandas as pd
import streamlit as st

from config.buildings import ASHRAE_BUILDINGS


def render_valuation_table():
    """Render recent valuation history — only selected buildings."""
    st.markdown(
        '<div class="exchange-header">📝 Recent Valuation History</div>',
        unsafe_allow_html=True,
    )

    selected = st.session_state.selected_buildings
    all_rows = []
    for bkey in selected:
        logs = st.session_state.data_logs.get(bkey, [])
        for d in logs[-5:]:
            all_rows.append({
                "Building": ASHRAE_BUILDINGS[bkey]["name"][:18],
                "Time": d["timestamp"],
                "RTPMV": f"${d['valuation']['rtpmv']:,.0f}",
                "Health": f"{d['valuation']['health_factor']*100:.1f}%",
                "SHF": f"{d['indicators']['SHF']:.3f}",
                "ESF": f"{d['indicators']['ESF']:.3f}",
                "USS": f"{d['indicators']['USS']:.3f}",
                "PDP": f"{d['indicators']['PDP']:.3f}",
                "CI": f"{d['indicators']['CI']:.3f}",
                "MV Source": "Override" if d["valuation"]["is_override"] else "Govt",
                "Token": d["verification"]["token_id"],
            })

    if all_rows:
        df = pd.DataFrame(all_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
