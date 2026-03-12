"""
Hash-chain audit trail tab.
"""

import pandas as pd
import streamlit as st

from config.buildings import ASHRAE_BUILDINGS


def render_hash_chain_tab():
    """Render hash chain verification details — only selected buildings."""
    st.markdown(
        '<div class="exchange-header">🔗 Hash-Chain Audit Trail (Element G/G1)</div>',
        unsafe_allow_html=True,
    )

    selected = st.session_state.selected_buildings
    tabs = st.tabs([ASHRAE_BUILDINGS[k]["name"][:20] for k in selected])
    for tab, bkey in zip(tabs, selected):
        with tab:
            orch = st.session_state.orchestrators[bkey]
            chain = orch.verification

            is_valid, break_idx = chain.verify_chain_integrity()
            if is_valid:
                st.success(
                    f"✅ Chain integrity verified — {len(chain.chain)} blocks, no tampering detected"
                )
            else:
                st.error(f"❌ Chain broken at block {break_idx}!")

            recent = chain.get_recent_chain(8)
            if recent:
                rows = []
                for b in recent:
                    rows.append({
                        "Block": b["block_index"],
                        "Timestamp": b["timestamp"],
                        "RTPMV": f"${b['rtpmv']:,.0f}",
                        "Health": f"{b['health_factor']:.3f}",
                        "Hash": b["block_hash"][:24] + "...",
                        "Prev Hash": b["prev_hash"][:24] + "...",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No blocks yet — start the system to begin recording.")
