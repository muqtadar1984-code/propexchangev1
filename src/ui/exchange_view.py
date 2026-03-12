"""
Exchange order book view.
"""

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config.buildings import ASHRAE_BUILDINGS


def render_exchange_view(exchange_data):
    """Render the full exchange order book view — only selected buildings."""
    st.markdown(
        '<div class="exchange-header">📊 PROPERTY EXCHANGE — Live Order Book</div>',
        unsafe_allow_html=True,
    )

    selected = st.session_state.selected_buildings
    all_listings = exchange_data["listings"]
    listings = {k: v for k, v in all_listings.items() if k in selected}

    if not listings:
        st.info("No buildings selected.")
        return

    # --- Top-level exchange metrics ---
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    sel_total_value = sum(l["latest_rtpmv"] for l in listings.values())
    sel_active = sum(1 for l in listings.values() if l["status"] == "ACTIVE")
    sel_halted = sum(1 for l in listings.values() if l["status"] in ("HALTED", "CIRCUIT_BREAKER"))
    with mcol1:
        st.metric("Total Market Value", f"${sel_total_value:,.0f}")
    with mcol2:
        st.metric("Listed Properties", len(listings))
    with mcol3:
        st.metric("Active", sel_active)
    with mcol4:
        st.metric(
            "Halted/Restricted", sel_halted,
            delta=f"-{sel_halted}" if sel_halted > 0 else None,
            delta_color="inverse" if sel_halted > 0 else "off",
        )

    # --- Per-property exchange cards ---
    cols = st.columns(len(listings))
    for i, (pid, listing) in enumerate(listings.items()):
        with cols[i]:
            rtpmv = listing["latest_rtpmv"]
            prev = listing["prev_rtpmv"]
            change_pct = ((rtpmv - prev) / prev * 100) if prev else 0

            status = listing["status"]
            if status == "ACTIVE":
                badge = "🟢 ACTIVE"
            elif status == "RESTRICTED":
                badge = "🟡 RESTRICTED"
            elif status == "CIRCUIT_BREAKER":
                badge = "🔴 CIRCUIT BREAKER"
            else:
                badge = "⛔ HALTED"

            st.markdown(f"**{listing['name'][:22]}** · {badge}")
            st.metric("RTPMV", f"${rtpmv:,.0f}", delta=f"{change_pct:+.2f}%")

            st.markdown(
                f'<span class="bid-price">BID ${listing["bid"]:,.0f}</span>'
                f' — '
                f'<span class="ask-price">ASK ${listing["ask"]:,.0f}</span>'
                f'<br/><small>Spread: {listing["spread_pct"]:.1f}%</small>',
                unsafe_allow_html=True,
            )

            if listing["restrictions"]:
                for r in listing["restrictions"]:
                    st.warning(r, icon="⚠️")

    # --- Order Book Depth Chart ---
    st.markdown("---")
    st.markdown("**📖 Order Book Depth**")
    book_cols = st.columns(len(listings))
    for i, (pid, listing) in enumerate(listings.items()):
        with book_cols[i]:
            ob = exchange_data["order_books"].get(pid, {"bids": [], "asks": []})
            if ob["bids"] and ob["asks"]:
                bid_prices = [b["price"] for b in reversed(ob["bids"])]
                bid_cum = list(np.cumsum([b["quantity"] for b in reversed(ob["bids"])]))
                ask_prices = [a["price"] for a in ob["asks"]]
                ask_cum = list(np.cumsum([a["quantity"] for a in ob["asks"]]))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=bid_prices, y=bid_cum, fill='tozeroy',
                    name='Bids', line=dict(color='#2e7d32', width=2),
                    fillcolor='rgba(46,125,50,0.2)',
                ))
                fig.add_trace(go.Scatter(
                    x=ask_prices, y=ask_cum, fill='tozeroy',
                    name='Asks', line=dict(color='#c62828', width=2),
                    fillcolor='rgba(198,40,40,0.2)',
                ))
                fig.add_vline(
                    x=listing["latest_rtpmv"],
                    line_dash="dash", line_color="blue",
                    annotation_text="RTPMV",
                )
                fig.update_layout(
                    height=200, margin=dict(l=10, r=10, t=25, b=10),
                    title=dict(text=listing["name"][:18], font=dict(size=11)),
                    showlegend=False,
                    xaxis=dict(title="", tickformat="$,.0f", tickfont=dict(size=8)),
                    yaxis=dict(title="", tickfont=dict(size=8)),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"depth_{pid}_{time.time()}")
