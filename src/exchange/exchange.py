"""
PATENT ELEMENT H/H1/H1.1: Property Exchange & Order Book
"""

import numpy as np
from collections import deque


class PropertyExchange:
    """
    PATENT ELEMENT H/H1/H1.1: Multi-property real-estate exchange.

    Full multi-property order book with:
    - Per-property bid/ask bands derived from sensor-based RTPMV
    - Cross-property relative value tracking
    - Circuit breakers triggered by sensor confidence drops
    - Market-wide controls (halt, restrict, widen spreads)
    """

    def __init__(self):
        self.listings = {}       # property_id → listing data
        self.order_book = {}     # property_id → {bids: [], asks: []}
        self.trade_history = []
        self.market_status = "OPEN"
        self.circuit_breakers = {}
        self.tick_size = 10_000  # $10K minimum increment

    def register_property(self, property_id, initial_rtpmv, building_info):
        """Register a property on the exchange."""
        self.listings[property_id] = {
            "property_id": property_id,
            "name": building_info["name"],
            "primary_use": building_info["primary_use"],
            "square_feet": building_info["square_feet"],
            "latest_rtpmv": initial_rtpmv,
            "prev_rtpmv": initial_rtpmv,
            "listing_price": initial_rtpmv,
            "bid": initial_rtpmv * 0.97,
            "ask": initial_rtpmv * 1.03,
            "spread_pct": 6.0,
            "confidence": 1.0,
            "health_factor": 1.0,
            "status": "ACTIVE",
            "restrictions": [],
            "last_update": None,
            "price_history": deque(maxlen=200),
            "volume_24h": 0,
        }
        self.order_book[property_id] = {"bids": [], "asks": []}
        self.circuit_breakers[property_id] = {
            "triggered": False, "reason": "", "cooldown_until": None
        }

    def update_property(self, property_id, rtpmv, ci, health_factor, timestamp):
        """
        ELEMENT H: Publish valuation signal → update listing, bid/ask, controls.
        ELEMENT H1.1: Automatically modify market parameters based on CI + health.
        """
        if property_id not in self.listings:
            return None

        listing = self.listings[property_id]
        prev = listing["latest_rtpmv"]
        listing["prev_rtpmv"] = prev
        listing["latest_rtpmv"] = rtpmv
        listing["confidence"] = ci
        listing["health_factor"] = health_factor
        listing["last_update"] = timestamp
        listing["price_history"].append(rtpmv)

        # --- Dynamic listing price (confidence-weighted) ---
        listing["listing_price"] = self._round_tick(rtpmv)

        # --- Dynamic bid/ask spread (Element H) ---
        base_spread = 0.03
        ci_multiplier = max(1.0, 2.0 - ci)
        health_multiplier = max(1.0, 1.5 - health_factor * 0.5)
        total_spread = base_spread * ci_multiplier * health_multiplier

        listing["bid"] = self._round_tick(rtpmv * (1 - total_spread / 2))
        listing["ask"] = self._round_tick(rtpmv * (1 + total_spread / 2))
        listing["spread_pct"] = round(total_spread * 100, 2)

        # --- Market controls (Element H1.1) ---
        listing["restrictions"] = []
        listing["status"] = "ACTIVE"

        if ci < 0.4:
            listing["status"] = "HALTED"
            listing["restrictions"].append("HALT: Sensor confidence < 40%")
        elif ci < 0.6:
            listing["status"] = "RESTRICTED"
            listing["restrictions"].append("RESTRICTED: Manual approval required")

        if health_factor < 0.35:
            listing["restrictions"].append("ALERT: Physical inspection mandatory")
        elif health_factor < 0.55:
            listing["restrictions"].append("NOTICE: Condition disclosure required")

        # Circuit breaker — price volatility check (needs warmup period)
        prices = list(listing["price_history"])
        if len(prices) >= 20:
            recent_vol = np.std(prices[-15:]) / np.mean(prices[-15:])
            if recent_vol > 0.10:
                listing["status"] = "CIRCUIT_BREAKER"
                listing["restrictions"].append(
                    f"CIRCUIT BREAKER: Volatility {recent_vol*100:.1f}% > 10% threshold"
                )
                self.circuit_breakers[property_id] = {
                    "triggered": True,
                    "reason": f"Volatility {recent_vol*100:.1f}%",
                    "cooldown_until": timestamp,
                }

        self._generate_order_depth(property_id, rtpmv, ci)
        return listing

    def _generate_order_depth(self, property_id, rtpmv, ci):
        """Simulate order book depth around current RTPMV."""
        bids = []
        asks = []
        depth_levels = 5

        for i in range(depth_levels):
            bid_price = self._round_tick(rtpmv * (1 - 0.01 * (i + 1) * (2 - ci)))
            bid_vol = max(1, int(5 * (depth_levels - i) * ci))
            bids.append({"price": bid_price, "quantity": bid_vol, "orders": max(1, bid_vol // 2)})

            ask_price = self._round_tick(rtpmv * (1 + 0.01 * (i + 1) * (2 - ci)))
            ask_vol = max(1, int(4 * (depth_levels - i) * ci))
            asks.append({"price": ask_price, "quantity": ask_vol, "orders": max(1, ask_vol // 2)})

        self.order_book[property_id] = {"bids": bids, "asks": asks}

    def _round_tick(self, price):
        return round(price / self.tick_size) * self.tick_size

    def get_exchange_summary(self):
        """Return all listings for exchange view."""
        active = sum(1 for l in self.listings.values() if l["status"] == "ACTIVE")
        halted = sum(1 for l in self.listings.values() if l["status"] in ("HALTED", "CIRCUIT_BREAKER"))
        total_value = sum(l["latest_rtpmv"] for l in self.listings.values())
        return {
            "total_properties": len(self.listings),
            "active": active,
            "halted": halted,
            "total_market_value": total_value,
            "listings": dict(self.listings),
            "order_books": dict(self.order_book),
        }
