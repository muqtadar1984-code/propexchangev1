"""
TwinVal State API — Element H Bridge
=====================================
Bridges the Streamlit Digital Twin Engine to the React REIT Dashboard.

Two modes:
  LOCAL (dev)  — imported by app.py; runs as background thread on port 8502.
                 Streamlit calls update_state() directly (no HTTP overhead).
  CLOUD        — deployed standalone on Railway / Render (free tier).
                 Streamlit POSTs to TWINVAL_API_URL env var; React reads from it.

Endpoints:
  GET  /api/state   — React polls this every 2s
  POST /api/update  — Streamlit pushes state here (cloud mode)
  GET  /health      — health check
"""

import os
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Shared state (thread-safe) ────────────────────────────────────────────────
_lock  = threading.Lock()
_state: dict = {
    "engine_running":  False,
    "buildings":       {},
    "active_scenario": None,
    "timestamp":       None,
}


def update_state(new_state: dict) -> None:
    """Called directly by Streamlit in local mode (same process)."""
    with _lock:
        _state.update(new_state)


def get_state() -> dict:
    with _lock:
        return dict(_state)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="TwinVal State API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Vercel URL in production if desired
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/state")
def read_state():
    return get_state()


@app.post("/api/update")
def write_state(payload: dict):
    """Cloud mode: Streamlit POSTs here instead of calling update_state()."""
    update_state(payload)
    return {"ok": True}


@app.get("/")
def root():
    return {
        "service": "TwinVal State API",
        "version": "1.0",
        "endpoints": {
            "state":  "GET  /api/state  — live engine data (polled by React dashboard)",
            "update": "POST /api/update — push new state (called by Streamlit engine)",
            "health": "GET  /health",
        },
        "engine_running": get_state().get("engine_running", False),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Background thread launcher (local dev) ────────────────────────────────────
_server_started = False
_server_lock    = threading.Lock()


def start_background(port: int = 8502) -> None:
    """
    Start the API server in a daemon thread.
    Called once at Streamlit startup for local development.
    Safe to call multiple times — only starts one thread.
    """
    global _server_started
    with _server_lock:
        if _server_started:
            return
        _server_started = True

    t = threading.Thread(
        target=lambda: uvicorn.run(
            app, host="0.0.0.0", port=port, log_level="error"
        ),
        daemon=True,
        name="twinval-api",
    )
    t.start()


# ── Standalone entry point (Railway / Render) ─────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
