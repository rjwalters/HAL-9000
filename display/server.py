"""
FastAPI server for HAL-9000 eye display webapp.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config

logger = logging.getLogger(__name__)

# Get the directory containing this file
DISPLAY_DIR = Path(__file__).parent
STATIC_DIR = DISPLAY_DIR / "static"

app = FastAPI(title="HAL-9000 Eye Display")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class DisplayState:
    """Manages the current display state and connected clients."""

    def __init__(self):
        self.state: str = config.State.IDLE
        self.amplitude: float = 0.0
        self.connected_clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def update_state(self, new_state: str) -> None:
        """Update the current state and broadcast to clients."""
        async with self._lock:
            self.state = new_state
            await self._broadcast()

    async def update_amplitude(self, amplitude: float) -> None:
        """Update the current amplitude and broadcast to clients."""
        async with self._lock:
            self.amplitude = min(1.0, max(0.0, amplitude))
            await self._broadcast()

    async def connect(self, websocket: WebSocket) -> None:
        """Add a new client connection."""
        await websocket.accept()
        async with self._lock:
            self.connected_clients.add(websocket)
            # Send current state immediately
            await websocket.send_json({
                "state": self.state,
                "amplitude": self.amplitude,
            })
        logger.info(f"Display client connected. Total: {len(self.connected_clients)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a client connection."""
        async with self._lock:
            self.connected_clients.discard(websocket)
        logger.info(f"Display client disconnected. Total: {len(self.connected_clients)}")

    async def _broadcast(self) -> None:
        """Broadcast current state to all connected clients."""
        if not self.connected_clients:
            return

        message = json.dumps({
            "state": self.state,
            "amplitude": self.amplitude,
        })

        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.add(client)

        # Clean up disconnected clients
        self.connected_clients -= disconnected


# Global display state
display_state = DisplayState()


@app.get("/")
async def serve_index():
    """Serve the main eye display page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time state updates."""
    await display_state.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle any incoming messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                # Client can send ping/pong for keepalive
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                try:
                    await websocket.send_text(json.dumps({
                        "state": display_state.state,
                        "amplitude": display_state.amplitude,
                    }))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await display_state.disconnect(websocket)


# API functions to be called from main orchestrator

async def set_state(state: str) -> None:
    """Set the display state (called from orchestrator)."""
    await display_state.update_state(state)


async def set_amplitude(amplitude: float) -> None:
    """Set the audio amplitude (called from orchestrator)."""
    await display_state.update_amplitude(amplitude)


def get_app() -> FastAPI:
    """Get the FastAPI app instance."""
    return app


async def run_server(host: str = config.DISPLAY_HOST, port: int = config.DISPLAY_PORT):
    """Run the display server (for standalone testing)."""
    import uvicorn

    config_uvicorn = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=config.DISPLAY_HOST,
        port=config.DISPLAY_PORT,
        reload=True,
    )
