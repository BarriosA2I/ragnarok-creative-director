"""
================================================================================
RAGNAROK WEBSOCKET SERVER
================================================================================
FastAPI WebSocket server for video generation and discovery pipelines.

Endpoints:
- /ws/video-generator  - Video production WebSocket
- /ws/antigravity     - Discovery agent WebSocket (proxied from Railway)
- /health             - Health check
- /health/video       - Video service health

Author: Barrios A2I
Version: 3.0.0 (Tier 2 Consolidated)
================================================================================
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import the consolidated video producer
from .video_agents import (
    ConsolidatedVideoProducer,
    VideoRequest,
    VideoStyle,
    create_progress_callback
)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("RagnarokServer")

# ============================================================================
# METRICS
# ============================================================================

if PROMETHEUS_AVAILABLE:
    ws_connections_total = Counter(
        'ragnarok_ws_connections_total',
        'Total WebSocket connections',
        ['endpoint', 'status']
    )

    ws_active_connections = Gauge(
        'ragnarok_ws_active_connections',
        'Currently active WebSocket connections',
        ['endpoint']
    )

    ws_messages_total = Counter(
        'ragnarok_ws_messages_total',
        'Total WebSocket messages',
        ['endpoint', 'direction', 'type']
    )

# ============================================================================
# CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections with tracking"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.active_connections: Dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Accept and track connection"""
        try:
            await websocket.accept()
            async with self._lock:
                self.active_connections[client_id] = websocket

            if PROMETHEUS_AVAILABLE:
                ws_connections_total.labels(endpoint=self.endpoint, status="connected").inc()
                ws_active_connections.labels(endpoint=self.endpoint).set(len(self.active_connections))

            logger.info(f"[{self.endpoint}] Client {client_id} connected. Active: {len(self.active_connections)}")
            return True
        except Exception as e:
            logger.error(f"[{self.endpoint}] Connection failed for {client_id}: {e}")
            return False

    async def disconnect(self, client_id: str):
        """Remove connection"""
        async with self._lock:
            self.active_connections.pop(client_id, None)

        if PROMETHEUS_AVAILABLE:
            ws_connections_total.labels(endpoint=self.endpoint, status="disconnected").inc()
            ws_active_connections.labels(endpoint=self.endpoint).set(len(self.active_connections))

        logger.info(f"[{self.endpoint}] Client {client_id} disconnected. Active: {len(self.active_connections)}")

    async def send_json(self, client_id: str, data: dict):
        """Send JSON to specific client"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send_json(data)
                if PROMETHEUS_AVAILABLE:
                    ws_messages_total.labels(
                        endpoint=self.endpoint,
                        direction="outbound",
                        type=data.get("type", "unknown")
                    ).inc()
            except Exception as e:
                logger.warning(f"[{self.endpoint}] Send to {client_id} failed: {e}")
                await self.disconnect(client_id)


# Global managers
video_manager = ConnectionManager("video-generator")

# Global VideoProducer instance
video_producer: ConsolidatedVideoProducer = None

# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global video_producer

    logger.info("=" * 60)
    logger.info("RAGNAROK v3.0 - Tier 2 Consolidated Architecture")
    logger.info("=" * 60)

    # Initialize VideoProducer
    video_producer = ConsolidatedVideoProducer()
    logger.info("VideoProducer initialized")

    # Create data directories
    video_dir = os.getenv("VIDEO_OUTPUT_DIR", "/var/data/videos")
    logs_dir = os.getenv("LOGS_DIR", "/var/data/logs")

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Logs directory: {logs_dir}")
    logger.info("=" * 60)

    yield

    # Cleanup
    logger.info("Shutting down RAGNAROK server")


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="RAGNAROK Video Production API",
    description="Tier 2 Consolidated Video Generation Platform",
    version="3.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HEALTH ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Main health check"""
    return {
        "status": "healthy",
        "service": "ragnarok-api",
        "version": "3.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health/video")
async def video_health():
    """Video generator health check"""
    global video_producer

    kie_configured = bool(os.getenv("KIE_API_KEY"))
    anthropic_configured = bool(os.getenv("ANTHROPIC_API_KEY"))

    return {
        "status": "healthy" if kie_configured else "degraded",
        "service": "video-generator",
        "active_connections": len(video_manager.active_connections),
        "producer_stats": video_producer.stats if video_producer else None,
        "configuration": {
            "kie_api_key": kie_configured,
            "anthropic_api_key": anthropic_configured,
            "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
            "elevenlabs_api_key": bool(os.getenv("ELEVENLABS_API_KEY"))
        },
        "timestamp": datetime.now().isoformat()
    }


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

# ============================================================================
# VIDEO GENERATOR WEBSOCKET
# ============================================================================

@app.websocket("/ws/video-generator")
async def video_generator_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for video generation.

    Protocol:
    - Client sends: {"type": "generate", "business": "...", "style": "professional"}
    - Server sends progress: {"type": "progress", "stage": "SCRIPTING", "percent": 20}
    - Server sends logs: {"type": "log", "message": "..."}
    - Server sends completion: {"type": "complete", "video_url": "https://..."}
    - Server sends errors: {"type": "error", "message": "..."}
    """
    global video_producer

    # Generate unique client ID
    client_id = f"video_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(websocket) % 10000}"

    # Accept connection
    if not await video_manager.connect(websocket, client_id):
        return

    # Send welcome message
    await video_manager.send_json(client_id, {
        "type": "log",
        "message": f"Connected to RAGNAROK Render Node (session: {client_id})"
    })

    try:
        while True:
            # Receive message
            try:
                data = await websocket.receive_json()

                if PROMETHEUS_AVAILABLE:
                    ws_messages_total.labels(
                        endpoint="video-generator",
                        direction="inbound",
                        type=data.get("type", "unknown")
                    ).inc()

            except Exception as e:
                logger.warning(f"Receive error from {client_id}: {e}")
                break

            # Handle message types
            msg_type = data.get("type")

            if msg_type == "ping":
                await video_manager.send_json(client_id, {"type": "pong"})

            elif msg_type == "generate":
                # Validate request
                try:
                    business = data.get("business", "").strip()
                    style_str = data.get("style", "professional").lower()

                    # Map style string to enum
                    style_map = {
                        "professional": VideoStyle.PROFESSIONAL,
                        "cinematic": VideoStyle.CINEMATIC,
                        "energetic": VideoStyle.ENERGETIC,
                        "minimalist": VideoStyle.MINIMALIST,
                        "testimonial": VideoStyle.TESTIMONIAL,
                        "explainer": VideoStyle.EXPLAINER,
                    }
                    style = style_map.get(style_str, VideoStyle.PROFESSIONAL)

                    # Create validated request
                    request = VideoRequest(
                        business=business,
                        style=style,
                        duration=data.get("duration", 5),
                        include_voice=data.get("include_voice", False)
                    )

                except ValidationError as e:
                    await video_manager.send_json(client_id, {
                        "type": "error",
                        "message": f"Invalid request: {str(e)}"
                    })
                    continue
                except Exception as e:
                    await video_manager.send_json(client_id, {
                        "type": "error",
                        "message": f"Request validation failed: {str(e)}"
                    })
                    continue

                # Create progress callback
                async def send_update(stage: str, percent: int, log: str):
                    await video_manager.send_json(client_id, {
                        "type": "progress",
                        "stage": stage,
                        "percent": percent
                    })
                    await video_manager.send_json(client_id, {
                        "type": "log",
                        "message": log
                    })

                # Run video generation
                try:
                    logger.info(f"Starting video generation for {client_id}")

                    result = await video_producer.produce(
                        request=request,
                        on_progress=send_update
                    )

                    # Send completion
                    await video_manager.send_json(client_id, {
                        "type": "complete",
                        "video_url": result.video_url,
                        "audio_url": result.audio_url,
                        "duration": result.duration_seconds,
                        "cost": result.cost_usd,
                        "generation_time": result.generation_time_seconds,
                        "metrics": result.metrics
                    })

                    logger.info(f"Video generation complete for {client_id}: {result.video_url}")

                except Exception as e:
                    logger.error(f"Video generation failed for {client_id}: {e}")
                    await video_manager.send_json(client_id, {
                        "type": "error",
                        "message": str(e)
                    })

            else:
                await video_manager.send_json(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        await video_manager.disconnect(client_id)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        ws_ping_interval=20,
        ws_ping_timeout=30,
        timeout_keep_alive=120
    )
