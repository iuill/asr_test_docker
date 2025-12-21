"""
FastAPI WebSocket Proxy Server for ASR models.

This module provides the HTTP and WebSocket endpoints for the
unified Web UI, proxying requests to backend ASR models.
"""

import asyncio
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Set

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.hash import bcrypt
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed

from .config import MODELS, DEFAULT_MODEL, get_model, get_all_models

logger = logging.getLogger(__name__)

# Authentication settings
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "password")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

# Hash the password at startup for comparison
_hashed_password = bcrypt.hash(AUTH_PASSWORD)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


# Session sharing models
class CreateSessionRequest(BaseModel):
    viewer_password: str
    selected_models: list[str]


class SessionResponse(BaseModel):
    session_id: str
    share_url: str
    selected_models: list[str]
    created_at: str


class ViewerAuthRequest(BaseModel):
    password: str


class ViewerAuthResponse(BaseModel):
    token: str
    models: list[str]


# Data classes for session management
@dataclass
class TranscriptionSegment:
    """A single transcription segment."""
    text: str
    is_final: bool
    speaker_tag: int = 0
    model_id: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class Session:
    """A transcription sharing session."""
    id: str
    host_user: str
    viewer_password_hash: str
    created_at: datetime
    last_activity: datetime
    selected_models: list[str] = field(default_factory=list)
    transcriptions: dict[str, list[TranscriptionSegment]] = field(default_factory=dict)
    viewers: Set[WebSocket] = field(default_factory=set)
    host_ws: WebSocket | None = None
    timeout_seconds: int = 3600


class SessionManager:
    """Manages transcription sharing sessions."""

    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self._cleanup_task: asyncio.Task | None = None

    def create_session(
        self,
        host_user: str,
        viewer_password: str,
        selected_models: list[str],
        timeout_seconds: int = 3600
    ) -> Session:
        """Create a new sharing session."""
        session_id = secrets.token_urlsafe(16)
        now = datetime.now(timezone.utc)

        session = Session(
            id=session_id,
            host_user=host_user,
            viewer_password_hash=bcrypt.hash(viewer_password),
            created_at=now,
            last_activity=now,
            selected_models=selected_models,
            timeout_seconds=timeout_seconds,
        )

        # Initialize transcription lists for each model
        for model_id in selected_models:
            session.transcriptions[model_id] = []

        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {host_user}")
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def verify_viewer_password(self, session_id: str, password: str) -> bool:
        """Verify viewer password for a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        return bcrypt.verify(password, session.viewer_password_hash)

    def close_session(self, session_id: str) -> bool:
        """Close and remove a session."""
        session = self.sessions.pop(session_id, None)
        if session:
            logger.info(f"Closed session {session_id}")
            return True
        return False

    def update_activity(self, session_id: str) -> None:
        """Update last activity timestamp for a session."""
        session = self.get_session(session_id)
        if session:
            session.last_activity = datetime.now(timezone.utc)

    def add_transcription(
        self,
        session_id: str,
        model_id: str,
        text: str,
        is_final: bool,
        speaker_tag: int = 0
    ) -> None:
        """Add a transcription segment to a session."""
        session = self.get_session(session_id)
        if not session:
            return

        if model_id not in session.transcriptions:
            session.transcriptions[model_id] = []

        segment = TranscriptionSegment(
            text=text,
            is_final=is_final,
            speaker_tag=speaker_tag,
            model_id=model_id,
        )
        session.transcriptions[model_id].append(segment)
        session.last_activity = datetime.now(timezone.utc)

    async def broadcast_to_viewers(
        self,
        session_id: str,
        message: dict
    ) -> None:
        """Broadcast a message to all viewers of a session."""
        session = self.get_session(session_id)
        if not session:
            return

        disconnected = set()
        for viewer_ws in session.viewers:
            try:
                await viewer_ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to viewer: {e}")
                disconnected.add(viewer_ws)

        # Remove disconnected viewers
        session.viewers -= disconnected

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        now = datetime.now(timezone.utc)
        expired = []

        for session_id, session in self.sessions.items():
            elapsed = (now - session.last_activity).total_seconds()
            if elapsed > session.timeout_seconds:
                expired.append(session_id)

        for session_id in expired:
            self.close_session(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    async def start_cleanup_task(self) -> None:
        """Start background task for session cleanup."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes
                self.cleanup_expired_sessions()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    def stop_cleanup_task(self) -> None:
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None


# Global session manager
session_manager = SessionManager()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=JWT_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> bool:
    if username != AUTH_USERNAME:
        return False
    return verify_password(password, _hashed_password)


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> str | None:
    if not AUTH_ENABLED:
        return "anonymous"

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_ws_token(token: str | None) -> bool:
    """Verify WebSocket token. Returns True if valid or auth disabled."""
    if not AUTH_ENABLED:
        return True

    if not token:
        return False

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        return username is not None
    except JWTError:
        return False

app = FastAPI(
    title="ASR Test Web UI",
    description="Unified Web UI for ASR models",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main web UI."""
    web_dir = Path(__file__).parent / "web"
    index_path = web_dir / "index.html"

    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html>
            <head><title>ASR Test</title></head>
            <body>
                <h1>ASR Test Web UI</h1>
                <p>Web UI not found. Please ensure the web files are in place.</p>
            </body>
            </html>
            """,
            status_code=500,
        )


@app.get("/health")
async def health_check():
    """Health check endpoint for the Web UI service."""
    return {"status": "healthy", "service": "webui"}


@app.get("/api/auth/status")
async def auth_status():
    """Check if authentication is enabled."""
    return {"auth_enabled": AUTH_ENABLED}


@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT token."""
    if not AUTH_ENABLED:
        return Token(access_token=create_access_token({"sub": "anonymous"}), token_type="bearer")

    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token({"sub": form_data.username})
    return Token(access_token=access_token, token_type="bearer")


@app.get("/api/auth/me")
async def get_me(current_user: str = Depends(get_current_user)):
    """Get current user info."""
    return {"username": current_user, "auth_enabled": AUTH_ENABLED}


# Session API endpoints
@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    current_user: str = Depends(get_current_user)
):
    """Create a new sharing session."""
    if not request.selected_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one model must be selected"
        )

    session = session_manager.create_session(
        host_user=current_user,
        viewer_password=request.viewer_password,
        selected_models=request.selected_models,
    )

    return SessionResponse(
        session_id=session.id,
        share_url=f"/view/{session.id}",
        selected_models=session.selected_models,
        created_at=session.created_at.isoformat(),
    )


@app.get("/api/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session info (for viewers before auth)."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Build model info with names for viewer display
    model_info = []
    for model_id in session.selected_models:
        model_config = get_model(model_id)
        model_info.append({
            "id": model_id,
            "name": model_config.name if model_config else model_id,
        })

    return {
        "session_id": session.id,
        "models": session.selected_models,
        "model_info": model_info,
        "created_at": session.created_at.isoformat(),
    }


@app.post("/api/sessions/{session_id}/auth", response_model=ViewerAuthResponse)
async def authenticate_viewer(session_id: str, request: ViewerAuthRequest):
    """Authenticate a viewer for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if not session_manager.verify_viewer_password(session_id, request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password"
        )

    # Create viewer token
    token = create_access_token({
        "sub": f"viewer:{session_id}",
        "role": "viewer",
        "session_id": session_id,
    })

    return ViewerAuthResponse(
        token=token,
        models=session.selected_models,
    )


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """End a sharing session (host only)."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if session.host_user != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the session host can end this session"
        )

    # Notify viewers that session is ending
    await session_manager.broadcast_to_viewers(session_id, {
        "type": "session_end",
        "message": "Session has been ended by host"
    })

    session_manager.close_session(session_id)
    return {"status": "ok", "message": "Session ended"}


@app.get("/api/models")
async def get_models(current_user: str = Depends(get_current_user)):
    """
    Get list of available ASR models with their status.

    Returns model info and health status from each backend.
    """
    models_info = []

    async with httpx.AsyncClient(timeout=5.0) as client:
        for model in get_all_models():
            model_data = {
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "speed": model.speed,
                "append_newline_on_final": model.append_newline_on_final,
                "status": "unknown",
                "model_loaded": False,
            }

            try:
                response = await client.get(f"{model.url}/health")
                if response.status_code == 200:
                    health = response.json()
                    model_data["status"] = health.get("status", "healthy")
                    model_data["model_loaded"] = health.get("model_loaded", False)
                else:
                    model_data["status"] = "error"
            except Exception as e:
                logger.warning(f"Failed to check health for {model.id}: {e}")
                model_data["status"] = "offline"

            models_info.append(model_data)

    return JSONResponse(content={"models": models_info, "default": DEFAULT_MODEL})


@app.websocket("/ws/asr")
async def websocket_asr(
    websocket: WebSocket,
    model: str = Query(default=DEFAULT_MODEL),
    token: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
):
    """
    WebSocket endpoint for real-time ASR.

    Proxies WebSocket connections to the selected backend model.

    Args:
        websocket: Client WebSocket connection
        model: Model ID to use for transcription
        token: JWT token for authentication
        session_id: Optional session ID for sharing
    """
    # Verify authentication
    if not verify_ws_token(token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Validate model
    model_config = get_model(model)
    if not model_config:
        await websocket.close(code=4000, reason=f"Unknown model: {model}")
        return

    # Validate session if provided
    session = None
    if session_id:
        session = session_manager.get_session(session_id)
        if not session:
            await websocket.close(code=4002, reason="Session not found")
            return
        # Dynamically add new model to session if not already present
        if model not in session.selected_models:
            session.selected_models.append(model)
            session.transcriptions[model] = []
            # Notify viewers about the new model (include model names)
            model_info = []
            for mid in session.selected_models:
                mc = get_model(mid)
                model_info.append({
                    "id": mid,
                    "name": mc.name if mc else mid,
                })
            await session_manager.broadcast_to_viewers(session_id, {
                "type": "models_updated",
                "models": session.selected_models,
                "model_info": model_info,
            })

    await websocket.accept()
    logger.info(f"Client connected, using model: {model}, session: {session_id}")

    # Convert HTTP URL to WebSocket URL
    backend_url = model_config.url.replace("http://", "ws://").replace("https://", "wss://")
    backend_ws_url = f"{backend_url}/ws/asr"

    try:
        async with websockets.connect(backend_ws_url) as backend_ws:
            # Create tasks for bidirectional proxying
            client_to_backend = asyncio.create_task(
                proxy_client_to_backend(websocket, backend_ws)
            )
            backend_to_client = asyncio.create_task(
                proxy_backend_to_client_with_session(
                    websocket, backend_ws, model, session_id
                )
            )

            # Wait for either direction to complete
            done, pending = await asyncio.wait(
                [client_to_backend, backend_to_client],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"Failed to connect to backend {model}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Backend model {model} is not available"
            })
        except Exception:
            pass
    except ConnectionRefusedError:
        logger.error(f"Connection refused by backend {model}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Backend model {model} is offline"
            })
        except Exception:
            pass
    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"Client disconnected from model: {model}")


def verify_viewer_token(token: str | None, session_id: str) -> bool:
    """Verify viewer token for WebSocket connection."""
    if not token:
        return False

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        role = payload.get("role")
        token_session_id = payload.get("session_id")
        return role == "viewer" and token_session_id == session_id
    except JWTError:
        return False


@app.websocket("/ws/view/{session_id}")
async def websocket_viewer(
    websocket: WebSocket,
    session_id: str,
    token: str | None = Query(default=None),
):
    """
    WebSocket endpoint for viewers to receive transcription updates.

    Args:
        websocket: Viewer WebSocket connection
        session_id: Session ID to join
        token: Viewer JWT token
    """
    # Verify viewer token
    if not verify_viewer_token(token, session_id):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Get session
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4002, reason="Session not found")
        return

    await websocket.accept()
    logger.info(f"Viewer connected to session: {session_id}")

    # Add to session viewers
    session.viewers.add(websocket)

    try:
        # Send initial data (existing transcriptions)
        init_data = {
            "type": "init",
            "models": session.selected_models,
            "transcriptions": {
                model_id: [
                    {
                        "text": seg.text,
                        "is_final": seg.is_final,
                        "speaker_tag": seg.speaker_tag,
                        "timestamp": seg.timestamp,
                    }
                    for seg in segments
                ]
                for model_id, segments in session.transcriptions.items()
            }
        }
        await websocket.send_json(init_data)

        # Keep connection alive, handle ping/pong
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # Handle ping
                if message == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send ping to keep alive
                try:
                    await websocket.send_text("ping")
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.debug(f"Viewer disconnected from session: {session_id}")
    except Exception as e:
        logger.error(f"Viewer WebSocket error: {e}")
    finally:
        # Remove from session viewers
        session.viewers.discard(websocket)
        logger.info(f"Viewer removed from session: {session_id}")


async def proxy_client_to_backend(
    client_ws: WebSocket,
    backend_ws: websockets.WebSocketClientProtocol,
):
    """Proxy messages from client to backend."""
    try:
        while True:
            message = await client_ws.receive()

            if "bytes" in message:
                # Binary audio data
                await backend_ws.send(message["bytes"])
            elif "text" in message:
                # Text/JSON control message
                await backend_ws.send(message["text"])
            elif message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.debug("Client disconnected")
    except ConnectionClosed:
        logger.debug("Backend connection closed")
    except Exception as e:
        logger.error(f"Error proxying client to backend: {e}")


async def proxy_backend_to_client(
    client_ws: WebSocket,
    backend_ws: websockets.WebSocketClientProtocol,
):
    """Proxy messages from backend to client."""
    try:
        async for message in backend_ws:
            if isinstance(message, bytes):
                await client_ws.send_bytes(message)
            else:
                logger.debug(f"Proxying to client: {message[:200] if len(message) > 200 else message}")
                await client_ws.send_text(message)

    except ConnectionClosed:
        logger.debug("Backend connection closed")
    except WebSocketDisconnect:
        logger.debug("Client disconnected")
    except Exception as e:
        logger.error(f"Error proxying backend to client: {e}")


async def proxy_backend_to_client_with_session(
    client_ws: WebSocket,
    backend_ws: websockets.WebSocketClientProtocol,
    model_id: str,
    session_id: str | None,
):
    """Proxy messages from backend to client, with session broadcasting."""
    import json

    try:
        async for message in backend_ws:
            if isinstance(message, bytes):
                await client_ws.send_bytes(message)
            else:
                logger.debug(f"Proxying to client: {message[:200] if len(message) > 200 else message}")
                await client_ws.send_text(message)

                # If session is active, broadcast to viewers
                if session_id:
                    try:
                        data = json.loads(message)
                        if data.get("type") == "transcription":
                            # Store in session
                            text = data.get("text", "")
                            is_final = data.get("is_final", False)

                            # Handle both speaker_tag (integer, Google) and speaker_id (string, Azure)
                            speaker_tag = data.get("speaker_tag", 0)
                            if not speaker_tag and data.get("speaker_id"):
                                # Convert Azure speaker_id (e.g., "Guest_1") to integer
                                match = re.search(r'\d+', data.get("speaker_id", ""))
                                if match:
                                    speaker_tag = int(match.group())

                            # Check if model should append newline on final
                            broadcast_text = text
                            if is_final:
                                model_config = get_model(model_id)
                                if model_config and model_config.append_newline_on_final:
                                    broadcast_text = text + '\n'

                                # Store final results (with newline if applicable)
                                session_manager.add_transcription(
                                    session_id,
                                    model_id,
                                    broadcast_text,
                                    is_final,
                                    speaker_tag,
                                )

                            # Broadcast to viewers (both partial and final)
                            broadcast_msg = {
                                "type": "transcription",
                                "model_id": model_id,
                                "text": broadcast_text,
                                "is_final": is_final,
                                "speaker_tag": speaker_tag,
                                "timestamp": datetime.now().timestamp(),
                            }
                            # Include provider_info if present (needed for Google STT V1 result_index)
                            if "provider_info" in data:
                                broadcast_msg["provider_info"] = data["provider_info"]
                            await session_manager.broadcast_to_viewers(
                                session_id, broadcast_msg
                            )
                    except json.JSONDecodeError:
                        pass

    except ConnectionClosed:
        logger.debug("Backend connection closed")
    except WebSocketDisconnect:
        logger.debug("Client disconnected")
    except Exception as e:
        logger.error(f"Error proxying backend to client: {e}")


@app.get("/view/{session_id}")
async def viewer_page(session_id: str):
    """Serve the viewer page."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    web_dir = Path(__file__).parent / "web"
    viewer_path = web_dir / "viewer.html"

    if viewer_path.exists():
        return FileResponse(viewer_path, media_type="text/html")
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Viewer page not found"
        )


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    # Mount static files if available
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    return app
