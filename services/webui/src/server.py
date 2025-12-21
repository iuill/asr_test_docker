"""
FastAPI WebSocket Proxy Server for ASR models.

This module provides the HTTP and WebSocket endpoints for the
unified Web UI, proxying requests to backend ASR models.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
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
):
    """
    WebSocket endpoint for real-time ASR.

    Proxies WebSocket connections to the selected backend model.

    Args:
        websocket: Client WebSocket connection
        model: Model ID to use for transcription
        token: JWT token for authentication
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

    await websocket.accept()
    logger.info(f"Client connected, using model: {model}")

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
                proxy_backend_to_client(websocket, backend_ws)
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
