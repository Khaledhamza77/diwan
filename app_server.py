"""
Root FastAPI application.

Mounts Chainlit at /chainlit and serves the React frontend.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Then start the React frontend separately:
    cd frontend && npm run dev       (runs at http://localhost:5173)
"""
from pathlib import Path
from urllib.parse import unquote

from chainlit.utils import mount_chainlit
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

_DOCUMENTS_DIR = Path(__file__).parent / "data"

app = FastAPI(title="Regulatory AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alternative dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/custom-auth")
async def custom_auth():
    """Auth endpoint required by Chainlit React client. Auth disabled for POC."""
    return {"message": "Authentication disabled"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/documents/{filename:path}")
async def download_document(filename: str):
    """Serve source documents for download."""
    # Decode percent-encoded characters (e.g. Arabic filenames encoded by browsers)
    decoded = unquote(filename)
    file_path = _DOCUMENTS_DIR / decoded
    try:
        file_path.resolve().relative_to(_DOCUMENTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(
        path=str(file_path),
        filename=decoded,
        media_type="application/pdf",
    )


# Mount Chainlit at /chainlit — React client connects here via WebSocket
mount_chainlit(app=app, target="diwan/main.py", path="/chainlit")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
