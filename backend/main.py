from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    try:
        from utils.db_connect import init_pool
        await init_pool()
    except Exception as e:
        print(f"Database initialization warning: {e}")
    
    yield
    
    # Shutdown
    try:
        from utils.db_connect import close_pool
        await close_pool()
    except Exception as e:
        print(f"Database cleanup warning: {e}")

app = FastAPI(title="AI Buddy Backend", version="1.0.0", lifespan=lifespan)

# CORS configuration for local Electron app and dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "app://*"  # Electron app protocol
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
from routes import auth, assistant, media

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(assistant.router, prefix="/api/assistant", tags=["Assistant"])
app.include_router(media.router, prefix="/api/media", tags=["Media"])


@app.get("/")
async def root():
    return {"message": "AI Buddy Backend is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "backend": "online"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
