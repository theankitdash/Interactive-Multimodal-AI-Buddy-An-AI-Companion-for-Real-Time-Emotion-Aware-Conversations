from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting up...")
    try:
        from utils.db_connect import init_pool
        await init_pool()
        logger.info("Database pool initialized.")
    except Exception as e:
        logger.error(f"Database initialization warning: {e}")
    
    # Initialize session registry
    # (No explicit init needed for now, but good placeholder)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
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
from routes import auth, assistant, media, cognition

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(assistant.router, prefix="/api/assistant", tags=["Assistant - Audio Socket"])
app.include_router(cognition.router, prefix="/api/cognition", tags=["Cognition - Reasoning Socket"])
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

# cd frontend && npm run dev