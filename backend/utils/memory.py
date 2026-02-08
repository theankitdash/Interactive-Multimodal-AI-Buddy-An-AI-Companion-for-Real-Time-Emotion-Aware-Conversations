import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from utils.db_connect import get_pool
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Global embedder (CPU-bound operations will be run in thread pool)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

async def get_user_profile(username: str) -> Optional[Dict[str, str]]:
    """Fetch basic user details."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                "SELECT username, name FROM user_details WHERE username = $1",
                username
            )
            if row:
                return {"username": row["username"], "name": row["name"]}
            return None
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None

async def store_knowledge(username: str, fact: str, category: str = "other") -> None:
    """Store a fact with vector embedding."""
    # Run embedder in thread pool to avoid blocking event loop
    embedding = await asyncio.to_thread(embedder.encode, fact)
    embedding_list = embedding.tolist()
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            # Check if fact exists to avoid duplicates (optional, based on unique constraint)
            await conn.execute(
                """
                INSERT INTO user_knowledge (username, fact, category, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (username, fact) DO UPDATE 
                SET last_updated = CURRENT_TIMESTAMP
                """,
                username, fact, category, embedding_list
            )
        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
            raise

async def retrieve_knowledge(username: str, query: str, k: int = 5) -> List[Dict[str, str]]:
    """Semantic search for relevant user knowledge."""
    # Run embedder in thread pool to avoid blocking event loop
    query_emb = await asyncio.to_thread(embedder.encode, query)
    query_emb_list = query_emb.tolist()
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT fact, category 
                FROM user_knowledge
                WHERE username = $1
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                username, query_emb_list, k
            )
            return [{"fact": r["fact"], "category": r["category"]} for r in rows]
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []

async def store_event(username: str, description: str, event_time: datetime, event_type: str = "task") -> None:
    """Create a new event."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                """
                INSERT INTO events (username, description, event_time, type, status)
                VALUES ($1, $2, $3, $4, 'pending')
                """,
                username, description, event_time, event_type
            )
        except Exception as e:
            logger.error(f"Error storing event: {e}")
            raise

async def get_upcoming_events(username: str, limit: int = 5) -> List[Dict]:
    """Get upcoming pending events."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT description, event_time, type
                FROM events
                WHERE username = $1 AND status = 'pending' AND event_time > CURRENT_TIMESTAMP
                ORDER BY event_time ASC
                LIMIT $2
                """,
                username, limit
            )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            return []
