import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from utils.db_connect import get_pool
import google.genai as genai
import os
import logging

logger = logging.getLogger(__name__)

# Initialize Gemini Client for Embeddings
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
    """Store a fact with Gemini vector embedding (768 dims)."""
    try:
        # Generate embedding using Gemini
        result = await client.aio.models.embed_content(
            model="text-embedding-004",
            contents=fact
        )
        embedding_list = result.embeddings[0].values
        
        pool = await get_pool()
        async with pool.acquire() as conn:
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
            logger.info(f"Stored knowledge for {username}: {fact[:30]}...")
            
    except Exception as e:
        logger.error(f"Error storing knowledge: {e}")
        # Don't raise, just log error to allow flow to continue
        pass

async def retrieve_knowledge(username: str, query: str, k: int = 5) -> List[Dict[str, str]]:
    """Semantic search using Gemini embeddings."""
    try:
        # Generate embedding for query
        result = await client.aio.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        query_emb_list = result.embeddings[0].values
        
        pool = await get_pool()
        async with pool.acquire() as conn:
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
