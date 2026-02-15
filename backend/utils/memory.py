import json
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from utils.db_connect import get_pool
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Initialize local embedding model (768 dims, matches DB schema)
_embed_model = SentenceTransformer("all-mpnet-base-v2")
logger.info("[Memory] Loaded embedding model: all-mpnet-base-v2 (768 dims)")

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

def _get_embedding(text: str) -> np.ndarray:
    """Generate embedding locally using sentence-transformers."""
    return _embed_model.encode(text, normalize_embeddings=True).astype(np.float32)

async def store_knowledge(username: str, fact: str, category: str = "other") -> bool:
    """Store a fact with sentence-transformer vector embedding (768 dims). Returns True on success."""
    try:
        # Generate embedding locally (runs on CPU, fast for short text)
        logger.info(f"[Memory] Generating embedding for: {fact[:50]}...")
        embedding_array = await asyncio.to_thread(_get_embedding, fact)
        logger.info(f"[Memory] Embedding generated: {len(embedding_array)} dims")
        
        pool = await get_pool()
        async with pool.acquire() as conn:
            logger.info(f"[Memory] Inserting into DB: username={username}, category={category}")
            await conn.execute(
                """
                INSERT INTO user_knowledge (username, fact, category, embedding)
                VALUES ($1, $2, $3::knowledge_category, $4)
                ON CONFLICT (username, fact) DO UPDATE 
                SET last_updated = CURRENT_TIMESTAMP
                """,
                username, fact, category, embedding_array
            )
            
            # Verify the row was actually saved
            row = await conn.fetchrow(
                "SELECT knowledge_id, fact, category FROM user_knowledge WHERE username = $1 AND fact = $2",
                username, fact
            )
            if row:
                logger.info(f"[Memory] ✓ VERIFIED in DB: id={row['knowledge_id']}, fact={row['fact'][:30]}")
            else:
                logger.error(f"[Memory] ✗ ROW NOT FOUND after insert! username={username}, fact={fact[:30]}")
            
            return True
            
    except Exception as e:
        logger.error(f"[Memory] Error storing knowledge: {e}", exc_info=True)
        return False

async def retrieve_knowledge(username: str, query: str, k: int = 5) -> List[Dict[str, str]]:
    """Semantic search using sentence-transformer embeddings."""
    try:
        # Generate embedding locally
        query_emb_array = await asyncio.to_thread(_get_embedding, query)
        
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
                username, query_emb_array, k
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
