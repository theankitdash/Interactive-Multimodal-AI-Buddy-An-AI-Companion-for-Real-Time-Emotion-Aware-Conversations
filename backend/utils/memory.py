import json
from datetime import datetime
from utils.db_connect import connect_db
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

async def get_user_profile(username: str):
    """Fetch basic user details."""
    conn = await connect_db()
    try:
        row = await conn.fetchrow(
            "SELECT username, name FROM user_details WHERE username = $1",
            username
        )
        if row:
            return {"username": row["username"], "name": row["name"]}
        return None
    except Exception as e:
        import logging
        logging.error(f"Error fetching user profile: {e}")
        return None
    finally:
        await conn.close()

async def store_knowledge(username: str, fact: str, category: str = "other"):
    """Store a fact with vector embedding."""
    embedding = embedder.encode(fact).tolist()
    conn = await connect_db()
    try:
        # Check if fact exists to avoid duplicates (optional, based on unique constraint)
        await conn.execute(
            """
            INSERT INTO user_knowledge (username, fact, category, embedding)
            VALUES ($1, $2, $3, $4::vector)
            ON CONFLICT (username, fact) DO UPDATE 
            SET last_updated = CURRENT_TIMESTAMP
            """,
            username, fact, category, embedding
        )
    finally:
        await conn.close()

async def retrieve_knowledge(username: str, query: str, k: int = 5):
    """Semantic search for relevant user knowledge."""
    query_emb = embedder.encode(query).tolist()
    conn = await connect_db()
    try:
        rows = await conn.fetch(
            """
            SELECT fact, category 
            FROM user_knowledge
            WHERE username = $1
            ORDER BY embedding <=> $2::vector
            LIMIT $3
            """,
            username, query_emb, k
        )
        return [{"fact": r["fact"], "category": r["category"]} for r in rows]
    finally:
        await conn.close()

async def store_event(username: str, description: str, event_time: datetime, event_type: str = "task"):
    """Create a new event."""
    conn = await connect_db()
    try:
        await conn.execute(
            """
            INSERT INTO events (username, description, event_time, type, status)
            VALUES ($1, $2, $3, $4, 'pending')
            """,
            username, description, event_time, event_type
        )
    finally:
        await conn.close()

async def get_upcoming_events(username: str, limit: int = 5):
    """Get upcoming pending events."""
    conn = await connect_db()
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
    finally:
        await conn.close()
