import os
import asyncpg
from pgvector.asyncpg import register_vector

# Global connection pool
_pool = None

async def _init_connection(conn):
    """Initialize each connection with pgvector type codec."""
    await register_vector(conn)

async def init_pool():
    """Initialize the connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            min_size=5,
            max_size=20,
            command_timeout=60,
            init=_init_connection  # Register pgvector on every connection
        )

async def get_pool():
    """Get the connection pool, creating it if necessary."""
    if _pool is None:
        await init_pool()
    return _pool



async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

async def init_db():
    """Initialize database schema."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            # EXTENSIONS
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pgcrypto;
            """)

            await conn.execute("""
                
                CREATE TABLE IF NOT EXISTS user_details (
                    username TEXT PRIMARY KEY NOT NULL,
                    name TEXT NOT NULL,
                    face_embedding vector(512)        
                );
                               
            """)

            # ENUM TYPES
            await conn.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'event_type') THEN
                        CREATE TYPE event_type AS ENUM (
                            'task', 'reminder', 'meeting', 'birthday', 'other'
                        );
                    END IF;

                    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'event_status') THEN
                        CREATE TYPE event_status AS ENUM (
                            'pending', 'in-progress', 'completed', 'dismissed'
                        );
                    END IF;

                    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'knowledge_category') THEN
                        CREATE TYPE knowledge_category AS ENUM (
                            'preference', 'memory', 'skill', 'habit', 'other'
                        );
                    END IF;
                END $$ LANGUAGE plpgsql;
            """)

            await conn.execute("""

                CREATE TABLE IF NOT EXISTS events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                               
                    username TEXT NOT NULL REFERENCES user_details(username) ON DELETE CASCADE,

                    type event_type NOT NULL,
                    description TEXT NOT NULL,

                    event_time TIMESTAMPTZ NOT NULL,
                    repeat_interval INTERVAL,

                    priority SMALLINT CHECK (priority BETWEEN 1 AND 5) DEFAULT 3, 

                    status event_status NOT NULL DEFAULT 'pending',

                    completed_at TIMESTAMPTZ, 

                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_username
                ON events(username);
            """)

            
            await conn.execute("""

                CREATE TABLE IF NOT EXISTS user_knowledge (
                    knowledge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                               
                    username TEXT NOT NULL REFERENCES user_details(username) ON DELETE CASCADE,
                               
                    fact TEXT NOT NULL, 
                               
                    category knowledge_category DEFAULT 'other',
                    importance SMALLINT CHECK (importance BETWEEN 1 AND 5) DEFAULT 3,
                                
                    embedding vector(768),
                               
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                               
                    UNIQUE (username, fact)
                );
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_knowledge_username
                ON user_knowledge(username);
            """)

            # TIMESTAMP TRIGGERS
            await conn.execute("""
                CREATE OR REPLACE FUNCTION set_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            await conn.execute("""
                CREATE OR REPLACE FUNCTION set_last_updated()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.last_updated = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            await conn.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_events_updated'
                    ) THEN
                        CREATE TRIGGER trg_events_updated
                        BEFORE UPDATE ON events
                        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_user_knowledge_updated'
                    ) THEN
                        CREATE TRIGGER trg_user_knowledge_updated
                        BEFORE UPDATE ON user_knowledge
                        FOR EACH ROW EXECUTE FUNCTION set_last_updated();
                    END IF;
                END $$ LANGUAGE plpgsql;
            """)
        except Exception as e:
            raise e
