import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def connect_db():
    conn = await asyncpg.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

    await conn.execute("""

        CREATE TABLE IF NOT EXISTS user_details (
            username TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            face_embedding FLOAT8[]
        );

    """)

    await conn.execute("""

        CREATE TABLE IF NOT EXISTS events (
            event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username TEXT NOT NULL REFERENCES user_details(username) ON DELETE CASCADE,

            type TEXT NOT NULL CHECK (type IN ('task', 'reminder', 'meeting', 'birthday', 'other')),
            description TEXT NOT NULL,

            event_time TIMESTAMP NOT NULL, -- due date / reminder time
            repeat_interval INTERVAL, -- NULL if one-time, e.g., '1 day', '1 week'

            priority SMALLINT CHECK (priority BETWEEN 1 AND 5) DEFAULT 3, -- urgency

            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'in-progress', 'completed', 'dismissed')),

            completed_at TIMESTAMP, -- for completed tasks or acknowledged reminders

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

    """)

    await conn.execute("""

        CREATE TABLE IF NOT EXISTS user_knowledge (
            knowledge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username TEXT NOT NULL REFERENCES user_details(username) ON DELETE CASCADE,
            fact TEXT NOT NULL, -- the information about the user
            category TEXT CHECK (category IN ('preference', 'memory', 'skill', 'habit', 'other')) DEFAULT 'other',
            importance SMALLINT CHECK (importance BETWEEN 1 AND 5) DEFAULT 3, -- priority for recall
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

    """)
    
    return conn 
