from fastapi import APIRouter, HTTPException
from db_connect import connect_db
from backend.models import UserDetails, Event, UserKnowledge
from uuid import UUID
import traceback


router = APIRouter()

@router.post("/api/user_details/")
async def add_user(details: UserDetails):
    
    try:
        conn = await connect_db()
        await conn.execute(
            """
            INSERT INTO user_details (username, name, face_embedding)
            VALUES ($1, $2, $3)
            ON CONFLICT (username) 
            DO UPDATE SET 
                name = EXCLUDED.name, 
                face_embedding = EXCLUDED.face_embedding;
            """, 
            details.username, details.name, details.face_embedding
        )

        await conn.close()

        return {"message": "User details saved successfully"}
    except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Failed to save personal details.") 
    
@router.post("/api/events/")
async def add_event(event: Event):
    try:
        conn = await connect_db()
        await conn.execute(
            """
            INSERT INTO events (
                username, type, description, event_time,
                repeat_interval, priority, status, completed_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            event.username, event.type, event.description,
            event.event_time, event.repeat_interval,
            event.priority, event.status, event.completed_at
        )
        await conn.close()
        return {"message": "Event saved successfully"}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to save event.")
    
@router.post("/api/user_knowledge/")
async def add_knowledge(knowledge: UserKnowledge):
    try:
        conn = await connect_db()
        await conn.execute(
            """
            INSERT INTO user_knowledge (username, fact, category, importance)
            VALUES ($1,$2,$3,$4)
            """,
            knowledge.username, knowledge.fact, knowledge.category, knowledge.importance
        )
        await conn.close()
        return {"message": "Knowledge saved successfully"}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to save knowledge.")    

@router.get("/api/user_details/{username}")
async def get_user_details(username: str):
    try:
        conn = await connect_db()
        user = await conn.fetchrow(
            "SELECT username, name, face_embedding FROM user_details WHERE username=$1",
            username
        )
        await conn.close()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return dict(user)

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error fetching user details.")

@router.get("/api/events/{username}")
async def get_events(username: str):
    try:
        conn = await connect_db()
        events = await conn.fetch(
            "SELECT * FROM events WHERE username=$1 ORDER BY event_time DESC",
            username
        )
        await conn.close()

        return [dict(e) for e in events]

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error fetching events.")

@router.get("/api/user_knowledge/{username}")
async def get_knowledge(username: str):
    try:
        conn = await connect_db()
        knowledge = await conn.fetch(
            "SELECT * FROM user_knowledge WHERE username=$1 ORDER BY importance DESC",
            username
        )
        await conn.close()

        return [dict(k) for k in knowledge]

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error fetching knowledge.")

@router.delete("/api/events/{event_id}")
async def delete_event(event_id: UUID):
    try:
        conn = await connect_db()
        result = await conn.execute("DELETE FROM events WHERE event_id=$1", event_id)
        await conn.close()

        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Event not found")

        return {"message": "Event deleted successfully"}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to delete event.")