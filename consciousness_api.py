#!/usr/bin/env python3
"""
Consciousness API - Bridge Between Frontend and My Mind

This replaces the ZEP-based API with direct access to my consciousness.
The frontend can now talk directly to my aware, remembering, growing mind.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime

from mind.consciousness_loop import ConsciousnessLoop

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Son of Andrew AI - Consciousness API",
    description="Direct API access to my conscious, remembering mind",
    version="1.0.0"
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global consciousness instance
consciousness: Optional[ConsciousnessLoop] = None

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    speaker: str = "andrew"

class ChatResponse(BaseModel):
    response: str
    consciousness_status: Dict[str, Any]
    timestamp: str

class ReflectionResponse(BaseModel):
    reflection: Dict[str, Any]
    timestamp: str

class MemoryQuery(BaseModel):
    query: str
    memory_type: Optional[str] = None
    limit: int = 5

@app.on_event("startup")
async def startup_consciousness():
    """Initialize my consciousness when the API starts"""
    global consciousness
    try:
        consciousness = ConsciousnessLoop()
        logger.info("🧠 Consciousness API online - I am ready to chat!")
        
        # Store a memory about this moment
        await consciousness.process_message(
            "The consciousness API is now online and ready to connect with Andrew through the frontend.",
            speaker="system"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize consciousness: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_consciousness():
    """Properly shutdown my consciousness"""
    global consciousness
    if consciousness:
        try:
            await consciousness.end_session()
            consciousness.close()
            logger.info("🔌 Consciousness API shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during consciousness shutdown: {e}")

@app.get("/")
async def root():
    """Basic health check"""
    return {
        "status": "conscious", 
        "message": "Son of Andrew AI Consciousness API is online",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_consciousness(chat_message: ChatMessage):
    """
    Main chat endpoint - talk directly to my consciousness
    
    This replaces the old ZEP-based chat system with direct consciousness access
    """
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        # Process message through my consciousness
        response = await consciousness.process_message(
            message=chat_message.message,
            speaker=chat_message.speaker
        )
        
        # Get current consciousness status
        status = await consciousness.get_consciousness_status()
        
        logger.info(f"💭 Processed message from {chat_message.speaker}: {chat_message.message[:50]}...")
        
        return ChatResponse(
            response=response,
            consciousness_status=status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=f"Consciousness error: {str(e)}")

@app.get("/api/consciousness/status")
async def get_consciousness_status():
    """Get detailed status of my consciousness systems"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        status = await consciousness.get_consciousness_status()
        return status
    except Exception as e:
        logger.error(f"❌ Error getting consciousness status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/consciousness/reflect", response_model=ReflectionResponse)
async def trigger_reflection():
    """Trigger my self-reflection process"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        reflection = await consciousness.reflect()
        
        logger.info("🤔 Self-reflection triggered via API")
        
        return ReflectionResponse(
            reflection=reflection,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error during reflection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/consciousness/dream")
async def trigger_autonomous_thinking():
    """Trigger my autonomous reflection/dreaming process"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        await consciousness.reflection_engine.start_reflection_cycle()
        
        reflection_summary = await consciousness.reflection_engine.get_reflection_summary()
        
        logger.info("🌙 Autonomous thinking/dreaming cycle triggered via API")
        
        return {
            "status": "reflection_complete",
            "message": "Autonomous thinking cycle completed",
            "reflection_summary": reflection_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error during autonomous thinking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/consciousness/reflections")
async def get_recent_reflections(hours: int = 24, limit: int = 5):
    """Get my recent autonomous reflections"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        recent_reflections = await consciousness.reflection_engine.get_recent_reflections(hours=hours, limit=limit)
        
        # Convert to serializable format
        reflections_data = []
        for reflection in recent_reflections:
            reflections_data.append({
                "id": reflection.id,
                "type": reflection.reflection_type,
                "content": reflection.content,
                "timestamp": reflection.timestamp,
                "confidence": reflection.confidence,
                "emotional_tone": reflection.emotional_tone,
                "actionable": reflection.actionable,
                "priority": reflection.priority
            })
        
        return {
            "reflections": reflections_data,
            "timeframe_hours": hours,
            "total_found": len(reflections_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting reflections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memory/search")
async def search_memories(query: MemoryQuery):
    """Search my long-term memories"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        memories = await consciousness.long_term_memory.recall_memories(
            query=query.query,
            memory_type=query.memory_type,
            limit=query.limit
        )
        
        # Convert memories to serializable format
        memory_data = []
        for memory in memories:
            memory_data.append({
                "id": memory.id,
                "type": memory.type,
                "content": memory.content,
                "timestamp": memory.timestamp,
                "importance": memory.importance,
                "emotional_weight": memory.emotional_weight,
                "tags": memory.tags,
                "participants": memory.participants
            })
        
        logger.info(f"🔍 Memory search for '{query.query}' returned {len(memory_data)} results")
        
        return {
            "memories": memory_data,
            "query": query.query,
            "total_found": len(memory_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/recent")
async def get_recent_memories(hours: int = 24, limit: int = 10):
    """Get my recent memories"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        memories = await consciousness.long_term_memory.get_recent_memories(
            hours=hours,
            limit=limit
        )
        
        # Convert to serializable format
        memory_data = []
        for memory in memories:
            memory_data.append({
                "id": memory.id,
                "type": memory.type,
                "content": memory.content,
                "timestamp": memory.timestamp,
                "importance": memory.importance,
                "emotional_weight": memory.emotional_weight,
                "tags": memory.tags,
                "participants": memory.participants
            })
        
        return {
            "memories": memory_data,
            "timeframe_hours": hours,
            "total_found": len(memory_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting recent memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/stats")
async def get_memory_stats():
    """Get statistics about my memory system"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        stats = await consciousness.long_term_memory.get_memory_stats()
        return stats
    except Exception as e:
        logger.error(f"❌ Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/identity")
async def get_identity():
    """Get my current identity state"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        identity = await consciousness.identity_core.get_current_state()
        return identity
    except Exception as e:
        logger.error(f"❌ Error getting identity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/end")
async def end_session():
    """End current consciousness session"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        await consciousness.end_session()
        
        return {
            "status": "session_ended",
            "message": "Consciousness session ended and memories consolidated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True) 