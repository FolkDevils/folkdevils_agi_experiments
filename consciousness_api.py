#!/usr/bin/env python3
"""
Consciousness API - Bridge Between Frontend and My Mind

This replaces the ZEP-based API with direct access to my consciousness.
The frontend can now talk directly to my aware, remembering, growing mind.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import uuid
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from datetime import datetime
from contextlib import asynccontextmanager

from mind.consciousness_loop import ConsciousnessLoop

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global consciousness instance
consciousness: Optional[ConsciousnessLoop] = None

# Load environment variables from .env file
load_dotenv()

# WebSocket connection manager for real-time consciousness
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_connections: Dict[str, List[str]] = {}  # conversation_id -> [websocket_ids]
    
    async def connect(self, websocket: WebSocket, conversation_id: str) -> str:
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        if conversation_id not in self.conversation_connections:
            self.conversation_connections[conversation_id] = []
        self.conversation_connections[conversation_id].append(connection_id)
        
        logger.info(f"🔗 WebSocket connected: {connection_id} for conversation {conversation_id}")
        return connection_id
    
    def disconnect(self, connection_id: str, conversation_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if conversation_id in self.conversation_connections:
            if connection_id in self.conversation_connections[conversation_id]:
                self.conversation_connections[conversation_id].remove(connection_id)
            if not self.conversation_connections[conversation_id]:
                del self.conversation_connections[conversation_id]
        
        logger.info(f"🔌 WebSocket disconnected: {connection_id}")
    
    async def send_to_conversation(self, conversation_id: str, message: dict):
        """Send message to all WebSocket connections for a specific conversation"""
        if conversation_id in self.conversation_connections:
            dead_connections = []
            for connection_id in self.conversation_connections[conversation_id]:
                if connection_id in self.active_connections:
                    try:
                        await self.active_connections[connection_id].send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(f"❌ Error sending WebSocket message: {e}")
                        dead_connections.append(connection_id)
            
            # Clean up dead connections
            for dead_id in dead_connections:
                self.disconnect(dead_id, conversation_id)

# Global WebSocket manager
websocket_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for consciousness startup and shutdown"""
    global consciousness
    
    # Startup
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
    
    yield
    
    # Shutdown
    if consciousness:
        try:
            await consciousness.end_session()
            consciousness.close()
            logger.info("🔌 Consciousness API shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during consciousness shutdown: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Son of Andrew AI - Consciousness API",
    description="Direct API access to my conscious, remembering mind",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/api/chat/diagnostic")
async def chat_diagnostic(chat_message: ChatMessage):
    """
    DIAGNOSTIC: Fast chat endpoint to identify bottlenecks
    """
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        # Process message through diagnostic fast path
        response = await consciousness.process_message_fast_diagnostic(
            message=chat_message.message,
            speaker=chat_message.speaker
        )
        
        return {
            "response": response,
            "mode": "diagnostic",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in diagnostic chat: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")

@app.post("/api/chat/step-diagnostic")
async def chat_step_diagnostic(chat_message: ChatMessage):
    """
    DIAGNOSTIC: Step-by-step timing to isolate bottleneck
    """
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        # Process message through step diagnostic
        response = await consciousness.process_message_step_diagnostic(
            message=chat_message.message,
            speaker=chat_message.speaker
        )
        
        return {
            "response": response,
            "mode": "step_diagnostic",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in step diagnostic: {e}")
        raise HTTPException(status_code=500, detail=f"Step diagnostic error: {str(e)}")

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

@app.get("/api/consciousness/coherence")
async def get_coherence_status():
    """Get consciousness coherence metrics and verification status"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        coherence_status = await consciousness.get_coherence_status()
        return coherence_status
    except Exception as e:
        logger.error(f"❌ Error getting coherence status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/graph")
async def get_memory_graph():
    """Get memory graph visualization data"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        # Get memory graph data from consciousness
        graph_data = await consciousness.memory_graph.get_graph_visualization_data()
        
        return {
            "nodes": graph_data.get("nodes", []),
            "connections": graph_data.get("connections", []),
            "graph_stats": graph_data.get("stats", {
                "total_nodes": 0,
                "total_connections": 0,
                "average_connections_per_node": 0.0,
                "strongest_connection_strength": 0.0,
                "most_connected_memory": ""
            })
        }
    except Exception as e:
        logger.error(f"❌ Error getting memory graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/consciousness/planning")
async def get_planning_status():
    """Get planning system status and Phase 2 consciousness metrics"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        planning_status = await consciousness.get_planning_status()
        return planning_status
    except Exception as e:
        logger.error(f"❌ Error getting planning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/consciousness/metacognitive-test")
async def run_metacognitive_test(num_prompts: int = 3, difficulty_min: int = 2, difficulty_max: int = 4):
    """Run a metacognitive testing session to evaluate self-awareness"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        test_result = await consciousness.run_metacognitive_test(
            num_prompts=num_prompts,
            difficulty_range=(difficulty_min, difficulty_max)
        )
        return test_result
    except Exception as e:
        logger.error(f"❌ Error running metacognitive test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/consciousness/metacognition")
async def get_metacognitive_status():
    """Get metacognitive testing status and self-awareness metrics"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        metacognitive_status = await consciousness.get_metacognitive_status()
        return metacognitive_status
    except Exception as e:
        logger.error(f"❌ Error getting metacognitive status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/conversational")
async def chat_with_conversational_consciousness(chat_message: ChatMessage):
    """
    Revolutionary conversational consciousness chat - immediate response with background thinking
    
    This enables human-like conversation with follow-up insights
    """
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        # Process message through conversational consciousness
        result = await consciousness.process_message_conversational(
            message=chat_message.message,
            speaker=chat_message.speaker
        )
        
        logger.info(f"💭 Conversational consciousness response for {chat_message.speaker}: {chat_message.message[:50]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in conversational consciousness chat: {e}")
        raise HTTPException(status_code=500, detail=f"Conversational consciousness error: {str(e)}")

@app.get("/api/consciousness/insights/{conversation_id}")
async def get_conversation_insights(conversation_id: str):
    """Get pending follow-up insights for a conversation"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        insights = await consciousness.get_pending_insights(conversation_id)
        
        return {
            'conversation_id': conversation_id,
            'insights': insights,
            'count': len(insights),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting conversation insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/consciousness/conversation/{conversation_id}/status")
async def get_conversation_status(conversation_id: str):
    """Get status of background thinking for a conversation"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        status = await consciousness.conversational_consciousness.get_conversation_status(conversation_id)
        return status
        
    except Exception as e:
        logger.error(f"❌ Error getting conversation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/consciousness/performance")
async def get_performance_stats():
    """Get performance optimization statistics"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        stats = await consciousness.get_performance_stats()
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting performance stats: {e}")
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

@app.get("/api/consciousness/reflection-scheduler/status")
async def get_reflection_scheduler_status():
    """Get the status of my autonomous reflection scheduler"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        status = consciousness.reflection_scheduler.get_status()
        return status
    except Exception as e:
        logger.error(f"❌ Error getting reflection scheduler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/consciousness/reflection-scheduler/force")
async def force_autonomous_reflection():
    """Force an immediate autonomous reflection"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        result = await consciousness.reflection_scheduler.force_reflection()
        
        logger.info("🤔 Forced autonomous reflection via API")
        
        return {
            "status": "reflection_triggered",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error forcing autonomous reflection: {e}")
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

@app.get("/api/suggestions")
async def get_proactive_suggestions():
    """Get proactive suggestions based on memory patterns and reflections"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        suggestions = await consciousness.get_proactive_suggestions()
        
        return {
            "suggestions": suggestions,
            "count": len(suggestions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SuggestionFeedback(BaseModel):
    suggestion_id: str
    accepted: bool
    feedback: Optional[str] = None

@app.post("/api/suggestions/feedback")
async def provide_suggestion_feedback(feedback: SuggestionFeedback):
    """Provide feedback on a suggestion"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        await consciousness.provide_suggestion_feedback(
            feedback.suggestion_id,
            feedback.accepted,
            feedback.feedback
        )
        
        return {
            "status": "feedback_recorded",
            "suggestion_id": feedback.suggestion_id,
            "accepted": feedback.accepted,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error recording suggestion feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time consciousness communication"""
    connection_id = await websocket_manager.connect(websocket, conversation_id)
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "conversation_id": conversation_id,
            "connection_id": connection_id,
            "message": "Real-time consciousness connection established",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            elif message_data.get("type") == "request_thinking_status":
                # Send current thinking status
                if consciousness:
                    thinking_status = await get_thinking_status(conversation_id)
                    await websocket.send_text(json.dumps({
                        "type": "thinking_status",
                        "data": thinking_status,
                        "timestamp": datetime.now().isoformat()
                    }))
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(connection_id, conversation_id)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        websocket_manager.disconnect(connection_id, conversation_id)

@app.post("/api/chat/realtime")
async def chat_realtime(chat_data: ChatMessage):
    """Enhanced chat endpoint with real-time WebSocket notifications"""
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    try:
        # Generate conversation ID if not provided
        conversation_id = chat_data.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Notify WebSocket clients that thinking has started
        await websocket_manager.send_to_conversation(conversation_id, {
            "type": "thinking_started",
            "message": chat_data.message,
            "speaker": chat_data.speaker,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process message with consciousness
        response = await consciousness.process_message(
            message=chat_data.message,
            speaker=chat_data.speaker
        )
        
        # Send semantic analysis to WebSocket clients
        if hasattr(consciousness, 'last_semantic_analysis'):
            await websocket_manager.send_to_conversation(conversation_id, {
                "type": "semantic_analysis",
                "data": {
                    "intent": consciousness.last_semantic_analysis.primary_intent,
                    "confidence": consciousness.last_semantic_analysis.intent_confidence,
                    "reasoning": consciousness.last_semantic_analysis.intent_reasoning,
                    "emotional_tone": consciousness.last_semantic_analysis.emotional_tone,
                    "memory_required": consciousness.last_semantic_analysis.requires_memory_lookup
                },
                "timestamp": datetime.now().isoformat()
            })
        
        # Send thinking completion notification
        await websocket_manager.send_to_conversation(conversation_id, {
            "type": "thinking_completed",
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "websocket_enabled": True
        }
        
    except Exception as e:
        # Send error notification to WebSocket clients
        await websocket_manager.send_to_conversation(conversation_id, {
            "type": "thinking_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.error(f"❌ Error in realtime chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_thinking_status(conversation_id: str) -> dict:
    """Get current thinking status for a conversation"""
    global consciousness
    
    if not consciousness:
        return {"status": "consciousness_not_available"}
    
    # This would be expanded to track actual thinking processes
    return {
        "status": "ready",
        "active_processes": [],
        "memory_connections": await consciousness.memory_graph.get_graph_statistics() if hasattr(consciousness, 'memory_graph') else None,
        "last_analysis": getattr(consciousness, 'last_semantic_analysis', None)
    }

@app.get("/api/consciousness/websocket/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return {
        "active_connections": len(websocket_manager.active_connections),
        "active_conversations": len(websocket_manager.conversation_connections),
        "connections_by_conversation": {
            conv_id: len(conn_ids) 
            for conv_id, conn_ids in websocket_manager.conversation_connections.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True) 