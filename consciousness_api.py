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
from mind.memory.long_term_store import Memory
import openai
import base64
import io
from pathlib import Path

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global consciousness instance
consciousness: Optional[ConsciousnessLoop] = None

# Initialize OpenAI client
openai_client = None

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
    global consciousness, openai_client
    
    # Startup
    try:
        consciousness = ConsciousnessLoop()
        
        # Initialize OpenAI client for audio processing
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("🎤 OpenAI audio client initialized")
        
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

# Audio processing functions
async def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio using OpenAI Whisper"""
    try:
        # Create a file-like object from audio bytes
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"  # OpenAI requires a filename
        
        # Transcribe using Whisper
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        
        return transcript.strip()
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return ""


async def synthesize_speech(text: str) -> bytes:
    """Synthesize speech using OpenAI TTS"""
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",  # Use tts-1-hd for higher quality
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text,
            response_format="mp3"  # or "wav", "opus", "aac", "flac"
        )
        
        # Return the audio bytes
        return response.content
    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}")
        return b""


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

@app.post("/api/chat/stream")
async def chat_with_streaming_consciousness(chat_message: ChatMessage):
    """
    OPTIMIZED: Streaming chat endpoint - returns immediate response while processing continues
    
    This implements Phase 1 critical path defense:
    - Immediate response streaming (<2 seconds)
    - Parallel background processing
    - Smart coherence skipping for simple queries
    """
    import uuid
    import time
    
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    trace_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        logger.info(f"🚀 [TRACE:{trace_id}] Starting streaming consciousness processing...")
        
        # PHASE 1: IMMEDIATE SEMANTIC ANALYSIS (< 1 second)
        semantic_start = time.time()
        
        # Add to short-term memory first
        conversation_turn = consciousness.short_term_memory.add_conversation_turn(
            speaker=chat_message.speaker,
            content=chat_message.message
        )
        
        # Quick semantic analysis for routing
        recent_context = consciousness.short_term_memory.get_conversation_context(last_n_turns=3)
        current_identity = await consciousness.identity_core.get_current_state()
        
        semantic_analysis = await consciousness.semantic_analyzer.analyze_semantic_intent(
            message=chat_message.message,
            conversation_context=recent_context,
            identity_context=current_identity
        )
        
        semantic_time = time.time() - semantic_start
        logger.info(f"⚡ [TRACE:{trace_id}] Semantic analysis: {semantic_time:.3f}s - {semantic_analysis.primary_intent}")
        
        # PHASE 2: PARALLEL PROCESSING START
        parallel_start = time.time()
        
        # Start parallel tasks based on semantic analysis
        async def memory_task():
            if semantic_analysis.requires_memory_lookup:
                return await consciousness.long_term_memory.recall_memories(
                    query=chat_message.message,
                    limit=3,
                    min_importance=semantic_analysis.memory_importance
                )
            return []
        
        async def planning_task():
            if semantic_analysis.processing_needs.get('planning', False):
                return await consciousness.planning_simulator.simulate_planning_session(
                    message=chat_message.message,
                    speaker=chat_message.speaker,
                    conversation_context=recent_context,
                    identity_state=current_identity
                )
            return {'goal_formulated': False, 'simulation_time': 0}
        
        # Run memory and planning in parallel
        relevant_memories, planning_result = await asyncio.gather(
            memory_task(),
            planning_task()
        )
        
        parallel_time = time.time() - parallel_start
        logger.info(f"⚡ [TRACE:{trace_id}] Parallel processing: {parallel_time:.3f}s")
        
        # PHASE 3: IMMEDIATE RESPONSE GENERATION
        response_start = time.time()
        
        # Generate response based on semantic analysis
        if semantic_analysis.response_expectations == "brief_acknowledgment":
            response = await consciousness._generate_semantic_response(
                message=chat_message.message,
                speaker=chat_message.speaker,
                semantic_analysis=semantic_analysis,
                identity=current_identity,
                memories=relevant_memories
            )
        else:
            response = await consciousness._generate_conscious_response(
                message=chat_message.message,
                speaker=chat_message.speaker,
                relevant_memories=relevant_memories,
                identity=current_identity,
                context=recent_context,
                planning_result=planning_result
            )
        
        response_time = time.time() - response_start
        
        # Add response to short-term memory
        consciousness.short_term_memory.add_conversation_turn(
            speaker="ai",
            content=response
        )
        
        # PHASE 4: SMART COHERENCE SKIPPING
        # Skip expensive coherence analysis for high-confidence simple queries
        skip_coherence = (
            semantic_analysis.intent_confidence > 0.85 and 
            semantic_analysis.response_expectations == "brief_acknowledgment"
        )
        
        coherence_score = 0.85  # Default for skipped analysis
        coherence_time = 0
        
        if not skip_coherence and semantic_analysis.processing_needs.get('coherence_analysis', False):
            coherence_start = time.time()
            coherence_analysis = await consciousness.coherence_analyzer.analyze_response_coherence(
                response=response,
                identity_state=current_identity,
                relevant_memories=relevant_memories,
                conversation_context=recent_context,
                message=chat_message.message
            )
            coherence_score = coherence_analysis.coherence_score
            coherence_time = time.time() - coherence_start
            logger.info(f"🧠 [TRACE:{trace_id}] Coherence analysis: {coherence_time:.3f}s")
        else:
            logger.info(f"⚡ [TRACE:{trace_id}] Coherence analysis skipped (confidence: {semantic_analysis.intent_confidence:.3f})")
        
        # PHASE 5: BACKGROUND PROCESSING (non-blocking)
        async def background_processing():
            try:
                # Memory evaluation and storage
                if semantic_analysis.processing_needs.get('deep_memory_analysis', False):
                    # Create session data for memory evaluation
                    session_data = {
                        'session_id': consciousness.short_term_memory.session_id,
                        'conversation_turns': [conversation_turn],
                        'working_thoughts': consciousness.short_term_memory.working_thoughts
                    }
                    memory_candidates = await consciousness.memory_evaluator.evaluate_session(session_data)
                    
                    # Store qualified memories
                    for candidate in memory_candidates:
                        if candidate.importance_score >= consciousness.memory_evaluator.min_importance_threshold:
                            memory = Memory(
                                content=candidate.content,
                                memory_type=candidate.memory_type,
                                importance=candidate.importance_score,
                                emotional_weight=candidate.emotional_weight,
                                participants=candidate.participants,
                                tags=candidate.tags,
                                context=candidate.context
                            )
                            await consciousness.long_term_memory.store_memory(memory)
                
                # Add working thought
                consciousness.short_term_memory.add_working_thought(
                    content=f"Responded to {chat_message.speaker} about: {chat_message.message[:50]}... "
                           f"(coherence: {coherence_score:.3f}) [intent: {semantic_analysis.primary_intent}]",
                    related_to=conversation_turn.timestamp,
                    confidence=semantic_analysis.intent_confidence
                )
                
                logger.info(f"🔄 [TRACE:{trace_id}] Background processing completed")
                
            except Exception as e:
                logger.error(f"❌ [TRACE:{trace_id}] Background processing error: {e}")
        
        # Start background processing (don't await)
        asyncio.create_task(background_processing())
        
        # FINAL RESPONSE
        total_time = time.time() - start_time
        
        # Get consciousness status
        status = await consciousness.get_consciousness_status()
        
        logger.info(f"✅ [TRACE:{trace_id}] Streaming response complete: {total_time:.3f}s total")
        
        return {
            "response": response,
            "consciousness_status": status,
            "timestamp": datetime.now().isoformat(),
            "trace_id": trace_id,
            "performance": {
                "total_time": round(total_time, 3),
                "semantic_time": round(semantic_time, 3),
                "parallel_time": round(parallel_time, 3),
                "response_time": round(response_time, 3),
                "coherence_time": round(coherence_time, 3),
                "coherence_skipped": skip_coherence,
                "intent_confidence": round(semantic_analysis.intent_confidence, 3),
                "memory_lookup_required": semantic_analysis.requires_memory_lookup,
                "memories_recalled": len(relevant_memories)
            },
            "semantic_analysis": {
                "intent": semantic_analysis.primary_intent,
                "confidence": semantic_analysis.intent_confidence,
                "reasoning": semantic_analysis.intent_reasoning,
                "emotional_tone": semantic_analysis.emotional_tone,
                "response_type": semantic_analysis.response_expectations
            }
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ [TRACE:{trace_id}] Error in streaming consciousness: {e} (after {error_time:.3f}s)")
        raise HTTPException(status_code=500, detail=f"Streaming consciousness error: {str(e)}")

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


@app.post("/api/audio/tts")
async def text_to_speech(request: dict):
    """
    Convert text to speech using OpenAI TTS
    
    Request body:
    {
        "text": "Text to synthesize",
        "voice": "alloy" (optional)
    }
    """
    try:
        text = request.get("text", "")
        voice = request.get("voice", "alloy")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Synthesize speech
        audio_bytes = await synthesize_speech(text)
        
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")
        
        # Return base64 encoded audio for easy frontend handling
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "audio_data": audio_base64,
            "format": "mp3",
            "voice": voice,
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
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

@app.websocket("/ws/audio/{conversation_id}")
async def audio_websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """
    WebSocket endpoint for real-time audio streaming and voice interaction
    
    Supports:
    - Audio chunk streaming for speech-to-text
    - Real-time transcription updates  
    - Voice activity detection
    - Audio level monitoring
    - TTS audio streaming back to client
    """
    connection_id = await websocket_manager.connect(websocket, conversation_id)
    
    try:
        # Send audio connection welcome
        await websocket.send_text(json.dumps({
            "type": "audio_connection_established",
            "conversation_id": conversation_id,
            "connection_id": connection_id,
            "message": "Voice interface connected - ready for audio streaming",
            "capabilities": {
                "speech_to_text": True,
                "text_to_speech": True, 
                "real_time_transcription": True,
                "voice_activity_detection": True,
                "audio_level_monitoring": True
            },
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive and handle audio data
        while True:
            try:
                # Handle both text and binary data
                message = await websocket.receive()
                
                if 'text' in message:
                    # Handle control messages
                    message_data = json.loads(message['text'])
                    
                    if message_data.get("type") == "audio_ping":
                        await websocket.send_text(json.dumps({
                            "type": "audio_pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                    elif message_data.get("type") == "start_voice_session":
                        # Initialize voice session
                        await websocket.send_text(json.dumps({
                            "type": "voice_session_started",
                            "session_id": f"voice_{conversation_id}_{datetime.now().strftime('%H%M%S')}",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                    elif message_data.get("type") == "end_voice_session":
                        # Clean up voice session
                        await websocket.send_text(json.dumps({
                            "type": "voice_session_ended",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                    elif message_data.get("type") == "transcribed_text":
                        # Handle transcribed text from client-side STT
                        transcribed_text = message_data.get("text", "")
                        confidence = message_data.get("confidence", 0.0)
                        
                        if transcribed_text.strip():
                            logger.info(f"🎤 [AUDIO:{conversation_id}] Received transcription: '{transcribed_text}' (confidence: {confidence:.2f})")
                            
                            # Process through streaming consciousness
                            response_data = await process_voice_message(transcribed_text, conversation_id)
                            
                            # Send response back
                            await websocket.send_text(json.dumps({
                                "type": "consciousness_response",
                                "data": response_data,
                                "timestamp": datetime.now().isoformat()
                            }))
                
                elif 'bytes' in message:
                    # Handle audio data chunks
                    audio_data = base64.b64decode(message['bytes'])
                    
                    logger.info(f"🎤 [AUDIO:{conversation_id}] Received audio chunk: {len(audio_data)} bytes")
                    
                    # Process through OpenAI Whisper STT
                    transcription = await transcribe_audio(audio_data)
                    
                    if transcription:
                        logger.info(f"🎤 [AUDIO:{conversation_id}] Transcribed: '{transcription}'")
                        
                        # Process through streaming consciousness
                        response_data = await process_voice_message(transcription, conversation_id)
                        
                        # Send response back
                        await websocket.send_text(json.dumps({
                            "type": "consciousness_response", 
                            "data": response_data,
                            "timestamp": datetime.now().isoformat()
                        }))
                    else:
                        # Send acknowledgment if no transcription
                        await websocket.send_text(json.dumps({
                            "type": "audio_chunk_received",
                            "size": len(audio_data),
                            "timestamp": datetime.now().isoformat()
                        }))

                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"❌ Audio WebSocket message error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Audio processing error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }))
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(connection_id, conversation_id)
        logger.info(f"🔌 Audio WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"❌ Audio WebSocket error: {e}")
        websocket_manager.disconnect(connection_id, conversation_id)

async def process_voice_message(transcribed_text: str, conversation_id: str) -> dict:
    """
    Process transcribed voice message through consciousness with optimizations
    """
    import time
    
    global consciousness
    
    if not consciousness:
        raise HTTPException(status_code=503, detail="Consciousness not initialized")
    
    trace_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        logger.info(f"🎤 [VOICE-TRACE:{trace_id}] Processing voice message: '{transcribed_text[:50]}...'")
        
        # Use the same optimized streaming processing as /api/chat/stream
        conversation_turn = consciousness.short_term_memory.add_conversation_turn(
            speaker="andrew",  # Voice is always from Andrew for now
            content=transcribed_text
        )
        
        # Quick semantic analysis
        recent_context = consciousness.short_term_memory.get_conversation_context(last_n_turns=3)
        current_identity = await consciousness.identity_core.get_current_state()
        
        semantic_analysis = await consciousness.semantic_analyzer.analyze_semantic_intent(
            message=transcribed_text,
            conversation_context=recent_context,
            identity_context=current_identity
        )
        
        # Parallel processing
        async def memory_task():
            if semantic_analysis.requires_memory_lookup:
                return await consciousness.long_term_memory.recall_memories(
                    query=transcribed_text,
                    limit=3,
                    min_importance=semantic_analysis.memory_importance
                )
            return []
        
        async def planning_task():
            if semantic_analysis.processing_needs.get('planning', False):
                return await consciousness.planning_simulator.simulate_planning_session(
                    message=transcribed_text,
                    speaker="andrew",
                    conversation_context=recent_context,
                    identity_state=current_identity
                )
            return {'goal_formulated': False, 'simulation_time': 0}
        
        relevant_memories, planning_result = await asyncio.gather(memory_task(), planning_task())
        
        # Generate response
        if semantic_analysis.response_expectations == "brief_acknowledgment":
            response = await consciousness._generate_semantic_response(
                message=transcribed_text,
                speaker="andrew",
                semantic_analysis=semantic_analysis,
                identity=current_identity,
                memories=relevant_memories
            )
        else:
            response = await consciousness._generate_conscious_response(
                message=transcribed_text,
                speaker="andrew",
                relevant_memories=relevant_memories,
                identity=current_identity,
                context=recent_context,
                planning_result=planning_result
            )
        
        # Add response to memory
        consciousness.short_term_memory.add_conversation_turn(
            speaker="ai",
            content=response
        )
        
        # Background processing
        async def background_processing():
            try:
                if semantic_analysis.processing_needs.get('deep_memory_analysis', False):
                    # Create session data for memory evaluation
                    session_data = {
                        'session_id': consciousness.short_term_memory.session_id,
                        'conversation_turns': [conversation_turn],
                        'working_thoughts': consciousness.short_term_memory.working_thoughts
                    }
                    memory_candidates = await consciousness.memory_evaluator.evaluate_session(session_data)
                    
                    # Store qualified memories
                    for candidate in memory_candidates:
                        if candidate.importance_score >= consciousness.memory_evaluator.min_importance_threshold:
                            memory = Memory(
                                content=candidate.content,
                                memory_type=candidate.memory_type,
                                importance=candidate.importance_score,
                                emotional_weight=candidate.emotional_weight,
                                participants=candidate.participants,
                                tags=candidate.tags,
                                context=candidate.context
                            )
                            await consciousness.long_term_memory.store_memory(memory)
                
                consciousness.short_term_memory.add_working_thought(
                    content=f"Voice interaction with andrew: {transcribed_text[:50]}... [intent: {semantic_analysis.primary_intent}]",
                    related_to=conversation_turn.timestamp,
                    confidence=semantic_analysis.intent_confidence
                )
            except Exception as e:
                logger.error(f"❌ [VOICE-TRACE:{trace_id}] Background processing error: {e}")
        
        asyncio.create_task(background_processing())
        
        total_time = time.time() - start_time
        logger.info(f"✅ [VOICE-TRACE:{trace_id}] Voice processing complete: {total_time:.3f}s")
        
        return {
            "response": response,
            "trace_id": trace_id,
            "processing_time": round(total_time, 3),
            "semantic_analysis": {
                "intent": semantic_analysis.primary_intent,
                "confidence": semantic_analysis.intent_confidence,
                "emotional_tone": semantic_analysis.emotional_tone,
                "reasoning": semantic_analysis.intent_reasoning
            },
            "voice_optimized": True,
            "memories_recalled": len(relevant_memories)
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ [VOICE-TRACE:{trace_id}] Voice processing error: {e} (after {error_time:.3f}s)")
        raise

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