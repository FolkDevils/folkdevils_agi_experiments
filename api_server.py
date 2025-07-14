from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
import time
from datetime import datetime

# Import the MetaAgent system
from agents.meta_agent import meta_agent
from memory_manager import memory_manager, get_zep_status
from config import settings
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown lifecycle"""
    # Startup
    logger.info("🚀 Son of Andrew API starting up with MetaAgent architecture...")
    yield
    # Shutdown
    logger.info("🛑 Shutting down Son of Andrew API...")
    try:
        await memory_manager.cleanup_background_tasks()
        logger.info("✅ Background task cleanup completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown cleanup: {e}")

app = FastAPI(title="Son of Andrew API", version="2.0.0", lifespan=lifespan)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_context: Optional[list] = []  # New field for conversation history
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validate message content
        if not self.message or not self.message.strip():
            raise ValueError("Message cannot be empty")
        if len(self.message) > 5000:  # Reasonable limit
            raise ValueError("Message too long (max 5000 characters)")
        if len(self.message.strip()) < 2:
            raise ValueError("Message too short (min 2 characters)")
        
        # Validate session_id format if provided
        if self.session_id and not self.session_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Invalid session_id format")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    command_used: Optional[str] = None
    reasoning: Optional[str] = None
    memory_context: Optional[str] = None
    voice_compliance_score: Optional[int] = None
    learning_detected: Optional[bool] = None
    preference_learned: Optional[str] = None
    plan_steps: Optional[int] = None  # Number of execution steps taken
    timestamp: int  # Unix timestamp for cache validation
    cache_version: str  # Server version for cache invalidation
    
    # Enhanced error reporting fields
    intelligence_status: Optional[str] = None  # ✅ SUCCESS, ⚠️ FALLBACK, ❌ FAILED, 💥 CRITICAL_FAILURE
    intelligence_error: Optional[str] = None  # Error details when intelligence fails
    fallback_used: Optional[bool] = None  # Whether fallback processing was used

class SessionResponse(BaseModel):
    session_id: str
    message: str

@app.get("/")
async def root():
    return {"message": "Son of Andrew API v2.0 - MetaAgent Architecture", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/debug/memory-status")
async def memory_status():
    """Debug endpoint to check memory system status"""
    zep_status = get_zep_status()
    return {
        "zep_connection": zep_status,
        "memory_system": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response from Son of Andrew using the new plan-based architecture"""
    try:
        # Validate request first
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Sanitize input
        user_message = request.message.strip()[:5000]  # Truncate if too long
        
        # Ensure session exists with robust error handling
        session_id = request.session_id
        try:
            if not session_id:
                session_id = await memory_manager.create_session("conversation")
            else:
                session_id = await memory_manager.ensure_session_exists(session_id)
            
            # Fallback if session creation fails
            if not session_id:
                session_id = f"fallback_session_{int(time.time())}"
                logger.warning(f"Session creation failed, using fallback: {session_id}")
                
        except Exception as session_error:
            session_id = f"error_session_{int(time.time())}"
            logger.error(f"Session management failed: {session_error}, using fallback: {session_id}")
        
        # Use MetaAgent for intelligent request processing
        meta_result = await meta_agent.process_request(
            user_input=user_message,
            session_id=session_id,
            conversation_context=request.conversation_context or []
        )
        
        # Extract response details
        response_text = meta_result.get("content", "I apologize, but I couldn't process your request.")
        agent_used = meta_result.get("agent_used", "MetaAgent")
        execution_successful = meta_result.get("execution_successful", False)
        plan_steps = meta_result.get("plan_steps", 1)
        
        # Set intelligence status based on execution
        intelligence_status = "✅ SUCCESS" if execution_successful else "❌ FAILED"
        intelligence_error = meta_result.get("error", None)
        fallback_used = not execution_successful
        
        # Get memory context and session summary with error handling
        memory_context = ""
        session_summary = {}
        
        try:
            # 🚀 SHARED MEMORY CONTEXT: Use memory context from MetaAgent instead of duplicate call
            memory_context = meta_result.get("memory_context", "")
            if not memory_context:
                # Fallback to direct memory manager call only if MetaAgent didn't provide context
                logger.warning("⚠️ No memory context from MetaAgent, falling back to direct call")
                memory_context = await memory_manager.get_memory_context(session_id, user_message)
        except Exception as memory_error:
            logger.warning(f"Memory context retrieval failed: {memory_error}")
            memory_context = ""
        
        try:
            session_summary = await memory_manager.get_session_summary(session_id)
        except Exception as summary_error:
            logger.warning(f"Session summary retrieval failed: {summary_error}")
            session_summary = {}
        
        # Check if learning was detected
        learning_detected = (agent_used == "LearningAgent")
        preference_learned = "Information stored successfully" if learning_detected and execution_successful else None
        
        # Build reasoning
        plan_reasoning = f"MetaAgent orchestrated {agent_used} for this request"
        if intelligence_error:
            plan_reasoning = f"MetaAgent execution: {intelligence_status} - {intelligence_error}"
        
        # CRITICAL: Ensure session_id is never None with multiple checks
        if not session_id or session_id == "None" or not isinstance(session_id, str):
            session_id = f"emergency_session_{int(time.time())}"
            logger.error(f"Session ID was invalid ({session_id}), creating emergency session")
        
        # Double-check that meta_result doesn't have None session_id
        meta_session_id = meta_result.get("session_id")
        if meta_session_id and isinstance(meta_session_id, str):
            session_id = meta_session_id
        
        # FINAL CHECK: Absolutely ensure we have a valid string session_id
        if not session_id or not isinstance(session_id, str) or session_id.strip() == "":
            session_id = f"final_fallback_session_{int(time.time())}"
            logger.error(f"Final session_id check failed, using ultimate fallback: {session_id}")
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            command_used=agent_used,
            reasoning=f"MetaAgent orchestrated {agent_used} for this request",
            memory_context=memory_context,
            voice_compliance_score=session_summary.get("voice_compliance_score", 100),
            learning_detected=learning_detected,
            preference_learned=preference_learned,
            plan_steps=plan_steps,
            timestamp=int(time.time()),  # Current Unix timestamp
            cache_version="2.0.0",  # Server version for cache validation
            intelligence_status=intelligence_status,
            intelligence_error=intelligence_error,
            fallback_used=fallback_used
        )
        
    except ValueError as e:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        
        # Ensure we have a valid session_id for error response
        error_session_id = getattr(request, 'session_id', None) or f"error_session_{int(time.time())}"
        
        return ChatResponse(
            response="I'm experiencing technical difficulties with both my advanced and basic processing systems. Please try again in a moment.",
            session_id=error_session_id,
            command_used=None,
            reasoning=f"Critical system error: {str(e)}",
            memory_context=None,
            voice_compliance_score=None,
            learning_detected=None,
            preference_learned=None,
            plan_steps=None,
            timestamp=int(time.time()),
            cache_version="2.0.0",
            intelligence_status="💥 CRITICAL_FAILURE",
            intelligence_error=str(e),
            fallback_used=False
        )

@app.post("/api/session", response_model=SessionResponse)
async def create_session():
    """Create a new conversation session"""
    try:
        session_id = await memory_manager.create_session("conversation")
        return SessionResponse(
            session_id=session_id,
            message="New session created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.get("/api/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get summary and analytics for a session"""
    try:
        summary = await memory_manager.get_session_summary(session_id)
        if "error" in summary:
            raise HTTPException(status_code=404, detail="Session not found")
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session summary: {str(e)}")

@app.get("/api/session/{session_id}/memory")
async def get_memory_context(session_id: str, query: Optional[str] = None):
    """Get memory context for a session"""
    try:
        context = await memory_manager.get_memory_context(session_id, query or "")
        return {
            "session_id": session_id,
            "memory_context": context,
            "query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting memory context: {str(e)}")

@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    try:
        sessions = await memory_manager.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

# 🆕 Opportunistic Learning Endpoints

@app.get("/api/session/{session_id}/learning-summary")
async def get_learning_summary(session_id: str):
    """Get current learning candidates for a session"""
    try:
        summary = meta_agent.get_session_learning_summary(session_id)
        return {
            "status": "success",
            "learning_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get learning summary for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning summary: {str(e)}")

@app.post("/api/session/{session_id}/end-session-learning")
async def end_session_learning(session_id: str):
    """Perform comprehensive learning analysis at the end of a session"""
    try:
        summary = await meta_agent._end_of_session_learning(session_id)
        return {
            "status": "success", 
            "learning_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ End-of-session learning failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"End-of-session learning failed: {str(e)}")

@app.post("/api/session/{session_id}/confirm-learning")
async def confirm_learning_candidate(session_id: str, request: dict):
    """Confirm or reject a learning candidate"""
    try:
        candidate_index = request.get("candidate_index")
        action = request.get("action", "confirm")  # "confirm" or "reject"
        
        if candidate_index is None:
            raise HTTPException(status_code=400, detail="candidate_index is required")
        
        # Get current learning candidates
        candidates = meta_agent.session_learning_candidates.get(session_id, [])
        
        if candidate_index >= len(candidates):
            raise HTTPException(status_code=404, detail="Learning candidate not found")
        
        candidate = candidates[candidate_index]
        
        if action == "confirm":
            # Store the confirmed fact
            from agents.memory_agent import memory_agent
            success = await memory_agent.save_fact(
                candidate.get("fact", ""),
                candidate.get("category", "general")
            )
            
            if success:
                # Remove from candidates
                candidates.pop(candidate_index)
                message = f"Learning candidate confirmed and stored: {candidate.get('fact', '')[:50]}..."
            else:
                message = "Failed to store confirmed learning candidate"
                
        elif action == "reject":
            # Remove from candidates without storing
            candidates.pop(candidate_index)
            message = f"Learning candidate rejected: {candidate.get('fact', '')[:50]}..."
        else:
            raise HTTPException(status_code=400, detail="action must be 'confirm' or 'reject'")
        
        return {
            "status": "success",
            "message": message,
            "remaining_candidates": len(candidates),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to process learning confirmation for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process learning confirmation: {str(e)}")

@app.post("/test-meta-agent")
async def test_meta_agent(request: dict):
    """Test endpoint for the MetaAgent system."""
    start_time = time.time()
    
    try:
        user_input = request.get("message", "Write a quick status update")
        session_id = request.get("session_id", "test_session")
        
        # Run MetaAgent
        result = await meta_agent.process_request(
            user_input=user_input,
            session_id=session_id,
            conversation_context=[]
        )
        
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)  # milliseconds
        
        return {
            "success": True,
            "result": result,
            "performance": {
                "response_time_ms": response_time,
                "architecture": "meta-agent",
                "agent_used": result.get("agent_used", "Unknown"),
                "execution_successful": result.get("execution_successful", False)
            }
        }
        
    except Exception as e:
        logger.error(f"MetaAgent test failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/test-opportunistic-learning")
async def test_opportunistic_learning(request: dict):
    """Test endpoint for opportunistic learning functionality"""
    try:
        # Extract test parameters
        user_input = request.get("user_input", "Sarah is my design partner and I prefer concise communication")
        response_content = request.get("response_content", "Got it! I'll remember that Sarah is your design partner.")
        session_id = request.get("session_id", "test_learning_session")
        
        # Trigger opportunistic learning directly
        await meta_agent._opportunistic_learn(
            user_input=user_input,
            response_content=response_content,
            session_id=session_id,
            conversation_context=[]
        )
        
        # Get learning summary
        summary = meta_agent.get_session_learning_summary(session_id)
        
        return {
            "status": "success",
            "message": "Opportunistic learning test completed",
            "learning_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Opportunistic learning test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🚀 Starting Son of Andrew API server v2.0...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🎯 New Architecture: Plan-Based Execution")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 