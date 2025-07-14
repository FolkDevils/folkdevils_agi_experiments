"""
SessionState - Temporary Coordination State for Multi-Step Plans

Provides working memory for agents to coordinate during multi-step plan execution.
This is separate from persistent memory and only exists for the duration of a plan.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels for step results"""
    HIGH = "high"      # 90%+ confidence
    MEDIUM = "medium"  # 70-89% confidence  
    LOW = "low"        # 50-69% confidence
    UNCERTAIN = "uncertain"  # <50% confidence

@dataclass
class StepResult:
    """Result from a single step in a multi-step plan"""
    step_number: int
    agent_name: str
    action: str
    content: str
    confidence: ConfidenceLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easier handling"""
        return {
            "step_number": self.step_number,
            "agent_name": self.agent_name,
            "action": self.action,
            "content": self.content,
            "confidence": self.confidence.value,
            "metadata": self.metadata,
            "insights": self.insights,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms
        }

@dataclass
class PlanContext:
    """Overall context for the multi-step plan"""
    original_request: str
    overall_intent: str
    target_tone: Optional[str] = None
    target_length: Optional[str] = None
    content_type: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    plan_metadata: Dict[str, Any] = field(default_factory=dict)

class SessionState:
    """
    Temporary coordination state for multi-step plan execution.
    
    Provides working memory for agents to:
    - Understand overall plan context and intent
    - Access results from previous steps
    - Share insights and confidence levels
    - Coordinate error handling and recovery
    
    This state is:
    - Temporary (exists only during plan execution)
    - In-memory (no persistence)
    - Read-mostly with controlled writes
    """
    
    def __init__(self, session_id: str, plan_context: PlanContext):
        self.session_id = session_id
        self.plan_context = plan_context
        self.step_results: List[StepResult] = []
        self.error_context: Optional[Dict[str, Any]] = None
        self.plan_metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        
        logger.info(f"🧠 SessionState created for plan: {plan_context.overall_intent}")
    
    # === CONTEXT ACCESS (READ-ONLY) ===
    
    def get_original_intent(self) -> str:
        """Get the original user intent for the plan"""
        return self.plan_context.overall_intent
    
    def get_original_request(self) -> str:
        """Get the original user request"""
        return self.plan_context.original_request
    
    def get_target_tone(self) -> Optional[str]:
        """Get the target tone for content (e.g., 'formal', 'casual', 'professional')"""
        return self.plan_context.target_tone
    
    def get_target_length(self) -> Optional[str]:
        """Get the target length for content (e.g., 'shorter', 'longer', 'concise')"""
        return self.plan_context.target_length
    
    def get_content_type(self) -> Optional[str]:
        """Get the content type being worked on (e.g., 'email', 'copy', 'message')"""
        return self.plan_context.content_type
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences relevant to this plan"""
        return self.plan_context.user_preferences.copy()
    
    # === STEP RESULT ACCESS ===
    
    def get_previous_step_result(self) -> Optional[StepResult]:
        """Get the result from the immediately previous step"""
        if not self.step_results:
            return None
        return self.step_results[-1]
    
    def get_step_result(self, step_number: int) -> Optional[StepResult]:
        """Get result from a specific step number"""
        for result in self.step_results:
            if result.step_number == step_number:
                return result
        return None
    
    def get_step_content(self, step_number: int) -> str:
        """Get content from a specific step (convenience method)"""
        result = self.get_step_result(step_number)
        return result.content if result else ""
    
    def get_previous_content(self) -> str:
        """Get content from the previous step (convenience method)"""
        previous = self.get_previous_step_result()
        return previous.content if previous else ""
    
    def get_all_step_results(self) -> List[StepResult]:
        """Get all step results so far"""
        return self.step_results.copy()
    
    def get_step_count(self) -> int:
        """Get the number of completed steps"""
        return len(self.step_results)
    
    # === CONFIDENCE AND INSIGHTS ===
    
    def get_overall_confidence(self) -> ConfidenceLevel:
        """Get overall confidence based on all step results"""
        if not self.step_results:
            return ConfidenceLevel.UNCERTAIN
        
        # Calculate average confidence
        confidence_values = {
            ConfidenceLevel.HIGH: 4,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.LOW: 2,
            ConfidenceLevel.UNCERTAIN: 1
        }
        
        total_score = sum(confidence_values[result.confidence] for result in self.step_results)
        avg_score = total_score / len(self.step_results)
        
        if avg_score >= 3.5:
            return ConfidenceLevel.HIGH
        elif avg_score >= 2.5:
            return ConfidenceLevel.MEDIUM
        elif avg_score >= 1.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def get_step_insights(self) -> List[str]:
        """Get all insights collected from steps"""
        insights = []
        for result in self.step_results:
            insights.extend(result.insights)
        return insights
    
    def has_low_confidence_steps(self) -> bool:
        """Check if any steps had low confidence"""
        return any(
            result.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]
            for result in self.step_results
        )
    
    # === ERROR CONTEXT ===
    
    def get_error_context(self) -> Optional[Dict[str, Any]]:
        """Get error context if any step failed"""
        return self.error_context
    
    def has_errors(self) -> bool:
        """Check if there are any errors in the plan execution"""
        return self.error_context is not None
    
    # === PLAN METADATA ===
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get plan-level metadata"""
        return self.plan_metadata.get(key, default)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of plan execution"""
        return {
            "session_id": self.session_id,
            "overall_intent": self.get_original_intent(),
            "steps_completed": self.get_step_count(),
            "overall_confidence": self.get_overall_confidence().value,
            "has_errors": self.has_errors(),
            "total_insights": len(self.get_step_insights()),
            "execution_duration_ms": (datetime.now() - self.created_at).total_seconds() * 1000
        }
    
    # === CONTROLLED WRITE OPERATIONS ===
    
    def add_step_result(self, step_result: StepResult) -> None:
        """Add a step result (called by MetaAgent during execution)"""
        self.step_results.append(step_result)
        logger.info(f"🔄 Step {step_result.step_number} result added: {step_result.agent_name} - {step_result.confidence.value}")
    
    def set_error_context(self, error_context: Dict[str, Any]) -> None:
        """Set error context (called by MetaAgent if step fails)"""
        self.error_context = error_context
        logger.warning(f"❌ Error context set: {error_context.get('error', 'Unknown error')}")
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update plan metadata (for agents to contribute insights)"""
        self.plan_metadata[key] = value
        logger.debug(f"📝 Metadata updated: {key} = {value}")
    
    def add_insight(self, step_number: int, insight: str) -> None:
        """Add an insight to a specific step"""
        for result in self.step_results:
            if result.step_number == step_number:
                result.insights.append(insight)
                logger.info(f"💡 Insight added to step {step_number}: {insight}")
                break

class SessionStateManager:
    """
    Manages session states for different plan executions.
    
    This ensures clean separation between different multi-step plans
    and proper cleanup of temporary state.
    """
    
    def __init__(self):
        self._active_states: Dict[str, SessionState] = {}
        logger.info("🧠 SessionStateManager initialized")
    
    def create_session_state(self, session_id: str, plan_context: PlanContext) -> SessionState:
        """Create a new session state for a plan"""
        state = SessionState(session_id, plan_context)
        self._active_states[session_id] = state
        logger.info(f"🧠 Created session state for: {session_id}")
        return state
    
    def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """Get existing session state"""
        return self._active_states.get(session_id)
    
    def cleanup_session_state(self, session_id: str) -> None:
        """Clean up session state after plan completion"""
        if session_id in self._active_states:
            state = self._active_states.pop(session_id)
            logger.info(f"🧠 Cleaned up session state for: {session_id} (completed {state.get_step_count()} steps)")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self._active_states.keys())
    
    def cleanup_old_sessions(self, max_age_minutes: int = 60) -> None:
        """Clean up session states older than max_age_minutes"""
        now = datetime.now()
        to_cleanup = []
        
        for session_id, state in self._active_states.items():
            age_minutes = (now - state.created_at).total_seconds() / 60
            if age_minutes > max_age_minutes:
                to_cleanup.append(session_id)
        
        for session_id in to_cleanup:
            self.cleanup_session_state(session_id)
        
        if to_cleanup:
            logger.info(f"🧹 Cleaned up {len(to_cleanup)} old session states")

# Global instance
session_state_manager = SessionStateManager() 