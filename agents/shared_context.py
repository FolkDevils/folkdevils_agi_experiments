from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SharedContext:
    """
    Shared context available to all agents.
    
    This provides a clean interface to memory, tools, and other resources
    without agents needing to manage these dependencies directly.
    """
    session_id: str
    user_input: str
    previous_results: Dict[str, Any]
    memory_context: str = ""
    
    def __post_init__(self):
        # Initialize memory manager reference
        try:
            from memory_manager import memory_manager
            self._memory_manager = memory_manager
        except ImportError:
            logger.warning("Memory manager not available")
            self._memory_manager = None
    
    async def get_recent(self, content_type: str = "last_output") -> str:
        """Get recent content from memory"""
        if not self._memory_manager or not self.session_id:
            return ""
        
        try:
            if content_type == "last_output":
                return await self._memory_manager.get_last_output(self.session_id)
            elif content_type == "last_generated_content":
                # Try to get the most recent generated content
                result = await self._memory_manager.get_last_agent_output(self.session_id, "COMPLETE")
                return result.get("content", "")
            else:
                return ""
        except Exception as e:
            logger.warning(f"Failed to get recent content: {e}")
            return ""
    
    async def get_relevant(self, query: str) -> str:
        """Get relevant context from memory"""
        if not self._memory_manager or not self.session_id:
            return ""
        
        try:
            return await self._memory_manager.get_memory_context(self.session_id, query)
        except Exception as e:
            logger.warning(f"Failed to get relevant context: {e}")
            return ""
    
    def get_previous_result(self, step_name: str) -> Any:
        """Get result from a previous step in the current plan"""
        return self.previous_results.get(step_name)
    
    def set_result(self, step_name: str, result: Any):
        """Store result for future steps"""
        self.previous_results[step_name] = result
    
    def get_user_content(self) -> str:
        """Get the original user input"""
        return self.user_input
    
    async def store_preference(self, preference: Dict[str, Any]) -> bool:
        """Store a learned preference"""
        if not self._memory_manager:
            return False
        
        try:
            return await self._memory_manager.store_learned_preference(preference)
        except Exception as e:
            logger.warning(f"Failed to store preference: {e}")
            return False
    
    async def get_all_time_entries(self, period: str = "today", category: str = None) -> List[Dict[str, Any]]:
        """Get ALL time entries across all sessions for Andrew"""
        if not self._memory_manager:
            return []
        
        try:
            from dataclasses import asdict
            time_entries = await self._memory_manager.get_all_user_time_entries(period, category)
            return [asdict(entry) for entry in time_entries]
        except Exception as e:
            logger.warning(f"Failed to get all time entries: {e}")
            return []
    
    async def get_all_time_summary(self, period: str = "today", category: str = None) -> Dict[str, Any]:
        """Get time summary across ALL sessions for Andrew"""
        if not self._memory_manager:
            return {}
        
        try:
            return await self._memory_manager.get_all_user_time_summary(period, category)
        except Exception as e:
            logger.warning(f"Failed to get all time summary: {e}")
            return {}

class ContextManager:
    """Manages shared context across plan execution"""
    
    def __init__(self):
        self._contexts: Dict[str, SharedContext] = {}
    
    def create_context(self, session_id: str, user_input: str, memory_context: str = "") -> SharedContext:
        """Create a new shared context for plan execution"""
        context = SharedContext(
            session_id=session_id,
            user_input=user_input,
            previous_results={},
            memory_context=memory_context
        )
        self._contexts[session_id] = context
        return context
    
    def get_context(self, session_id: str) -> Optional[SharedContext]:
        """Get existing context"""
        return self._contexts.get(session_id)
    
    def cleanup_context(self, session_id: str):
        """Clean up context after plan execution"""
        self._contexts.pop(session_id, None)

# Global context manager instance
context_manager = ContextManager() 