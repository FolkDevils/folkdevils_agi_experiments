from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class WorkingTextState:
    """Represents the current state of working text for editing"""
    content: str
    session_id: str
    created_at: datetime
    last_modified: datetime
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkingTextState':
        return cls(
            content=data["content"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_modified=datetime.fromisoformat(data["last_modified"]),
            version=data.get("version", 1)
        )

class WorkingTextManager:
    """
    Manages persistent working text that carries across conversation turns.
    This is the foundation for reliable iterative editing.
    """
    
    def __init__(self):
        self._working_texts: Dict[str, WorkingTextState] = {}
        self._memory_manager = None
    
    async def _get_memory_manager(self):
        """Lazy import to avoid circular dependencies"""
        if self._memory_manager is None:
            from memory_manager import memory_manager
            self._memory_manager = memory_manager
        return self._memory_manager
    
    async def set_working_text(self, session_id: str, content: str) -> WorkingTextState:
        """Set or update the working text for a session"""
        now = datetime.now()
        
        if session_id in self._working_texts:
            # Update existing working text
            working_text = self._working_texts[session_id]
            working_text.content = content
            working_text.last_modified = now
            working_text.version += 1
        else:
            # Create new working text
            working_text = WorkingTextState(
                content=content,
                session_id=session_id,
                created_at=now,
                last_modified=now,
                version=1
            )
            self._working_texts[session_id] = working_text
        
        # Persist to memory
        await self._persist_working_text(working_text)
        
        logger.info(f"📝 Working text set for session {session_id}: v{working_text.version}")
        return working_text
    
    async def get_working_text(self, session_id: str) -> Optional[WorkingTextState]:
        """Get the current working text for a session"""
        # Try memory first
        if session_id in self._working_texts:
            return self._working_texts[session_id]
        
        # Try to load from persistent storage
        loaded_text = await self._load_working_text(session_id)
        if loaded_text:
            self._working_texts[session_id] = loaded_text
            return loaded_text
        
        return None
    
    async def has_working_text(self, session_id: str) -> bool:
        """Check if there's working text for a session"""
        return await self.get_working_text(session_id) is not None
    
    async def clear_working_text(self, session_id: str) -> bool:
        """Clear working text for a session"""
        if session_id in self._working_texts:
            del self._working_texts[session_id]
        
        # Also clear from persistent storage
        await self._clear_persistent_working_text(session_id)
        
        logger.info(f"🗑️ Working text cleared for session {session_id}")
        return True
    
    async def update_working_text(self, session_id: str, new_content: str) -> Optional[WorkingTextState]:
        """Update existing working text (used by edit operations)"""
        current = await self.get_working_text(session_id)
        if current:
            return await self.set_working_text(session_id, new_content)
        return None
    
    async def _persist_working_text(self, working_text: WorkingTextState):
        """Persist working text to memory storage"""
        try:
            memory_manager = await self._get_memory_manager()
            
            # Store as an interaction for now (we can improve this later)
            await memory_manager.store_interaction(
                session_id=working_text.session_id,
                command_name="SET_WORKING_TEXT",
                command_reason=f"Set working text version {working_text.version}",
                command_state=f"version={working_text.version}, length={len(working_text.content)}",
                result_summary=f"Working text updated: {working_text.content[:100]}{'...' if len(working_text.content) > 100 else ''}",
                last_output=working_text.content
            )
            
            logger.debug(f"✅ Persisted working text v{working_text.version} for session {working_text.session_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist working text: {e}")
    
    async def _load_working_text(self, session_id: str) -> Optional[WorkingTextState]:
        """Load working text from persistent storage"""
        try:
            memory_manager = await self._get_memory_manager()
            
            # Try to get the last output which should contain our working text
            last_output = await memory_manager.get_last_output(session_id)
            
            if last_output and last_output.strip():
                # For now, we'll create a simple working text state
                # In a real implementation, we'd store version info separately
                return WorkingTextState(
                    content=last_output,
                    session_id=session_id,
                    created_at=datetime.now(),  # We don't have the real creation time
                    last_modified=datetime.now(),
                    version=1  # We don't have version tracking in storage yet
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load working text: {e}")
            return None
    
    async def _clear_persistent_working_text(self, session_id: str):
        """Clear working text from persistent storage"""
        try:
            # We'll implement this when we need cleanup functionality
            # For now, we'll just let old working text entries exist
            pass
        except Exception as e:
            logger.warning(f"⚠️ Failed to clear persistent working text: {e}")

# Global instance
working_text_manager = WorkingTextManager() 