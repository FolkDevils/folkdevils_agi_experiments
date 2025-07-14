from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class ConfirmationType(Enum):
    REPLACE = "replace"
    REMOVE = "remove"

@dataclass
class PendingConfirmation:
    """Represents a pending semantic confirmation"""
    session_id: str
    confirmation_type: ConfirmationType
    original_target: str
    replacement_text: Optional[str]  # None for remove operations
    suggestions: List[Dict[str, Any]]
    instruction: str
    working_text_version: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "confirmation_type": self.confirmation_type.value,
            "original_target": self.original_target,
            "replacement_text": self.replacement_text,
            "suggestions": self.suggestions,
            "instruction": self.instruction,
            "working_text_version": self.working_text_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingConfirmation":
        """Create from dictionary"""
        return cls(
            session_id=data["session_id"],
            confirmation_type=ConfirmationType(data["confirmation_type"]),
            original_target=data["original_target"],
            replacement_text=data.get("replacement_text"),
            suggestions=data["suggestions"],
            instruction=data["instruction"],
            working_text_version=data["working_text_version"]
        )

class ConfirmationManager:
    """Manages pending semantic confirmations"""
    
    def __init__(self):
        self._pending_confirmations: Dict[str, PendingConfirmation] = {}
    
    def store_confirmation(self, confirmation: PendingConfirmation) -> None:
        """Store a pending confirmation"""
        self._pending_confirmations[confirmation.session_id] = confirmation
        logger.info(f"Stored pending confirmation for session {confirmation.session_id}")
    
    def get_pending_confirmation(self, session_id: str) -> Optional[PendingConfirmation]:
        """Get pending confirmation for a session"""
        return self._pending_confirmations.get(session_id)
    
    def has_pending_confirmation(self, session_id: str) -> bool:
        """Check if there's a pending confirmation for a session"""
        return session_id in self._pending_confirmations
    
    def clear_confirmation(self, session_id: str) -> None:
        """Clear pending confirmation for a session"""
        if session_id in self._pending_confirmations:
            del self._pending_confirmations[session_id]
            logger.info(f"Cleared pending confirmation for session {session_id}")
    
    def get_confirmed_text(self, session_id: str, option_number: int) -> Optional[str]:
        """Get the confirmed text for a specific option"""
        confirmation = self.get_pending_confirmation(session_id)
        if not confirmation:
            return None
        
        if 1 <= option_number <= len(confirmation.suggestions):
            return confirmation.suggestions[option_number - 1]["text"]
        
        return None
    
    def create_replace_confirmation(
        self, 
        session_id: str, 
        original_target: str, 
        replacement_text: str,
        suggestions: List[Dict[str, Any]], 
        instruction: str, 
        working_text_version: int
    ) -> PendingConfirmation:
        """Create a replace confirmation"""
        confirmation = PendingConfirmation(
            session_id=session_id,
            confirmation_type=ConfirmationType.REPLACE,
            original_target=original_target,
            replacement_text=replacement_text,
            suggestions=suggestions,
            instruction=instruction,
            working_text_version=working_text_version
        )
        self.store_confirmation(confirmation)
        return confirmation
    
    def create_remove_confirmation(
        self, 
        session_id: str, 
        original_target: str, 
        suggestions: List[Dict[str, Any]], 
        instruction: str, 
        working_text_version: int
    ) -> PendingConfirmation:
        """Create a remove confirmation"""
        confirmation = PendingConfirmation(
            session_id=session_id,
            confirmation_type=ConfirmationType.REMOVE,
            original_target=original_target,
            replacement_text=None,
            suggestions=suggestions,
            instruction=instruction,
            working_text_version=working_text_version
        )
        self.store_confirmation(confirmation)
        return confirmation

# Global instance
confirmation_manager = ConfirmationManager() 