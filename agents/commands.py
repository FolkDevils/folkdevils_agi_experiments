from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum

class CommandIntents(str, Enum):
    """Standard command intents for the workflow system."""
    NEEDS_WRITING = "NEEDS_WRITING"
    NEEDS_EDITING = "NEEDS_EDITING"
    NEEDS_ANALYSIS = "NEEDS_ANALYSIS"
    NEEDS_LEARNING = "NEEDS_LEARNING"
    NEEDS_CONVERSATION = "NEEDS_CONVERSATION"
    NEEDS_TIMETRACKING = "NEEDS_TIMETRACKING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    RETRY = "RETRY"

@dataclass
class Command:
    """
    Enhanced command structure with clean content separation.
    
    This eliminates the need for response parsing by keeping
    user-facing content separate from system metadata.
    """
    name: str
    state: Dict[str, Any]
    reason: str
    
    # NEW: Clean content separation
    content: Optional[str] = None  # Clean user-facing content
    metadata: Optional[Dict[str, Any]] = None  # System metadata (changes made, etc.)
    
    def get_user_content(self) -> str:
        """Get the clean content that should be shown to the user."""
        if self.content:
            return self.content
        
        # Fallback to state content if available
        return self.state.get("content", "")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get system metadata (changes made, analysis, etc.)."""
        return self.metadata or {}

def create_command(intent: str, **kwargs) -> Command:
    """Helper function to create commands with validation"""
    valid_intents = [
        CommandIntents.NEEDS_EDITING,
        CommandIntents.NEEDS_ANALYSIS, 
        CommandIntents.NEEDS_WRITING,
        CommandIntents.NEEDS_LEARNING,
        CommandIntents.COMPLETE,
        CommandIntents.ERROR,
        CommandIntents.RETRY,
        CommandIntents.NEEDS_TIMETRACKING,
        CommandIntents.NEEDS_CONVERSATION,
        CommandIntents.NEEDS_CLARIFICATION
    ]
    
    if intent not in valid_intents:
        raise ValueError(f"Unknown command intent: {intent}. Valid intents: {valid_intents}")
    
    return Command(name=intent, **kwargs)

# Convenience functions for common commands
def complete_command(
    state: Dict[str, Any], 
    reason: str = "Task completed successfully", 
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Command:
    """Create a COMPLETE command"""
    return Command(
        name=CommandIntents.COMPLETE, 
        state=state, 
        reason=reason,
        content=content,
        metadata=metadata
    )

def needs_editing_command(
    state: Dict[str, Any], 
    reason: str = "Content needs improvement",
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Command:
    """Create a NEEDS_EDITING command"""
    return Command(
        name=CommandIntents.NEEDS_EDITING, 
        state=state, 
        reason=reason,
        content=content,
        metadata=metadata
    )

def needs_analysis_command(
    state: Dict[str, Any], 
    reason: str = "Performance analysis needed",
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Command:
    """Create a NEEDS_ANALYSIS command"""
    return Command(
        name=CommandIntents.NEEDS_ANALYSIS, 
        state=state, 
        reason=reason,
        content=content,
        metadata=metadata
    )

def needs_writing_command(
    state: Dict[str, Any], 
    reason: str = "Content creation needed",
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Command:
    """Create a NEEDS_WRITING command"""
    return Command(
        name=CommandIntents.NEEDS_WRITING, 
        state=state, 
        reason=reason,
        content=content,
        metadata=metadata
    )

def needs_learning_command(
    state: Dict[str, Any], 
    reason: str = "Style learning needed",
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Command:
    """Create a NEEDS_LEARNING command"""
    return Command(
        name=CommandIntents.NEEDS_LEARNING, 
        state=state, 
        reason=reason,
        content=content,
        metadata=metadata
    )

def error_command(
    state: Dict[str, Any], 
    reason: str = "An error occurred",
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Command:
    """Create an ERROR command"""
    return Command(
        name=CommandIntents.ERROR, 
        state=state, 
        reason=reason,
        content=content,
        metadata=metadata
    )

def retry_command(
    state: Dict[str, Any], 
    reason: str = "Retry needed",
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Command:
    """Create a RETRY command"""
    return Command(
        name=CommandIntents.RETRY, 
        state=state, 
        reason=reason,
        content=content,
        metadata=metadata
    ) 