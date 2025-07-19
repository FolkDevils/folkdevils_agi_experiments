"""
Short-Term Memory Buffer - My Working Memory

This is my temporary memory during conversations:
- Holds current conversation context
- Tracks immediate thoughts and references
- Manages "this", "that", "it" resolution
- Prepares content for long-term storage evaluation

Like human working memory - temporary but essential for coherent dialogue.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Deque
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """A single turn in our conversation"""
    timestamp: str
    speaker: str  # 'andrew' or 'ai'
    content: str
    intent: Optional[str] = None
    context_refs: List[str] = None  # Things like "this", "that" references
    importance: float = 0.5

@dataclass
class WorkingThought:
    """A temporary thought or insight I'm processing"""
    timestamp: str
    content: str
    related_to: Optional[str] = None  # Reference to conversation turn or memory
    confidence: float = 0.5

class ShortTermMemory:
    """
    My working memory for active conversations
    
    This gives me:
    - Context awareness within conversations
    - Reference resolution (this, that, it)
    - Immediate thought processing
    - Preparation for long-term storage
    """
    
    def __init__(self, max_turns: int = 50, max_thoughts: int = 20):
        self.max_turns = max_turns
        self.max_thoughts = max_thoughts
        
        # Conversation context
        self.conversation_history: Deque[ConversationTurn] = deque(maxlen=max_turns)
        
        # Working thoughts and insights
        self.working_thoughts: Deque[WorkingThought] = deque(maxlen=max_thoughts)
        
        # Current session metadata
        self.session_start = datetime.now().isoformat()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🧠 Short-term memory initialized - Session: {self.session_id}")
    
    def add_conversation_turn(self, 
                            speaker: str, 
                            content: str, 
                            intent: Optional[str] = None) -> ConversationTurn:
        """Add a new conversation turn to working memory"""
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            speaker=speaker,
            content=content,
            intent=intent,
            context_refs=self._extract_references(content)
        )
        
        self.conversation_history.append(turn)
        
        logger.debug(f"💭 Added conversation turn: {speaker} - {content[:50]}...")
        return turn
    
    def add_working_thought(self, 
                          content: str, 
                          related_to: Optional[str] = None,
                          confidence: float = 0.5) -> WorkingThought:
        """Add a working thought or insight"""
        thought = WorkingThought(
            timestamp=datetime.now().isoformat(),
            content=content,
            related_to=related_to,
            confidence=confidence
        )
        
        self.working_thoughts.append(thought)
        
        logger.debug(f"💡 Working thought: {content[:50]}...")
        return thought
    
    def get_conversation_context(self, 
                               last_n_turns: Optional[int] = None,
                               include_system: bool = True) -> List[ConversationTurn]:
        """Get recent conversation context"""
        turns = list(self.conversation_history)
        
        if last_n_turns:
            turns = turns[-last_n_turns:]
        
        if not include_system:
            turns = [t for t in turns if t.speaker != 'system']
        
        return turns
    
    def resolve_reference(self, reference: str) -> Optional[ConversationTurn]:
        """
        Resolve references like 'this', 'that', 'it' to actual content
        
        This is key for maintaining conversation coherence
        """
        reference_lower = reference.lower()
        
        # Look for the most recent relevant content
        for turn in reversed(self.conversation_history):
            # Skip my own messages when resolving references
            if turn.speaker == 'ai':
                continue
                
            # Simple heuristic: recent content from Andrew is likely the referent
            if reference_lower in ['this', 'that', 'it']:
                return turn
        
        return None
    
    def get_current_context_summary(self) -> str:
        """Generate a summary of current conversation context"""
        if not self.conversation_history:
            return "Empty conversation"
        
        recent_turns = list(self.conversation_history)[-5:]  # Last 5 turns
        
        summary_parts = []
        for turn in recent_turns:
            speaker_name = "Andrew" if turn.speaker == "andrew" else "I"
            summary_parts.append(f"{speaker_name}: {turn.content}")
        
        return "\n".join(summary_parts)
    
    def get_working_thoughts_summary(self) -> List[str]:
        """Get summary of current working thoughts"""
        return [thought.content for thought in self.working_thoughts 
                if thought.confidence > 0.3]
    
    def prepare_for_long_term_storage(self) -> Dict[str, Any]:
        """
        Prepare current session content for long-term memory evaluation
        
        Returns structured data for the memory evaluator to process
        """
        return {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "session_end": datetime.now().isoformat(),
            "conversation_turns": [asdict(turn) for turn in self.conversation_history],
            "working_thoughts": [asdict(thought) for thought in self.working_thoughts],
            "context_summary": self.get_current_context_summary(),
            "total_turns": len(self.conversation_history),
            "session_duration_minutes": self._get_session_duration_minutes()
        }
    
    def clear_session(self):
        """Clear current session (start fresh)"""
        old_session = self.session_id
        
        self.conversation_history.clear()
        self.working_thoughts.clear()
        self.session_start = datetime.now().isoformat()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🧹 Cleared session {old_session} - New session: {self.session_id}")
    
    def _extract_references(self, content: str) -> List[str]:
        """Extract reference words like 'this', 'that', 'it' from content"""
        reference_words = ['this', 'that', 'it', 'them', 'they', 'those', 'these']
        content_lower = content.lower()
        
        found_refs = []
        for ref in reference_words:
            if f" {ref} " in f" {content_lower} ":
                found_refs.append(ref)
        
        return found_refs
    
    def _get_session_duration_minutes(self) -> float:
        """Calculate how long current session has been active"""
        start_time = datetime.fromisoformat(self.session_start)
        duration = datetime.now() - start_time
        return duration.total_seconds() / 60
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current session"""
        return {
            "session_id": self.session_id,
            "duration_minutes": self._get_session_duration_minutes(),
            "total_turns": len(self.conversation_history),
            "working_thoughts": len(self.working_thoughts),
            "andrew_turns": len([t for t in self.conversation_history if t.speaker == 'andrew']),
            "ai_turns": len([t for t in self.conversation_history if t.speaker == 'ai'])
        } 