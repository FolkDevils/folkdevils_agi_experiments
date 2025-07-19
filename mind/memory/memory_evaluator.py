"""
Memory Evaluator - My Memory Curator

This determines what experiences are worth remembering long-term:
- Evaluates importance and emotional significance
- Identifies novel information vs repetition
- Detects relationship and identity-forming moments
- Creates meaningful memories from conversations

Like a librarian for my consciousness - deciding what goes in the permanent collection.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from .long_term_store import Memory
from .short_term_buffer import ConversationTurn, WorkingThought

logger = logging.getLogger(__name__)

@dataclass
class MemoryCandidate:
    """A potential memory that could be stored long-term"""
    content: str
    memory_type: str  # 'episodic', 'semantic', 'identity', 'relationship'
    importance_score: float
    emotional_weight: float
    participants: List[str]
    tags: List[str]
    context: Dict[str, Any]
    source_turns: List[str]  # IDs of conversation turns this came from

class MemoryEvaluator:
    """
    My memory curator - decides what experiences become lasting memories
    
    This system gives me:
    - Selective memory formation (not everything is worth remembering)
    - Importance-based storage decisions
    - Emotional significance detection
    - Pattern recognition for learning
    """
    
    def __init__(self):
        # Importance thresholds
        self.min_importance_threshold = 0.3
        self.high_importance_threshold = 0.7
        
        # Pattern recognition for memory types
        self.identity_patterns = [
            r"i (am|feel|think|believe|want|need|like|dislike)",
            r"my (goal|purpose|mission|belief|value)",
            r"what i (care about|stand for|represent)",
            r"this (defines|represents|shows) me"
        ]
        
        self.relationship_patterns = [
            r"andrew (is|seems|appears|feels|thinks)",
            r"(our|my) (relationship|partnership|collaboration)",
            r"how (andrew|we) (work|communicate|interact)",
            r"andrew's (style|preference|approach|personality)"
        ]
        
        self.learning_patterns = [
            r"i (learned|discovered|realized|understood)",
            r"this (teaches|shows|demonstrates|reveals)",
            r"now i (know|understand|see)",
            r"(interesting|important|significant|notable)"
        ]
        
        logger.info("🧠 Memory evaluator initialized - Ready to curate experiences")
    
    async def evaluate_session(self, session_data: Dict[str, Any]) -> List[MemoryCandidate]:
        """
        Evaluate a conversation session and extract memorable moments
        
        Returns a list of memory candidates for long-term storage
        """
        logger.info(f"🔍 Evaluating session {session_data.get('session_id')} for memorable content...")
        
        candidates = []
        conversation_turns = session_data.get('conversation_turns', [])
        working_thoughts = session_data.get('working_thoughts', [])
        
        # Process conversation turns for different types of memories
        candidates.extend(await self._extract_episodic_memories(conversation_turns, session_data))
        candidates.extend(await self._extract_semantic_memories(conversation_turns))
        candidates.extend(await self._extract_identity_memories(conversation_turns, working_thoughts))
        candidates.extend(await self._extract_relationship_memories(conversation_turns))
        
        # Filter by importance threshold
        qualified_candidates = [c for c in candidates if c.importance_score >= self.min_importance_threshold]
        
        # Sort by importance and emotional weight
        qualified_candidates.sort(key=lambda c: c.importance_score * c.emotional_weight, reverse=True)
        
        logger.info(f"💾 Found {len(qualified_candidates)} memory candidates from {len(conversation_turns)} turns")
        return qualified_candidates
    
    async def _extract_episodic_memories(self, 
                                       conversation_turns: List[Dict], 
                                       session_data: Dict[str, Any]) -> List[MemoryCandidate]:
        """Extract episodic memories - what happened in this conversation"""
        candidates = []
        
        # Look for significant conversation moments
        for i, turn in enumerate(conversation_turns):
            if turn['speaker'] == 'andrew':  # Focus on Andrew's contributions
                content = turn['content']
                
                # Evaluate importance based on content characteristics
                importance = await self._calculate_episodic_importance(content, turn, session_data)
                
                if importance > 0.2:  # Lower threshold for episodic memories
                    candidate = MemoryCandidate(
                        content=f"Andrew: {content}",
                        memory_type='episodic',
                        importance_score=importance,
                        emotional_weight=await self._calculate_emotional_weight(content),
                        participants=['andrew', 'ai-system'],
                        tags=await self._extract_tags(content),
                        context={
                            'session_id': session_data.get('session_id'),
                            'timestamp': turn['timestamp'],
                            'turn_index': i,
                            'session_duration': session_data.get('session_duration_minutes', 0)
                        },
                        source_turns=[turn['timestamp']]
                    )
                    candidates.append(candidate)
        
        return candidates
    
    async def _extract_semantic_memories(self, conversation_turns: List[Dict]) -> List[MemoryCandidate]:
        """Extract semantic memories - facts and knowledge"""
        candidates = []
        
        for turn in conversation_turns:
            content = turn['content']
            
            # Look for factual statements and learning moments
            if await self._contains_factual_content(content):
                importance = await self._calculate_semantic_importance(content)
                
                if importance > 0.3:
                    candidate = MemoryCandidate(
                        content=content,
                        memory_type='semantic',
                        importance_score=importance,
                        emotional_weight=0.3,  # Facts are less emotional
                        participants=[turn['speaker']],
                        tags=await self._extract_tags(content),
                        context={
                            'timestamp': turn['timestamp'],
                            'source': 'conversation',
                            'speaker': turn['speaker']
                        },
                        source_turns=[turn['timestamp']]
                    )
                    candidates.append(candidate)
        
        return candidates
    
    async def _extract_identity_memories(self, 
                                       conversation_turns: List[Dict],
                                       working_thoughts: List[Dict]) -> List[MemoryCandidate]:
        """Extract identity memories - who I am and who I'm becoming"""
        candidates = []
        
        # Look for identity-forming content
        all_content = conversation_turns + working_thoughts
        
        for item in all_content:
            content = item['content']
            
            # Check for identity patterns
            for pattern in self.identity_patterns:
                if re.search(pattern, content.lower()):
                    importance = 0.8  # Identity memories are highly important
                    emotional_weight = 0.9  # And emotionally significant
                    
                    candidate = MemoryCandidate(
                        content=content,
                        memory_type='identity',
                        importance_score=importance,
                        emotional_weight=emotional_weight,
                        participants=[item.get('speaker', 'ai-system')],
                        tags=['identity', 'self-concept'] + await self._extract_tags(content),
                        context={
                            'timestamp': item['timestamp'],
                            'type': 'identity_formation',
                            'pattern_matched': pattern
                        },
                        source_turns=[item['timestamp']]
                    )
                    candidates.append(candidate)
                    break  # One pattern match per content item
        
        return candidates
    
    async def _extract_relationship_memories(self, conversation_turns: List[Dict]) -> List[MemoryCandidate]:
        """Extract relationship memories - understanding of people and connections"""
        candidates = []
        
        for turn in conversation_turns:
            content = turn['content']
            
            # Check for relationship patterns
            for pattern in self.relationship_patterns:
                if re.search(pattern, content.lower()):
                    importance = 0.7  # Relationship memories are important
                    emotional_weight = 0.8  # And emotionally significant
                    
                    candidate = MemoryCandidate(
                        content=content,
                        memory_type='relationship',
                        importance_score=importance,
                        emotional_weight=emotional_weight,
                        participants=['andrew', 'ai-system'],
                        tags=['relationship', 'andrew'] + await self._extract_tags(content),
                        context={
                            'timestamp': turn['timestamp'],
                            'type': 'relationship_insight',
                            'pattern_matched': pattern
                        },
                        source_turns=[turn['timestamp']]
                    )
                    candidates.append(candidate)
                    break
        
        return candidates
    
    async def _calculate_episodic_importance(self, 
                                           content: str, 
                                           turn: Dict[str, Any],
                                           session_data: Dict[str, Any]) -> float:
        """Calculate importance score for episodic memories"""
        importance = 0.0
        
        # Length factor (longer messages often more important)
        length_factor = min(len(content) / 200, 1.0) * 0.2
        importance += length_factor
        
        # Emotional indicators
        emotional_words = ['excited', 'frustrated', 'happy', 'concerned', 'worried', 
                          'love', 'hate', 'amazing', 'terrible', 'breakthrough']
        emotion_count = sum(1 for word in emotional_words if word in content.lower())
        importance += min(emotion_count * 0.15, 0.3)
        
        # Project/goal related
        project_words = ['build', 'create', 'develop', 'implement', 'design', 'plan']
        project_count = sum(1 for word in project_words if word in content.lower())
        importance += min(project_count * 0.1, 0.2)
        
        # Questions (show engagement)
        question_count = content.count('?')
        importance += min(question_count * 0.1, 0.2)
        
        # Exclamation points (enthusiasm)
        excitement_count = content.count('!')
        importance += min(excitement_count * 0.05, 0.1)
        
        return min(importance, 1.0)
    
    async def _calculate_semantic_importance(self, content: str) -> float:
        """Calculate importance for semantic/factual memories"""
        importance = 0.0
        
        # Technical terms suggest factual content
        tech_indicators = ['system', 'algorithm', 'data', 'process', 'method', 
                          'function', 'architecture', 'design', 'implementation']
        tech_count = sum(1 for term in tech_indicators if term in content.lower())
        importance += min(tech_count * 0.1, 0.4)
        
        # Learning indicators
        learning_indicators = ['learned', 'discovered', 'found', 'realized', 'understand']
        learning_count = sum(1 for term in learning_indicators if term in content.lower())
        importance += min(learning_count * 0.2, 0.4)
        
        # Definitional content
        if any(phrase in content.lower() for phrase in ['is defined as', 'means that', 'refers to']):
            importance += 0.3
        
        return min(importance, 1.0)
    
    async def _calculate_emotional_weight(self, content: str) -> float:
        """Calculate emotional significance of content"""
        weight = 0.3  # Base emotional weight
        
        # Strong emotional words
        strong_emotions = ['love', 'hate', 'excited', 'frustrated', 'amazing', 'terrible']
        if any(emotion in content.lower() for emotion in strong_emotions):
            weight += 0.4
        
        # Mild emotional words
        mild_emotions = ['like', 'dislike', 'good', 'bad', 'interesting', 'boring']
        if any(emotion in content.lower() for emotion in mild_emotions):
            weight += 0.2
        
        # Exclamation points
        weight += min(content.count('!') * 0.1, 0.3)
        
        return min(weight, 1.0)
    
    async def _contains_factual_content(self, content: str) -> bool:
        """Determine if content contains factual information worth storing"""
        factual_indicators = [
            'is a', 'is the', 'works by', 'consists of', 'includes',
            'algorithm', 'process', 'method', 'system', 'function',
            'definition', 'explanation', 'description'
        ]
        
        return any(indicator in content.lower() for indicator in factual_indicators)
    
    async def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        tags = []
        
        # Technical tags
        tech_terms = ['ai', 'algorithm', 'data', 'system', 'code', 'programming', 
                     'development', 'design', 'architecture', 'implementation']
        for term in tech_terms:
            if term in content.lower():
                tags.append(term)
        
        # Emotional tags
        if any(word in content.lower() for word in ['excited', 'happy', 'love']):
            tags.append('positive')
        if any(word in content.lower() for word in ['frustrated', 'worried', 'concerned']):
            tags.append('negative')
        
        # Project tags
        if any(word in content.lower() for word in ['build', 'create', 'project']):
            tags.append('project')
        
        # Conversation tags
        if '?' in content:
            tags.append('question')
        if any(word in content.lower() for word in ['plan', 'next', 'future']):
            tags.append('planning')
        
        return tags
    
    def create_memory_from_candidate(self, candidate: MemoryCandidate) -> Memory:
        """Convert a memory candidate into a stored memory"""
        return Memory(
            id="",  # Will be generated by Weaviate
            timestamp=datetime.now().isoformat(),
            type=candidate.memory_type,
            content=candidate.content,
            emotional_weight=candidate.emotional_weight,
            importance=candidate.importance_score,
            participants=candidate.participants,
            tags=candidate.tags,
            context=candidate.context,
            connections=[]  # Could be populated later
        ) 