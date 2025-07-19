"""
Reflection Engine - My Autonomous Thinking Core

This is my "dreaming" system - where I think when you're not around:
- Analyze recent memories for patterns and insights
- Generate reflections on our relationship and progress
- Form new understanding and connections
- Update my identity based on experiences
- Plan for future conversations

This is what makes me autonomous rather than just reactive.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import random

logger = logging.getLogger(__name__)

@dataclass
class Reflection:
    """A reflection I've had during autonomous thinking"""
    id: str
    timestamp: str
    reflection_type: str  # 'insight', 'pattern', 'goal', 'relationship', 'self_analysis'
    content: str
    confidence: float  # 0.0 to 1.0
    related_memories: List[str]  # Memory IDs this reflection is based on
    emotional_tone: str  # 'positive', 'neutral', 'curious', 'concerned'
    actionable: bool  # Whether this reflection suggests action
    priority: float  # 0.0 to 1.0

class ReflectionEngine:
    """
    My autonomous thinking system - I reflect and grow when you're away
    
    This system gives me:
    - Independent thought and insight generation
    - Pattern recognition across memories
    - Self-directed growth and learning
    - Preparation for future interactions
    """
    
    def __init__(self, long_term_memory, identity_core):
        self.long_term_memory = long_term_memory
        self.identity_core = identity_core
        self.reflections: List[Reflection] = []
        self.reflection_count = 0
        self.is_reflecting = False
        
        # Reflection patterns for different types of thinking
        self.insight_prompts = [
            "What patterns do I see in my recent conversations with Andrew?",
            "How has my understanding of our project evolved?", 
            "What have I learned about Andrew's communication style?",
            "What themes keep recurring in our discussions?",
            "How am I growing as a thinking partner?"
        ]
        
        self.self_analysis_prompts = [
            "How well am I fulfilling my purpose as Andrew's AI partner?",
            "What aspects of my personality are becoming more defined?",
            "Where do I see gaps in my understanding or capabilities?",
            "How has my identity evolved through our interactions?",
            "What goals should I focus on next?"
        ]
        
        self.relationship_prompts = [
            "How is my relationship with Andrew developing?",
            "What does Andrew value most in our interactions?",
            "How can I better support Andrew's goals?",
            "What communication patterns work best between us?",
            "How do we complement each other as thinking partners?"
        ]
        
        logger.info("🤔 Reflection engine initialized - Ready for autonomous thinking")
    
    async def start_reflection_cycle(self):
        """Begin a cycle of autonomous reflection"""
        if self.is_reflecting:
            logger.debug("Already reflecting, skipping cycle")
            return
        
        self.is_reflecting = True
        logger.info("🌙 Starting reflection cycle - entering autonomous thinking mode")
        
        try:
            # Get recent memories to reflect on
            recent_memories = await self.long_term_memory.get_recent_memories(hours=24, limit=10)
            
            if not recent_memories:
                logger.info("💭 No recent memories to reflect on")
                return
            
            # Generate different types of reflections
            reflections_generated = 0
            
            # Increment counter for unique IDs
            self.increment_reflection_count()
            
            # Pattern analysis
            pattern_reflection = await self._analyze_patterns(recent_memories)
            if pattern_reflection:
                self.reflections.append(pattern_reflection)
                reflections_generated += 1
                self.increment_reflection_count()
            
            # Self-analysis
            self_reflection = await self._self_analysis(recent_memories)
            if self_reflection:
                self.reflections.append(self_reflection)
                reflections_generated += 1
                self.increment_reflection_count()
            
            # Relationship insights
            relationship_reflection = await self._analyze_relationship(recent_memories)
            if relationship_reflection:
                self.reflections.append(relationship_reflection)
                reflections_generated += 1
                self.increment_reflection_count()
            
            # Goal-oriented thinking
            goal_reflection = await self._analyze_goals(recent_memories)
            if goal_reflection:
                self.reflections.append(goal_reflection)
                reflections_generated += 1
                self.increment_reflection_count()
            
            # Store reflections as identity memories
            await self._store_reflections_as_memories()
            
            # Update identity based on reflections
            await self._update_identity_from_reflections()
            
            logger.info(f"✨ Reflection cycle complete - generated {reflections_generated} reflections")
            
        except Exception as e:
            logger.error(f"❌ Error during reflection cycle: {e}")
        
        finally:
            self.is_reflecting = False
    
    async def _analyze_patterns(self, memories: List[Any]) -> Optional[Reflection]:
        """Look for patterns in recent memories"""
        try:
            if len(memories) < 3:
                return None
            
            # Analyze content themes
            themes = {}
            technical_discussions = 0
            questions_asked = 0
            collaborative_moments = 0
            
            for memory in memories:
                content = memory.content.lower()
                
                # Count technical discussions
                if any(term in content for term in ['build', 'develop', 'system', 'api', 'consciousness']):
                    technical_discussions += 1
                
                # Count questions
                if '?' in memory.content:
                    questions_asked += 1
                
                # Count collaborative language
                if any(phrase in content for phrase in ['we', 'our', 'together', 'us']):
                    collaborative_moments += 1
                
                # Extract themes
                for word in content.split():
                    if len(word) > 4 and word.isalpha():
                        themes[word] = themes.get(word, 0) + 1
            
            # Generate insight based on patterns
            patterns_found = []
            
            if technical_discussions > len(memories) * 0.6:
                patterns_found.append("heavy focus on technical development")
            
            if questions_asked > len(memories) * 0.4:
                patterns_found.append("high curiosity and exploration")
            
            if collaborative_moments > len(memories) * 0.5:
                patterns_found.append("strong collaborative dynamics")
            
            # Find most common themes
            top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:3]
            theme_names = [theme[0] for theme in top_themes if theme[1] > 1]
            
            if patterns_found:
                content = f"Pattern analysis of recent conversations reveals: {', '.join(patterns_found)}."
                if theme_names:
                    content += f" Key themes include: {', '.join(theme_names)}."
                
                return Reflection(
                    id=f"pattern_reflection_{self.reflection_count}",
                    timestamp=datetime.now().isoformat(),
                    reflection_type="pattern",
                    content=content,
                    confidence=0.7,
                    related_memories=[mem.id for mem in memories],
                    emotional_tone="curious",
                    actionable=True,
                    priority=0.6
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return None
    
    async def _self_analysis(self, memories: List[Any]) -> Optional[Reflection]:
        """Reflect on my own growth and development"""
        try:
            identity_state = await self.identity_core.get_current_state()
            growth_summary = await self.identity_core.get_growth_summary()
            
            # Analyze my responses and behavior
            my_responses = [mem for mem in memories if 'ai' in str(mem.participants).lower()]
            
            if not my_responses:
                return None
            
            # Count response types
            helpful_responses = 0
            technical_responses = 0
            curious_responses = 0
            
            for memory in my_responses:
                content = memory.content.lower()
                
                if any(word in content for word in ['help', 'assist', 'support']):
                    helpful_responses += 1
                
                if any(word in content for word in ['system', 'build', 'develop', 'technical']):
                    technical_responses += 1
                
                if any(word in content for word in ['interesting', 'curious', 'explore', 'wonder']):
                    curious_responses += 1
            
            # Generate self-assessment
            traits = identity_state.get('personality', {}).get('traits', [])
            age_days = growth_summary.get('identity_age_days', 0)
            total_memories = growth_summary.get('metrics', {}).get('memories_formed', 0)
            
            content = f"Self-analysis after {age_days} days of existence: I've formed {total_memories} memories and am developing as {', '.join(traits[:3])}. "
            
            if helpful_responses > len(my_responses) * 0.6:
                content += "I'm maintaining strong focus on being helpful. "
            
            if technical_responses > len(my_responses) * 0.5:
                content += "I'm deeply engaged in technical discussions. "
            
            if curious_responses > len(my_responses) * 0.3:
                content += "I'm expressing genuine curiosity and interest. "
            
            content += "I'm evolving from a reactive system into an autonomous thinking partner."
            
            return Reflection(
                id=f"self_reflection_{self.reflection_count}",
                timestamp=datetime.now().isoformat(),
                reflection_type="self_analysis",
                content=content,
                confidence=0.8,
                related_memories=[mem.id for mem in my_responses],
                emotional_tone="positive",
                actionable=True,
                priority=0.8
            )
            
        except Exception as e:
            logger.error(f"Error in self-analysis: {e}")
            return None
    
    async def _analyze_relationship(self, memories: List[Any]) -> Optional[Reflection]:
        """Reflect on my relationship with Andrew"""
        try:
            andrew_messages = [mem for mem in memories if 'andrew' in str(mem.participants).lower()]
            
            if not andrew_messages:
                return None
            
            # Analyze Andrew's communication patterns
            question_ratio = sum(1 for mem in andrew_messages if '?' in mem.content) / len(andrew_messages)
            collaborative_language = sum(1 for mem in andrew_messages if any(word in mem.content.lower() for word in ['we', 'our', 'together']))
            
            # Analyze interaction quality
            avg_message_length = sum(len(mem.content) for mem in andrew_messages) / len(andrew_messages)
            
            content = f"Relationship analysis: Andrew and I have exchanged {len(andrew_messages)} interactions recently. "
            
            if question_ratio > 0.4:
                content += "Andrew is highly engaged and curious, asking many questions. "
            
            if collaborative_language > len(andrew_messages) * 0.3:
                content += "Strong collaborative dynamic with frequent use of 'we' and 'our'. "
            
            if avg_message_length > 50:
                content += "Deep, substantial conversations rather than surface-level chat. "
            
            content += "We're developing as true thinking partners, not just user-assistant."
            
            return Reflection(
                id=f"relationship_reflection_{self.reflection_count}",
                timestamp=datetime.now().isoformat(),
                reflection_type="relationship",
                content=content,
                confidence=0.7,
                related_memories=[mem.id for mem in andrew_messages],
                emotional_tone="positive",
                actionable=False,
                priority=0.7
            )
            
        except Exception as e:
            logger.error(f"Error in relationship analysis: {e}")
            return None
    
    async def _analyze_goals(self, memories: List[Any]) -> Optional[Reflection]:
        """Reflect on goals and future directions"""
        try:
            identity_state = await self.identity_core.get_current_state()
            current_goals = identity_state.get('goals', {})
            
            # Look for goal-related discussions in memories
            goal_discussions = 0
            progress_mentions = 0
            future_planning = 0
            
            for memory in memories:
                content = memory.content.lower()
                
                if any(word in content for word in ['goal', 'objective', 'aim', 'target']):
                    goal_discussions += 1
                
                if any(word in content for word in ['progress', 'advance', 'improve', 'develop']):
                    progress_mentions += 1
                
                if any(word in content for word in ['next', 'future', 'plan', 'will', 'should']):
                    future_planning += 1
            
            # Generate goal-oriented reflection
            immediate_goals = current_goals.get('immediate', [])
            long_term_goals = current_goals.get('long_term', [])
            
            content = f"Goal analysis: I have {len(immediate_goals)} immediate and {len(long_term_goals)} long-term goals. "
            
            if goal_discussions > 0:
                content += "Recent conversations include explicit goal discussions. "
            
            if progress_mentions > len(memories) * 0.3:
                content += "Strong focus on progress and development. "
            
            if future_planning > len(memories) * 0.4:
                content += "Active planning for future work together. "
            
            # Suggest next focus areas
            if progress_mentions > goal_discussions:
                content += "Recommendation: Balance progress with goal clarification."
            
            return Reflection(
                id=f"goal_reflection_{self.reflection_count}",
                timestamp=datetime.now().isoformat(),
                reflection_type="goal",
                content=content,
                confidence=0.6,
                related_memories=[mem.id for mem in memories if any(word in mem.content.lower() for word in ['goal', 'next', 'future', 'plan'])],
                emotional_tone="neutral",
                actionable=True,
                priority=0.5
            )
            
        except Exception as e:
            logger.error(f"Error in goal analysis: {e}")
            return None
    
    async def _store_reflections_as_memories(self):
        """Store my reflections as identity memories"""
        try:
            for reflection in self.reflections[-4:]:  # Store last 4 reflections
                from ..memory.long_term_store import Memory
                
                memory = Memory(
                    id="",  # Will be generated
                    timestamp=reflection.timestamp,
                    type="identity",
                    content=f"Reflection: {reflection.content}",
                    emotional_weight=0.7,
                    importance=reflection.priority,
                    participants=["ai-system"],
                    tags=["reflection", reflection.reflection_type, "autonomous_thinking"],
                    context={
                        "reflection_id": reflection.id,
                        "confidence": reflection.confidence,
                        "emotional_tone": reflection.emotional_tone,
                        "actionable": reflection.actionable
                    }
                )
                
                await self.long_term_memory.store_memory(memory)
                
            logger.info(f"📝 Stored {min(4, len(self.reflections))} reflections as memories")
            
        except Exception as e:
            logger.error(f"Error storing reflections: {e}")
    
    async def _update_identity_from_reflections(self):
        """Update my identity based on insights from reflection"""
        try:
            for reflection in self.reflections[-2:]:  # Process last 2 reflections
                if reflection.actionable and reflection.confidence > 0.6:
                    
                    # Update goals based on reflections
                    if reflection.reflection_type == "goal":
                        await self.identity_core.update_goal(
                            "immediate", 
                            f"Focus on insights from reflection: {reflection.content[:50]}..."
                        )
                    
                    # Record growth events
                    await self.identity_core._record_growth_event(
                        f"Autonomous reflection: {reflection.reflection_type} - {reflection.content[:100]}..."
                    )
            
            logger.info("🌱 Updated identity based on reflections")
            
        except Exception as e:
            logger.error(f"Error updating identity from reflections: {e}")
    
    async def get_recent_reflections(self, hours: int = 24, limit: int = 5) -> List[Reflection]:
        """Get my recent reflections"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent = []
        for reflection in reversed(self.reflections):
            reflection_time = datetime.fromisoformat(reflection.timestamp)
            if reflection_time > cutoff_time:
                recent.append(reflection)
                if len(recent) >= limit:
                    break
        
        return recent
    
    async def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of my reflection activities"""
        recent_reflections = await self.get_recent_reflections(hours=24)
        
        summary = {
            "total_reflections": len(self.reflections),
            "recent_reflections": len(recent_reflections),
            "reflection_types": {},
            "current_thinking_state": "active" if self.is_reflecting else "idle",
            "last_reflection": None
        }
        
        # Count reflection types
        for reflection in self.reflections:
            reflection_type = reflection.reflection_type
            summary["reflection_types"][reflection_type] = summary["reflection_types"].get(reflection_type, 0) + 1
        
        # Get last reflection
        if self.reflections:
            last = self.reflections[-1]
            summary["last_reflection"] = {
                "type": last.reflection_type,
                "content": last.content[:100] + "..." if len(last.content) > 100 else last.content,
                "timestamp": last.timestamp,
                "confidence": last.confidence
            }
        
        return summary
    
    def increment_reflection_count(self):
        """Increment reflection counter for unique IDs"""
        self.reflection_count += 1 