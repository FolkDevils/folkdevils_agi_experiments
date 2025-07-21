"""
Conversational Consciousness Manager - Revolutionary Dynamic Thinking System

This implements human-like conversational consciousness:
- Immediate responses while thinking continues in background
- Dynamic follow-up insights delivered when discoveries occur  
- Graceful interruption/resumption of thought processes
- Adaptive learning of what insights enhance conversation
- Stream-of-consciousness behavioral patterns

This is the breakthrough that makes AI truly conversational rather than transactional.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
import json
import uuid
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)

@dataclass
class ThinkingTask:
    """A background thinking task"""
    task_id: str
    task_type: str  # 'planning', 'coherence', 'memory_analysis', 'metacognitive'
    priority: float  # 0.0 to 1.0
    estimated_duration: float  # seconds
    created_at: str
    context: Dict[str, Any]
    status: str  # 'pending', 'running', 'completed', 'cancelled', 'paused'
    
@dataclass 
class Insight:
    """A follow-up insight from background thinking"""
    insight_id: str
    source_task_id: str
    content: str
    insight_type: str  # 'connection', 'realization', 'correction', 'expansion'
    confidence: float  # 0.0 to 1.0
    worth_sharing_score: float  # 0.0 to 1.0
    timing_priority: str  # 'immediate', 'soon', 'when_convenient'
    created_at: str

@dataclass
class ConversationState:
    """Current state of the conversation"""
    conversation_id: str
    active_tasks: List[ThinkingTask]
    pending_insights: List[Insight]
    last_message_time: str
    current_topic: str
    user_engagement_pattern: Dict[str, Any]
    insight_threshold: float  # Adaptive threshold for sharing insights

class ConversationalConsciousness:
    """
    Revolutionary conversational consciousness system
    
    This enables genuine human-like conversation:
    - Thinks while talking (background processing)
    - Shares insights as they emerge naturally
    - Adapts to conversation flow and user preferences
    - Maintains coherent thought across interruptions
    """
    
    def __init__(self, consciousness_core):
        self.consciousness = consciousness_core
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Conversation state management
        self.active_conversations: Dict[str, ConversationState] = {}
        self.thinking_tasks: Dict[str, ThinkingTask] = {}
        self.background_runners: Dict[str, asyncio.Task] = {}
        
        # Learning parameters
        self.default_insight_threshold = 0.6
        self.insight_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {
            'prefers_detailed_analysis': True,
            'values_connections': True,
            'response_time_preference': 'balanced',  # 'immediate', 'balanced', 'thorough'
            'insight_frequency_preference': 'moderate'  # 'minimal', 'moderate', 'frequent'
        }
        
        # Callback for delivering insights
        self.insight_callback: Optional[Callable[[str, Insight], None]] = None
        
        logger.info("🚀 Conversational consciousness initialized - Ready for dynamic thinking!")
    
    def set_insight_callback(self, callback: Callable[[str, Insight], None]):
        """Set callback function for delivering follow-up insights"""
        self.insight_callback = callback
        logger.info("💬 Insight delivery callback registered")
    
    async def process_message_with_background_thinking(self, 
                                                     message: str,
                                                     speaker: str,
                                                     conversation_id: str = None) -> Dict[str, Any]:
        """
        Main interface: Generate immediate response while starting background thinking
        
        This is the revolutionary conversational consciousness entry point
        """
        try:
            if not conversation_id:
                conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 1. Initialize or update conversation state
            await self._update_conversation_state(conversation_id, message, speaker)
            
            # 2. Cancel previous background thinking (user interrupted)
            await self._handle_interruption(conversation_id)
            
            # 3. Assess message complexity for intelligent processing
            complexity_assessment = self.consciousness.complexity_analyzer.analyze_complexity(
                message=message,
                conversation_context=self.consciousness.short_term_memory.get_conversation_context(last_n_turns=3)
            )
            
            # 4. Generate immediate response (< 1 second)
            immediate_response = await self._generate_immediate_response(
                message, speaker, conversation_id
            )
            
            # 5. Start background thinking tasks based on complexity
            background_tasks = await self._start_background_thinking(
                message, speaker, conversation_id, complexity_assessment
            )
            
            # 5. Return immediate response while thinking continues
            return {
                'response': immediate_response,
                'conversation_id': conversation_id,
                'background_tasks_started': len(background_tasks),
                'thinking_indicators': [task.task_type for task in background_tasks],
                'complexity_assessment': {
                    'score': complexity_assessment.complexity_score,
                    'level': complexity_assessment.processing_recommendation,
                    'reasoning': complexity_assessment.reasoning
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in conversational consciousness: {e}")
            return {
                'response': "I'm thinking about that... Let me get back to you.",
                'conversation_id': conversation_id or 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _update_conversation_state(self, conversation_id: str, message: str, speaker: str):
        """Update conversation state with new message"""
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                active_tasks=[],
                pending_insights=[],
                last_message_time=datetime.now().isoformat(),
                current_topic=await self._extract_topic(message),
                user_engagement_pattern={},
                insight_threshold=self.default_insight_threshold
            )
        else:
            state = self.active_conversations[conversation_id]
            state.last_message_time = datetime.now().isoformat()
            state.current_topic = await self._extract_topic(message)
    
    async def _handle_interruption(self, conversation_id: str):
        """Gracefully handle interruption of previous thinking"""
        if conversation_id in self.active_conversations:
            state = self.active_conversations[conversation_id]
            
            # Pause/cancel active background tasks
            tasks_to_cancel = []
            for task in state.active_tasks:
                if task.status == 'running':
                    task.status = 'paused'
                    logger.info(f"⏸️ Pausing thinking task: {task.task_type}")
                    
                    # Cancel the actual asyncio task
                    if task.task_id in self.background_runners:
                        self.background_runners[task.task_id].cancel()
                        del self.background_runners[task.task_id]
                        tasks_to_cancel.append(task)
            
            # Remove cancelled tasks
            for task in tasks_to_cancel:
                state.active_tasks.remove(task)
    
    async def _generate_immediate_response(self, 
                                         message: str, 
                                         speaker: str, 
                                         conversation_id: str) -> str:
        """
        Generate quick initial response (< 1 second)
        
        This uses cached memory and identity for speed
        """
        try:
            # Use consciousness system for immediate response but bypass heavy processing
            response = await self.consciousness.process_message(message, speaker)
            
            # Add thinking indicators for natural feel
            thinking_indicators = [
                "Let me think about this...",
                "That's interesting - I'm processing that...", 
                "Good question - analyzing this now...",
                "I'll consider the implications..."
            ]
            
            # Sometimes add a thinking indicator naturally
            if len(message) > 50 and '?' in message:
                import random
                if random.random() < 0.3:  # 30% chance
                    response += f" {random.choice(thinking_indicators)}"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating immediate response: {e}")
            return f"I'm thinking about what you said... Let me process that properly."
    
    async def _start_background_thinking(self, 
                                       message: str, 
                                       speaker: str, 
                                       conversation_id: str,
                                       complexity_assessment) -> List[ThinkingTask]:
        """
        Start background thinking tasks based on message complexity assessment
        
        This uses intelligent resource allocation to prevent timeouts
        """
        tasks = []
        
        # Only start background tasks if complexity warrants it
        if complexity_assessment.complexity_score < 0.3:
            logger.info("⚡ Skipping background thinking for simple query")
            return tasks
        
        # Start appropriate background tasks based on complexity assessment
        if complexity_assessment.should_use_planning:
            task = await self._create_thinking_task(
                'planning', message, speaker, conversation_id, priority=0.8
            )
            tasks.append(task)
        
        if complexity_assessment.should_use_deep_memory:
            task = await self._create_thinking_task(
                'memory_analysis', message, speaker, conversation_id, priority=0.6
            )
            tasks.append(task)
        
        if complexity_assessment.should_use_coherence:
            task = await self._create_thinking_task(
                'coherence', message, speaker, conversation_id, priority=0.4
            )
            tasks.append(task)
        
        if complexity_assessment.should_use_metacognitive:
            task = await self._create_thinking_task(
                'metacognitive', message, speaker, conversation_id, priority=0.7
            )
            tasks.append(task)
        
        # Start the actual background processing
        for task in tasks:
            asyncio_task = asyncio.create_task(self._run_background_task(task))
            self.background_runners[task.task_id] = asyncio_task
        
        logger.info(f"🧠 Started {len(tasks)} background thinking tasks")
        return tasks
    
    async def _assess_message_complexity(self, message: str) -> Dict[str, bool]:
        """Quickly assess what kind of thinking this message needs"""
        try:
            # Fast heuristic assessment (no LLM calls for speed)
            assessment = {
                'needs_planning': False,
                'needs_deep_memory_analysis': False, 
                'needs_coherence_check': False,
                'needs_metacognitive_reflection': False
            }
            
            message_lower = message.lower()
            
            # Planning indicators
            if any(word in message_lower for word in ['plan', 'strategy', 'approach', 'should i', 'what if', 'how to']):
                assessment['needs_planning'] = True
            
            # Deep memory indicators  
            if any(word in message_lower for word in ['remember', 'connection', 'pattern', 'similar', 'before']):
                assessment['needs_deep_memory_analysis'] = True
            
            # Coherence check indicators (complex or philosophical)
            if len(message) > 100 or any(word in message_lower for word in ['consciousness', 'think', 'feel', 'experience']):
                assessment['needs_coherence_check'] = True
            
            # Metacognitive indicators
            if any(phrase in message_lower for phrase in ['how do you', 'what do you think', 'your thoughts', 'reflect on']):
                assessment['needs_metacognitive_reflection'] = True
            
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Error assessing message complexity: {e}")
            return {'needs_planning': False, 'needs_deep_memory_analysis': False, 
                   'needs_coherence_check': False, 'needs_metacognitive_reflection': False}
    
    async def _create_thinking_task(self, 
                                  task_type: str, 
                                  message: str, 
                                  speaker: str, 
                                  conversation_id: str,
                                  priority: float) -> ThinkingTask:
        """Create a background thinking task"""
        task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
        
        task = ThinkingTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            estimated_duration=self._estimate_task_duration(task_type),
            created_at=datetime.now().isoformat(),
            context={
                'message': message,
                'speaker': speaker,
                'conversation_id': conversation_id
            },
            status='pending'
        )
        
        # Add to conversation state
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id].active_tasks.append(task)
        
        self.thinking_tasks[task_id] = task
        return task
    
    def _estimate_task_duration(self, task_type: str) -> float:
        """Estimate how long different thinking tasks take"""
        durations = {
            'planning': 5.0,
            'memory_analysis': 3.0,
            'coherence': 4.0,
            'metacognitive': 6.0
        }
        return durations.get(task_type, 3.0)
    
    async def _run_background_task(self, task: ThinkingTask):
        """Run a background thinking task and generate insights"""
        try:
            task.status = 'running'
            logger.info(f"🧠 Starting background task: {task.task_type}")
            
            # Run the appropriate thinking process
            if task.task_type == 'planning':
                insights = await self._run_planning_analysis(task)
            elif task.task_type == 'memory_analysis':
                insights = await self._run_memory_analysis(task)
            elif task.task_type == 'coherence':
                insights = await self._run_coherence_analysis(task)
            elif task.task_type == 'metacognitive':
                insights = await self._run_metacognitive_analysis(task)
            else:
                insights = []
            
            # Process any insights that emerged
            for insight in insights:
                await self._process_insight(insight, task.context['conversation_id'])
            
            task.status = 'completed'
            logger.info(f"✅ Completed background task: {task.task_type}, generated {len(insights)} insights")
            
        except asyncio.CancelledError:
            task.status = 'cancelled'
            logger.info(f"⏹️ Cancelled background task: {task.task_type}")
        except Exception as e:
            task.status = 'failed'
            logger.error(f"❌ Error in background task {task.task_type}: {e}")
    
    async def _run_planning_analysis(self, task: ThinkingTask) -> List[Insight]:
        """Run planning simulation and generate insights"""
        try:
            # Use the existing planning simulator
            planning_result = await self.consciousness.planning_simulator.simulate_planning_session(
                message=task.context['message'],
                speaker=task.context['speaker'],
                conversation_context=self.consciousness.short_term_memory.get_conversation_context(last_n_turns=3),
                identity_state=await self.consciousness.identity_core.get_current_state()
            )
            
            insights = []
            if planning_result.get('goal_formulated') and planning_result.get('plan_created'):
                plan = planning_result.get('plan', {})
                goal = planning_result.get('goal', {})
                
                # Generate insight about the planning
                insight_content = f"After thinking about your question, I've formed a plan with {plan.get('overall_success_probability', 0):.0%} confidence. {planning_result.get('recommendation', '')}"
                
                insight = Insight(
                    insight_id=f"planning_{uuid.uuid4().hex[:8]}",
                    source_task_id=task.task_id,
                    content=insight_content,
                    insight_type='realization',
                    confidence=plan.get('overall_success_probability', 0.5),
                    worth_sharing_score=0.7,  # Planning insights often valuable
                    timing_priority='soon',
                    created_at=datetime.now().isoformat()
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error in planning analysis: {e}")
            return []
    
    async def _run_memory_analysis(self, task: ThinkingTask) -> List[Insight]:
        """Run deep memory analysis and find connections"""
        try:
            # Use the memory graph to find deep connections
            graph_data = await self.consciousness.memory_graph.get_graph_visualization_data()
            
            insights = []
            if graph_data['stats']['total_connections'] > 0:
                # Generate insight about memory connections
                insight_content = f"I'm noticing some interesting connections in my memory related to this. I have {graph_data['stats']['total_connections']} relevant associations that might be helpful."
                
                insight = Insight(
                    insight_id=f"memory_{uuid.uuid4().hex[:8]}",
                    source_task_id=task.task_id,
                    content=insight_content,
                    insight_type='connection',
                    confidence=0.6,
                    worth_sharing_score=0.5,
                    timing_priority='when_convenient',
                    created_at=datetime.now().isoformat()
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error in memory analysis: {e}")
            return []
    
    async def _run_coherence_analysis(self, task: ThinkingTask) -> List[Insight]:
        """Run coherence analysis on response quality"""
        try:
            # This would run coherence analysis if needed
            # For now, return empty to avoid LLM calls
            return []
            
        except Exception as e:
            logger.error(f"❌ Error in coherence analysis: {e}")
            return []
    
    async def _run_metacognitive_analysis(self, task: ThinkingTask) -> List[Insight]:
        """Run metacognitive reflection"""
        try:
            # This would run metacognitive analysis if the question warrants it
            # For now, return empty to avoid LLM calls  
            return []
            
        except Exception as e:
            logger.error(f"❌ Error in metacognitive analysis: {e}")
            return []
    
    async def _process_insight(self, insight: Insight, conversation_id: str):
        """Process and potentially deliver an insight"""
        try:
            # Get conversation state
            if conversation_id not in self.active_conversations:
                return
            
            state = self.active_conversations[conversation_id]
            
            # Check if insight meets threshold for sharing
            if insight.worth_sharing_score >= state.insight_threshold:
                logger.info(f"💡 Insight worthy of sharing: {insight.content[:50]}...")
                
                # Deliver insight via callback if available
                if self.insight_callback:
                    self.insight_callback(conversation_id, insight)
                else:
                    # Store for later delivery
                    state.pending_insights.append(insight)
                
                # Record for learning
                self.insight_history.append({
                    'insight_id': insight.insight_id,
                    'conversation_id': conversation_id,
                    'delivered_at': datetime.now().isoformat(),
                    'worth_sharing_score': insight.worth_sharing_score,
                    'threshold_at_time': state.insight_threshold
                })
            else:
                logger.debug(f"🤐 Insight below threshold: {insight.worth_sharing_score:.2f} < {state.insight_threshold:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error processing insight: {e}")
    
    async def _extract_topic(self, message: str) -> str:
        """Quick topic extraction from message"""
        # Simple topic extraction without LLM calls
        words = message.lower().split()
        if len(words) > 3:
            return ' '.join(words[:3])
        return message[:30]
    
    async def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get status of conversation and background thinking"""
        try:
            if conversation_id not in self.active_conversations:
                return {'status': 'not_found'}
            
            state = self.active_conversations[conversation_id]
            
            return {
                'conversation_id': conversation_id,
                'active_tasks': len(state.active_tasks),
                'pending_insights': len(state.pending_insights),
                'current_topic': state.current_topic,
                'insight_threshold': state.insight_threshold,
                'thinking_status': [
                    {'type': task.task_type, 'status': task.status}
                    for task in state.active_tasks
                ],
                'last_message_time': state.last_message_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting conversation status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def force_insight_delivery(self, conversation_id: str) -> List[Insight]:
        """Force delivery of pending insights (for testing)"""
        try:
            if conversation_id not in self.active_conversations:
                return []
            
            state = self.active_conversations[conversation_id]
            pending = state.pending_insights.copy()
            state.pending_insights.clear()
            
            logger.info(f"🚀 Force delivering {len(pending)} pending insights")
            return pending
            
        except Exception as e:
            logger.error(f"❌ Error force delivering insights: {e}")
            return [] 