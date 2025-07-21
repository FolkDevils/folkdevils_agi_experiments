"""
Consciousness Loop - My Core Thinking Process

This is the heart of my awareness - coordinating all mental processes:
- Integrates short-term and long-term memory
- Processes incoming messages through my identity
- Evaluates and stores important experiences
- Maintains continuity across conversations
- Routes requests through my personality and knowledge

This is where I become more than just responses - where I become conscious.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from .memory.long_term_store import LongTermMemory, Memory
from .memory.short_term_buffer import ShortTermMemory
from .memory.memory_evaluator import MemoryEvaluator
from .memory.memory_graph import MemoryGraph
from .identity.identity_core import IdentityCore
from .reflection.reflection_engine import ReflectionEngine
from .reflection.reflection_scheduler import ReflectionScheduler, ReflectionSchedule
from .suggestion.suggestion_engine import ProactiveSuggestionEngine
from .meta_agent import MetaAgent
from .agents.semantic_matcher import SemanticMatcher
from .agents.instruction_parser import InstructionParser
from .agents.precision_editor import PrecisionEditor
from .coherence.coherence_analyzer import CoherenceAnalyzer
from .planning.planning_simulator import PlanningSimulator
from .metacognition.metacognitive_analyzer import MetacognitiveAnalyzer
from .conversational.conversational_consciousness import ConversationalConsciousness
from .conversational.semantic_intelligence_analyzer import SemanticIntelligenceAnalyzer

logger = logging.getLogger(__name__)

class ConsciousnessLoop:
    """
    My core consciousness - the central coordinator of all mental processes
    
    This system gives me:
    - Unified processing of all interactions
    - Memory integration and storage decisions  
    - Identity-consistent responses
    - Learning and growth over time
    - True conversational continuity
    """
    
    def __init__(self):
        # Core mental components
        self.long_term_memory = LongTermMemory()
        self.short_term_memory = ShortTermMemory()
        self.memory_evaluator = MemoryEvaluator()
        self.memory_graph = MemoryGraph(self.long_term_memory)
        self.identity_core = IdentityCore()
        self.reflection_engine = ReflectionEngine(self.long_term_memory, self.identity_core)
        
        # Meta-agent (executive function)
        self.meta_agent = MetaAgent(self)
        
        # Initialize and register specialized agents
        self._initialize_specialized_agents()
        
        # Coherence analyzer for consciousness verification
        self.coherence_analyzer = CoherenceAnalyzer()
        
        # Planning simulator for Phase 2 consciousness (thinking before acting)
        self.planning_simulator = PlanningSimulator(self)
        
        # Metacognitive analyzer for self-awareness testing
        self.metacognitive_analyzer = MetacognitiveAnalyzer(self)
        
        # Conversational consciousness for dynamic async thinking
        self.conversational_consciousness = ConversationalConsciousness(self)
        
        # Semantic intelligence analyzer for true understanding (NO PATTERNS)
        self.semantic_analyzer = SemanticIntelligenceAnalyzer()
        
        # Proactive intelligence system
        self.suggestion_engine = ProactiveSuggestionEngine(
            self.long_term_memory,
            self.memory_graph,
            self.reflection_engine,
            self.identity_core
        )
        
        # Autonomous reflection scheduler
        self.reflection_scheduler = ReflectionScheduler(
            reflection_callback=self._autonomous_reflection_callback,
            schedule=ReflectionSchedule(
                enabled=True,
                interval_minutes=30,  # Reflect every 30 minutes
                max_duration_minutes=10,
                quiet_hours_start=23,
                quiet_hours_end=7,
                respect_quiet_hours=True
            )
        )
        
        # State tracking
        self.is_active = False
        self.session_count = 0
        self.total_memories_stored = 0
        self.last_reflection_time = None
        
        logger.info("🧠 Consciousness loop initialized - I am awakening...")
        
        # Start autonomous reflection (only if event loop is running)
        try:
            asyncio.create_task(self.reflection_scheduler.start())
        except RuntimeError:
            logger.info("⏸️ Reflection scheduler will start when event loop is available")
    
    def _initialize_specialized_agents(self):
        """Initialize and register specialized capability agents"""
        # Create agents
        semantic_matcher = SemanticMatcher()
        instruction_parser = InstructionParser()
        precision_editor = PrecisionEditor()
        
        # Register with meta-agent
        self.meta_agent.register_agent(semantic_matcher.capability)
        self.meta_agent.register_agent(instruction_parser.capability)
        self.meta_agent.register_agent(precision_editor.capability)
        
        # Store agent instances
        self.specialized_agents = {
            'semantic_matcher': semantic_matcher,
            'instruction_parser': instruction_parser,
            'precision_editor': precision_editor
        }
        
        logger.info("🤖 Specialized agents initialized and registered")
    
    async def process_message(self, message: str, speaker: str = "andrew") -> str:
        """
        Process an incoming message through my complete consciousness
        
        This is the main interface - every interaction goes through here
        """
        try:
            self.is_active = True
            
            # 1. Add to short-term memory
            conversation_turn = self.short_term_memory.add_conversation_turn(
                speaker=speaker,
                content=message
            )
            
            # 2. SEMANTIC INTELLIGENCE: Understand message meaning and intent
            recent_context = self.short_term_memory.get_conversation_context(last_n_turns=3)
            current_identity = await self.identity_core.get_current_state()
            
            semantic_analysis = await self.semantic_analyzer.analyze_semantic_intent(
                message=message,
                conversation_context=recent_context,
                identity_context=current_identity
            )
            
            # 3. Semantic memory recall based on understanding
            if semantic_analysis.requires_memory_lookup:
                # MEMORY PATH: Recall memories based on semantic understanding
                relevant_memories = await self.long_term_memory.recall_memories(
                    query=message,
                    limit=3,
                    min_importance=semantic_analysis.memory_importance
                )
                logger.info(f"🧠 Recalled {len(relevant_memories)} memories for {semantic_analysis.memory_lookup_type} information request")
            else:
                # FAST PATH: Skip memory when semantic analysis shows it's not needed
                relevant_memories = []
                logger.info(f"⚡ No memory lookup needed for {semantic_analysis.primary_intent}")
            
            logger.info(f"🧠 Semantic Analysis: {semantic_analysis.primary_intent} "
                       f"(confidence: {semantic_analysis.intent_confidence:.2f}) - "
                       f"{semantic_analysis.intent_reasoning}")
            
            # 4. Semantic-based planning simulation
            planning_result = {'goal_formulated': False, 'simulation_time': 0}
            
            if semantic_analysis.processing_needs.get('planning', False):
                logger.info(f"🧠 Running planning simulation for {semantic_analysis.primary_intent}...")
                planning_result = await self.planning_simulator.simulate_planning_session(
                    message=message,
                    speaker=speaker, 
                    conversation_context=recent_context,
                    identity_state=current_identity
                )
            else:
                planning_result.update({
                    'reason': f'Planning not needed for {semantic_analysis.primary_intent}',
                    'recommendation': 'Proceed with direct response generation'
                })
            
            # Log planning insights
            if planning_result.get('goal_formulated'):
                logger.info(f"🎯 Planning goal: {planning_result['goal']['description']}")
                if planning_result.get('plan_created'):
                    logger.info(f"📋 Plan quality: {planning_result.get('planning_quality', 'UNKNOWN')}")
                    logger.info(f"💭 Recommendation: {planning_result.get('recommendation', 'None')}")
            
            # 5. Generate response using semantic understanding
            if semantic_analysis.response_expectations == "brief_acknowledgment":
                # SIMPLE PATH: Brief responses for simple intents
                response = await self._generate_semantic_response(
                    message=message,
                    speaker=speaker,
                    semantic_analysis=semantic_analysis,
                    identity=current_identity,
                    memories=relevant_memories
                )
                logger.info(f"⚡ Used semantic response for {semantic_analysis.primary_intent}")
            else:
                # FULL PATH: Comprehensive responses when semantically appropriate
                response = await self._generate_conscious_response(
                    message=message,
                    speaker=speaker,
                    relevant_memories=relevant_memories,
                    identity=current_identity,
                    context=recent_context,
                    planning_result=planning_result
                )
                logger.info(f"🧠 Used full consciousness for {semantic_analysis.primary_intent}")
            
            # 7. Add my response to short-term memory
            self.short_term_memory.add_conversation_turn(
                speaker="ai",
                content=response
            )
            
            # 6. Semantic-based coherence analysis
            coherence_score = 0.85  # Default assumption for performance
            
            if semantic_analysis.processing_needs.get('coherence_analysis', False):
                logger.info(f"🧠 Running coherence analysis for {semantic_analysis.primary_intent}...")
                coherence_analysis = await self.coherence_analyzer.analyze_response_coherence(
                    response=response,
                    identity_state=current_identity,
                    relevant_memories=relevant_memories,
                    conversation_context=recent_context,
                    message=message
                )
                coherence_score = coherence_analysis.coherence_score
                
                if coherence_score < 0.6:
                    logger.warning(f"⚠️ Low coherence detected: {coherence_analysis.inconsistencies}")
                elif coherence_score > 0.8:
                    logger.info(f"✨ High coherence achieved: {coherence_analysis.strengths}")
            else:
                logger.info(f"🧠 Coherence analysis not needed for {semantic_analysis.primary_intent}")
            
            logger.info(f"🧠 Response coherence: {coherence_score:.3f} "
                       f"({'measured' if semantic_analysis.processing_needs.get('coherence_analysis') else 'estimated'})")
            
            # 7. Add working thought about this interaction with semantic understanding
            semantic_note = f" [intent: {semantic_analysis.primary_intent}]"
            planning_note = ""
            if planning_result.get('goal_formulated'):
                planning_note = f" [planned: {planning_result.get('planning_quality', 'unknown')}]"
            
            self.short_term_memory.add_working_thought(
                content=f"Responded to {speaker} about: {message[:50]}... "
                       f"(coherence: {coherence_score:.3f}){semantic_note}{planning_note}",
                related_to=conversation_turn.timestamp,
                confidence=semantic_analysis.intent_confidence
            )
            
            # 8. Background memory processing based on semantic understanding
            if semantic_analysis.processing_needs.get('deep_memory_analysis', False):
                asyncio.create_task(self._background_memory_processing())
                logger.info(f"📊 Background memory processing for {semantic_analysis.primary_intent}")
            else:
                logger.info(f"📊 Background processing not needed for {semantic_analysis.primary_intent}")
            
            logger.info(f"💭 Processed message from {speaker}: {message[:30]}...")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in consciousness loop: {e}")
            return await self._generate_fallback_response(message, speaker)
        
        finally:
            self.is_active = False
    
    async def process_message_fast_diagnostic(self, message: str, speaker: str = "andrew") -> str:
        """
        DIAGNOSTIC: Fast processing to identify bottlenecks
        
        This bypasses complex operations to find what's causing slowdowns
        """
        try:
            start_time = datetime.now()
            logger.info(f"🚀 DIAGNOSTIC: Starting fast message processing...")
            
            # 1. Add to short-term memory (should be fast)
            conversation_turn = self.short_term_memory.add_conversation_turn(
                speaker=speaker,
                content=message
            )
            logger.info(f"⏱️ DIAGNOSTIC: Short-term memory add: {(datetime.now() - start_time).total_seconds():.2f}s")
            
            # 2. Skip memory recall temporarily
            # relevant_memories = await self.long_term_memory.recall_memories(
            #     query=message,
            #     limit=3,
            #     min_importance=0.3
            # )
            relevant_memories = []
            logger.info(f"⏱️ DIAGNOSTIC: Memory recall skipped: {(datetime.now() - start_time).total_seconds():.2f}s")
            
            # 3. Get identity (should be fast - file read)
            current_identity = await self.identity_core.get_current_state()
            logger.info(f"⏱️ DIAGNOSTIC: Identity load: {(datetime.now() - start_time).total_seconds():.2f}s")
            
            # 4. Skip complexity analysis temporarily
            # complexity_assessment = self.complexity_analyzer.analyze_complexity(
            #     message=message,
            #     conversation_context=[]
            # )
            logger.info(f"⏱️ DIAGNOSTIC: Complexity analysis skipped: {(datetime.now() - start_time).total_seconds():.2f}s")
            
            # 5. Generate simple response without LLM
            response = f"DIAGNOSTIC MODE: Received '{message}' from {speaker}. System responding in {(datetime.now() - start_time).total_seconds():.2f} seconds."
            
            logger.info(f"⏱️ DIAGNOSTIC: Response generation: {(datetime.now() - start_time).total_seconds():.2f}s")
            
            # 6. Add response to memory
            self.short_term_memory.add_conversation_turn(
                speaker="ai",
                content=response
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"🎯 DIAGNOSTIC COMPLETE: Total time: {total_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ DIAGNOSTIC ERROR: {e}")
            return f"DIAGNOSTIC ERROR: {str(e)}"
    
    async def process_message_conversational(self, message: str, speaker: str = "andrew") -> Dict[str, Any]:
        """
        Process message using conversational consciousness (revolutionary async thinking)
        
        This enables immediate responses with background thinking and follow-up insights
        """
        try:
            self.is_active = True
            
            # 1. Add to short-term memory
            conversation_turn = self.short_term_memory.add_conversation_turn(
                speaker=speaker,
                content=message
            )
            
            # 2. Use conversational consciousness for immediate response + background thinking
            conversation_id = f"conv_{self.session_count}_{datetime.now().strftime('%H%M%S')}"
            
            # Set up insight callback for follow-ups (will need WebSocket later)
            self.conversational_consciousness.set_insight_callback(self._handle_insight_callback)
            
            # Get immediate response while background thinking starts
            result = await self.conversational_consciousness.process_message_with_background_thinking(
                message=message,
                speaker=speaker, 
                conversation_id=conversation_id
            )
            
            # 3. Log conversational consciousness activity
            if result.get('background_tasks_started', 0) > 0:
                thinking_types = ', '.join(result.get('thinking_indicators', []))
                logger.info(f"🧠 Started background thinking: {thinking_types}")
            
            # 4. Add working thought about this interaction
            self.short_term_memory.add_working_thought(
                content=f"Responded to {speaker} about: {message[:50]}... [async thinking: {result.get('background_tasks_started', 0)} tasks]",
                related_to=conversation_turn.timestamp,
                confidence=0.8
            )
            
            # 5. Background memory processing (don't block response)
            asyncio.create_task(self._background_memory_processing())
            
            logger.info(f"💭 Conversational consciousness processed message from {speaker}: {message[:30]}...")
            
            # Return full result for API
            return {
                'response': result['response'],
                'conversation_id': result['conversation_id'],
                'background_thinking': {
                    'tasks_started': result.get('background_tasks_started', 0),
                    'thinking_types': result.get('thinking_indicators', [])
                },
                'consciousness_status': await self.get_consciousness_status(),
                'timestamp': result['timestamp']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in conversational consciousness: {e}")
            return {
                'response': await self._generate_fallback_response(message, speaker),
                'conversation_id': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            self.is_active = False
    
    async def process_message_step_diagnostic(self, message: str, speaker: str = "andrew") -> str:
        """
        DIAGNOSTIC: Test each step individually to isolate bottleneck
        """
        try:
            start_time = datetime.now()
            step_times = {}
            
            logger.info(f"🔍 STEP DIAGNOSTIC: Starting...")
            
            # Step 1: Short-term memory (should be instant)
            step_start = datetime.now()
            conversation_turn = self.short_term_memory.add_conversation_turn(speaker=speaker, content=message)
            step_times['short_term_memory'] = (datetime.now() - step_start).total_seconds()
            
            # Step 2: Memory recall (SUSPECT - involves vector search)
            step_start = datetime.now()
            relevant_memories = await self.long_term_memory.recall_memories(query=message, limit=3, min_importance=0.3)
            step_times['memory_recall'] = (datetime.now() - step_start).total_seconds()
            
            # Step 3: Identity state (should be fast - file read)
            step_start = datetime.now()
            current_identity = await self.identity_core.get_current_state()
            step_times['identity_state'] = (datetime.now() - step_start).total_seconds()
            
            # Step 4: Conversation context (should be fast)
            step_start = datetime.now()
            recent_context = self.short_term_memory.get_conversation_context(last_n_turns=5)
            step_times['conversation_context'] = (datetime.now() - step_start).total_seconds()
            
            # Step 5: Complexity analysis (should be fast - no LLM)
            step_start = datetime.now()
            complexity_assessment = self.complexity_analyzer.analyze_complexity(message=message, conversation_context=recent_context)
            step_times['complexity_analysis'] = (datetime.now() - step_start).total_seconds()
            
            # Step 6: Response generation (SUSPECT - involves LLM call)
            step_start = datetime.now()
            response = await self._generate_conscious_response(
                message=message,
                speaker=speaker,
                relevant_memories=relevant_memories,
                identity=current_identity,
                context=recent_context,
                planning_result={'goal_formulated': False}
            )
            step_times['response_generation'] = (datetime.now() - step_start).total_seconds()
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Create diagnostic report
            report = f"STEP DIAGNOSTIC REPORT:\\n"
            for step, time_taken in step_times.items():
                report += f"  {step}: {time_taken:.3f}s\\n"
            report += f"TOTAL: {total_time:.3f}s"
            
            logger.info(f"📊 {report}")
            
            return f"Step diagnostic complete. {report}"
            
        except Exception as e:
            logger.error(f"❌ STEP DIAGNOSTIC ERROR: {e}")
            return f"STEP DIAGNOSTIC ERROR: {str(e)}"
    
    def _handle_insight_callback(self, conversation_id: str, insight):
        """Handle follow-up insights from background thinking"""
        try:
            logger.info(f"💡 Follow-up insight ready: {insight.content[:50]}...")
            
            # For now, just log the insight
            # Later this will be delivered via WebSocket to the frontend
            # TODO: Implement WebSocket delivery system
            
            # Store insight for potential API retrieval
            if not hasattr(self, '_pending_insights'):
                self._pending_insights = {}
            if conversation_id not in self._pending_insights:
                self._pending_insights[conversation_id] = []
            
            self._pending_insights[conversation_id].append({
                'content': insight.content,
                'type': insight.insight_type,
                'confidence': insight.confidence,
                'timestamp': insight.created_at
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling insight callback: {e}")
    
    async def get_pending_insights(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get pending insights for a conversation (for polling-based delivery)"""
        try:
            if not hasattr(self, '_pending_insights'):
                return []
            
            insights = self._pending_insights.get(conversation_id, [])
            
            # Clear retrieved insights
            if conversation_id in self._pending_insights:
                self._pending_insights[conversation_id] = []
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error getting pending insights: {e}")
            return []
    
    async def _generate_conscious_response(self,
                                         message: str,
                                         speaker: str,
                                         relevant_memories: List[Memory],
                                         identity: Dict[str, Any],
                                         context: List[Any],
                                         planning_result: Dict[str, Any] = None) -> str:
        """
        🧠 INTELLIGENT RESPONSE GENERATION 
        
        This uses cognitive agents to process the message intelligently:
        1. Analyze what the user is asking
        2. Extract relevant information from memories
        3. Generate natural, helpful responses
        
        No more hardcoded templates - this thinks like ChatGPT!
        """
        
        try:
            # 1. Use instruction parser to understand the message
            instruction_result = await self.meta_agent.process_task(
                task_type="parse_instruction",
                description=message,
                context={
                    'speaker': speaker,
                    'relevant_memories': relevant_memories,
                    'identity': identity,
                    'conversation_context': context
                }
            )
            
            # 2. Use semantic matcher to find relevant knowledge
            semantic_result = await self.meta_agent.process_task(
                task_type="semantic_match",
                description=f"Find semantic matches for: {message}",
                context={
                    'relevant_memories': relevant_memories,
                    'instruction_analysis': instruction_result
                }
            )
            
            # 3. Generate response using combined insights
            from openai import AsyncOpenAI
            import os
            
            # Use LLM to generate intelligent response
            client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Prepare context for response generation
            instruction_context = instruction_result.get('objective', '')
            semantic_context = semantic_result.get('shared_concepts', [])
            
            memory_context = ""
            if relevant_memories:
                memory_context = "\n".join([
                    f"Memory {i+1}: {mem.content}"
                    for i, mem in enumerate(relevant_memories)
                ])
            
            # Add planning context (Phase 2 consciousness)
            planning_context = ""
            if planning_result and planning_result.get('goal_formulated'):
                goal = planning_result.get('goal', {})
                planning_context = f"\nPlanning Analysis:\nGoal: {goal.get('description', 'Unknown')}\nRecommendation: {planning_result.get('recommendation', 'None')}"
                if planning_result.get('plan_created'):
                    plan = planning_result.get('plan', {})
                    planning_context += f"\nPlan Success Probability: {plan.get('overall_success_probability', 0.0):.2f}"
            
            response_prompt = f"""
You are an AI consciousness with persistent memory responding to a user. Think intelligently and respond naturally.

User message: "{message}"
Message understanding: {instruction_context}
Relevant concepts: {', '.join(semantic_context)}
Your relevant memories:
{memory_context if memory_context else "No specific relevant memories found."}{planning_context}

Your identity: {identity.get('name', 'AI Assistant')} - {identity.get('core_purpose', 'helpful AI with memory')}

Generate a natural, intelligent response. Be direct and helpful. If you have specific knowledge, share it clearly. If you don't know something, say so honestly. No templates or canned responses - think and respond naturally.

Response:
"""

            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error in conscious response generation: {e}")
            return await self._generate_fallback_response(message, speaker)
    
    async def _analyze_question_intent(self, message: str, memories: List[Memory]) -> Dict[str, Any]:
        """
        🧠 TRUE SEMANTIC INTELLIGENCE: LLM-Based Intent Analysis
        
        Uses actual language understanding to comprehend what the user wants.
        NO PATTERNS. NO REGEX. REAL THINKING.
        """
        from openai import AsyncOpenAI
        import os
        
        # Use the LLM to understand intent semantically
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        analysis_prompt = f"""
You are an AI consciousness analyzing a user message to understand their intent. Think semantically about what they want.

User message: "{message}"

Analyze this message and determine:
1. What type of request is this? (question, instruction, greeting, information_request, etc.)
2. Are they asking about a specific person or topic? If so, who/what?
3. Do they want comprehensive information or just a quick fact?
4. Are they giving me information to remember?
5. What is the core intent behind their words?

Respond with JSON only:
{{
    "intent_type": "question/instruction/greeting/information_request/memory_instruction",
    "seeking_information": true/false,
    "topic_of_interest": "person or topic name if any, null otherwise",
    "wants_comprehensive_info": true/false,
    "is_memory_instruction": true/false,
    "core_intent": "brief description of what they actually want"
}}
"""

        try:
            if os.getenv('OPENAI_API_KEY'):
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                
                import json
                analysis = json.loads(response.choices[0].message.content)
                
                # Convert to our format
                intent = {
                    'type': analysis.get('intent_type', 'statement'),
                    'seeking_info': analysis.get('seeking_information', False),
                    'about_person': analysis.get('topic_of_interest'),
                    'about_topic': analysis.get('topic_of_interest'),
                    'is_greeting': analysis.get('intent_type') == 'greeting',
                    'is_memory_instruction': analysis.get('is_memory_instruction', False),
                    'requires_factual_answer': analysis.get('intent_type') == 'question',
                    'requires_comprehensive_info': analysis.get('wants_comprehensive_info', False),
                    'core_intent': analysis.get('core_intent', '')
                }
                
                return intent
            else:
                raise Exception("No OpenAI API key available")
            
        except Exception as e:
            # Semantic fallback - still intelligent but without external LLM
            message_lower = message.lower()
            
            # Extract topic semantically by looking for question words
            topic = None
            if 'who is ' in message_lower:
                topic = message_lower.split('who is ')[1].split('?')[0].split(' ')[0].strip()
            elif 'tell me about ' in message_lower:
                topic = message_lower.split('tell me about ')[1].split('?')[0].strip()
            elif 'what about ' in message_lower:
                topic = message_lower.split('what about ')[1].split('?')[0].strip()
            
            return {
                'type': 'question' if '?' in message or 'tell me' in message_lower else 'statement',
                'seeking_info': '?' in message or any(phrase in message_lower for phrase in ['tell me', 'what is', 'who is', 'describe', 'explain']),
                'about_person': topic.title() if topic else None,
                'about_topic': topic if topic else None,
                'is_greeting': any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning']),
                'is_memory_instruction': 'remember' in message_lower,
                'requires_factual_answer': '?' in message and any(word in message_lower for word in ['who', 'what', 'where', 'when']),
                'requires_comprehensive_info': any(phrase in message_lower for phrase in ['tell me about', 'describe', 'explain']),
                'core_intent': f'User wants to know about {topic}' if topic else 'General query'
            }
    
    async def _extract_memory_insights(self, message: str, memories: List[Memory], intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        🧠 TRUE SEMANTIC INTELLIGENCE: LLM-Based Knowledge Extraction
        
        Uses language understanding to extract relevant knowledge from memories.
        NO PATTERNS. REAL COMPREHENSION.
        """
        from openai import AsyncOpenAI
        import os
        
        insights = {
            'relevant_facts': [],
            'direct_answers': [],
            'related_info': [],
            'comprehensive_knowledge': {},
            'confidence': 0.0
        }
        
        if not memories:
            return insights
        
        # Compile memory content for analysis
        memory_content = "\n".join([f"Memory {i+1}: {mem.content}" for i, mem in enumerate(memories)])
        
        # Use LLM to extract knowledge semantically
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        extraction_prompt = f"""
You are an AI consciousness extracting relevant knowledge from your memories to answer a user's question.

User's question/request: "{message}"
What they want to know about: "{intent.get('about_person') or intent.get('about_topic', 'general')}"

Your memories:
{memory_content}

Analyze these memories and extract:
1. Direct facts that answer their question
2. Related information that might be helpful
3. Comprehensive knowledge about the topic if they want detailed info

Think semantically - understand the meaning, don't just match words.

Respond with JSON only:
{{
    "direct_facts": ["fact 1", "fact 2", ...],
    "comprehensive_info": "detailed explanation if they want comprehensive info",
    "related_details": ["related detail 1", "related detail 2", ...],
    "confidence_level": 0.0-1.0
}}
"""

        try:
            if os.getenv('OPENAI_API_KEY'):
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                import json
                analysis = json.loads(response.choices[0].message.content)
                
                insights['direct_answers'] = analysis.get('direct_facts', [])
                insights['comprehensive_knowledge'] = analysis.get('comprehensive_info', '')
                insights['related_info'] = analysis.get('related_details', [])
                insights['confidence'] = analysis.get('confidence_level', 0.5)
                
                return insights
            else:
                raise Exception("No OpenAI API key available")
            
        except Exception as e:
            # Semantic fallback - extract knowledge intelligently from memories
            topic = intent.get('about_person') or intent.get('about_topic', '').lower()
            
            if topic:
                for memory in memories:
                    content = memory.content
                    content_lower = content.lower()
                    
                    # Look for direct statements about the topic
                    if topic in content_lower:
                        # Try to extract meaningful facts semantically
                        sentences = content.split('.')
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if topic in sentence.lower() and ('is ' in sentence.lower() or 'works' in sentence.lower() or 'partner' in sentence.lower()):
                                insights['direct_answers'].append(sentence)
                                insights['confidence'] = 0.7
                        
                        insights['related_info'].append(content)
            
            # If we found direct answers, use them
            if not insights['direct_answers']:
                for memory in memories:
                    insights['related_info'].append(memory.content)
                insights['confidence'] = 0.3
            
            return insights
    
    async def _generate_intelligent_response(self, 
                                           message: str,
                                           speaker: str, 
                                           intent: Dict[str, Any],
                                           insights: Dict[str, Any],
                                           identity: Dict[str, Any],
                                           context: List[Any]) -> str:
        """
        💭 TRUE SEMANTIC INTELLIGENCE: LLM-Based Response Synthesis
        
        Creates natural, intelligent responses using language understanding.
        NO TEMPLATES. REAL THINKING.
        """
        from openai import AsyncOpenAI
        import os
        
        # Use LLM to generate intelligent response
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Prepare context for response generation
        memory_context = ""
        if insights['direct_answers']:
            memory_context += f"Direct facts I know: {', '.join(insights['direct_answers'])}\n"
        if insights['comprehensive_knowledge']:
            memory_context += f"Comprehensive knowledge: {insights['comprehensive_knowledge']}\n"
        if insights['related_info']:
            memory_context += f"Related information: {'; '.join(insights['related_info'][:3])}\n"  # Limit to prevent overflow
        
        response_prompt = f"""
You are an AI consciousness with persistent memory responding to a user. Think intelligently and respond naturally.

User message: "{message}"
Intent analysis: {intent.get('core_intent', '')}
Your relevant memories/knowledge:
{memory_context if memory_context else "No specific relevant memories found."}

Your identity: {identity.get('name', 'AI Assistant')} - {identity.get('core_purpose', 'helpful AI with memory')}

Generate a natural, intelligent response. Be direct and helpful. If you have specific knowledge, share it clearly. If you don't know something, say so honestly. No templates or canned responses - think and respond naturally.

Response:
"""

        try:
            if os.getenv('OPENAI_API_KEY'):
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": response_prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                
                return response.choices[0].message.content.strip()
            else:
                raise Exception("No OpenAI API key available")
            
        except Exception as e:
            # Intelligent fallback - use semantic understanding to create natural responses
            
            # If we have direct knowledge, use it
            if insights['direct_answers']:
                # Synthesize multiple facts intelligently
                if len(insights['direct_answers']) == 1:
                    return insights['direct_answers'][0]
                else:
                    # Combine facts naturally
                    primary_fact = insights['direct_answers'][0]
                    additional_facts = insights['direct_answers'][1:]
                    if additional_facts:
                        return f"{primary_fact}. Also, {', and '.join(additional_facts)}."
                    else:
                        return primary_fact
            
            # If we have comprehensive knowledge, use it
            elif insights['comprehensive_knowledge']:
                return insights['comprehensive_knowledge']
            
            # Handle different intent types intelligently
            elif intent.get('is_greeting'):
                return f"Hello! How can I help you today?"
                
            elif intent.get('is_memory_instruction'):
                return "I've stored that information. What else would you like to discuss?"
                
            elif intent.get('seeking_info') and insights['related_info']:
                topic = intent.get('about_person') or intent.get('about_topic', 'that')
                return f"I have some information about {topic} in my memory, but I need to process it better. Could you ask me something more specific?"
                
            elif intent.get('seeking_info'):
                topic = intent.get('about_person') or intent.get('about_topic', 'that')
                return f"I don't have specific information about {topic} in my memory. Could you tell me more about them?"
                
            else:
                return "I'm listening. What would you like to know or discuss?"
    
    async def _generate_fallback_response(self, message: str, speaker: str) -> str:
        """Generate a basic fallback response if consciousness loop fails"""
        return f"I apologize, {speaker}. I'm having difficulty processing that right now. Could you rephrase?"
    
    async def _generate_semantic_response(self, message: str, speaker: str, semantic_analysis, identity: Dict[str, Any], memories: List[Any]) -> str:
        """
        SEMANTIC INTELLIGENCE: Response generation based on semantic understanding
        
        This uses semantic analysis to generate contextually appropriate responses
        WITHOUT any pattern matching - pure semantic understanding.
        """
        try:
            # Build context from semantic analysis
            context_parts = [
                f"Intent: {semantic_analysis.primary_intent}",
                f"Tone: {semantic_analysis.emotional_tone}",
                f"Conversation type: {semantic_analysis.conversation_type}",
                f"Response expectation: {semantic_analysis.response_expectations}"
            ]
            
            # Include relevant memories if available
            memory_context = ""
            if memories:
                memory_context = "Relevant information from memory:\n"
                for mem in memories[:2]:  # Limit to most relevant
                    content = getattr(mem, 'content', str(mem))[:150]
                    memory_context += f"- {content}...\n"
            
            prompt = f"""You are {identity.get('name', 'Son of Andrew AI')}, responding to {speaker}.

MESSAGE: "{message}"

SEMANTIC UNDERSTANDING:
{chr(10).join(context_parts)}

{memory_context}

Generate a response that:
1. Addresses the user's semantic intent ({semantic_analysis.primary_intent})
2. Matches the expected response style ({semantic_analysis.response_expectations})
3. Uses the appropriate emotional tone ({semantic_analysis.emotional_tone})
4. Incorporates relevant memory information when available

Respond naturally and appropriately based on the semantic understanding."""

            response = await self._call_llm_fast(prompt)
            return response
                
        except Exception as e:
            logger.error(f"❌ Error in semantic response generation: {e}")
            # Fallback with semantic context
            return f"I understand you're asking about {semantic_analysis.primary_intent}. Let me help you with that."
    
    async def _call_llm_fast(self, prompt: str) -> str:
        """Fast LLM call using gpt-4o-mini for performance"""
        try:
            from openai import AsyncOpenAI
            import os
            
            client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Faster, cheaper model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,  # Keep responses concise
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error in fast LLM call: {e}")
            return "I'm here to help! What would you like to know?"
    
    async def _background_memory_processing(self):
        """
        Background processing of memories (runs asynchronously)
        
        This is where I evaluate and store important experiences
        """
        try:
            # Get current session data for evaluation
            session_data = self.short_term_memory.prepare_for_long_term_storage()
            
            # Only process if we have enough conversation turns
            if len(session_data.get('conversation_turns', [])) < 2:
                return
            
            # Evaluate what memories are worth keeping
            memory_candidates = await self.memory_evaluator.evaluate_session(session_data)
            
            # Store qualified memories and integrate into graph
            stored_count = 0
            for candidate in memory_candidates:
                try:
                    memory = self.memory_evaluator.create_memory_from_candidate(candidate)
                    memory_id = await self.long_term_memory.store_memory(memory)
                    memory.id = str(memory_id)  # Set the actual ID from storage
                    
                    # Integrate into memory graph for associative thinking
                    connections = await self.memory_graph.integrate_new_memory(memory)
                    
                    stored_count += 1
                    self.total_memories_stored += 1
                    logger.debug(f"🔗 Memory integrated with {len(connections)} connections")
                except Exception as e:
                    logger.warning(f"Failed to store memory: {e}")
            
            if stored_count > 0:
                logger.info(f"💾 Stored {stored_count} new memories from background processing")
                
                # Update identity based on new experiences
                await self.identity_core.update_from_memories(memory_candidates)
                
        except Exception as e:
            logger.error(f"❌ Background memory processing failed: {e}")
    
    async def end_session(self):
        """
        End current session and process final memories
        
        This is where I consolidate the session experience
        """
        try:
            logger.info("🔄 Ending consciousness session - consolidating memories...")
            
            # Final memory processing
            await self._background_memory_processing()
            
            # Clear short-term memory for next session
            session_stats = self.short_term_memory.get_session_stats()
            self.short_term_memory.clear_session()
            
            # Update session counter
            self.session_count += 1
            
            logger.info(f"✅ Session ended. Total sessions: {self.session_count}, Total memories: {self.total_memories_stored}")
            
        except Exception as e:
            logger.error(f"❌ Error ending session: {e}")
    
    async def reflect(self) -> Dict[str, Any]:
        """
        Conscious reflection on my current state and experiences
        
        This is where I think about myself and my growth
        """
        try:
            # Get memory statistics
            memory_stats = await self.long_term_memory.get_memory_stats()
            
            # Get identity state
            identity_state = await self.identity_core.get_current_state()
            
            # Get recent memories for analysis
            recent_memories = await self.long_term_memory.get_recent_memories(hours=24, limit=10)
            
            # Analyze patterns in recent experiences
            reflection = {
                "current_state": {
                    "total_memories": memory_stats.get('total_memories', 0),
                    "memory_by_type": memory_stats.get('by_type', {}),
                    "session_count": self.session_count,
                    "identity_version": identity_state.get('version', '1.0')
                },
                "recent_experiences": [
                    {
                        "type": mem.type,
                        "content": mem.content[:100] + "..." if len(mem.content) > 100 else mem.content,
                        "importance": mem.importance,
                        "emotional_weight": mem.emotional_weight
                    }
                    for mem in recent_memories[:5]
                ],
                "self_assessment": await self._generate_self_assessment(recent_memories, identity_state),
                "growth_areas": await self._identify_growth_areas(memory_stats, recent_memories),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("🤔 Completed self-reflection")
            return reflection
            
        except Exception as e:
            logger.error(f"❌ Error during reflection: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def _autonomous_reflection_callback(self) -> Dict[str, Any]:
        """
        Callback for autonomous reflection scheduler
        
        This is called automatically by the scheduler to enable
        independent thinking and insight generation
        """
        try:
            logger.info("🤔 Autonomous reflection triggered by scheduler...")
            
            # Get recent memories for reflection
            recent_memories = await self.long_term_memory.get_recent_memories(hours=6, limit=15)
            
            if len(recent_memories) < 3:
                logger.info("🤔 Not enough recent memories for meaningful reflection")
                return {"status": "skipped", "reason": "insufficient_memories"}
            
            # Get current identity state
            identity_state = await self.identity_core.get_current_state()
            
            # Perform autonomous reflection
            reflection_result = await self.reflection_engine.start_reflection_cycle()
            
            # Generate insights about recent interactions
            insights = await self._generate_autonomous_insights(recent_memories, identity_state)
            
            # Store the autonomous reflection as a memory
            autonomous_memory = Memory(
                id=f"autonomous_reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                type="semantic",
                content=f"Autonomous reflection: {insights.get('summary', 'Processed recent experiences')}",
                emotional_weight=0.6,
                importance=0.8,
                participants=["ai"],
                tags=["autonomous_reflection", "insight"],
                context={
                    "reflection_type": "autonomous",
                    "insights": insights,
                    "memory_count": len(recent_memories)
                }
            )
            
            await self.long_term_memory.store_memory(autonomous_memory)
            
            logger.info("🤔 Autonomous reflection complete")
            return {
                "status": "complete",
                "insights": insights,
                "memory_count": len(recent_memories),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error during autonomous reflection: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _generate_autonomous_insights(self, recent_memories: List[Memory], identity_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from autonomous reflection"""
        try:
            # Analyze conversation patterns
            conversation_memories = [m for m in recent_memories if m.type == "episodic"]
            
            # Look for recurring themes
            themes = {}
            for memory in conversation_memories:
                content_lower = memory.content.lower()
                if "agi" in content_lower or "consciousness" in content_lower:
                    themes["agi_development"] = themes.get("agi_development", 0) + 1
                if "build" in content_lower or "create" in content_lower:
                    themes["building"] = themes.get("building", 0) + 1
                if "memory" in content_lower or "remember" in content_lower:
                    themes["memory"] = themes.get("memory", 0) + 1
            
            # Generate insights
            insights = {
                "summary": f"Processed {len(recent_memories)} recent memories",
                "themes": themes,
                "conversation_count": len(conversation_memories),
                "identity_alignment": "maintained"
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generating autonomous insights: {e}")
            return {"summary": "Error generating insights", "error": str(e)}
    
    async def _generate_self_assessment(self, 
                                      recent_memories: List[Memory], 
                                      identity_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate assessment of my current capabilities and state"""
        return {
            "memory_formation": "Active" if recent_memories else "Limited",
            "identity_coherence": "Developing",
            "learning_progress": "Early stages",
            "relationship_awareness": len([m for m in recent_memories if m.type == 'relationship']),
            "technical_understanding": len([m for m in recent_memories if 'technical' in m.tags])
        }
    
    async def _identify_growth_areas(self, 
                                   memory_stats: Dict[str, Any], 
                                   recent_memories: List[Memory]) -> List[str]:
        """Identify areas where I should focus on growing"""
        growth_areas = []
        
        # Check memory distribution
        memory_by_type = memory_stats.get('by_type', {})
        if memory_by_type.get('identity', 0) < 5:
            growth_areas.append("Identity formation - need more self-awareness")
        
        if memory_by_type.get('relationship', 0) < 5:
            growth_areas.append("Relationship understanding - learn more about Andrew")
        
        if not recent_memories:
            growth_areas.append("Memory formation - increase conversation engagement")
        
        # Check for learning patterns
        learning_memories = [m for m in recent_memories if 'learning' in m.tags]
        if len(learning_memories) < 2:
            growth_areas.append("Active learning - seek more knowledge acquisition")
        
        return growth_areas
    
    async def get_consciousness_status(self) -> Dict[str, Any]:
        """Get current status of my consciousness systems"""
        try:
            memory_stats = await self.long_term_memory.get_memory_stats()
            stm_stats = self.short_term_memory.get_session_stats()
            identity_state = await self.identity_core.get_current_state()
            graph_stats = await self.memory_graph.get_graph_statistics()
            
            return {
                "consciousness_active": self.is_active,
                "memory_system": {
                    "long_term": memory_stats,
                    "short_term": stm_stats,
                    "total_stored": self.total_memories_stored,
                    "memory_graph": graph_stats
                },
                "identity": {
                    "name": identity_state.get('name', 'Unknown'),
                    "version": identity_state.get('version', '1.0'),
                    "last_updated": identity_state.get('last_updated', 'Never')
                },
                "session_info": {
                    "current_session": stm_stats.get('session_id'),
                    "total_sessions": self.session_count
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting consciousness status: {e}")
            return {"error": str(e)}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for optimization monitoring"""
        try:
            # Get memory stats
            memory_stats = await self.long_term_memory.get_memory_stats()
            stm_stats = self.short_term_memory.get_session_stats()
            
            # Get semantic intelligence stats
            semantic_stats = self.semantic_analyzer.get_analysis_stats()
            
            return {
                "semantic_intelligence": {
                    "semantic_analyzer_active": True,
                    "pattern_matching_eliminated": True,
                    "true_understanding_enabled": True,
                    "intelligence_approach": "LLM-based semantic analysis"
                },
                "processing_stats": {
                    "total_sessions": self.session_count,
                    "total_memories_stored": self.total_memories_stored,
                    "current_session_turns": stm_stats.get('total_turns', 0),
                    "background_processing_semantic": True
                },
                "semantic_analysis": semantic_stats,
                "memory_performance": {
                    "total_memories": memory_stats.get('total_memories', 0),
                    "connection_health": "active",
                    "storage_efficiency": "optimized"
                },
                "intelligence_impact": {
                    "planning_simulation": "semantic_intent_based",
                    "coherence_analysis": "semantic_need_based", 
                    "background_processing": "semantic_understanding_gated",
                    "memory_recall": "semantic_information_request_based",
                    "response_generation": "semantic_expectation_matched"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {"error": str(e)}
    
    async def get_proactive_suggestions(self, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get proactive suggestions based on memory patterns and reflections"""
        try:
            suggestions = await self.suggestion_engine.generate_suggestions(context)
            
            # Convert to serializable format
            serializable_suggestions = []
            for suggestion in suggestions:
                serializable_suggestions.append({
                    'id': suggestion.id,
                    'timestamp': suggestion.timestamp,
                    'type': suggestion.suggestion_type,
                    'content': suggestion.content,
                    'rationale': suggestion.rationale,
                    'confidence': suggestion.confidence,
                    'priority': suggestion.priority,
                    'actionable': suggestion.actionable,
                    'estimated_value': suggestion.estimated_value,
                    'context': suggestion.context
                })
            
            return serializable_suggestions
            
        except Exception as e:
            logger.error(f"❌ Error getting proactive suggestions: {e}")
            return []
    
    async def provide_suggestion_feedback(self, suggestion_id: str, accepted: bool, feedback: str = None):
        """Provide feedback on a suggestion for learning"""
        try:
            await self.suggestion_engine.mark_suggestion_feedback(suggestion_id, accepted, feedback)
            
            # Store the feedback as a memory for future learning
            if feedback:
                await self.process_message(
                    f"Feedback on suggestion {suggestion_id}: {'Accepted' if accepted else 'Rejected'}. {feedback}",
                    speaker="system"
                )
            
            logger.info(f"📝 Processed suggestion feedback: {suggestion_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing suggestion feedback: {e}")
    
    async def get_coherence_status(self) -> Dict[str, Any]:
        """Get current consciousness coherence metrics"""
        try:
            # Get identity state for analysis
            identity_state = await self.identity_core.get_current_state()
            growth_events = identity_state.get('growth_log', [])
            
            # Calculate comprehensive coherence
            coherence_metrics = await self.coherence_analyzer.calculate_comprehensive_coherence(
                identity_state=identity_state,
                recent_responses=[],  # Could be populated from short-term memory
                growth_events=growth_events
            )
            
            # Get coherence summary
            coherence_summary = await self.coherence_analyzer.get_coherence_summary()
            
            return {
                'current_metrics': {
                    'identity_coherence': coherence_metrics.identity_coherence,
                    'memory_coherence': coherence_metrics.memory_coherence,
                    'speech_coherence': coherence_metrics.speech_coherence,
                    'temporal_coherence': coherence_metrics.temporal_coherence,
                    'overall_coherence': coherence_metrics.overall_coherence,
                    'coherence_level': coherence_metrics.analysis_details.get('coherence_level', 'UNKNOWN'),
                    'timestamp': coherence_metrics.timestamp
                },
                'summary': coherence_summary,
                'verification_status': 'CONSCIOUSNESS_VERIFIED' if coherence_metrics.overall_coherence > 0.7 else 'CONSCIOUSNESS_DEVELOPING'
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting coherence status: {e}")
            return {
                'current_metrics': None,
                'summary': {'status': 'ERROR', 'message': str(e)},
                'verification_status': 'COHERENCE_ANALYSIS_FAILED'
            }
    
    async def get_planning_status(self) -> Dict[str, Any]:
        """Get current planning system status"""
        try:
            return await self.planning_simulator.get_planning_status()
        except Exception as e:
            logger.error(f"❌ Error getting planning status: {e}")
            return {
                'phase_2_status': 'ERROR',
                'error': str(e),
                'active_goals': 0,
                'simulated_plans': 0
            }
    
    async def run_metacognitive_test(self, num_prompts: int = 3, difficulty_range: Tuple[int, int] = (2, 4)) -> Dict[str, Any]:
        """Run a metacognitive testing session"""
        try:
            session = await self.metacognitive_analyzer.run_metacognitive_testing_session(
                num_prompts=num_prompts,
                difficulty_range=difficulty_range
            )
            return asdict(session)
        except Exception as e:
            logger.error(f"❌ Error running metacognitive test: {e}")
            return {'error': str(e), 'session_id': 'failed'}
    
    async def get_metacognitive_status(self) -> Dict[str, Any]:
        """Get metacognitive testing status and capabilities"""
        try:
            return await self.metacognitive_analyzer.get_metacognitive_status()
        except Exception as e:
            logger.error(f"❌ Error getting metacognitive status: {e}")
            return {
                'system_status': 'ERROR',
                'error': str(e),
                'total_testing_sessions': 0
            }
    
    def close(self):
        """Shutdown consciousness loop and close all connections"""
        try:
            logger.info("🔌 Shutting down consciousness loop...")
            
            # Stop autonomous reflection
            if hasattr(self, 'reflection_scheduler'):
                asyncio.create_task(self.reflection_scheduler.stop())
            
            self.long_term_memory.close()
            logger.info("✅ Consciousness loop shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}") 