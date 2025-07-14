"""
MetaAgent - The Core Intelligence Layer

This agent is responsible for understanding user requests and orchestrating
all other agents and tools to fulfill them. Instead of rigid routing logic,
it uses intelligent decision-making to determine what should happen next.
"""

import json
import logging
import asyncio
import re
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from openai import OpenAI
from config import settings

# Import langchain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Import SessionState components
from agents.session_state import (
    SessionState, 
    SessionStateManager, 
    PlanContext, 
    StepResult, 
    ConfidenceLevel,
    session_state_manager
)

# Import MemoryAgent for opportunistic learning
from agents.memory_agent import memory_agent

# Import Agent Registry for pre-compiled agents
from agents.agent_registry import agent_registry

# Import Model Selector for intelligent model selection (Phase 1g)
from agents.model_selector import model_selector

# Import WorkingTextManager
from agents.working_text_manager import WorkingTextManager

# Import EditHistoryManager
from agents.edit_history_manager import EditHistoryManager

# Import ConfirmationManager
from agents.confirmation_manager import ConfirmationManager

logger = logging.getLogger(__name__)

class MetaAgent:
    """
    The core intelligent agent that orchestrates all other agents and tools.
    
    This agent:
    1. Analyzes user input and context
    2. Determines what needs to be done
    3. Orchestrates the appropriate agents/tools
    4. Validates and executes the plan
    5. Returns the result
    6. 🆕 Performs opportunistic learning to build knowledge over time
    """
    
    def __init__(self):
        # Model selection configuration
        self.model_selector = model_selector
        
        # Initialize OpenAI client for routing decisions
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize agent registry
        self.agent_registry = agent_registry
        
        # Initialize session state manager
        self.session_state_manager = SessionStateManager()
        
        # Initialize working text manager
        self.working_text_manager = WorkingTextManager()
        
        # Initialize managers 
        self.edit_history_manager = EditHistoryManager()
        self.confirmation_manager = ConfirmationManager()
        
        # Retry configuration
        self.retry_config = {
            "max_attempts": 3,
            "initial_delay": 1.0,
            "backoff_factor": 2.0,
            "max_delay": 30.0
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_content_length": 10,
            "min_word_count": 3,
            "max_error_indicators": 2
        }
        
        # Learning configurations
        self.learning_triggers = {
            "response_quality": 0.7,
            "user_correction": True,
            "explicit_feedback": True,
            "style_preference": 0.8
        }
        
        # 🚀 AGENT STATE TEMPLATES: Pre-built state templates to reduce object construction overhead
        self.state_templates = {
            "precision_editor": {
                "user_input": None,
                "content": None,
                "session_id": None,
                "modification_request": None
            },
            "editor": {
                "user_input": None,
                "content": None,
                "session_id": None,
                "modification_request": None
            },
            "writer": {
                "prompt": None,
                "context": None,
                "original_request": None,
                "session_id": None,
                "memory_context": "",
                "conversation_context": "",
                "relevant_memories": {}
            },
            "learning": {
                "user_input": None,
                "session_id": None,
                "fact_to_store": None
            },
            "timekeeper": {
                "user_input": None,
                "session_id": None
            },
            "conversational": {
                "user_input": None,
                "session_id": None,
                "context": "",
                "memory_context": ""
            },
            "error": {
                "error": None,
                "user_input": None,
                "session_id": None,
                "agent_used": None
            },
            "success": {
                "content": None,
                "agent_used": None,
                "execution_successful": True
            },
            "failure": {
                "content": None,
                "agent_used": None,
                "execution_successful": False,
                "error": None
            }
        }
        
        # 🚀 TEMPLATE CACHE: Cache frequently used state combinations
        self.template_cache = {}
        self.template_cache_hits = 0
        self.template_cache_misses = 0
        
        # 🆕 Opportunistic Learning Configuration
        self.learning_patterns = {
            # Names and people
            'names': [
                r'\b[A-Z][a-z]+ (?:is|was)\b',  # "Sarah is my partner"
                r'\bmy (?:partner|colleague|manager|boss|client|designer|developer|friend) [A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+ (?:works|worked) (?:at|for|with)\b',
                r'\bI work with [A-Z][a-z]+\b'
            ],
            # Roles and relationships
            'roles': [
                r'\bmy (?:partner|colleague|manager|boss|client|designer|developer|assistant)\b',
                r'\bis my (?:partner|colleague|manager|boss|client|designer|developer|assistant)\b',
                r'\bworks as (?:a|an|the)\b',
                r'\bour (?:team|company|client|project)\b'
            ],
            # Preferences and habits
            'preferences': [
                r'\bI (?:prefer|like|love|hate|dislike|always|usually|never|typically)\b',
                r'\bI tend to\b',
                r'\bI\'m (?:more|better|worse) (?:at|with)\b',
                r'\bmy style is\b',
                r'\bI work best\b'
            ],
            # Projects and work
            'projects': [
                r'\bworking on [A-Z][a-zA-Z0-9\s]+\b',
                r'\bproject called [A-Z][a-zA-Z0-9\s]+\b',
                r'\bclient [A-Z][a-zA-Z0-9\s]+\b',
                r'\bfor [A-Z][a-zA-Z0-9\s]+ (?:project|client|company)\b'
            ],
            # Temporal patterns and schedules
            'schedules': [
                r'\bevery (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                r'\busually (?:at|around) \d+(?::\d+)?(?:am|pm)?\b',
                r'\bI have (?:meetings|calls|work) (?:at|on)\b',
                r'\bmy schedule is\b'
            ]
        }
        
        # Session-based learning accumulation
        self.session_learning_candidates: Dict[str, List[Dict[str, Any]]] = {}
        
        # 🔁 Intelligent Retry System Configuration (Phase 1e)
        self.retry_config = {
            "max_attempts": 3,                    # Maximum retry attempts per step
            "confidence_threshold": 0.4,          # 🚀 REDUCED: From 0.6 to 0.4 (fewer unnecessary retries)
            "backoff_intervals": [1, 2, 4],       # Exponential backoff in seconds
            "enable_retries": True,               # Global retry toggle
            "retry_strategies": [
                "different_agent",                # Try a different agent for the same task
                "rephrased_request",              # Rephrase the request for better understanding
                "additional_context"              # Add more context from conversation history
            ],
            # 🆕 Simple request detection settings
            "simple_request_threshold": 50,       # Requests under 50 chars = simple
            "skip_rephrasing_for_simple": True,   # Don't rephrase simple requests
            "confidence_boost_enabled": True,     # Enable local confidence boosting
            "obvious_success_patterns": [         # Patterns that indicate clear success
                r"✅",                            # Checkmark indicates success
                r"successfully",                   # "Successfully completed"
                r"here is",                       # "Here is your content"
                r"i've",                          # "I've created/updated"
                r"completed",                     # "Task completed"
                r"done",                          # "Done! Here's your"
            ]
        }
        
        # Registry of available agents and their capabilities
        self.available_agents = {
            "PrecisionEditorAgent": {
                "description": "Performs precise, instruction-aware editing with persistent working text",
                "capabilities": ["replace", "remove", "shorten", "edit_instruction", "working_text", "precise_edit"],
                "module": "agents.precision_editor_agent",
                "function": "precision_editor_agent"
            },
            "EditorAgent": {
                "description": "Rewrites, edits, polishes, and improves text content",
                "capabilities": ["rewrite", "edit", "polish", "improve", "revise", "update"],
                "module": "agents.editor_agent",
                "function": "editor_agent"
            },
            "WriterAgent": {
                "description": "Creates new content from scratch in Andrew's voice",
                "capabilities": ["write", "create", "draft", "compose", "email", "message", "content"],
                "module": "agents.writer_agent", 
                "function": "writer_agent"
            },
            "ConversationalAgent": {
                "description": "Handles general conversation, questions, and chat",
                "capabilities": ["chat", "question", "conversation", "general"],
                "module": "agents.conversational_agent", 
                "function": "conversational_agent"
            },
            "LearningAgent": {
                "description": "Stores and retrieves memories, facts, information, tasks, file references, and resources",
                "capabilities": ["remember", "store", "recall", "memory", "fact", "task", "todo", "file", "resource", "link", "bookmark"],
                "module": "agents.learning_agent",
                "function": "learning_agent"
            },
            "TimekeeperAgent": {
                "description": "Tracks time, logs hours, provides current time, manages schedules",
                "capabilities": ["time", "schedule", "when", "date", "clock", "hours", "logged", "log", "track", "tracking", "timekeeper", "timer"],
                "module": "agents.timekeeper_agent",
                "function": "timekeeper_agent"
            }
        }
        
        # Available tools
        self.available_tools = {
            "memory_search": "Search through stored memories and facts",
            "memory_store": "Store new facts or information",
            "current_time": "Get the current date and time",
            "context_resolve": "Resolve references like 'this', 'that', 'it'"
        }
    
    # 🚀 AGENT STATE TEMPLATE METHODS: Reduce object construction overhead
    
    def _create_agent_state(self, template_name: str, **kwargs) -> Dict[str, Any]:
        """
        Create agent state from template with provided values.
        🚀 PERFORMANCE: Uses pre-built templates to reduce object construction overhead.
        """
        # Check template cache first
        cache_key = f"{template_name}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key in self.template_cache:
            self.template_cache_hits += 1
            return self.template_cache[cache_key].copy()
        
        # Cache miss - create from template
        self.template_cache_misses += 1
        
        if template_name not in self.state_templates:
            logger.warning(f"Unknown template: {template_name}, using empty dict")
            state = {}
        else:
            # Create state from template
            state = self.state_templates[template_name].copy()
        
        # Fill in provided values
        for key, value in kwargs.items():
            state[key] = value
        
        # Cache the created state if cache is not too large
        if len(self.template_cache) < 100:  # Limit cache size
            self.template_cache[cache_key] = state.copy()
        
        return state
    
    def _create_precision_editor_state(self, user_input: str, content: str, 
                                     session_id: str, modification_request: str) -> Dict[str, Any]:
        """Create PrecisionEditorAgent state using template."""
        return self._create_agent_state(
            "precision_editor",
            user_input=user_input,
            content=content,
            session_id=session_id,
            modification_request=modification_request
        )
    
    def _create_editor_state(self, user_input: str, content: str, 
                           session_id: str, modification_request: str) -> Dict[str, Any]:
        """Create EditorAgent state using template."""
        return self._create_agent_state(
            "editor",
            user_input=user_input,
            content=content,
            session_id=session_id,
            modification_request=modification_request
        )
    
    def _create_writer_state(self, prompt: str, context: str, session_id: str, 
                           original_request: str = None, memory_context: str = "",
                           conversation_context: str = "", relevant_memories: Dict = None) -> Dict[str, Any]:
        """Create WriterAgent state using template."""
        return self._create_agent_state(
            "writer",
            prompt=prompt,
            context=context,
            original_request=original_request or prompt,
            session_id=session_id,
            memory_context=memory_context,
            conversation_context=conversation_context,
            relevant_memories=relevant_memories or {}
        )
    
    def _create_learning_state(self, user_input: str, session_id: str, 
                             fact_to_store: str = None) -> Dict[str, Any]:
        """Create LearningAgent state using template."""
        return self._create_agent_state(
            "learning",
            user_input=user_input,
            session_id=session_id,
            fact_to_store=fact_to_store
        )
    
    def _create_timekeeper_state(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """Create TimekeeperAgent state using template."""
        return self._create_agent_state(
            "timekeeper",
            user_input=user_input,
            session_id=session_id
        )
    
    def _create_conversational_state(self, user_input: str, session_id: str, 
                                   context: str = "", memory_context: str = "") -> Dict[str, Any]:
        """Create ConversationalAgent state using template."""
        return self._create_agent_state(
            "conversational",
            user_input=user_input,
            session_id=session_id,
            context=context,
            memory_context=memory_context
        )
    
    def _create_success_response(self, content: str, agent_used: str) -> Dict[str, Any]:
        """Create success response using template."""
        return self._create_agent_state(
            "success",
            content=content,
            agent_used=agent_used
        )
    
    def _create_failure_response(self, content: str, agent_used: str, error: str) -> Dict[str, Any]:
        """Create failure response using template."""
        return self._create_agent_state(
            "failure",
            content=content,
            agent_used=agent_used,
            error=error
        )
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get template performance statistics."""
        total_requests = self.template_cache_hits + self.template_cache_misses
        hit_rate = (self.template_cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "template_cache_hits": self.template_cache_hits,
            "template_cache_misses": self.template_cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self.template_cache),
            "available_templates": list(self.state_templates.keys())
        }
    
    async def process_request(self, 
                            user_input: str, 
                            session_id: str,
                            conversation_context: List[Dict[str, Any]] = []) -> Dict[str, Any]:
        """
        Main entry point - analyze request and orchestrate response
        🚀 PARALLEL OPTIMIZATION: Run understanding and memory retrieval in parallel
        """
        try:
            logger.info(f"🤖 METAAGENT: Processing request for session {session_id}")
            logger.info(f"📝 INPUT: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
            
            # 🚀 PARALLEL OPTIMIZATION: Run understanding and memory pre-fetching in parallel
            # This reduces latency by ~200-400ms per request
            understanding_task = self._understand_request(user_input, conversation_context)
            memory_prefetch_task = self._prefetch_memory_context(user_input, conversation_context)
            
            # Execute both tasks in parallel
            understanding, prefetched_memory = await asyncio.gather(
                understanding_task, 
                memory_prefetch_task
            )
            
            # Step 2: Create execution plan (now with understanding ready)
            plan = await self._create_execution_plan(understanding, user_input, conversation_context)
            
            # Step 3: Validate the plan
            validated_plan = self._validate_plan(plan)
            
            # Step 4: Execute the plan (with prefetched memory context)
            result = await self._execute_plan(
                validated_plan, 
                session_id, 
                user_input, 
                conversation_context,
                prefetched_memory=prefetched_memory
            )
            
            # 🆕 Step 5: Parallel Intent Processing (async, non-blocking)
            if result.get("execution_successful", False):
                # 🚨 PARALLEL PROCESSING: Handle multiple intents simultaneously
                # These tasks run in background and DO NOT affect the primary response
                result_state = result.get("state", {})
                
                try:
                    # Start opportunistic learning task (fire-and-forget)
                    learning_task = asyncio.create_task(
                        self._opportunistic_learn(
                            user_input=user_input,
                            response_content=result.get("content", ""),
                            session_id=session_id,
                            result_state=result_state
                        )
                    )
                    
                    # 🚨 CRITICAL FIX: Properly track background tasks to prevent cancellation
                    # Register the task with memory manager for lifecycle management
                    from memory_manager import memory_manager
                    memory_manager._track_background_task(learning_task, f"opportunistic_learning_{session_id}")
                    
                    # Don't await - let it run in background
                    logger.info(f"🔄 BACKGROUND LEARNING: Started opportunistic learning task for session {session_id}")
                    
                    # Start parallel intent detection and processing (fire-and-forget)
                    parallel_task = asyncio.create_task(
                        self._detect_and_process_parallel_intents(
                            user_input=user_input,
                            session_id=session_id,
                            conversation_context=conversation_context,
                            primary_result=result
                        )
                    )
                    
                    # 🚨 CRITICAL FIX: Track parallel intent processing task too
                    memory_manager._track_background_task(parallel_task, f"parallel_intents_{session_id}")
                    
                    # Add done callbacks to handle any errors in background tasks
                    learning_task.add_done_callback(
                        lambda task: logger.error(f"❌ Learning task failed: {task.exception()}") 
                        if task.exception() else None
                    )
                    parallel_task.add_done_callback(
                        lambda task: logger.error(f"❌ Parallel intent task failed: {task.exception()}") 
                        if task.exception() else None
                    )
                    
                    logger.info(f"🔍 BACKGROUND TASKS: Started learning and parallel intent processing")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to start background learning: {e}")
                    # Don't fail the main request if background learning fails
                    
                except Exception as e:
                    logger.error(f"❌ Failed to start background tasks: {e}")
                    # Don't let background task failures affect the primary response
            
            # 🚨 PRIMARY RESPONSE: Return immediately, don't wait for background tasks
            logger.info(f"✅ METAAGENT: Successfully completed request")
            return {
                "content": result.get("content", ""),
                "session_id": session_id,
                "agent_used": result.get("agent_used", "MetaAgent"),
                "execution_successful": True,
                "plan_steps": len(validated_plan.get("steps", [])),
                "understanding": understanding,
                "memory_context": prefetched_memory.get("memory_context", ""),  # 🚀 SHARED CONTEXT: Return prefetched memory
                "conversation_context": prefetched_memory.get("conversation_context", ""),
                "relevant_memories": prefetched_memory.get("relevant_memories", {})
            }
            
        except Exception as e:
            logger.error(f"❌ METAAGENT ERROR: {e}")
            return await self._handle_error(e, user_input, session_id)
    
    async def _understand_request(self, user_input: str, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to understand what the user actually wants, including multi-step requests
        """
        # 🚀 DETERMINISTIC CLASSIFICATION: Force simple requests to be single-step
        user_input_lower = user_input.lower()
        
        # Force rewrite/edit requests to be single-step (these should never be multi-step)
        if any(keyword in user_input_lower for keyword in [
            "rewrite", "edit", "improve", "make better", "make more", "make it",
            "revise", "rephrase", "reword", "fix this", "enhance", "refine"
        ]) and not any(multi_word in user_input_lower for multi_word in [
            "and then", "after that", "then make", "then save", "then send"
        ]):
            logger.info(f"🎯 DETERMINISTIC CLASSIFICATION: '{user_input}' → SINGLE-STEP (rewrite/edit detected)")
            return {
                "is_multi_step": False,
                "intent": "rewrite/edit content",
                "content_to_work_with": user_input,
                "references_resolved": user_input,
                "action_needed": "rewrite",
                "confidence": 0.95
            }
        
        # Build context summary
        context_summary = self._build_context_summary(conversation_context)
        
        prompt = f"""
You are a MetaAgent that needs to understand what a user wants to accomplish. Some requests may require multiple steps.

CONVERSATION CONTEXT:
{context_summary}

USER INPUT: "{user_input}"

Analyze this request and determine if it requires one or multiple steps. Respond with ONLY a JSON object:

For SINGLE-STEP requests:
{{
    "is_multi_step": false,
    "intent": "what the user wants to accomplish",
    "content_to_work_with": "any specific content they mentioned",
    "references_resolved": "resolve any 'this', 'that', 'it' references",
    "action_needed": "specific action required",
    "confidence": 0.95
}}

For MULTI-STEP requests:
{{
    "is_multi_step": true,
    "overall_intent": "what the user wants to accomplish overall",
    "content_to_work_with": "any specific content they mentioned",
    "references_resolved": "resolve any 'this', 'that', 'it' references",
    "steps": [
        {{"step_number": 1, "action": "first action", "description": "what to do first"}},
        {{"step_number": 2, "action": "second action", "description": "what to do second"}}
    ],
    "confidence": 0.95
}}

MULTI-STEP EXAMPLES:
- "Replace 'Hello' with 'Hi' and then make it more formal" → Step 1: replace text, Step 2: make formal
- "Rewrite this copy and then make it shorter" → Step 1: rewrite, Step 2: make shorter
- "Update that email to be more friendly and then save it" → Step 1: make friendly, Step 2: save/remember

SINGLE-STEP EXAMPLES:
- "Please rewrite this copy: Hello world" → single rewrite action
- "What time is it?" → single time query
- "Make this more friendly" → single editing action

Look for connecting words like "and then", "after that", "also", "and make it", "then" to identify multi-step requests.

Return ONLY the JSON, no other text.
"""
        
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on request complexity
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="MetaAgent",
                context="request understanding"
            )
            
            logger.info(f"🧠 UNDERSTANDING REQUEST: '{user_input[:50]}...' using model {selected_model}")
            
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"🧠 LLM UNDERSTANDING RESPONSE: {content}")
            
            understanding = json.loads(content)
            
            if understanding.get("is_multi_step"):
                logger.info(f"🧠 MULTI-STEP REQUEST: {understanding['overall_intent']} with {len(understanding.get('steps', []))} steps")
                for step in understanding.get("steps", []):
                    logger.info(f"  Step {step['step_number']}: {step['action']} - {step['description']}")
            else:
                logger.info(f"🧠 SINGLE-STEP REQUEST: {understanding['intent']} (confidence: {understanding.get('confidence', 'unknown')})")
                logger.info(f"🧠 ACTION NEEDED: {understanding.get('action_needed', 'unknown')}")
            
            return understanding
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Understanding JSON parse failed: {e}")
            logger.error(f"❌ Raw LLM response: {content}")
            # Create a proper fallback for rewrite requests
            if "rewrite" in user_input.lower():
                return {
                    "is_multi_step": False,
                    "intent": "rewrite content",
                    "content_to_work_with": user_input,
                    "references_resolved": user_input,
                    "action_needed": "rewrite",
                    "confidence": 0.8
                }
            else:
                return {
                    "is_multi_step": False,
                    "intent": "process request",
                    "content_to_work_with": user_input,
                    "references_resolved": user_input,
                    "action_needed": "respond",
                    "confidence": 0.3
                }
        except Exception as e:
            logger.error(f"❌ Understanding failed completely: {type(e).__name__}: {str(e)}")
            # Create a proper fallback for rewrite requests
            if "rewrite" in user_input.lower():
                return {
                    "is_multi_step": False,
                    "intent": "rewrite content",
                    "content_to_work_with": user_input,
                    "references_resolved": user_input,
                    "action_needed": "rewrite",
                    "confidence": 0.8
                }
            else:
                return {
                    "is_multi_step": False,
                    "intent": "process request",
                    "content_to_work_with": user_input,
                    "references_resolved": user_input,
                    "action_needed": "respond",
                    "confidence": 0.3
                }
    
    async def _create_execution_plan(self, understanding: Dict[str, Any], user_input: str, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a plan for how to fulfill the user's request (single or multi-step)
        """
        is_multi_step = understanding.get("is_multi_step", False)
        
        # Resolve any references using conversation context
        resolved_content = self._resolve_references(
            understanding.get("content_to_work_with", user_input) or user_input, 
            conversation_context
        )
        
        if is_multi_step:
            # Create multi-step plan
            overall_intent = understanding.get("overall_intent", "")
            steps_info = understanding.get("steps", [])
            
            plan_steps = []
            for i, step_info in enumerate(steps_info):
                step_action = step_info.get("action", "")
                step_description = step_info.get("description", "")
                
                # Determine agent for this step
                agent_for_step = self._determine_best_agent(step_description, step_action, f"{step_action} {step_description}")
                
                # For the first step, use the resolved content
                # For subsequent steps, they'll get the output from the previous step
                step_input = resolved_content if i == 0 else "{{output_from_previous_step}}"
                
                plan_steps.append({
                    "step": i + 1,
                    "action": step_action,
                    "description": step_description,
                    "agent": agent_for_step,
                    "input": step_input
                })
            
            plan = {
                "is_multi_step": True,
                "overall_intent": overall_intent,
                "content": resolved_content,
                "original_input": user_input,
                "steps": plan_steps
            }
            
            logger.info(f"📋 MULTI-STEP PLAN: {overall_intent}")
            for step in plan_steps:
                logger.info(f"  Step {step['step']}: {step['agent']} will '{step['action']}' - {step['description']}")
                
        else:
            # Create single-step plan (existing logic)
            intent = understanding.get("intent", "")
            action_needed = understanding.get("action_needed", "")
            
            # Determine which agent to use
            agent_to_use = self._determine_best_agent(intent, action_needed, user_input)
            
            plan = {
                "is_multi_step": False,
                "agent": agent_to_use,
                "action": action_needed,
                "content": resolved_content,
                "original_input": user_input,
                "steps": [
                    {
                        "step": 1,
                        "action": action_needed,
                        "agent": agent_to_use,
                        "input": resolved_content
                    }
                ]
            }
            
            logger.info(f"📋 SINGLE-STEP PLAN: Use {agent_to_use} to '{action_needed}' with content: '{resolved_content[:50]}{'...' if len(resolved_content) > 50 else ''}'")
        
        return plan
    
    def _determine_best_agent(self, intent: str, action_needed: str, user_input: str) -> str:
        """
        Intelligently determine which agent is best for this request using semantic understanding
        ACTION VERBS TAKE PRIORITY over content analysis to fix URL-containing rewrite requests
        """
        # 🚀 DETERMINISTIC ROUTING: Handle obvious cases first to avoid LLM inconsistency
        user_input_lower = user_input.lower()
        
        # Editor Agent - rewrite/edit actions (HIGHEST PRIORITY)
        if any(keyword in user_input_lower for keyword in [
            "rewrite", "edit", "improve", "make better", "make more", "make it", 
            "revise", "rephrase", "reword", "fix this", "enhance", "refine"
        ]):
            logger.info(f"🎯 DETERMINISTIC ROUTING: '{user_input}' → EditorAgent (rewrite/edit detected)")
            return "EditorAgent"
        
        # Precision Editor Agent - precise edits
        if any(keyword in user_input_lower for keyword in [
            "replace", "change", "substitute", "swap", "remove", "delete", "add", "insert"
        ]) and any(quote in user_input for quote in ["'", '"']):
            logger.info(f"🎯 DETERMINISTIC ROUTING: '{user_input}' → PrecisionEditorAgent (precise edit detected)")
            return "PrecisionEditorAgent"
        
        # Writer Agent - creation actions
        if any(keyword in user_input_lower for keyword in [
            "write", "create", "draft", "compose", "generate", "make an", "write an", "create an"
        ]):
            logger.info(f"🎯 DETERMINISTIC ROUTING: '{user_input}' → WriterAgent (creation detected)")
            return "WriterAgent"
        
        # Learning Agent - memory/task/file operations
        if any(keyword in user_input_lower for keyword in [
            "remember", "save", "store", "add to", "todo", "task", "remind me",
            "what files", "show files", "list files", "figma files", "design files",
            "what am i working", "current files", "files i have"
        ]):
            logger.info(f"🎯 DETERMINISTIC ROUTING: '{user_input}' → LearningAgent (memory/task/file detected)")
            return "LearningAgent"
        
        # Timekeeper Agent - time tracking
        if any(keyword in user_input_lower for keyword in [
            "time", "hours", "logged", "worked", "log time", "track time", "time spent"
        ]):
            logger.info(f"🎯 DETERMINISTIC ROUTING: '{user_input}' → TimekeeperAgent (time tracking detected)")
            return "TimekeeperAgent"
        
        # If no deterministic match, use LLM for complex cases
        logger.info(f"🧠 FALLBACK TO LLM ROUTING: '{user_input}' (no deterministic match)")
        
        # Use LLM to understand the semantic intent and route intelligently
        routing_prompt = f"""You are an intelligent routing system. Analyze this user request and determine the best agent to handle it based on SEMANTIC UNDERSTANDING.

CRITICAL RULE: **ACTION VERBS TAKE PRIORITY OVER CONTENT ANALYSIS**
- If the user asks to "rewrite", "edit", "improve", "make better" → EditorAgent (even if content contains URLs)
- If the user asks to "write", "create", "draft" → WriterAgent (even if content contains URLs)
- If the user asks to "save", "remember", "store" → LearningAgent
- If the user asks about "tasks", "todos", "what do I need to do" → LearningAgent
- If the user asks about "time", "hours", "logged" → TimekeeperAgent

**🚨 CRITICAL FIX: FILE/MEMORY QUERIES → LearningAgent**
- If the user asks about "files", "what files", "show files", "list files" → LearningAgent
- If the user asks about "Figma files", "design files", "documents" → LearningAgent  
- If the user asks about "what am I working on", "current files" → LearningAgent
- If the user asks about stored information, memories, or references → LearningAgent

USER REQUEST: "{user_input}"
EXTRACTED INTENT: "{intent}"
ACTION NEEDED: "{action_needed}"

AVAILABLE AGENTS:
1. **ConversationalAgent** - General conversation, current information, system status, meta-discussion
2. **WriterAgent** - Creating NEW content from scratch (emails, documents, posts, articles, copy)
3. **EditorAgent** - Improving/rewriting EXISTING content (make it better, more formal, friendlier, longer)
4. **PrecisionEditorAgent** - Precise edits to EXISTING content (replace specific words, remove text, exact changes)
5. **TimekeeperAgent** - Time tracking analysis (logged hours, work patterns, productivity analysis of STORED time data)
6. **LearningAgent** - Learning and remembering facts, preferences, managing stored memories, TASK MANAGEMENT (to-do lists, tasks, what needs to be done), FILE REFERENCES (tracking files you're working on), RESOURCE MANAGEMENT (saving links and bookmarks), **FILE QUERIES** (what files do I have, show my files, list my design files)

ROUTING LOGIC (ACTION VERB PRIORITY):
- "Please rewrite this: [content with URL]" → EditorAgent (ACTION: rewrite takes priority over URL)
- "Can you rewrite this message: [content with URL]" → EditorAgent (ACTION: rewrite takes priority over URL)
- "Make this more formal: [content with URL]" → EditorAgent (ACTION: improve takes priority over URL)
- "Improve this copy: [content with URL]" → EditorAgent (ACTION: improve takes priority over URL)
- "Write an email" → WriterAgent (ACTION: write/create)
- "Replace 'hello' with 'hi'" → PrecisionEditorAgent (ACTION: precise edit)
- "What are my tasks?" → LearningAgent (ACTION: query tasks)
- "Add to my todo" → LearningAgent (ACTION: save task)
- "Save this link for later" → LearningAgent (ACTION: save/store)
- "Remember that I like..." → LearningAgent (ACTION: store information)
- "I'm working on config.py" → LearningAgent (ACTION: implicit file sharing)
- "Here's the Figma file for homepage" → LearningAgent (ACTION: explicit file sharing)
- "What file am I working on?" → LearningAgent (ACTION: query file references)
- **"What Figma files do I have?" → LearningAgent (ACTION: query file references)**
- **"Show me my design files" → LearningAgent (ACTION: query file references)**  
- **"List my files" → LearningAgent (ACTION: query file references)**
- **"What files am I working with?" → LearningAgent (ACTION: query file references)**
- "How many hours have I logged?" → TimekeeperAgent (ACTION: query time data)
- "What time is it?" → ConversationalAgent (ACTION: current time query)

CRITICAL DISTINCTIONS:
- **ACTION VERBS (rewrite, edit, improve, make better)** → EditorAgent (regardless of content)
- **ACTION VERBS (write, create, draft)** → WriterAgent (regardless of content)
- **ACTION VERBS (save, remember, store)** → LearningAgent
- **TASK QUERIES (tasks, todo, what do I need to do)** → LearningAgent
- **FILE QUERIES (what files, show files, list files, design files, Figma files)** → LearningAgent
- **TIME QUERIES (hours, time logged, time spent)** → TimekeeperAgent
- **CURRENT INFO (what time, weather, general questions)** → ConversationalAgent

Return ONLY the agent name that best matches the PRIMARY ACTION VERB or QUERY TYPE, not the content type.
"""

        try:
            # Use fast model for routing decisions
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": routing_prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            selected_agent = response.choices[0].message.content.strip()
            
            # Validate the response is a valid agent
            valid_agents = [
                "ConversationalAgent", 
                "WriterAgent", 
                "EditorAgent", 
                "PrecisionEditorAgent", 
                "TimekeeperAgent", 
                "LearningAgent"
            ]
            
            if selected_agent in valid_agents:
                logger.info(f"🧠 INTELLIGENT ROUTING: '{user_input}' → {selected_agent}")
                return selected_agent
            else:
                logger.warning(f"⚠️ Invalid agent returned: {selected_agent}, falling back to ConversationalAgent")
                return "ConversationalAgent"
                
        except Exception as e:
            logger.error(f"❌ Intelligent routing failed: {e}, falling back to ConversationalAgent")
            return "ConversationalAgent"
    
    def _resolve_references(self, content: str, conversation_context: List[Dict[str, Any]]) -> str:
        """
        Resolve pronouns and references in the content, with special handling for editing requests
        """
        if not conversation_context:
            return content
        
        # Get the last few messages for context
        recent_messages = conversation_context[-5:]  # Look at more messages
        
        # Find the most recent content that might be referenced
        last_user_message = None
        last_assistant_message = None
        
        for msg in reversed(recent_messages):
            if msg.get("role") == "user" and not last_user_message:
                last_user_message = msg.get("content", "")
            elif msg.get("role") == "assistant" and not last_assistant_message:
                last_assistant_message = msg.get("content", "")
        
        # Check if this is an editing request with a reference
        user_input_lower = content.lower()
        is_editing_request = any(phrase in user_input_lower for phrase in [
            "update that", "change that", "edit that", "revise that", "modify that",
            "make that", "rewrite that", "improve that", "fix that"
        ])
        
        if is_editing_request and last_assistant_message:
            # For editing requests, return the content to be edited directly
            logger.info(f"🔄 REFERENCE RESOLVED: Editing request detected, using previous response as content")
            return last_assistant_message
        
        # For non-editing requests, do simple replacement
        resolved = content
        if "this" in resolved.lower() or "that" in resolved.lower() or "it" in resolved.lower():
            if last_assistant_message:
                # Usually "this/that/it" refers to the assistant's last response
                resolved = resolved.replace("this", f'"{last_assistant_message}"')
                resolved = resolved.replace("that", f'"{last_assistant_message}"')
                resolved = resolved.replace("it", f'"{last_assistant_message}"')
            elif last_user_message:
                # Fallback to user's last message
                resolved = resolved.replace("this", f'"{last_user_message}"')
                resolved = resolved.replace("that", f'"{last_user_message}"')
                resolved = resolved.replace("it", f'"{last_user_message}"')
        
        return resolved
    
    def _resolve_instruction_references(self, instruction: str, conversation_context: List[Dict[str, Any]]) -> str:
        """
        Resolve references in editing instructions like "update that to be shorter"
        """
        if not conversation_context:
            return instruction
        
        # Check if this is an editing instruction with a reference
        instruction_lower = instruction.lower()
        editing_patterns_with_references = [
            "update that", "change that", "make that", "edit that", 
            "rewrite that", "shorten that", "improve that", "fix that"
        ]
        
        has_editing_reference = any(pattern in instruction_lower for pattern in editing_patterns_with_references)
        
        if not has_editing_reference:
            return instruction
        
        # Get the most recent assistant message (the content to edit)
        last_assistant_message = None
        for msg in reversed(conversation_context):
            if msg.get("role") == "assistant":
                last_assistant_message = msg.get("content", "")
                break
        
        if not last_assistant_message:
            logger.warning("🔄 No previous assistant message found for reference resolution")
            return instruction
        
        # For editing instructions, we need to set the content as working text
        # and modify the instruction to be more direct
        if "update that to be shorter" in instruction_lower:
            # Set the content and change instruction to just "make it shorter"
            return "make it shorter"
        elif "update that to make it" in instruction_lower:
            # Extract the modification part
            match = re.search(r'update that to make it (.+)', instruction_lower)
            if match:
                modification = match.group(1)
                return f"make it {modification}"
        elif "make that" in instruction_lower:
            # Change "make that X" to "make it X"
            return re.sub(r'\bthat\b', 'it', instruction, flags=re.IGNORECASE)
        elif "update that" in instruction_lower:
            # Change "update that" to "make it" or "rewrite it"
            return re.sub(r'update that', 'make it', instruction, flags=re.IGNORECASE)
        elif "change that" in instruction_lower:
            # Change "change that" to "make it"
            return re.sub(r'change that', 'make it', instruction, flags=re.IGNORECASE)
        else:
            # General reference resolution - replace "that" with "it"
            return re.sub(r'\bthat\b', 'it', instruction, flags=re.IGNORECASE)
    
    def _validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the plan is executable and safe (single or multi-step)
        """
        is_multi_step = plan.get("is_multi_step", False)
        
        if is_multi_step:
            # Validate multi-step plan
            steps = plan.get("steps", [])
            if not steps:
                logger.warning("⚠️ Multi-step plan has no steps, converting to single-step")
                plan["is_multi_step"] = False
                plan["agent"] = "ConversationalAgent"
                plan["steps"] = [{
                    "step": 1,
                    "action": "process request",
                    "agent": "ConversationalAgent",
                    "input": plan.get("content", "")
                }]
            else:
                # Validate each step
                for step in steps:
                    agent_name = step.get("agent", "")
                    if agent_name not in self.available_agents:
                        logger.warning(f"⚠️ Unknown agent '{agent_name}' in step {step.get('step', 'unknown')}, using ConversationalAgent")
                        step["agent"] = "ConversationalAgent"
                
                logger.info(f"✅ MULTI-STEP PLAN VALIDATED: {len(steps)} steps")
                for step in steps:
                    logger.info(f"  Step {step.get('step', '?')}: {step.get('agent', 'Unknown')} - {step.get('action', 'Unknown')}")
        else:
            # Validate single-step plan
            agent_name = plan.get("agent", "")
            
            # Check if agent exists
            if agent_name not in self.available_agents:
                logger.warning(f"⚠️ Unknown agent '{agent_name}', falling back to ConversationalAgent")
                plan["agent"] = "ConversationalAgent"
                for step in plan.get("steps", []):
                    step["agent"] = "ConversationalAgent"
            
            # Ensure we have content to work with
            if not plan.get("content"):
                plan["content"] = plan.get("original_input", "")
            
            logger.info(f"✅ SINGLE-STEP PLAN VALIDATED: {plan['agent']} will handle request")
        
        return plan
    
    async def _execute_plan(self, plan: Dict[str, Any], session_id: str, original_input: str, 
                           conversation_context: List[Dict[str, Any]] = None,
                           prefetched_memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the validated plan (single or multi-step)
        """
        is_multi_step = plan.get("is_multi_step", False)
        
        if is_multi_step:
            # Execute multi-step plan with SessionState coordination
            logger.info(f"🎯 EXECUTING MULTI-STEP PLAN: {plan.get('overall_intent', '')}")
            
            # Create SessionState for coordination
            plan_context = PlanContext(
                original_request=original_input,
                overall_intent=plan.get('overall_intent', ''),
                target_tone=self._extract_target_tone(original_input),
                target_length=self._extract_target_length(original_input),
                content_type=self._extract_content_type(original_input)
            )
            
            session_state = session_state_manager.create_session_state(session_id, plan_context)
            
            try:
                current_content = plan.get("content", "")
                step_results = []
                
                for step in plan.get("steps", []):
                    step_number = step.get("step", 0)
                    agent_name = step.get("agent", "")
                    step_action = step.get("action", "")
                    step_description = step.get("description", "")
                    
                    # Replace placeholder with actual content from previous step
                    step_input = step.get("input", "")
                    if step_input == "{{output_from_previous_step}}":
                        step_input = current_content
                    
                    logger.info(f"🎯 EXECUTING STEP {step_number}: {agent_name} - {step_action}")
                    logger.info(f"   Input: {step_input[:100]}{'...' if len(step_input) > 100 else ''}")
                    
                    # Execute this step with SessionState context and retry logic
                    step_start_time = datetime.now()
                    step_result = await self._execute_step_with_state_and_retry(
                        agent_name, step_input, session_id, original_input, 
                        step_action, step_description, session_state, conversation_context, prefetched_memory
                    )
                    step_end_time = datetime.now()
                    
                    if not step_result.get("execution_successful", False):
                        logger.error(f"❌ STEP {step_number} FAILED: {step_result.get('error', 'Unknown error')}")
                        
                        # Set error context in SessionState
                        session_state.set_error_context({
                            "failed_step": step_number,
                            "failed_agent": agent_name,
                            "error": step_result.get('error', 'Unknown error'),
                            "step_action": step_action
                        })
                        
                        return {
                            "content": f"I encountered an issue at step {step_number}: {step_result.get('content', 'Unknown error')}",
                            "session_id": session_id,
                            "agent_used": "MetaAgent",
                            "execution_successful": False,
                            "failed_at_step": step_number,
                            "session_state_summary": session_state.get_execution_summary()
                        }
                    
                    # Update current content for next step
                    current_content = step_result.get("content", "")
                    
                    # Create StepResult and add to SessionState
                    execution_time_ms = int((step_end_time - step_start_time).total_seconds() * 1000)
                    confidence = self._determine_step_confidence(step_result, step_action)
                    
                    step_result_obj = StepResult(
                        step_number=step_number,
                        agent_name=agent_name,
                        action=step_action,
                        content=current_content,
                        confidence=confidence,
                        metadata={
                            "step_description": step_description,
                            "input_length": len(step_input),
                            "output_length": len(current_content)
                        },
                        execution_time_ms=execution_time_ms
                    )
                    
                    session_state.add_step_result(step_result_obj)
                    
                    step_results.append({
                        "step": step_number,
                        "agent": agent_name,
                        "action": step_action,
                        "result": current_content,
                        "confidence": confidence.value
                    })
                    
                    logger.info(f"✅ STEP {step_number} COMPLETED: {current_content[:100]}{'...' if len(current_content) > 100 else ''}")
                
                logger.info(f"✅ MULTI-STEP PLAN COMPLETED: All {len(step_results)} steps executed successfully")
                
                execution_summary = session_state.get_execution_summary()
                
                return {
                    "content": current_content,
                    "session_id": session_id,
                    "agent_used": "MetaAgent (Multi-Step)",
                    "execution_successful": True,
                    "steps_executed": len(step_results),
                    "step_results": step_results,
                    "overall_confidence": session_state.get_overall_confidence().value,
                    "session_state_summary": execution_summary,
                    "insights": session_state.get_step_insights()
                }
                
            finally:
                # Clean up SessionState
                session_state_manager.cleanup_session_state(session_id)
            
        else:
            # Execute single-step plan (existing logic)
            agent_name = plan.get("agent", "")
            content = plan.get("content", "")
            
            logger.info(f"🎯 EXECUTING SINGLE-STEP: {agent_name} with content: '{content[:50]}{'...' if len(content) > 50 else ''}'")
            
            step_result = await self._execute_step_with_retry(agent_name, content, session_id, original_input, "", "", conversation_context, prefetched_memory)
            
            return {
                "content": step_result.get("content", ""),
                "session_id": session_id,
                "agent_used": step_result.get("agent_used", agent_name),
                "execution_successful": step_result.get("execution_successful", False)
            }
    
    def _extract_content_to_edit(self, user_input: str) -> str:
        """Extract content to edit from patterns like 'Please rewrite this: Hello world'"""
        import re
        
        # Patterns for extracting content to edit
        patterns = [
            r'(?:rewrite|edit|improve|polish|revise|update|fix)\s+(?:this|that):\s*(.+)',
            r'(?:please\s+)?(?:rewrite|edit|improve|polish|revise|update|fix)\s+(?:this|that):\s*(.+)',
            r'(?:make\s+this|make\s+that)\s+(?:shorter|longer|more\s+friendly|better):\s*(.+)',
            r'(?:please\s+)?(?:rewrite|edit|improve)\s*:\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content:
                    return content
        
        return ""
    
    def _extract_modification_request(self, user_input: str) -> str:
        """Extract the type of modification requested"""
        user_lower = user_input.lower()
        
        if 'shorter' in user_lower:
            return 'make shorter'
        elif 'longer' in user_lower:
            return 'make longer'
        elif 'friendly' in user_lower or 'friendlier' in user_lower:
            return 'make more friendly'
        elif 'formal' in user_lower:
            return 'make more formal'
        elif 'casual' in user_lower:
            return 'make more casual'
        elif 'professional' in user_lower:
            return 'make more professional'
        elif 'improve' in user_lower:
            return 'general improvement'
        elif 'polish' in user_lower:
            return 'polish and refine'
        elif 'fix' in user_lower:
            return 'fix issues'
        else:
            return 'general improvement'

    def _build_context_summary(self, conversation_context: List[Dict[str, Any]]) -> str:
        """Build a summary of recent conversation context"""
        if not conversation_context:
            return "No previous conversation."
        
        recent = conversation_context[-3:]  # Last 3 messages
        summary_parts = []
        
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]  # Truncate long messages
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    async def _handle_error(self, error: Exception, user_input: str, session_id: str) -> Dict[str, Any]:
        """Handle errors gracefully"""
        logger.error(f"❌ MetaAgent error: {error}")
        
        return {
            "content": "I apologize, but I encountered an issue processing your request. Could you please try rephrasing it?",
            "session_id": session_id,
            "agent_used": "MetaAgent",
            "execution_successful": False,
            "error": str(error)
        }

    async def _execute_step_with_retry(self, agent_name: str, content: str, session_id: str, original_input: str,
                                     step_action: str = "", step_description: str = "",
                                     conversation_context: List[Dict[str, Any]] = None,
                                     prefetched_memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single step with retry logic for low-confidence results
        🚀 SHARED MEMORY CONTEXT: Uses prefetched memory to avoid duplicate API calls
        """
        attempt_number = 1
        last_result = None
        
        while attempt_number <= self.retry_config["max_attempts"]:
            try:
                # Execute the step
                result = await self._execute_step(
                    agent_name, content, session_id, original_input, 
                    step_action, step_description, conversation_context, prefetched_memory
                )
                
                # Check if retry is needed
                if not self._should_retry_step(result, attempt_number, original_input, agent_name):
                    # Success! Return the result
                    if attempt_number > 1:
                        logger.info(f"🔁 RETRY SUCCESSFUL on attempt {attempt_number}")
                    return result
                
                # Store the result for potential fallback
                last_result = result
                
                # If this was the last attempt, return the best result we have
                if attempt_number >= self.retry_config["max_attempts"]:
                    logger.warning(f"🔁 MAX RETRIES REACHED ({self.retry_config['max_attempts']}), using last result")
                    return result
                
                # Retry with strategy
                logger.info(f"🔁 RETRY NEEDED: Attempt {attempt_number} had low confidence")
                attempt_number += 1
                
                # Try with retry strategy
                retry_result = await self._retry_with_strategy(
                    agent_name, content, session_id, original_input,
                    step_action, step_description, attempt_number, conversation_context
                )
                
                # Update agent name if strategy changed it
                if retry_result.get("agent_used") != agent_name:
                    agent_name = retry_result.get("agent_used", agent_name)
                
                result = retry_result
                
            except Exception as e:
                logger.error(f"❌ Step execution failed on attempt {attempt_number}: {e}")
                
                # If this was the last attempt, return error
                if attempt_number >= self.retry_config["max_attempts"]:
                    return {
                        "content": f"I encountered an issue processing your request: {str(e)}",
                        "agent_used": agent_name,
                        "execution_successful": False,
                        "error": str(e)
                    }
                
                # Try next attempt
                attempt_number += 1
                
                # Add a small delay before retry
                await asyncio.sleep(self.retry_config["backoff_intervals"][min(attempt_number-2, len(self.retry_config["backoff_intervals"])-1)])
        
        # Should not reach here, but fallback to last result if available
        return last_result or {
            "content": "I encountered an issue processing your request.",
            "agent_used": agent_name,
            "execution_successful": False,
            "error": "Max retries exceeded"
        }

    async def _execute_step(self, agent_name: str, content: str, session_id: str, original_input: str, 
                           step_action: str = "", step_description: str = "", 
                           conversation_context: List[Dict[str, Any]] = None,
                           prefetched_memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single step with the specified agent
        🚀 SHARED MEMORY CONTEXT: Uses prefetched memory to avoid duplicate API calls
        """
        try:
            logger.info(f"🎯 EXECUTING: {agent_name} with content: '{content[:50]}{'...' if len(content) > 50 else ''}'")
            
            # Get pre-compiled agent
            agent = agent_registry.get_agent(agent_name)
            if not agent:
                logger.error(f"❌ Agent '{agent_name}' not found in registry")
                return {
                    "content": f"I'm sorry, but I couldn't find the '{agent_name}' agent.",
                    "agent_used": agent_name,
                    "execution_successful": False,
                    "error": f"Agent '{agent_name}' not found"
                }
            
            # Agent-specific execution with shared memory context
            if agent_name == "PrecisionEditorAgent":
                # Extract content and modification request
                content_to_edit = self._extract_content_to_edit(original_input) or content
                modification_request = self._extract_modification_request(original_input)
                
                # 🚀 TEMPLATE OPTIMIZATION: Use state template instead of manual construction
                state = self._create_precision_editor_state(
                    user_input=original_input,
                    content=content_to_edit,
                    session_id=session_id,
                    modification_request=modification_request
                )
                result = await agent.ainvoke(state)
                
            elif agent_name == "EditorAgent":
                # Extract content and modification request
                content_to_edit = self._extract_content_to_edit(original_input) or content
                modification_request = self._extract_modification_request(original_input)
                
                # 🚀 TEMPLATE OPTIMIZATION: Use state template instead of manual construction
                state = self._create_editor_state(
                    user_input=original_input,
                    content=content_to_edit,
                    session_id=session_id,
                    modification_request=modification_request
                )
                result = await agent.ainvoke(state)
                logger.info(f"📝 EditorAgent result type: {type(result)}")
                
            elif agent_name == "WriterAgent":
                # 🚀 SHARED MEMORY CONTEXT: Use prefetched memory instead of fetching again
                if prefetched_memory:
                    memory_context = prefetched_memory.get("memory_context", "")
                    conversation_context_formatted = prefetched_memory.get("conversation_context", "")
                    relevant_memories = prefetched_memory.get("relevant_memories", {})
                    
                    logger.info(f"🧠 USING PREFETCHED MEMORY: {len(memory_context)} chars, {len(relevant_memories)} categories")
                else:
                    # Fallback to individual memory search (should rarely happen)
                    logger.warning(f"⚠️ No prefetched memory available, falling back to individual search")
                    relevant_memories = await self._get_relevant_memories(original_input, agent_name, content)
                    memory_context = self._format_memory_context(relevant_memories, original_input)
                    conversation_context_formatted = self._format_conversation_context(conversation_context or [])
                
                # Enhance context with both memory injection and conversation history
                enhanced_context = content
                context_parts = []
                
                if conversation_context_formatted:
                    context_parts.append(conversation_context_formatted)
                if memory_context:
                    context_parts.append(memory_context)
                if content:
                    context_parts.append(content)
                    
                enhanced_context = "\n\n".join(context_parts) if context_parts else content
                
                # 🚀 TEMPLATE OPTIMIZATION: Use state template instead of manual construction
                state = self._create_writer_state(
                    prompt=original_input,
                    context=enhanced_context,
                    session_id=session_id,
                    original_request=original_input,
                    memory_context=memory_context,
                    conversation_context=conversation_context_formatted,
                    relevant_memories=relevant_memories
                )
                if hasattr(agent, 'ainvoke'):
                    result = await agent.ainvoke(state)
                else:
                    result = await agent.invoke(state)
                
            elif agent_name == "LearningAgent":
                # 🚀 TEMPLATE OPTIMIZATION: Use state template instead of manual construction
                state = self._create_learning_state(
                    user_input=original_input,
                    session_id=session_id,
                    fact_to_store=content
                )
                if hasattr(agent, 'ainvoke'):
                    result = await agent.ainvoke(state)
                else:
                    result = agent.invoke(state)
                    if hasattr(result, '__await__'):
                        result = await result
                
            elif agent_name == "TimekeeperAgent":
                # 🚀 TEMPLATE OPTIMIZATION: Use state template instead of manual construction
                state = self._create_timekeeper_state(
                    user_input=original_input,
                    session_id=session_id
                )
                if hasattr(agent, 'ainvoke'):
                    result = await agent.ainvoke(state)
                else:
                    result = agent.invoke(state)
                    if hasattr(result, '__await__'):
                        result = await result
                
            else:  # ConversationalAgent (default)
                # 🚀 SHARED MEMORY CONTEXT: Pass prefetched memory context to ConversationalAgent
                memory_context = ""
                if prefetched_memory:
                    memory_context = prefetched_memory.get("memory_context", "")
                
                # 🚀 TEMPLATE OPTIMIZATION: Use state template instead of manual construction
                state = self._create_conversational_state(
                    user_input=original_input,
                    session_id=session_id,
                    context="",
                    memory_context=memory_context
                )
                if hasattr(agent, 'ainvoke'):
                    result = await agent.ainvoke(state)
                else:
                    result = agent.invoke(state)
                    if hasattr(result, '__await__'):
                        result = await result

            # Extract content from result with robust Command object handling
            response_content = None
            logger.info(f"🔍 Extracting content from {type(result)}: {result}")
            
            # Priority 1: Check Command.content first (new preferred method)
            if hasattr(result, 'content') and result.content and str(result.content).strip():
                response_content = str(result.content)
                logger.info(f"✅ Extracted via Command.content: {response_content[:100]}...")
            
            # Priority 2: Use get_user_content() method
            elif hasattr(result, 'get_user_content'):
                try:
                    response_content = result.get_user_content()
                    if response_content and str(response_content).strip():
                        logger.info(f"✅ Extracted via get_user_content(): {response_content[:100]}...")
                    else:
                        response_content = None
                        logger.warning(f"⚠️ get_user_content() returned empty content")
                except Exception as e:
                    logger.warning(f"⚠️ get_user_content() failed: {e}")
            
            # Priority 3: Handle Command objects with state content  
            if not response_content and hasattr(result, 'name') and hasattr(result, 'state'):
                logger.info(f"🔍 Checking Command state: name={result.name}, state_keys={list(result.state.keys())}")
                
                if result.name == "ERROR":
                    error_msg = result.state.get('error', 'Unknown error')
                    response_content = f"I apologize, but I encountered an issue: {error_msg}"
                    logger.info(f"⚠️ Extracted error message: {response_content[:100]}...")
                elif 'content' in result.state and result.state['content'] and str(result.state['content']).strip():
                    response_content = str(result.state['content'])
                    logger.info(f"✅ Extracted from Command.state['content']: {response_content[:100]}...")
                elif 'learning_result' in result.state and result.state['learning_result'] and str(result.state['learning_result']).strip():
                    # Handle Learning Agent responses
                    learning_result = str(result.state['learning_result'])
                    if learning_result != "No learning opportunities detected":
                        response_content = learning_result
                        logger.info(f"✅ Extracted from Command.state['learning_result']: {response_content[:100]}...")
                    else:
                        # Learning Agent found no opportunities - this means we need to route to a different agent
                        logger.warning(f"⚠️ Learning Agent found no opportunities - request should be routed to a different agent")
                        response_content = None  # Force retry with different agent
                elif 'created_content' in result.state and result.state['created_content']:
                    response_content = str(result.state['created_content'])
                    logger.info(f"✅ Extracted from Command.state['created_content']: {response_content[:100]}...")
                elif 'edited_content' in result.state and result.state['edited_content']:
                    response_content = str(result.state['edited_content'])
                    logger.info(f"✅ Extracted from Command.state['edited_content']: {response_content[:100]}...")
            
            # Priority 4: Handle dictionary responses
            if not response_content and isinstance(result, dict):
                logger.info(f"🔍 Checking dict result: keys={list(result.keys())}")
                if 'content' in result and result['content'] and str(result['content']).strip():
                    response_content = str(result['content'])
                    logger.info(f"✅ Extracted from dict['content']: {response_content[:100]}...")
                elif 'edited_content' in result and result['edited_content']:
                    response_content = str(result['edited_content'])
                    logger.info(f"✅ Extracted from dict['edited_content']: {response_content[:100]}...")
                elif 'created_content' in result and result['created_content']:
                    response_content = str(result['created_content'])
                    logger.info(f"✅ Extracted from dict['created_content']: {response_content[:100]}...")
            
            # Priority 5: Handle string results
            if not response_content and isinstance(result, str) and result.strip():
                response_content = result
                logger.info(f"✅ Using string result: {response_content[:100]}...")
            
            # Ensure we never return None or empty
            if not response_content or not response_content.strip():
                logger.error(f"❌ No content extracted from agent result. Result type: {type(result)}, has_content: {hasattr(result, 'content')}, has_state: {hasattr(result, 'state')}")
                if hasattr(result, 'state'):
                    logger.error(f"❌ State contents: {result.state}")
                response_content = "I processed your request but couldn't generate a proper response."
            
            # 🚨 CRITICAL FIX: Propagate the state from the Command object
            result_state = {}
            if hasattr(result, 'state') and isinstance(result.state, dict):
                result_state = result.state

            return {
                "content": response_content,
                "agent_used": agent_name,
                "execution_successful": True,
                "state": result_state
            }
            
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")
            return {
                "content": f"I apologize, but I encountered an issue processing your request: {str(e)}",
                "agent_used": agent_name,
                "execution_successful": False,
                "error": str(e)
            }

    async def _execute_step_with_state_and_retry(self, agent_name: str, content: str, session_id: str, original_input: str,
                                               step_action: str = "", step_description: str = "",
                                               session_state: Optional[SessionState] = None,
                                               conversation_context: List[Dict[str, Any]] = None,
                                               prefetched_memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single step with SessionState context and retry logic for low-confidence results
        """
        attempt_number = 1
        last_result = None
        
        while attempt_number <= self.retry_config["max_attempts"]:
            try:
                # Execute the step with state
                result = await self._execute_step_with_state(
                    agent_name, content, session_id, original_input, 
                    step_action, step_description, session_state, conversation_context, prefetched_memory
                )
                
                # Check if retry is needed
                if not self._should_retry_step(result, attempt_number, original_input, agent_name):
                    # Success! Return the result
                    if attempt_number > 1:
                        logger.info(f"🔁 RETRY SUCCESSFUL on attempt {attempt_number}")
                    return result
                
                # Store the result for potential fallback
                last_result = result
                
                # If this was the last attempt, return the best result we have
                if attempt_number >= self.retry_config["max_attempts"]:
                    logger.warning(f"🔁 MAX RETRIES REACHED ({self.retry_config['max_attempts']}), using last result")
                    return result
                
                # Retry with strategy
                logger.info(f"🔁 RETRY NEEDED: Attempt {attempt_number} had low confidence")
                attempt_number += 1
                
                # Try with retry strategy (note: we lose SessionState context in retry strategies)
                # This is a limitation we could improve later
                retry_result = await self._retry_with_strategy(
                    agent_name, content, session_id, original_input,
                    step_action, step_description, attempt_number, conversation_context
                )
                
                # Update agent name if strategy changed it
                if retry_result.get("agent_used"):
                    agent_name = retry_result["agent_used"]
                
                # Update the result for next iteration
                result = retry_result
                
            except Exception as e:
                logger.error(f"🔁 RETRY ATTEMPT {attempt_number} FAILED: {e}")
                attempt_number += 1
                
                if attempt_number > self.retry_config["max_attempts"]:
                    # Return the last successful result or an error
                    if last_result and last_result.get("execution_successful"):
                        logger.warning(f"🔁 Using last successful result after {self.retry_config['max_attempts']} attempts")
                        return last_result
                    else:
                        return {
                            "content": f"I apologize, but I encountered repeated issues processing your request after {self.retry_config['max_attempts']} attempts: {str(e)}",
                            "agent_used": agent_name,
                            "execution_successful": False,
                            "error": str(e),
                            "retry_attempts": attempt_number - 1
                        }
        
        # Fallback (shouldn't reach here, but just in case)
        return last_result or {
            "content": "I apologize, but I couldn't process your request successfully.",
            "agent_used": agent_name,
            "execution_successful": False,
            "retry_attempts": attempt_number - 1
        }

    async def _execute_step_with_state(self, agent_name: str, content: str, session_id: str, original_input: str, 
                                      step_action: str = "", step_description: str = "", 
                                      session_state: Optional[SessionState] = None,
                                      conversation_context: List[Dict[str, Any]] = None,
                                      prefetched_memory: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single step with SessionState context passed to the agent
        """
        try:
            logger.info(f"🎯 EXECUTING STEP WITH STATE: {agent_name} with content: '{content[:50]}{'...' if len(content) > 50 else ''}'")
            
            # Prepare base state for the agent
            base_state = {
                "user_input": original_input,
                "session_id": session_id,
                "content": content,
                "step_action": step_action,
                "step_description": step_description
            }
            
            # Add SessionState context if available
            if session_state:
                base_state.update({
                    "session_state": session_state,
                    "plan_context": {
                        "original_intent": session_state.get_original_intent(),
                        "target_tone": session_state.get_target_tone(),
                        "target_length": session_state.get_target_length(),
                        "content_type": session_state.get_content_type(),
                        "previous_step_count": session_state.get_step_count(),
                        "overall_confidence": session_state.get_overall_confidence().value if session_state.get_step_count() > 0 else None
                    },
                    "previous_content": session_state.get_previous_content(),
                    "previous_step_result": session_state.get_previous_step_result().to_dict() if session_state.get_previous_step_result() else None
                })
            
            # Get agent from registry (eliminates dynamic import overhead)
            try:
                agent = agent_registry.get_agent(agent_name)
            except KeyError:
                # Fallback to ConversationalAgent if agent not found
                logger.warning(f"⚠️ Agent '{agent_name}' not found in registry, using ConversationalAgent")
                agent = agent_registry.get_agent("ConversationalAgent")
                agent_name = "ConversationalAgent"
            
            # Execute agent with the enhanced state
            if agent_name == "PrecisionEditorAgent":
                # For editing requests with references like "update that to be shorter",
                # we need to resolve both the content and update the instruction
                resolved_instruction = self._resolve_instruction_references(original_input, conversation_context)
                
                state = {
                    **base_state,
                    "user_instruction": resolved_instruction,
                    "resolved_content": content  # Pass resolved content if needed
                }
                logger.info(f"🎯 PrecisionEditorAgent state with SessionState")
                result = await agent.ainvoke(state)
                
            elif agent_name == "EditorAgent":
                # For multi-step, use the content passed from previous step
                # For single-step, extract content from the original input
                if step_action:
                    # Multi-step: use content directly and create modification request from step info
                    content_to_edit = content
                    modification_request = f"{step_action}: {step_description}"
                else:
                    # Single-step: extract content from original input
                    content_to_edit = self._extract_content_to_edit(original_input)
                    modification_request = self._extract_modification_request(original_input)
                
                state = {
                    **base_state,
                    "content": content_to_edit or content,
                    "modification_request": modification_request
                }
                logger.info(f"📝 EditorAgent state with SessionState")
                result = await agent.ainvoke(state)
                
            elif agent_name == "WriterAgent":
                # 🚀 SHARED MEMORY CONTEXT: Use prefetched memory instead of fetching again
                if prefetched_memory:
                    memory_context = prefetched_memory.get("memory_context", "")
                    conversation_context_formatted = prefetched_memory.get("conversation_context", "")
                    relevant_memories = prefetched_memory.get("relevant_memories", {})
                    
                    logger.info(f"🧠 USING PREFETCHED MEMORY: {len(memory_context)} chars, {len(relevant_memories)} categories")
                else:
                    # Fallback to individual memory search (should rarely happen)
                    logger.warning(f"⚠️ No prefetched memory available, falling back to individual search")
                    relevant_memories = await self._get_relevant_memories(original_input, agent_name, content)
                    memory_context = self._format_memory_context(relevant_memories, original_input)
                    conversation_context_formatted = self._format_conversation_context(conversation_context or [])
                
                # Enhance context with both memory injection and conversation history
                enhanced_context = content
                context_parts = []
                
                if conversation_context_formatted:
                    context_parts.append(conversation_context_formatted)
                if memory_context:
                    context_parts.append(memory_context)
                if content:
                    context_parts.append(content)
                    
                enhanced_context = "\n\n".join(context_parts) if context_parts else content
                
                state = {
                    **base_state,
                    "prompt": original_input,
                    "context": enhanced_context,
                    "original_request": original_input,
                    "memory_context": memory_context,
                    "conversation_context": conversation_context_formatted,
                    "relevant_memories": relevant_memories
                }
                if hasattr(agent, 'ainvoke'):
                    result = await agent.ainvoke(state)
                else:
                    result = await agent.invoke(state)
                
            elif agent_name == "LearningAgent":
                state = {
                    **base_state,
                    "fact_to_store": content
                }
                if hasattr(agent, 'ainvoke'):
                    result = await agent.ainvoke(state)
                else:
                    result = agent.invoke(state)
                    if hasattr(result, '__await__'):
                        result = await result
                
            elif agent_name == "TimekeeperAgent":
                state = base_state  # TimekeeperAgent doesn't need content parameter
                if hasattr(agent, 'ainvoke'):
                    result = await agent.ainvoke(state)
                else:
                    result = agent.invoke(state)
                    if hasattr(result, '__await__'):
                        result = await result
                
            else:  # ConversationalAgent (default)
                # 🚀 SHARED MEMORY CONTEXT: Pass prefetched memory context to ConversationalAgent
                memory_context = ""
                if prefetched_memory:
                    memory_context = prefetched_memory.get("memory_context", "")
                
                state = {
                    **base_state,
                    "context": "",
                    "memory_context": memory_context  # Pass prefetched memory context
                }
                if hasattr(agent, 'ainvoke'):
                    result = await agent.ainvoke(state)
                else:
                    result = agent.invoke(state)
                    if hasattr(result, '__await__'):
                        result = await result
            
            # Extract content from result with robust Command object handling
            response_content = None
            logger.info(f"🔍 Extracting content from {type(result)}: {result}")
            
            # Priority 1: Check Command.content first (new preferred method)
            if hasattr(result, 'content') and result.content and str(result.content).strip():
                response_content = str(result.content)
                logger.info(f"✅ Extracted via Command.content: {response_content[:100]}...")
            
            # Priority 2: Use get_user_content() method
            elif hasattr(result, 'get_user_content'):
                try:
                    response_content = result.get_user_content()
                    if response_content and str(response_content).strip():
                        logger.info(f"✅ Extracted via get_user_content(): {response_content[:100]}...")
                    else:
                        response_content = None
                        logger.warning(f"⚠️ get_user_content() returned empty content")
                except Exception as e:
                    logger.warning(f"⚠️ get_user_content() failed: {e}")
            
            # Priority 3: Handle Command objects with state content  
            if not response_content and hasattr(result, 'name') and hasattr(result, 'state'):
                logger.info(f"🔍 Checking Command state: name={result.name}, state_keys={list(result.state.keys())}")
                
                if result.name == "ERROR":
                    error_msg = result.state.get('error', 'Unknown error')
                    response_content = f"I apologize, but I encountered an issue: {error_msg}"
                    logger.info(f"⚠️ Extracted error message: {response_content[:100]}...")
                elif 'content' in result.state and result.state['content'] and str(result.state['content']).strip():
                    response_content = str(result.state['content'])
                    logger.info(f"✅ Extracted from Command.state['content']: {response_content[:100]}...")
                elif 'learning_result' in result.state and result.state['learning_result'] and str(result.state['learning_result']).strip():
                    # Handle Learning Agent responses
                    learning_result = str(result.state['learning_result'])
                    if learning_result != "No learning opportunities detected":
                        response_content = learning_result
                        logger.info(f"✅ Extracted from Command.state['learning_result']: {response_content[:100]}...")
                    else:
                        # Learning Agent found no opportunities - this means we need to route to a different agent
                        logger.warning(f"⚠️ Learning Agent found no opportunities - request should be routed to a different agent")
                        response_content = None  # Force retry with different agent
                elif 'created_content' in result.state and result.state['created_content']:
                    response_content = str(result.state['created_content'])
                    logger.info(f"✅ Extracted from Command.state['created_content']: {response_content[:100]}...")
                elif 'edited_content' in result.state and result.state['edited_content']:
                    response_content = str(result.state['edited_content'])
                    logger.info(f"✅ Extracted from Command.state['edited_content']: {response_content[:100]}...")
            
            # Priority 4: Handle dictionary responses
            if not response_content and isinstance(result, dict):
                logger.info(f"🔍 Checking dict result: keys={list(result.keys())}")
                if 'content' in result and result['content'] and str(result['content']).strip():
                    response_content = str(result['content'])
                    logger.info(f"✅ Extracted from dict['content']: {response_content[:100]}...")
                elif 'edited_content' in result and result['edited_content']:
                    response_content = str(result['edited_content'])
                    logger.info(f"✅ Extracted from dict['edited_content']: {response_content[:100]}...")
                elif 'created_content' in result and result['created_content']:
                    response_content = str(result['created_content'])
                    logger.info(f"✅ Extracted from dict['created_content']: {response_content[:100]}...")
            
            # Priority 5: Handle string results
            if not response_content and isinstance(result, str) and result.strip():
                response_content = result
                logger.info(f"✅ Using string result: {response_content[:100]}...")
            
            # Ensure we never return None or empty
            if not response_content or not response_content.strip():
                logger.error(f"❌ No content extracted from agent result. Result type: {type(result)}, has_content: {hasattr(result, 'content')}, has_state: {hasattr(result, 'state')}")
                if hasattr(result, 'state'):
                    logger.error(f"❌ State contents: {result.state}")
                response_content = "I processed your request but couldn't generate a proper response."
            
            # 🚨 CRITICAL FIX: Propagate the state from the Command object
            result_state = {}
            if hasattr(result, 'state') and isinstance(result.state, dict):
                result_state = result.state

            return {
                "content": response_content,
                "agent_used": agent_name,
                "execution_successful": True,
                "state": result_state
            }
            
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")
            return {
                "content": f"I apologize, but I encountered an issue processing your request: {str(e)}",
                "agent_used": agent_name,
                "execution_successful": False,
                "error": str(e)
            }
    
    def _determine_step_confidence(self, step_result: Dict[str, Any], step_action: str) -> ConfidenceLevel:
        """
        Determine confidence level for a step result based on various factors
        """
        try:
            # Check if the agent provided confidence information
            if 'confidence' in step_result:
                confidence_value = step_result['confidence']
                if isinstance(confidence_value, str):
                    # Parse string confidence levels
                    if confidence_value.lower() in ['high', 'good', 'excellent']:
                        return ConfidenceLevel.HIGH
                    elif confidence_value.lower() in ['medium', 'moderate', 'ok']:
                        return ConfidenceLevel.MEDIUM
                    elif confidence_value.lower() in ['low', 'poor', 'uncertain']:
                        return ConfidenceLevel.LOW
                elif isinstance(confidence_value, (int, float)):
                    # Parse numeric confidence (0-1 or 0-100 scale)
                    if confidence_value > 1:
                        confidence_value = confidence_value / 100  # Convert from 0-100 to 0-1
                    
                    if confidence_value >= 0.9:
                        return ConfidenceLevel.HIGH
                    elif confidence_value >= 0.7:
                        return ConfidenceLevel.MEDIUM
                    elif confidence_value >= 0.5:
                        return ConfidenceLevel.LOW
                    else:
                        return ConfidenceLevel.UNCERTAIN
            
            # Check execution success
            if not step_result.get("execution_successful", False):
                return ConfidenceLevel.UNCERTAIN
            
            # Check content quality indicators
            content = step_result.get("content", "")
            if not content or len(content.strip()) < 10:
                return ConfidenceLevel.LOW
            
            # Check for error indicators in content
            error_indicators = [
                "i apologize", "i'm sorry", "couldn't", "failed", "error", 
                "unable to", "not sure", "uncertain", "might be"
            ]
            content_lower = content.lower()
            if any(indicator in content_lower for indicator in error_indicators):
                return ConfidenceLevel.LOW
            
            # Default to medium confidence for successful steps
            return ConfidenceLevel.MEDIUM
            
        except Exception as e:
            logger.warning(f"Failed to determine step confidence: {e}")
            return ConfidenceLevel.UNCERTAIN
    
    def _extract_target_tone(self, user_input: str) -> Optional[str]:
        """Extract target tone from user input"""
        user_lower = user_input.lower()
        
        tone_indicators = {
            'formal': ['formal', 'professional', 'business'],
            'casual': ['casual', 'informal', 'friendly', 'relaxed'],
            'professional': ['professional', 'business', 'corporate'],
            'friendly': ['friendly', 'warm', 'approachable'],
            'concise': ['concise', 'brief', 'short', 'succinct']
        }
        
        for tone, indicators in tone_indicators.items():
            if any(indicator in user_lower for indicator in indicators):
                return tone
        
        return None
    
    def _extract_target_length(self, user_input: str) -> Optional[str]:
        """Extract target length from user input"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['shorter', 'shorten', 'brief', 'concise']):
            return 'shorter'
        elif any(word in user_lower for word in ['longer', 'expand', 'elaborate']):
            return 'longer'
        elif any(word in user_lower for word in ['concise', 'tight', 'succinct']):
            return 'concise'
        
        return None
    
    def _extract_content_type(self, user_input: str) -> Optional[str]:
        """Extract content type from user input"""
        user_lower = user_input.lower()
        
        content_types = {
            'email': ['email', 'message', 'mail'],
            'copy': ['copy', 'text', 'content'],
            'document': ['document', 'doc', 'report'],
            'message': ['message', 'note', 'communication']
        }
        
        for content_type, indicators in content_types.items():
            if any(indicator in user_lower for indicator in indicators):
                return content_type
        
        return None

    # 🆕 Memory Injection Methods
    
    async def _get_relevant_memories(self, user_input: str, agent_name: str, content_context: str = "") -> Dict[str, Any]:
        """
        Intelligently retrieve relevant memories using single LLM-powered search
        
        Args:
            user_input: The user's original request
            agent_name: Name of the agent that will use these memories
            content_context: Additional context (e.g., content from previous step)
            
        Returns:
            Dictionary containing relevant memories organized by category
        """
        try:
            logger.info(f"🧠 MEMORY INJECTION: Retrieving relevant memories for {agent_name}")
            
            # Create search context for memory retrieval
            search_context = f"{user_input} {content_context}".strip()
            
            # 🚀 INTELLIGENT SINGLE SEARCH: Use LLM to understand what memories are relevant
            # This replaces 5 separate category searches with one intelligent search
            search_results = await self._intelligent_memory_search(search_context, agent_name)
            
            # Organize results by category using LLM-based classification
            relevant_memories = await self._categorize_memories(search_results, search_context)
            
            if relevant_memories:
                logger.info(f"🧠 FOUND MEMORIES: {list(relevant_memories.keys())}")
                for category, facts in relevant_memories.items():
                    logger.info(f"  - {category}: {len(facts)} facts")
                    for fact in facts[:2]:  # Log first 2 facts
                        logger.info(f"    • {fact[:60]}...")
            else:
                logger.info(f"🧠 No relevant memories found for context")
            
            return relevant_memories
            
        except Exception as e:
            logger.error(f"❌ Memory injection failed: {e}")
            return {}
    
    async def _intelligent_memory_search(self, search_context: str, agent_name: str) -> List[str]:
        """
        Single intelligent memory search using LLM to understand relevance
        """
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Use fast model for memory search
            selected_model = model_selector.select_model(
                user_input=search_context,
                agent_name="MetaAgent_MemorySearch",
                context="Memory search operation"
            )
            
            # Create LLM for intelligent search query generation
            llm = ChatOpenAI(
                model=selected_model,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1,
                streaming=False
            )
            
            # Generate intelligent search query
            search_prompt = f"""You are an intelligent memory search system. Based on the user's request and the agent that will use the memories, generate the most effective search query to find relevant memories.

USER REQUEST: "{search_context}"
TARGET AGENT: {agent_name}

AGENT CONTEXT:
- WriterAgent: Needs style preferences, tone, communication patterns, writing examples
- EditorAgent: Needs editing preferences, quality standards, revision patterns
- TimekeeperAgent: Needs work patterns, project information, time tracking habits
- ConversationalAgent: Needs personality traits, communication style, relationships
- LearningAgent: Needs learning preferences, feedback patterns, knowledge gaps
- PrecisionEditorAgent: Needs specific editing rules, precision requirements

Generate a focused search query that will find the most relevant memories for this request and agent. Include key people, preferences, projects, or facts that would be useful.

SEARCH QUERY:"""

            response = await llm.ainvoke([HumanMessage(content=search_prompt)])
            intelligent_query = response.content.strip()
            
            logger.info(f"🧠 INTELLIGENT QUERY: {intelligent_query}")
            
            # Use the intelligent query to search memories
            search_results = await memory_agent.search_facts(intelligent_query, limit=15)
            
            if search_results:
                # Extract and clean the actual facts from search results
                facts = []
                for result in search_results[:10]:  # Limit to top 10 results
                    fact_content = ""
                    
                    if isinstance(result, dict):
                        # Try multiple possible content keys
                        for key in ['content', 'fact', 'text', 'message']:
                            if key in result and result[key]:
                                fact_content = str(result[key]).strip()
                                break
                    elif isinstance(result, str):
                        fact_content = result.strip()
                    
                    # Clean up and filter facts
                    if fact_content and len(fact_content) > 10:
                        # Remove FACT: prefix if present
                        if fact_content.startswith("FACT: "):
                            fact_content = fact_content[6:]
                        
                        facts.append(fact_content)
                
                return facts
            
            return []
            
        except Exception as e:
            logger.warning(f"⚠️ Intelligent memory search failed: {e}")
            return []
    
    async def _categorize_memories(self, facts: List[str], search_context: str) -> Dict[str, List[str]]:
        """
        Intelligently categorize memories using LLM instead of keyword matching
        """
        if not facts:
            return {}
            
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Use fast model for categorization
            selected_model = model_selector.select_model(
                user_input=search_context,
                agent_name="MetaAgent_MemoryCateg",
                context="Memory categorization operation"
            )
            
            # Create LLM for intelligent categorization
            llm = ChatOpenAI(
                model=selected_model,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1,
                streaming=False
            )
            
            # Create categorization prompt
            facts_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts)])
            
            categorization_prompt = f"""You are an intelligent memory categorization system. Categorize these facts into relevant categories based on the user's request context.

USER REQUEST CONTEXT: "{search_context}"

FACTS TO CATEGORIZE:
{facts_text}

AVAILABLE CATEGORIES:
- people: Information about specific people, relationships, contacts
- preferences: User preferences, styles, communication patterns, habits
- projects: Work projects, clients, business information
- communication: Communication patterns, meeting styles, contact methods
- general: General facts, notes, or information that doesn't fit other categories

For each fact, determine the most appropriate category. Some facts may fit multiple categories - choose the primary one.

Respond in this exact format:
people: [fact numbers that belong to people category]
preferences: [fact numbers that belong to preferences category]  
projects: [fact numbers that belong to projects category]
communication: [fact numbers that belong to communication category]
general: [fact numbers that belong to general category]

Example: preferences: 1, 3, 5
         people: 2, 4
         projects: 6

Response:"""

            response = await llm.ainvoke([HumanMessage(content=categorization_prompt)])
            categorization_result = response.content.strip()
            
            # Parse the categorization result
            categories = {}
            for line in categorization_result.split('\n'):
                if ':' in line:
                    category, numbers_str = line.split(':', 1)
                    category = category.strip()
                    
                    if category in ['people', 'preferences', 'projects', 'communication', 'general']:
                        # Extract fact numbers
                        numbers = []
                        for num_str in numbers_str.split(','):
                            try:
                                num = int(num_str.strip())
                                if 1 <= num <= len(facts):
                                    numbers.append(num - 1)  # Convert to 0-based index
                            except ValueError:
                                continue
                        
                        if numbers:
                            categories[category] = [facts[i] for i in numbers]
            
            return categories
            
        except Exception as e:
            logger.warning(f"⚠️ Memory categorization failed: {e}")
            # Fallback: put all facts in general category
            return {"general": facts}
    
    def _format_memory_context(self, relevant_memories: Dict[str, List[str]], user_input: str) -> str:
        """
        Format relevant memories into a context string for the agent
        """
        if not relevant_memories:
            return ""
        
        context_parts = []
        
        # Add introductory line
        context_parts.append("## Context from Previous Conversations:")
        
        # Add each category of memories
        for category, facts in relevant_memories.items():
            if facts:
                category_title = category.replace('_', ' ').title()
                context_parts.append(f"\n**{category_title}:**")
                for fact in facts:
                    context_parts.append(f"- {fact}")
        
        context_parts.append("")  # Add blank line before main content
        
        context_string = "\n".join(context_parts)
        logger.info(f"🧠 FORMATTED MEMORY CONTEXT: {len(context_string)} characters")
        
        return context_string
    
    def _format_conversation_context(self, conversation_context: List[Dict[str, Any]]) -> str:
        """
        Format recent conversation history for WriterAgent context
        """
        if not conversation_context:
            return ""
        
        # Get the last 5 messages for context (balance between context and brevity)
        recent_messages = conversation_context[-5:]
        
        context_parts = []
        context_parts.append("## Recent Conversation:")
        
        for i, msg in enumerate(recent_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Truncate very long messages but keep reasonable context
            if len(content) > 300:
                content = content[:300] + "..."
            
            # Format role for readability
            if role == "user":
                role_display = "You"
            elif role == "assistant":
                role_display = "Andrew"
            else:
                role_display = role.title()
            
            context_parts.append(f"\n**{role_display}:** {content}")
        
        context_parts.append("")  # Add blank line before main content
        
        context_string = "\n".join(context_parts)
        logger.info(f"💬 FORMATTED CONVERSATION CONTEXT: {len(context_string)} characters from {len(recent_messages)} messages")
        
        return context_string

    # 🔁 Retry System Methods
    
    def _should_retry_step(self, step_result: Dict[str, Any], attempt_number: int, 
                          original_input: str = "", agent_name: str = "") -> bool:
        """
        🚀 INTELLIGENT RETRY DECISION: Determine if a step should be retried
        Includes confidence boosting and smart success detection (Phase 1e)
        """
        if not self.retry_config["enable_retries"]:
            return False
            
        if attempt_number >= self.retry_config["max_attempts"]:
            return False
            
        if not step_result.get("execution_successful", False):
            return True  # Always retry failed executions
        
        content = step_result.get("content", "")
        
        # 🆕 SMART SUCCESS DETECTION: Don't retry obvious successes
        if self._is_obvious_success(content):
            logger.info(f"🚀 OBVIOUS SUCCESS detected, skipping retry")
            return False
            
        # 🆕 LOCAL CONFIDENCE BOOSTING: Try to boost confidence before expensive retry
        original_confidence = self._extract_confidence_score(step_result)
        boosted_confidence = self._boost_confidence_locally(step_result, content)
        
        if boosted_confidence != original_confidence:
            logger.info(f"🚀 CONFIDENCE BOOSTED: {original_confidence:.2f} → {boosted_confidence:.2f}")
        
        # Use boosted confidence for threshold comparison
        # 🚀 ADAPTIVE CONFIDENCE SCORING: Dynamic threshold based on context
        adaptive_threshold = self._calculate_adaptive_confidence_threshold(
            step_result, original_input, agent_name, attempt_number
        )
        
        if boosted_confidence < adaptive_threshold:
            logger.info(f"🔁 LOW CONFIDENCE ({boosted_confidence:.2f} < {adaptive_threshold:.2f}), will retry")
            return True
            
        # Check for quality indicators in content (unchanged)
        if self._has_quality_issues(content):
            logger.info(f"🔁 QUALITY ISSUES detected in content, will retry")
            return True
            
        logger.info(f"🚀 CONFIDENCE SUFFICIENT ({boosted_confidence:.2f} >= {adaptive_threshold:.2f}), no retry needed")
        return False
    
    def _calculate_adaptive_confidence_threshold(self, step_result: Dict[str, Any], 
                                               original_input: str, agent_name: str, 
                                               attempt_number: int) -> float:
        """
        🚀 ADAPTIVE CONFIDENCE SCORING: Calculate dynamic confidence threshold based on context
        """
        # Start with base threshold
        base_threshold = 0.4
        
        # 🚀 AGENT-SPECIFIC THRESHOLDS: Different agents have different quality requirements
        agent_adjustments = {
            "WriterAgent": 0.1,           # Higher threshold for content creation
            "EditorAgent": 0.05,          # Slightly higher for editing
            "PrecisionEditorAgent": 0.0,  # Standard threshold for precision edits
            "ConversationalAgent": -0.1,  # Lower threshold for conversation
            "TimekeeperAgent": -0.1,      # Lower threshold for time queries
            "LearningAgent": -0.05        # Slightly lower for learning
        }
        
        agent_adjustment = agent_adjustments.get(agent_name, 0.0)
        
        # 🚀 COMPLEXITY-BASED ADJUSTMENTS: More complex requests need higher confidence
        complexity_adjustment = 0.0
        if original_input:
            request_length = len(original_input)
            if request_length > 200:
                complexity_adjustment += 0.1  # Complex requests need higher confidence
            elif request_length < 50:
                complexity_adjustment -= 0.05  # Simple requests can have lower confidence
        
        # 🚀 ATTEMPT-BASED ADJUSTMENTS: Lower threshold on later attempts
        attempt_adjustment = 0.0
        if attempt_number > 1:
            attempt_adjustment = -0.05 * (attempt_number - 1)  # Progressively lower threshold
        
        # 🚀 CONTEXT-BASED ADJUSTMENTS: Adjust based on task type
        context_adjustment = 0.0
        if original_input:
            input_lower = original_input.lower()
            
            # Creative tasks need higher confidence
            if any(word in input_lower for word in ['write', 'create', 'compose', 'draft']):
                context_adjustment += 0.05
            
            # Analytical tasks need higher confidence
            elif any(word in input_lower for word in ['analyze', 'report', 'summary', 'insights']):
                context_adjustment += 0.05
            
            # Simple queries can have lower confidence
            elif any(word in input_lower for word in ['what', 'how', 'when', 'where', 'who']):
                context_adjustment -= 0.05
        
        # 🚀 EXECUTION SUCCESS BONUS: Lower threshold if execution was successful
        execution_adjustment = 0.0
        if step_result.get("execution_successful", False):
            execution_adjustment = -0.1  # Lower threshold for successful executions
        
        # Calculate final adaptive threshold
        adaptive_threshold = base_threshold + agent_adjustment + complexity_adjustment + attempt_adjustment + context_adjustment + execution_adjustment
        
        # Ensure threshold stays within reasonable bounds
        adaptive_threshold = max(0.1, min(0.8, adaptive_threshold))
        
        # Log the adaptive threshold calculation for debugging
        logger.info(f"🧠 ADAPTIVE THRESHOLD: base={base_threshold:.2f} + agent={agent_adjustment:.2f} + complexity={complexity_adjustment:.2f} + attempt={attempt_adjustment:.2f} + context={context_adjustment:.2f} + execution={execution_adjustment:.2f} = {adaptive_threshold:.2f}")
        
        return adaptive_threshold
    
    def _is_obvious_success(self, content: str) -> bool:
        """
        🚀 SMART SUCCESS DETECTION: Check if content shows obvious success patterns
        """
        if not content:
            return False
            
        content_lower = content.lower()
        
        # Check for obvious success patterns
        for pattern in self.retry_config["obvious_success_patterns"]:
            if isinstance(pattern, str):
                if pattern.lower() in content_lower:
                    return True
            else:
                import re
                if re.search(pattern, content_lower):
                    return True
        
        # Additional success indicators
        success_indicators = [
            "✅", "✓", "done", "complete", "finished", "success", "here is", "here's", 
            "i've", "i have", "created", "updated", "wrote", "written", "generated",
            "result:", "output:", "final version:", "here you go"
        ]
        
        for indicator in success_indicators:
            if indicator in content_lower:
                return True
                
        return False
    
    def _boost_confidence_locally(self, step_result: Dict[str, Any], content: str) -> float:
        """
        🚀 LOCAL CONFIDENCE BOOSTING: Boost confidence without expensive LLM calls
        """
        original_confidence = self._extract_confidence_score(step_result)
        boosted_confidence = original_confidence
        
        if not content:
            return boosted_confidence
            
        # Boost confidence based on content quality indicators
        content_lower = content.lower()
        
        # Length-based confidence boost
        if len(content) > 100:
            boosted_confidence += 0.1  # Longer content often indicates better response
        
        # Structure-based confidence boost
        if any(marker in content for marker in ["**", "*", "•", "1.", "2.", "3.", "-"]):
            boosted_confidence += 0.05  # Structured content is usually better
        
        # Domain-specific confidence boost
        if any(term in content_lower for term in ["here is", "here's", "i've", "created", "updated"]):
            boosted_confidence += 0.15  # Clear completion phrases
        
        # Format-based confidence boost
        if any(marker in content for marker in ["✅", "✓", "🎯", "📝"]):
            boosted_confidence += 0.1  # Emoji markers indicate structured output
        
        # Successful execution indicators
        if step_result.get("execution_successful", False):
            boosted_confidence += 0.05
        
        # Cap at 1.0
        boosted_confidence = min(boosted_confidence, 1.0)
        
        return boosted_confidence
    
    def _is_simple_request(self, original_input: str) -> bool:
        """
        🚀 SIMPLE REQUEST DETECTION: Determine if request is simple enough to skip expensive rephrasing
        """
        if not original_input:
            return True
            
        # Check length threshold
        if len(original_input) < self.retry_config["simple_request_threshold"]:
            return True
            
        # Check for simple patterns
        simple_patterns = [
            r"^(what|how|when|where|who|why)\s+",     # Question words
            r"^(make|create|write|edit|update|fix)",  # Simple action words  
            r"^(please|can you|could you)",           # Polite requests
            r"^\w+\s+time",                           # Time-related queries
            r"^(hello|hi|thanks|ok|yes|no)",          # Simple greetings/responses
        ]
        
        import re
        for pattern in simple_patterns:
            if re.search(pattern, original_input.lower()):
                return True
                
        # Check for single words or very short phrases
        words = original_input.split()
        if len(words) <= 3:
            return True
            
        return False
    
    def _extract_confidence_score(self, step_result: Dict[str, Any]) -> float:
        """
        Extract confidence score from step result
        """
        # Check if agent provided explicit confidence
        if "confidence" in step_result:
            confidence = step_result["confidence"]
            if isinstance(confidence, (int, float)):
                return float(confidence) if confidence <= 1.0 else float(confidence) / 100.0
            elif isinstance(confidence, str):
                confidence_lower = confidence.lower()
                if confidence_lower in ["high", "excellent", "good"]:
                    return 0.9
                elif confidence_lower in ["medium", "moderate", "ok"]:
                    return 0.7
                elif confidence_lower in ["low", "poor"]:
                    return 0.4
                else:
                    return 0.3
        
        # Analyze content for confidence indicators
        content = step_result.get("content", "")
        if not content:
            return 0.1
            
        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i'm not sure", "i don't know", "might be", "could be", "possibly",
            "i think", "maybe", "perhaps", "uncertain", "unclear", "confused"
        ]
        
        content_lower = content.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in content_lower)
        
        if uncertainty_count > 0:
            return max(0.3, 0.8 - (uncertainty_count * 0.2))
        
        # Check for error indicators
        error_indicators = [
            "error", "failed", "couldn't", "unable", "apologize", "sorry", "issue", "problem"
        ]
        
        error_count = sum(1 for indicator in error_indicators if indicator in content_lower)
        if error_count > 0:
            return max(0.2, 0.6 - (error_count * 0.1))
        
        # Content length as quality indicator
        if len(content.strip()) < 20:
            return 0.4
        elif len(content.strip()) < 50:
            return 0.6
        
        # Default to reasonable confidence for successful steps
        return 0.8
    
    def _has_quality_issues(self, content: str) -> bool:
        """
        Check if content has quality issues that warrant a retry
        """
        if not content or len(content.strip()) < 10:
            return True
            
        content_lower = content.lower()
        
        # Check for generic/unhelpful responses
        generic_responses = [
            "i processed your request but couldn't generate",
            "i apologize, but i encountered an issue",
            "something went wrong",
            "please try again",
            "i'm not able to help with that",
            "i don't understand what you're asking"
        ]
        
        for generic in generic_responses:
            if generic in content_lower:
                return True
                
        # Check for repetitive content
        words = content.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:  # Less than 50% unique words
                return True
        
        return False
    
    async def _retry_with_strategy(self, 
                                 original_agent: str, 
                                 content: str, 
                                 session_id: str, 
                                 original_input: str,
                                 step_action: str = "",
                                 step_description: str = "",
                                 attempt_number: int = 1,
                                 conversation_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retry step execution with different strategies
        """
        strategy_index = (attempt_number - 1) % len(self.retry_config["retry_strategies"])
        strategy = self.retry_config["retry_strategies"][strategy_index]
        
        logger.info(f"🔁 RETRY ATTEMPT {attempt_number} with strategy: {strategy}")
        
        # 🚀 ADAPTIVE BACKOFF: Use context-aware retry timing
        if attempt_number > 1:
            backoff_time = self._calculate_adaptive_backoff(
                attempt_number=attempt_number,
                original_agent=original_agent,
                strategy=strategy,
                original_input=original_input,
                step_action=step_action
            )
            logger.info(f"🔁 ADAPTIVE BACKOFF: Backing off for {backoff_time} seconds (strategy: {strategy}, agent: {original_agent})")
            await asyncio.sleep(backoff_time)
        
        if strategy == "different_agent":
            return await self._retry_with_different_agent(
                original_agent, content, session_id, original_input, step_action, step_description, conversation_context
            )
        elif strategy == "rephrased_request":
            return await self._retry_with_rephrased_request(
                original_agent, content, session_id, original_input, step_action, step_description, conversation_context
            )
        elif strategy == "additional_context":
            return await self._retry_with_additional_context(
                original_agent, content, session_id, original_input, step_action, step_description, conversation_context
            )
        else:
            # Fallback to different agent
            return await self._retry_with_different_agent(
                original_agent, content, session_id, original_input, step_action, step_description, conversation_context
            )
    
    def _calculate_adaptive_backoff(self, attempt_number: int, original_agent: str, 
                                  strategy: str, original_input: str, step_action: str) -> float:
        """
        🚀 ADAPTIVE BACKOFF: Calculate context-aware retry timing based on error type and agent characteristics
        """
        # Base backoff intervals
        base_intervals = [1, 2, 4, 8, 16]  # Extended for longer sequences
        base_backoff = base_intervals[min(attempt_number - 2, len(base_intervals) - 1)]
        
        # Agent-specific multipliers (some agents need more time to recover)
        agent_multipliers = {
            "PrecisionEditorAgent": 1.5,    # Complex editing needs more time
            "WriterAgent": 1.3,             # Content creation needs more time
            "EditorAgent": 1.2,             # Regular editing needs slight delay
            "ConversationalAgent": 0.8,     # Conversation is fast
            "TimekeeperAgent": 0.9,         # Time operations are typically fast
            "LearningAgent": 1.1            # Memory operations need slight delay
        }
        
        # Strategy-specific multipliers
        strategy_multipliers = {
            "different_agent": 0.7,         # Different agent might be faster
            "rephrased_request": 1.4,       # Rephrasing takes extra processing
            "additional_context": 1.2       # More context needs more processing
        }
        
        # Request complexity multipliers
        complexity_multiplier = 1.0
        input_length = len(original_input)
        if input_length > 500:
            complexity_multiplier = 1.5     # Long requests need more time
        elif input_length > 200:
            complexity_multiplier = 1.2     # Medium requests need some extra time
        elif input_length < 50:
            complexity_multiplier = 0.8     # Short requests can be faster
        
        # Task type multipliers
        task_multiplier = 1.0
        if step_action:
            action_lower = step_action.lower()
            if "edit" in action_lower or "modify" in action_lower:
                task_multiplier = 1.3       # Editing tasks are complex
            elif "create" in action_lower or "write" in action_lower:
                task_multiplier = 1.4       # Creation tasks are complex
            elif "search" in action_lower or "find" in action_lower:
                task_multiplier = 0.9       # Search tasks are typically fast
        
        # Progressive penalty for repeated failures
        failure_penalty = 1.0 + (attempt_number - 1) * 0.2  # Increase by 20% each attempt
        
        # Calculate final backoff time
        agent_multiplier = agent_multipliers.get(original_agent, 1.0)
        strategy_multiplier = strategy_multipliers.get(strategy, 1.0)
        
        adaptive_backoff = (
            base_backoff * 
            agent_multiplier * 
            strategy_multiplier * 
            complexity_multiplier * 
            task_multiplier * 
            failure_penalty
        )
        
        # Apply bounds (minimum 0.5s, maximum 30s)
        adaptive_backoff = max(0.5, min(30.0, adaptive_backoff))
        
        return adaptive_backoff
    
    async def _retry_with_different_agent(self, 
                                        original_agent: str, 
                                        content: str, 
                                        session_id: str, 
                                        original_input: str,
                                        step_action: str = "",
                                        step_description: str = "",
                                        conversation_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retry with a different agent that might be better suited for the task
        """
        # Get alternative agent suggestions
        alternative_agents = self._get_alternative_agents(original_agent, original_input, step_action, step_description)
        
        if not alternative_agents:
            logger.warning(f"🔁 No alternative agents available for {original_agent}")
            return await self._execute_step(original_agent, content, session_id, original_input, 
                                          step_action, step_description, conversation_context)
        
        # Try the first alternative agent
        alternative_agent = alternative_agents[0]
        logger.info(f"🔁 Retrying with alternative agent: {alternative_agent} (was {original_agent})")
        
        return await self._execute_step(alternative_agent, content, session_id, original_input,
                                      step_action, step_description, conversation_context)
    
    async def _retry_with_rephrased_request(self,
                                          original_agent: str,
                                          content: str,
                                          session_id: str,
                                          original_input: str,
                                          step_action: str = "",
                                          step_description: str = "",
                                          conversation_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        🚀 INTELLIGENT RETRY WITH REPHRASING: Skip expensive LLM rephrasing for simple requests
        """
        try:
            # 🆕 SIMPLE REQUEST DETECTION: Skip expensive rephrasing for simple requests
            if self._is_simple_request(original_input):
                logger.info(f"🚀 SIMPLE REQUEST detected, skipping expensive LLM rephrasing")
                # Just retry with original request (fast)
                return await self._execute_step(original_agent, content, session_id, original_input,
                                              step_action, step_description, conversation_context)
            
            # For complex requests, use LLM to rephrase
            rephrased_input = await self._rephrase_request(original_input, step_action, step_description)
            
            if rephrased_input and rephrased_input != original_input:
                logger.info(f"🔁 Retrying with rephrased request: {rephrased_input[:50]}...")
                return await self._execute_step(original_agent, content, session_id, rephrased_input,
                                              step_action, step_description, conversation_context)
            else:
                logger.warning(f"🔁 Failed to rephrase request, using original")
                return await self._execute_step(original_agent, content, session_id, original_input,
                                              step_action, step_description, conversation_context)
                
        except Exception as e:
            logger.error(f"🔁 Rephrasing failed: {e}")
            return await self._execute_step(original_agent, content, session_id, original_input,
                                          step_action, step_description, conversation_context)
    
    async def _retry_with_additional_context(self,
                                           original_agent: str,
                                           content: str,
                                           session_id: str,
                                           original_input: str,
                                           step_action: str = "",
                                           step_description: str = "",
                                           conversation_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retry with additional context from conversation history
        """
        # Enhance context with more conversation history
        enhanced_context = conversation_context or []
        
        # Add more context if available (expand from 5 to 10 messages)
        if len(enhanced_context) < 10:
            logger.info(f"🔁 Context already includes {len(enhanced_context)} messages")
        else:
            enhanced_context = enhanced_context[-10:]  # Use last 10 messages instead of 5
            logger.info(f"🔁 Retrying with enhanced context: {len(enhanced_context)} messages")
        
        return await self._execute_step(original_agent, content, session_id, original_input,
                                      step_action, step_description, enhanced_context)
    
    def _get_alternative_agents(self, original_agent: str, original_input: str, step_action: str, step_description: str) -> List[str]:
        """
        🚀 INTELLIGENT FALLBACK: Use LLM-based semantic analysis to determine optimal fallback agents
        """
        try:
            # Use intelligent semantic analysis to determine best fallback
            return self._intelligent_fallback_selection(original_agent, original_input, step_action, step_description)
        except Exception as e:
            logger.error(f"❌ Intelligent fallback failed: {e}, using simple fallback")
            # Fallback to simple approach on error
            return self._simple_fallback_selection(original_agent, original_input, step_action, step_description)
    
    def _intelligent_fallback_selection(self, original_agent: str, original_input: str, 
                                       step_action: str, step_description: str) -> List[str]:
        """
        🚀 INTELLIGENT FALLBACK: Use LLM to analyze task context and select optimal fallback agents
        """
        try:
            # Build comprehensive context for analysis
            task_context = f"""
            Original Agent: {original_agent}
            User Input: {original_input}
            Step Action: {step_action}
            Step Description: {step_description}
            """
            
            # Create agent capability descriptions
            agent_descriptions = []
            for agent_name, info in self.available_agents.items():
                if agent_name != original_agent:  # Exclude original agent
                    capabilities = ", ".join(info["capabilities"])
                    agent_descriptions.append(f"- {agent_name}: {info['description']} (Capabilities: {capabilities})")
            
            agents_info = "\n".join(agent_descriptions)
            
            prompt = f"""You are an intelligent agent routing system. The original agent '{original_agent}' failed to complete a task. 
            
Analyze the task context and select the 2-3 most appropriate fallback agents in order of preference.

TASK CONTEXT:
{task_context}

AVAILABLE AGENTS:
{agents_info}

Consider:
1. Task complexity and type
2. Content creation vs editing vs conversation
3. Agent capabilities and specializations
4. Context understanding requirements

Return ONLY a JSON array of agent names in order of preference (most suitable first):
["Agent1", "Agent2", "Agent3"]

Example: If the task is about editing but the original editor failed, you might prefer:
["PrecisionEditorAgent", "WriterAgent", "ConversationalAgent"]

If the task is conversational but failed, you might prefer:
["WriterAgent", "EditorAgent"]"""

            # 🚀 INTELLIGENT MODEL SELECTION: Use fast model for fallback selection
            selected_model = model_selector.select_model(
                user_input=original_input,
                agent_name="MetaAgent",
                context="fallback selection",
                is_retry=True  # This is a retry operation, use fast model
            )
            
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.choices[0].message.content.strip())
            
            # Validate that all returned agents exist
            valid_agents = [agent for agent in result if agent in self.available_agents and agent != original_agent]
            
            if not valid_agents:
                raise ValueError("No valid agents returned from LLM")
            
            logger.info(f"🚀 INTELLIGENT FALLBACK: Selected {valid_agents} for {original_agent}")
            return valid_agents
            
        except Exception as e:
            logger.error(f"❌ Intelligent fallback selection failed: {e}")
            raise e
    
    def _simple_fallback_selection(self, original_agent: str, original_input: str, 
                                  step_action: str, step_description: str) -> List[str]:
        """
        Simple fallback selection as backup when intelligent selection fails
        """
        # Build context for alternative agent selection
        context = f"{original_input} {step_action} {step_description}".lower()
        
        # Agent fallback chains (simplified and improved)
        fallback_chains = {
            "PrecisionEditorAgent": ["EditorAgent", "WriterAgent", "ConversationalAgent"],
            "EditorAgent": ["PrecisionEditorAgent", "WriterAgent", "ConversationalAgent"],
            "WriterAgent": ["EditorAgent", "ConversationalAgent"],
            "ConversationalAgent": ["WriterAgent", "EditorAgent"],
            "TimekeeperAgent": ["ConversationalAgent"],
            "LearningAgent": ["ConversationalAgent"]
        }
        
        # Get fallback chain for the original agent
        alternatives = fallback_chains.get(original_agent, ["ConversationalAgent"])
        
        # Filter out the original agent
        alternatives = [agent for agent in alternatives if agent != original_agent]
        
        # Re-rank based on context relevance
        ranked_alternatives = []
        for agent in alternatives:
            if agent in self.available_agents:
                capabilities = self.available_agents[agent]["capabilities"]
                relevance_score = sum(1 for capability in capabilities if capability in context)
                ranked_alternatives.append((agent, relevance_score))
        
        # Sort by relevance score (descending)
        ranked_alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return [agent for agent, _ in ranked_alternatives]
    
    async def _rephrase_request(self, original_input: str, step_action: str, step_description: str) -> str:
        """
        Use LLM to rephrase a request for better understanding
        """
        try:
            context = f"Action: {step_action}, Description: {step_description}" if step_action else ""
            
            prompt = f"""
Rephrase the following request to be clearer and more specific while preserving the original intent:

ORIGINAL REQUEST: "{original_input}"
CONTEXT: {context}

Guidelines:
- Make it more specific and actionable
- Remove ambiguity
- Add helpful details
- Keep the same intent and tone
- Make it easier for an AI agent to understand

Return only the rephrased request, no explanations.
"""
            
            # 🚀 INTELLIGENT MODEL SELECTION: Use fast model for retry operations
            selected_model = model_selector.select_model(
                user_input=original_input,
                agent_name="MetaAgent",
                context="request rephrasing",
                is_retry=True  # This is a retry operation, use fast model
            )
            
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            rephrased = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            if rephrased.startswith('"') and rephrased.endswith('"'):
                rephrased = rephrased[1:-1]
            
            return rephrased
            
        except Exception as e:
            logger.error(f"❌ Request rephrasing failed: {e}")
            return original_input

    # 🆕 Opportunistic Learning Methods
    
    def _should_trigger_learning(self, user_input: str, response_content: str) -> Set[str]:
        """
        Determine if opportunistic learning should trigger based on pattern detection
        
        Returns:
            Set of learning categories that should be analyzed
        """
        triggered_categories = set()
        combined_text = f"{user_input} {response_content}".lower()
        
        for category, patterns in self.learning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    triggered_categories.add(category)
                    break  # Found one pattern for this category, move to next category
        
        if triggered_categories:
            logger.info(f"🧠 LEARNING TRIGGERED: Categories {triggered_categories}")
        
        return triggered_categories
    
    async def _opportunistic_learn(self, 
                                 user_input: str, 
                                 response_content: str,
                                 session_id: str,
                                 conversation_context: List[Dict[str, Any]] = [],
                                 result_state: Dict[str, Any] = {}) -> None:
        """
        Perform opportunistic learning after successful request execution
        
        This runs asynchronously and doesn't block the main response flow
        """
        try:
            # 🚨 CRITICAL FIX: Do not learn from search results to prevent feedback loops
            if result_state.get("is_search_result", False):
                logger.info("🧠 SKIPPING OPPORTUNISTIC LEARNING: Response is a search result.")
                return

            logger.info(f"🧠 OPPORTUNISTIC LEARNING: Analyzing for learning opportunities")
            
            # Check if learning should be triggered
            triggered_categories = self._should_trigger_learning(user_input, response_content)
            
            if not triggered_categories:
                logger.info(f"🧠 No learning patterns detected, skipping opportunistic learning")
                return
            
            # Analyze for learning opportunities using LLM
            learning_candidates = await self._analyze_for_learning(
                user_input, response_content, triggered_categories, conversation_context
            )
            
            if not learning_candidates:
                logger.info(f"🧠 No learning opportunities found")
                return
            
            # Process each learning candidate based on confidence
            for candidate in learning_candidates:
                await self._process_learning_candidate(candidate, session_id)
            
            logger.info(f"🧠 OPPORTUNISTIC LEARNING COMPLETED: Processed {len(learning_candidates)} candidates")
            
        except Exception as e:
            logger.error(f"❌ Opportunistic learning failed: {e}")
            # Don't let learning failures affect the main response
    
    async def _analyze_for_learning(self, 
                                  user_input: str, 
                                  response_content: str,
                                  triggered_categories: Set[str],
                                  conversation_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to analyze conversation for learning opportunities
        """
        try:
            # Build context for analysis
            context_summary = self._build_context_summary(conversation_context[-3:])  # Last 3 messages
            
            categories_str = ", ".join(triggered_categories)
            
            prompt = f"""
You are analyzing a conversation for opportunistic learning. Look for facts, relationships, preferences, and other information that should be stored in Andrew's memory.

TRIGGERED CATEGORIES: {categories_str}

CONVERSATION CONTEXT:
{context_summary}

USER INPUT: "{user_input}"

ASSISTANT RESPONSE: "{response_content}"

Extract any learnable information and classify it by confidence level. Return ONLY a JSON array:

[
  {{
    "fact": "specific fact to store",
    "category": "people|preferences|projects|schedules|general",
    "confidence": "high|medium|low",
    "reasoning": "why this should be stored",
    "source": "user_input|response|context"
  }}
]

CONFIDENCE LEVELS:
- high (95%+): Clear, explicit facts like "Sarah is my design partner"
- medium (70-95%): Implied information that's likely true
- low (50-70%): Uncertain information that needs confirmation

CATEGORIES:
- people: Names, relationships, roles
- preferences: Communication style, work habits, likes/dislikes
- projects: Current work, clients, tasks
- schedules: Meeting times, work patterns, availability
- general: Other factual information

Only extract information that would be useful for future conversations. Ignore temporary information.

Return empty array [] if no learning opportunities found.
"""

            # 🚀 INTELLIGENT MODEL SELECTION: Learning analysis can use fast model for efficiency
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="MetaAgent",
                context="learning analysis"
            )
            
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"🧠 RAW LEARNING ANALYSIS RESPONSE: {content}")
            
            # 🚨 FIX: Handle empty or invalid responses
            if not content or content == "":
                logger.warning(f"⚠️ Empty response from learning analysis")
                return []
            
            # 🚨 FIX: Add robust JSON parsing with fallback
            try:
                candidates = json.loads(content)
                
                # Validate that we got a list
                if not isinstance(candidates, list):
                    logger.warning(f"⚠️ Learning analysis returned non-list: {type(candidates)}")
                    return []
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing failed in learning analysis: {e}")
                logger.error(f"❌ Raw content: {content}")
                
                # Try to extract JSON from the response if it's wrapped in text
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    try:
                        candidates = json.loads(json_match.group(0))
                        logger.info(f"✅ Extracted JSON from wrapped response")
                    except json.JSONDecodeError:
                        logger.error(f"❌ Failed to extract JSON even from match")
                        return []
                else:
                    logger.error(f"❌ No JSON array found in response")
                    return []
            
            if candidates:
                logger.info(f"🧠 FOUND {len(candidates)} learning candidates")
                for candidate in candidates:
                    logger.info(f"  - {candidate.get('confidence', 'unknown').upper()}: {candidate.get('fact', 'unknown')[:50]}...")
            
            return candidates
            
        except Exception as e:
            logger.error(f"❌ Learning analysis failed: {e}")
            return []
    
    async def _process_learning_candidate(self, candidate: Dict[str, Any], session_id: str) -> None:
        """
        Process a single learning candidate based on confidence level with intelligent conflict resolution
        """
        try:
            fact = candidate.get("fact", "")
            category = candidate.get("category", "general")
            confidence = candidate.get("confidence", "low")
            reasoning = candidate.get("reasoning", "")
            
            if not fact:
                return

            # 🚨 ENHANCED: Check for conflicts before storing
            conflicts = await memory_agent.detect_conflicts(fact)
            
            if conflicts:
                logger.info(f"⚠️ CONFLICT DETECTED for fact: {fact[:50]}...")
                await self._handle_learning_conflict(fact, conflicts, confidence, category, session_id)
                return

            if confidence == "high":
                # Auto-store high confidence facts (no conflicts detected)
                success = await memory_agent.save_fact(fact, category)
                if success:
                    logger.info(f"🧠 AUTO-STORED (HIGH confidence): {fact[:50]}...")
                else:
                    logger.warning(f"⚠️ Failed to store high confidence fact: {fact[:50]}...")
                    
            elif confidence == "medium":
                # Store medium confidence facts but accumulate for potential review
                success = await memory_agent.save_fact(fact, category)
                if success:
                    logger.info(f"🧠 STORED (MEDIUM confidence): {fact[:50]}...")
                    # Add to session learning candidates for end-of-session review
                    if session_id not in self.session_learning_candidates:
                        self.session_learning_candidates[session_id] = []
                    self.session_learning_candidates[session_id].append({
                        **candidate,
                        "stored": True,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    logger.warning(f"⚠️ Failed to store medium confidence fact: {fact[:50]}...")
                    
            elif confidence == "low":
                # Queue low confidence facts for potential user confirmation
                logger.info(f"🧠 QUEUED (LOW confidence): {fact[:50]}... - {reasoning}")
                if session_id not in self.session_learning_candidates:
                    self.session_learning_candidates[session_id] = []
                self.session_learning_candidates[session_id].append({
                    **candidate,
                    "stored": False,
                    "needs_confirmation": True,
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"❌ Failed to process learning candidate: {e}")
    
    async def _handle_learning_conflict(self, new_fact: str, conflicts: List[Dict[str, Any]], 
                                      confidence: str, category: str, session_id: str) -> None:
        """
        Handle conflicts detected during opportunistic learning
        """
        try:
            if not conflicts:
                return
            
            primary_conflict = conflicts[0]
            existing_fact = primary_conflict.get("content", "")
            conflict_type = primary_conflict.get("conflict_type", "general_contradiction")
            
            logger.info(f"🔧 RESOLVING CONFLICT: {conflict_type} between '{existing_fact[:30]}...' and '{new_fact[:30]}...'")
            
            if confidence == "high":
                # High confidence new facts should update existing ones
                if conflict_type in ["temporal", "negation", "status_change"]:
                    # Auto-resolve these conflict types
                    update_result = await memory_agent.update_fact(existing_fact, new_fact, category)
                    if update_result["success"]:
                        logger.info(f"🔧 AUTO-UPDATED: {existing_fact[:30]}... → {new_fact[:30]}...")
                    else:
                        logger.warning(f"⚠️ Failed to update conflicting fact")
                else:
                    # Store as conflicting information for manual resolution
                    conflict_fact = f"CONFLICT_DETECTED: {new_fact} (conflicts with: {existing_fact})"
                    await memory_agent.save_fact(conflict_fact, "conflicts")
                    logger.info(f"🔧 STORED CONFLICT for manual resolution")
                    
            elif confidence == "medium":
                # Medium confidence - store as potential update
                if conflict_type in ["temporal", "status_change"]:
                    # Likely an update, store it
                    await memory_agent.save_fact(f"POTENTIAL_UPDATE: {new_fact} (may replace: {existing_fact})", "potential_updates")
                    logger.info(f"🔧 STORED as potential update")
                else:
                    # Store as conflicting information
                    conflict_fact = f"CONFLICT_NOTED: {new_fact} (conflicts with: {existing_fact})"
                    await memory_agent.save_fact(conflict_fact, "conflicts")
                    logger.info(f"🔧 NOTED conflict for review")
                    
            else:  # low confidence
                # Low confidence - just note the conflict
                if session_id not in self.session_learning_candidates:
                    self.session_learning_candidates[session_id] = []
                
                self.session_learning_candidates[session_id].append({
                    "fact": new_fact,
                    "category": category,
                    "confidence": confidence,
                    "conflict_detected": True,
                    "conflicting_fact": existing_fact,
                    "conflict_type": conflict_type,
                    "needs_confirmation": True,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"🔧 QUEUED conflicting fact for user confirmation")
                
        except Exception as e:
            logger.error(f"❌ Failed to handle learning conflict: {e}")
    
    async def _end_of_session_learning(self, session_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive learning analysis at the end of a session
        
        This can be called when a session is being closed or after a period of inactivity
        """
        try:
            candidates = self.session_learning_candidates.get(session_id, [])
            if not candidates:
                return {"learning_summary": "No learning opportunities accumulated"}
            
            # Separate by confidence and status
            high_confidence = [c for c in candidates if c.get("confidence") == "high" and c.get("stored")]
            medium_confidence = [c for c in candidates if c.get("confidence") == "medium" and c.get("stored")]
            needs_confirmation = [c for c in candidates if c.get("needs_confirmation", False)]
            
            summary = {
                "session_id": session_id,
                "total_candidates": len(candidates),
                "auto_stored": len(high_confidence),
                "stored_medium": len(medium_confidence),
                "needs_confirmation": len(needs_confirmation),
                "learning_summary": f"Learned {len(high_confidence) + len(medium_confidence)} facts",
                "confirmation_candidates": needs_confirmation[:5] if needs_confirmation else []  # Limit to 5 for review
            }
            
            # Clean up session learning candidates
            if session_id in self.session_learning_candidates:
                del self.session_learning_candidates[session_id]
            
            if summary["total_candidates"] > 0:
                logger.info(f"🧠 END-OF-SESSION LEARNING: {summary['learning_summary']} from {summary['total_candidates']} candidates")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ End-of-session learning failed: {e}")
            return {"error": str(e)}
    
    def get_session_learning_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get current learning candidates for a session (for potential user review)
        """
        candidates = self.session_learning_candidates.get(session_id, [])
        needs_confirmation = [c for c in candidates if c.get("needs_confirmation", False)]
        
        return {
            "session_id": session_id,
            "total_candidates": len(candidates),
            "needs_confirmation": len(needs_confirmation),
            "confirmation_candidates": needs_confirmation
        }

    async def _prefetch_memory_context(self, user_input: str, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Pre-fetch memory context that can be shared across agents
        🚀 PARALLEL OPTIMIZATION: This runs in parallel with understanding
        """
        try:
            logger.info(f"🧠 PREFETCH: Pre-fetching memory context in parallel")
            
            # 🚀 INTELLIGENT MEMORY SEARCH: Use generic agent context for prefetch
            # This provides a base set of memories that can be used by any agent
            search_results = await self._intelligent_memory_search(user_input, "MetaAgent_Prefetch")
            
            # Categorize memories using LLM-based classification
            categorized_memories = await self._categorize_memories(search_results, user_input)
            
            # Format conversation context once for reuse
            conversation_context_formatted = self._format_conversation_context(conversation_context)
            
            # Return structured context that can be reused
            return {
                "relevant_memories": categorized_memories,
                "memory_context": self._format_memory_context(categorized_memories, user_input),
                "conversation_context": conversation_context_formatted,
                "raw_memories": search_results
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Memory prefetch failed: {e}")
            # Return empty context on failure
            return {
                "relevant_memories": {},
                "memory_context": "",
                "conversation_context": self._format_conversation_context(conversation_context),
                "raw_memories": []
            }
    
    async def _detect_and_process_parallel_intents(self, user_input: str, session_id: str, 
                                                 conversation_context: List[Dict[str, Any]], 
                                                 primary_result: Dict[str, Any]) -> None:
        """
        🚀 PARALLEL INTENT PROCESSING: Detect and process secondary intents in parallel
        
        This allows the system to handle multiple intents simultaneously:
        - Primary: "Please rewrite this" → EditorAgent
        - Secondary: File reference in content → LearningAgent (opportunistic)
        """
        try:
            logger.info(f"🔍 PARALLEL INTENT DETECTION: Analyzing for secondary intents")
            
            # Use LLM to detect parallel intents
            parallel_intents = await self._analyze_parallel_intents(user_input, primary_result)
            
            if not parallel_intents:
                logger.info(f"🔍 No parallel intents detected")
                return
            
            # Process each parallel intent
            for intent_info in parallel_intents:
                intent_type = intent_info.get("intent_type", "")
                confidence = intent_info.get("confidence", 0.0)
                
                # Only process high-confidence parallel intents
                if confidence < 0.7:
                    logger.info(f"🔍 Skipping low-confidence parallel intent: {intent_type} ({confidence:.2f})")
                    continue
                
                logger.info(f"🔍 Processing parallel intent: {intent_type} (confidence: {confidence:.2f})")
                
                # Route to appropriate agent for parallel processing
                if intent_type == "file_reference":
                    await self._process_file_reference_intent(intent_info, session_id, user_input)
                elif intent_type == "task_creation":
                    await self._process_task_creation_intent(intent_info, session_id, user_input)
                elif intent_type == "resource_saving":
                    await self._process_resource_saving_intent(intent_info, session_id, user_input)
                elif intent_type == "time_logging":
                    await self._process_time_logging_intent(intent_info, session_id, user_input)
                    
            logger.info(f"✅ PARALLEL INTENT PROCESSING: Completed {len(parallel_intents)} intents")
            
        except Exception as e:
            logger.error(f"❌ Parallel intent processing failed: {e}")
            # Don't let parallel processing failures affect the main response
    
    async def _analyze_parallel_intents(self, user_input: str, primary_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to analyze if there are secondary intents that should be processed in parallel
        """
        try:
            primary_agent = primary_result.get("agent_used", "")
            
            prompt = f"""You are analyzing a user request to detect secondary intents that should be processed in parallel with the primary intent.

PRIMARY INTENT: Handled by {primary_agent}
USER REQUEST: "{user_input}"

Look for secondary intents that should be processed in parallel:

1. **FILE REFERENCE**: User mentions files (Figma, design files, code files, documents) in context
   - "Please rewrite this: [content about Figma file]" → Secondary: save file reference
   - "Edit this copy for the homepage design" → Secondary: save file reference

2. **TASK CREATION**: User mentions tasks or things to do
   - "Rewrite this email and remind me to send it" → Secondary: create task
   - "Make this better, I need to review it later" → Secondary: create task

3. **RESOURCE SAVING**: User mentions links, tools, or resources to remember
   - "Improve this copy from https://example.com" → Secondary: save resource
   - "Edit this and save that link for later" → Secondary: save resource

4. **TIME LOGGING**: User mentions time spent or work done
   - "Rewrite this, I spent 2 hours on it" → Secondary: log time
   - "Make this better, worked on it this morning" → Secondary: log time

Return ONLY a JSON array of secondary intents:
[
  {{
    "intent_type": "file_reference|task_creation|resource_saving|time_logging",
    "content": "specific content to process",
    "confidence": 0.0-1.0,
    "reasoning": "why this should be processed in parallel"
  }}
]

IMPORTANT: Only detect intents that are SECONDARY to the primary action, not competing with it.
Return empty array [] if no parallel intents found.
"""

            # 🚀 INTELLIGENT MODEL SELECTION: Use fast model for intent analysis
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="MetaAgent",
                context="parallel intent analysis"
            )
            
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"🔍 PARALLEL INTENT ANALYSIS: {content}")
            
            # Parse JSON response
            import json
            try:
                # Handle empty or whitespace-only responses
                if not content or not content.strip():
                    logger.info(f"🔍 Empty parallel intent response - no secondary intents detected")
                    return []
                
                # 🚨 FIX: Handle markdown code blocks that wrap JSON
                content_to_parse = content.strip()
                if content_to_parse.startswith('```json'):
                    # Extract JSON from markdown code block
                    lines = content_to_parse.split('\n')
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip() == '```json':
                            in_json = True
                            continue
                        elif line.strip() == '```':
                            break
                        elif in_json:
                            json_lines.append(line)
                    content_to_parse = '\n'.join(json_lines)
                
                parallel_intents = json.loads(content_to_parse)
                if isinstance(parallel_intents, list):
                    logger.info(f"🔍 Found {len(parallel_intents)} parallel intents")
                    return parallel_intents
                else:
                    logger.warning(f"⚠️ Unexpected parallel intent format: {type(parallel_intents)}")
                    return []
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Failed to parse parallel intents JSON: {e}")
                logger.warning(f"⚠️ Raw content: '{content}'")
                return []
                
        except Exception as e:
            logger.error(f"❌ Parallel intent analysis failed: {e}")
            return []
    
    async def _process_file_reference_intent(self, intent_info: Dict[str, Any], session_id: str, user_input: str) -> None:
        """Process file reference intent in parallel"""
        try:
            from agents.learning_agent import IntelligentLearningAgent
            learning_agent = IntelligentLearningAgent()
            
            # Create state for file reference processing
            state = {
                "user_input": user_input,
                "session_id": session_id,
                "parallel_intent": True,
                "intent_type": "save_file_reference",
                "extracted_content": intent_info.get("content", "")
            }
            
            result = await learning_agent.invoke(state)
            logger.info(f"📁 Parallel file reference processed: {result.get('content', '')[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Parallel file reference processing failed: {e}")
    
    async def _process_task_creation_intent(self, intent_info: Dict[str, Any], session_id: str, user_input: str) -> None:
        """Process task creation intent in parallel"""
        try:
            from agents.learning_agent import IntelligentLearningAgent
            learning_agent = IntelligentLearningAgent()
            
            # Create state for task creation processing
            state = {
                "user_input": user_input,
                "session_id": session_id,
                "parallel_intent": True,
                "intent_type": "save_task",
                "extracted_content": intent_info.get("content", "")
            }
            
            result = await learning_agent.invoke(state)
            logger.info(f"📋 Parallel task creation processed: {result.get('content', '')[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Parallel task creation processing failed: {e}")
    
    async def _process_resource_saving_intent(self, intent_info: Dict[str, Any], session_id: str, user_input: str) -> None:
        """Process resource saving intent in parallel"""
        try:
            from agents.learning_agent import IntelligentLearningAgent
            learning_agent = IntelligentLearningAgent()
            
            # Create state for resource saving processing
            state = {
                "user_input": user_input,
                "session_id": session_id,
                "parallel_intent": True,
                "intent_type": "save_resource",
                "extracted_content": intent_info.get("content", "")
            }
            
            result = await learning_agent.invoke(state)
            logger.info(f"🔗 Parallel resource saving processed: {result.get('content', '')[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Parallel resource saving processing failed: {e}")
    
    async def _process_time_logging_intent(self, intent_info: Dict[str, Any], session_id: str, user_input: str) -> None:
        """Process time logging intent in parallel"""
        try:
            from agents.learning_agent import IntelligentLearningAgent
            learning_agent = IntelligentLearningAgent()
            
            # Create state for time logging processing
            state = {
                "user_input": user_input,
                "session_id": session_id,
                "parallel_intent": True,
                "intent_type": "log_time",
                "extracted_content": intent_info.get("content", "")
            }
            
            result = await learning_agent.invoke(state)
            logger.info(f"⏱️ Parallel time logging processed: {result.get('content', '')[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Parallel time logging processing failed: {e}")

# Global instance
meta_agent = MetaAgent() 