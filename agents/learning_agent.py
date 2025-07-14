from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .commands import Command, CommandIntents, create_command, complete_command
from .tone_of_voice import get_andrew_voice
from .tools import AVAILABLE_TOOLS
# Intent classification removed - using simple pattern matching instead
from .model_selector import model_selector
from config import settings
from .memory_agent import memory_agent
import os
import json
import re
import logging
from datetime import datetime
import aiofiles

# Configure logging
logger = logging.getLogger(__name__)

class IntelligentLearningAgent:
    """
    Enhanced learning agent that detects both explicit and implicit learning opportunities.
    
    Now uses intent classification and contextual understanding to pick up on:
    - Implicit information sharing ("By the way, I work with Sarah on design")
    - Behavioral patterns (user always asks for shorter content = preference)
    - Correction patterns (user often shortens responses = style preference)
    - Relationship mentions in natural conversation
    """
    
    def __init__(self):
        self.tools = AVAILABLE_TOOLS
        self.voice_guidelines = get_andrew_voice()
        # Note: Model will be selected dynamically based on request complexity
        self.base_llm_config = {
            "api_key": settings.OPENAI_API_KEY,
            "temperature": settings.LEARN_TEMP,
            "streaming": settings.STREAM
        }
        
        # Enhanced learning patterns
        self.implicit_learning_triggers = [
            # Relationship mentions
            r"(.+?)\s+(?:and\s+)?(?:i|we)\s+(?:work\s+together|collaborate|are\s+(?:working\s+)?on)",
            r"(?:i|we)\s+(?:work\s+with|collaborate\s+with)\s+(.+)",
            r"(.+?)\s+is\s+(?:my|our)\s+(.+?)(?:\s+(?:at|on|for))?",
            r"(.+?)\s+(?:works?\s+(?:at|for|with)|handles?|manages?)\s+(.+)",
            
            # Company/project mentions  
            r"(?:at|for)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
            r"(?:working\s+on|building|developing)\s+(.+?)(?:\s+project)?",
            
            # Preference hints
            r"i\s+(?:prefer|like|want|need)\s+(.+)",
            r"(?:please\s+)?(?:make\s+it|keep\s+it)\s+(more|less)\s+(.+)",
            r"(?:always|usually|typically)\s+(.+)",
        ]
        
        # Behavioral learning patterns
        self.behavioral_patterns = {
            "style_preferences": [
                "shorter", "longer", "more casual", "more formal", "friendlier", 
                "more direct", "less corporate", "simpler", "clearer"
            ],
            "content_types": [
                "emails", "messages", "updates", "reports", "notes", "summaries"
            ],
            "time_patterns": [
                "morning", "afternoon", "evening", "weekends", "weekdays"
            ]
        }
    
    async def _classify_learning_intent(self, user_input: str, previous_output: str = "") -> Dict[str, Any]:
        """
        🚀 INTELLIGENT CLASSIFICATION: Use LLM to classify learning intent.
        NO FALLBACKS - fail clearly if LLM fails.
        
        Returns:
            Dict with 'intent', 'confidence', and 'extracted_content' keys
        """
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on complexity
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="LearningAgent",
                context=previous_output
            )
            
            logger.info(f"🔍 MODEL SELECTED: {selected_model}")
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            # Build context for classification
            context_info = ""
            if previous_output:
                context_info = f"\nPrevious response: {previous_output[:200]}..."
            
            system_prompt = f"""You are an intelligent learning intent classifier. Analyze user input to determine what type of learning action is needed.

DECISION LOGIC (Apply in this exact order):

1. **REWRITE/EDIT PRIORITY RULE**: If the input contains action verbs like "rewrite", "edit", "improve", "make better" → ALWAYS classify as `no_intent`. This allows the meta agent to handle these requests.
   - "Please rewrite this: [content with URL]" → no_intent (let meta agent handle rewrite)
   - "Can you edit this message with Figma URL" → no_intent (let meta agent handle edit)
   - "Make this better: [content]" → no_intent (let meta agent handle improve)

2. **TASK KEYWORD PRIORITY RULE**: If the input contains explicit task keywords ("task", "to-do", "todo", "remind me", "add to my list") → ALWAYS classify as `save_task`. This rule OVERRIDES all others.
   - "Add to my todo: research https://grafana.com" → save_task (keyword "todo" takes priority over URL)
   - "Remind me to review the Figma file" → save_task (keyword "remind me" takes priority)

3. **EXPLICIT FILE SHARING RULE**: Only classify as `save_file_reference` if the user is EXPLICITLY sharing or mentioning files for work tracking.
   - "Here's the Figma file for homepage" → save_file_reference (EXPLICIT sharing)
   - "I'm working on config.py" → save_file_reference (EXPLICIT work context)
   - "Remember this file: https://www.figma.com/design/homepage" → save_file_reference (EXPLICIT file storage)

4. **WORK CONTEXT RULE**: Only if user is explicitly stating current work activities (not requesting edits).
   - Keywords: "working on", "editing", "building", "designing", "developing", "coding" (without edit requests)
   - "I'm currently working on config.py" → save_file_reference (explicit work statement)
   - "Please edit this config.py content" → no_intent (edit request, not work statement)

5. **TASK QUERY/CREATION**: If no edit/rewrite verbs and no explicit file sharing, but is clearly a task operation → `get_tasks` or `save_task`.
   - "What are my tasks?" → get_tasks
   - "Fix the auth bug" → save_task (only if not "edit this auth bug code")

6. **TIME LOGGING RULE**: If mentions hours/time spent → `log_time`.
   - "I worked 2 hours on..." → log_time

7. **QUERY RULE**: If asking questions about stored information → `semantic_query`.
   - "What file am I working on for..." → semantic_query
   - "Show me my..." → semantic_query

8. **MEMORY OPERATIONS**: If explicitly managing facts → `forget_fact`, `update_fact`, `store_memory`.
   - "Forget that" → forget_fact
   - "Remember that..." → store_memory

9. **GENERAL BOOKMARK RULE**: Only if NOT a design file and explicitly mentions "later", "bookmark", "reference" → `save_resource`.
   - "Save this article for later" → save_resource

CLASSIFICATION CATEGORIES:

1. **forget_fact** - User wants to delete/remove information from memory
2. **update_fact** - User wants to correct/modify existing information  
3. **store_memory** - User wants to save new information/facts
4. **log_time** - User wants to log work hours or time spent
5. **save_task** - User wants to create a task or to-do item
6. **get_tasks** - User wants to see their tasks or to-do list
7. **save_file_reference** - User EXPLICITLY mentions files for CURRENT WORK (not edit requests)
8. **save_resource** - User wants to save a link/URL as a bookmark for LATER USE (non-design files)
9. **semantic_query** - User is asking questions about stored information
10. **no_intent** - None of the above apply, or user is making edit/rewrite requests

CRITICAL RULE: Edit/rewrite requests should ALWAYS be no_intent to let the meta agent handle them.

Respond with ONLY a JSON object:
{{
    "intent": "category_name",
    "confidence": 0.0-1.0,
    "extracted_content": "relevant content extracted from input",
    "reasoning": "brief explanation of classification and which rule applied"
}}

USER INPUT: "{user_input}"{context_info}"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]

            logger.info(f"🔍 CLASSIFYING INTENT: '{user_input[:100]}...'")
            logger.info(f"📝 PROMPT LENGTH: {len(system_prompt)} chars")
            
            response = await llm.ainvoke(messages)
            
            logger.info(f"🔍 LLM RAW RESPONSE TYPE: {type(response)}")
            logger.info(f"🔍 LLM RAW RESPONSE: {response.content}")
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                logger.info(f"✅ JSON PARSED SUCCESSFULLY: {result}")
                
                # Validate result structure
                if not isinstance(result, dict) or "intent" not in result:
                    error_msg = f"Invalid LLM response structure. Expected dict with 'intent', got: {type(result)} = {result}"
                    logger.error(f"❌ STRUCTURE ERROR: {error_msg}")
                    raise ValueError(error_msg)
                
                # Ensure required fields with defaults
                result.setdefault("confidence", 0.5)
                result.setdefault("extracted_content", "")
                result.setdefault("reasoning", "")
                
                # Validate intent is one of expected values
                valid_intents = ["forget_fact", "update_fact", "store_memory", "log_time", "save_task", "get_tasks", "save_file_reference", "save_resource", "semantic_query", "no_intent"]
                if result["intent"] not in valid_intents:
                    error_msg = f"Invalid intent '{result['intent']}'. Must be one of: {valid_intents}"
                    logger.error(f"❌ INVALID INTENT: {error_msg}")
                    raise ValueError(error_msg)
                
                logger.info(f"✅ INTENT CLASSIFIED: {result['intent']} (confidence: {result['confidence']}) - {result['reasoning']}")
                return result
                
            except json.JSONDecodeError as e:
                error_msg = f"LLM returned invalid JSON. Parse error: {e}\nRaw response: {response.content}"
                logger.error(f"❌ JSON PARSE ERROR: {error_msg}")
                raise ValueError(error_msg)
            
        except Exception as e:
            error_msg = f"LLM Classification failed completely: {type(e).__name__}: {str(e)}"
            logger.error(f"❌ CLASSIFICATION FAILURE: {error_msg}")
            raise RuntimeError(error_msg)


    async def invoke(self, state: Dict[str, Any]) -> Command:
        """
        Enhanced learning that picks up both explicit and implicit information.
        """
        try:
            user_input = state.get("user_input", "")
            previous_output = state.get("previous_output", "")
            session_id = state.get("session_id", "")
            
            # 🚀 PARALLEL INTENT PROCESSING: Check if this is a parallel intent call
            parallel_intent = state.get("parallel_intent", False)
            if parallel_intent:
                intent_type = state.get("intent_type", "")
                extracted_content = state.get("extracted_content", "")
                
                logger.info(f"🔍 PARALLEL INTENT: Processing {intent_type} with content: {extracted_content[:100]}...")
                
                # Handle parallel intents directly without re-classification
                if intent_type == "save_file_reference":
                    return await self._handle_file_reference_creation(session_id, user_input, {"extracted_content": extracted_content})
                elif intent_type == "save_task":
                    return await self._handle_task_creation(session_id, user_input, {"extracted_content": extracted_content})
                elif intent_type == "save_resource":
                    return await self._handle_resource_creation(session_id, user_input, {"extracted_content": extracted_content})
                elif intent_type == "log_time":
                    return await self._handle_time_logging(session_id, user_input, {"extracted_content": extracted_content})
                else:
                    logger.warning(f"⚠️ Unknown parallel intent type: {intent_type}")
                    return complete_command(
                        state={"learning_result": f"Unknown parallel intent type: {intent_type}"},
                        reason="Unknown parallel intent type"
                    )
            
            # First check if this is a time insight from TimekeeperAgent
            time_insight = state.get("time_insight")
            if time_insight:
                return await self._handle_time_insight_learning(session_id, time_insight, state)
            
            # 🚀 INTELLIGENT CLASSIFICATION: Use LLM-based semantic understanding instead of keyword matching
            classification = await self._classify_learning_intent(user_input, previous_output)
            
            if classification["intent"] == "forget_fact":
                return await self._handle_forget_fact(session_id, user_input)
            elif classification["intent"] == "update_fact":
                return await self._handle_update_fact(session_id, user_input)
            elif classification["intent"] == "store_memory":
                return await self._handle_explicit_memory_storage(session_id, user_input, classification)
            elif classification["intent"] == "log_time":
                return await self._handle_time_logging(session_id, user_input, classification)
            elif classification["intent"] == "save_task":
                return await self._handle_task_creation(session_id, user_input, classification)
            elif classification["intent"] == "get_tasks":
                return await self._handle_task_query(session_id, user_input, classification)
            elif classification["intent"] == "save_file_reference":
                return await self._handle_file_reference_creation(session_id, user_input, classification)
            elif classification["intent"] == "save_resource":
                return await self._handle_resource_creation(session_id, user_input, classification)
            elif classification["intent"] == "semantic_query":
                return await self._handle_semantic_query(session_id, user_input, classification)
            
            else:
                # Check for implicit learning opportunities
                implicit_learning = await self._detect_implicit_learning(user_input, previous_output, session_id)
                if implicit_learning:
                    return implicit_learning
                
                # Check for behavioral patterns and corrections
                behavioral_learning = await self._detect_behavioral_learning(user_input, previous_output, session_id)
                if behavioral_learning:
                    return behavioral_learning
                
                # No learning detected
                return complete_command(
                    state={
                        "learning_result": "No learning opportunities detected",
                        "correction_detected": False
                    },
                    reason="No explicit or implicit learning detected in this interaction"
                )
                
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Learning agent error: {str(e)}"
            )

    async def _handle_explicit_memory_storage(self, session_id: str, user_input: str, classified_intent=None) -> Command:
        """Handle explicit memory storage requests with enhanced entity extraction"""
        try:
            # Extract fact content from user input using simple patterns
            fact_content = self._extract_fact_content(user_input)
            entities = {}
            
            # Enhanced entity extraction
            enhanced_entities = await self._extract_enhanced_entities(fact_content)
            entities.update(enhanced_entities)
            
            # Store as persistent fact
            success = await memory_agent.save_fact(
                fact=fact_content,
                category=self._determine_fact_category(entities)
            )
            
            if success:
                # Create natural response
                response = self._create_natural_memory_response(fact_content, entities)
                
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "fact_stored": fact_content,
                        "entities": entities,
                        "category": self._determine_fact_category(entities)
                    },
                    content=response,
                    reason=f"Successfully stored fact: {fact_content[:50]}..."
                )
            else:
                return create_command(
                    CommandIntents.ERROR,
                    state={"error": "Failed to store fact"},
                    reason="Memory storage failed"
                )
                
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Explicit memory storage failed: {str(e)}"
            )

    async def _handle_time_logging(self, session_id: str, user_input: str, classified_intent=None) -> Command:
        """Handle time logging with semantic understanding - NO FALLBACKS"""
        try:
            # 🚀 SEMANTIC EXTRACTION: Use intelligent understanding for time extraction
            duration, task, category = await self._extract_time_info_semantically(user_input)
            
            if duration > 0 and task:
                # Store time entry
                entry_id = await memory_agent.log_time(
                    task=task,
                    duration_hours=duration,
                    category=category,
                    notes=f"Auto-logged from: {user_input}"
                )
                
                # Also store as a fact for future reference
                time_fact = f"I worked {duration} hours on {task} (category: {category})"
                await memory_agent.save_fact(time_fact, "time_tracking")
                
                # Create natural response
                response = f"✅ Logged {duration} hours for {task}. I'll remember this for future time tracking."
                
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "time_logged": {
                            "duration": duration,
                            "task": task,
                            "category": category,
                            "entry_id": entry_id
                        }
                    },
                    content=response,
                    reason=f"Successfully logged {duration}h for {task}"
                )
            else:
                return create_command(
                    CommandIntents.ERROR,
                    state={"error": "Invalid time logging data"},
                    reason="Could not extract valid duration and task from time logging request"
                )
                
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Time logging failed: {str(e)}"
            )
    
    async def _handle_task_creation(self, session_id: str, user_input: str, classified_intent=None) -> Command:
        """Handle task creation requests using semantic understanding"""
        try:
            # 🚀 SEMANTIC EXTRACTION: Use intelligent understanding instead of patterns
            conversation_context = classified_intent.get("extracted_content", "") if classified_intent else ""
            task_info = await self._extract_task_info_semantically(user_input, conversation_context)
            
            if not task_info or not task_info[0]:  # No task detected
                return create_command(
                    CommandIntents.ERROR,
                    state={"error": "No task detected"},
                    reason="Could not extract valid task from request"
                )
            
            task_text, priority, due_date, project, semantic_tags = task_info
            
            # Store task with enhanced semantic information
            task_id = await memory_agent.save_task_with_context(
                task=task_text,
                priority=priority,
                due_date=due_date,
                project=project,
                semantic_tags=semantic_tags,
                conversation_context=user_input,
                notes=f"Created from: {user_input}"
            )
            
            # Create natural response
            response = f"✅ Added task: {task_text}"
            if priority != "medium":
                response += f" (priority: {priority})"
            if due_date:
                response += f" (due: {due_date})"
            if project:
                response += f" [project: {project}]"
            response += ". I'll remember this for you."
            
            # Add semantic tags to response for transparency
            if semantic_tags:
                response += f"\n\n🏷️ Tagged as: {', '.join(semantic_tags)}"
            
            return complete_command(
                state={
                    "learning_result": response,
                    "content": response,
                    "task_created": {
                        "task": task_text,
                        "priority": priority,
                        "due_date": due_date,
                        "project": project,
                        "semantic_tags": semantic_tags,
                        "task_id": task_id
                    }
                },
                content=response,
                reason=f"Successfully created semantic task: {task_text}"
            )
                
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Task creation failed: {str(e)}"
            )
    
    async def _handle_task_query(self, session_id: str, user_input: str, classified_intent=None) -> Command:
        """Handle requests to get tasks or to-do list using NEW semantic system"""
        try:
            # 🚀 NEW SEMANTIC SYSTEM: Use find_tasks_by_intent instead of get_tasks
            conversation_context = classified_intent.get("extracted_content", "") if classified_intent else ""
            tasks = await memory_agent.find_tasks_by_intent(user_input, conversation_context)
            
            if tasks:
                response = "Here are your tasks:\n\n"
                for i, task in enumerate(tasks, 1):
                    response += f"{i}. **{task['task']}**"
                    if task.get('priority', 'medium') != "medium":
                        response += f" (Priority: {task['priority']})"
                    if task.get('due_date'):
                        response += f" (Due: {task['due_date']})"
                    if task.get('project'):
                        response += f" [{task['project']}]"
                    if task.get('semantic_tags'):
                        response += f"\n   🏷️ Tags: {', '.join(task['semantic_tags'])}"
                    response += "\n\n"
                response += "I'll remember this for you."
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "tasks_retrieved": tasks,
                        "search_results": tasks,
                        "search_type": "task_search"
                    },
                    content=response,
                    reason="Successfully retrieved tasks using semantic search"
                )
            else:
                return complete_command(
                    state={
                        "learning_result": "You have no tasks currently.",
                        "content": "You have no tasks currently."
                    },
                    content="You have no tasks currently.",
                    reason="No tasks found"
                )
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Task query failed: {str(e)}"
            )

    async def _handle_file_reference_creation(self, session_id: str, user_input: str, classified_intent=None) -> Command:
        """Handle file reference creation requests using semantic understanding"""
        try:
            # 🚀 SEMANTIC EXTRACTION: Use intelligent understanding instead of patterns
            conversation_context = classified_intent.get("extracted_content", "") if classified_intent else ""
            file_info = await self._extract_file_info_semantically(user_input, conversation_context)
            
            if not file_info or not file_info[0]:  # No file detected
                return create_command(
                    CommandIntents.ERROR,
                    state={"error": "No file reference detected"},
                    reason="Could not extract valid file from request"
                )
            
            file_path, file_type, purpose, project, semantic_tags = file_info
            
            # Store file reference with enhanced semantic information
            ref_id = await memory_agent.save_file_reference_with_context(
                file_path=file_path,
                file_type=file_type,
                purpose=purpose,
                project=project,
                semantic_tags=semantic_tags,
                conversation_context=user_input,
                notes=f"Referenced from: {user_input}"
            )
            
            # Create natural response
            response = f"✅ I'll remember that you're working with {file_path}"
            if purpose:
                response += f" for {purpose}"
            if project:
                response += f" in the {project} project"
            response += ". I can help you find it later."
            
            # Add semantic tags to response for transparency
            if semantic_tags:
                response += f"\n\n🏷️ Tagged as: {', '.join(semantic_tags)}"
            
            return complete_command(
                state={
                    "learning_result": response,
                    "content": response,
                    "file_reference_created": {
                        "file_path": file_path,
                        "file_type": file_type,
                        "purpose": purpose,
                        "project": project,
                        "semantic_tags": semantic_tags,
                        "ref_id": ref_id
                    }
                },
                content=response,
                reason=f"Successfully created semantic file reference: {file_path}"
            )
                
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"File reference creation failed: {str(e)}"
            )
    
    async def _handle_resource_creation(self, session_id: str, user_input: str, classified_intent=None) -> Command:
        """Handle resource/link creation requests using semantic understanding"""
        try:
            # 🚀 SEMANTIC EXTRACTION: Use intelligent understanding instead of patterns
            conversation_context = classified_intent.get("extracted_content", "") if classified_intent else ""
            resource_info = await self._extract_resource_info_semantically(user_input, conversation_context)
            
            if not resource_info or not resource_info[0]:  # No resource detected
                return create_command(
                    CommandIntents.ERROR,
                    state={"error": "No resource detected"},
                    reason="Could not extract valid URL from request"
                )
            
            url, title, description, category, semantic_tags = resource_info
            
            # Store resource with enhanced semantic information
            resource_id = await memory_agent.save_resource_with_context(
                url=url,
                title=title,
                description=description,
                category=category,
                semantic_tags=semantic_tags,
                conversation_context=user_input,
                notes=f"Saved from: {user_input}"
            )
            
            # Create natural response
            response = f"✅ Saved resource: {url}"
            if title:
                response += f" ({title})"
            if category != "general":
                response += f" in {category} category"
            if description:
                response += f"\n\n📄 Description: {description}"
            response += "\n\nI'll remember this for you."
            
            # Add semantic tags to response for transparency
            if semantic_tags:
                response += f"\n\n🏷️ Tagged as: {', '.join(semantic_tags)}"
            
            return complete_command(
                state={
                    "learning_result": response,
                    "content": response,
                    "resource_created": {
                        "url": url,
                        "title": title,
                        "description": description,
                        "category": category,
                        "semantic_tags": semantic_tags,
                        "resource_id": resource_id
                    }
                },
                content=response,
                reason=f"Successfully created semantic resource: {url}"
            )
                
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Resource creation failed: {str(e)}"
            )
    
    async def _handle_forget_fact(self, session_id: str, user_input: str) -> Command:
        """Handle requests to forget/delete facts from memory"""
        try:
            # Extract what to forget from the user input
            fact_query = self._extract_forget_query(user_input)
            
            if not fact_query:
                return complete_command(
                    state={
                        "learning_result": "I need more specific information about what you'd like me to forget.",
                        "content": "I need more specific information about what you'd like me to forget."
                    },
                    content="I need more specific information about what you'd like me to forget.",
                    reason="Forget query too vague"
                )
            
            # Use MemoryAgent to forget the fact
            result = await memory_agent.forget_fact(fact_query)
            
            if result["success"]:
                response = f"✅ I've forgotten {result['deleted_count']} fact(s) about: {fact_query}"
                if result['deleted_count'] > 1:
                    response += f"\n\nDeleted facts:"
                    for fact in result['deleted_facts'][:3]:  # Show first 3
                        response += f"\n• {fact['content'][:100]}..."
                    if result['deleted_count'] > 3:
                        response += f"\n• ... and {result['deleted_count'] - 3} more"
                
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "facts_deleted": result['deleted_facts'],
                        "delete_count": result['deleted_count']
                    },
                    content=response,
                    reason=f"Successfully forgot {result['deleted_count']} facts"
                )
            else:
                response = f"❌ {result['message']}"
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "error": result['message']
                    },
                    content=response,
                    reason="Forget operation failed"
                )
                
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Forget fact failed: {str(e)}"
            )
    
    async def _handle_update_fact(self, session_id: str, user_input: str) -> Command:
        """Handle requests to update/correct facts in memory"""
        try:
            # Extract old and new fact information
            old_query, new_fact = self._extract_update_info(user_input)
            
            if not old_query or not new_fact:
                return complete_command(
                    state={
                        "learning_result": "I need more information about what to update and what the new information should be.",
                        "content": "I need more information about what to update and what the new information should be."
                    },
                    content="I need more information about what to update and what the new information should be.",
                    reason="Update query incomplete"
                )
            
            # Check for conflicts with existing facts
            conflicts = await memory_agent.detect_conflicts(new_fact)
            
            if conflicts:
                # Handle conflicts intelligently
                conflict_resolution = await self._resolve_fact_conflicts(conflicts, new_fact, user_input)
                if conflict_resolution:
                    return conflict_resolution
            
            # Use MemoryAgent to update the fact
            result = await memory_agent.update_fact(old_query, new_fact)
            
            if result["success"]:
                response = f"✅ I've updated the information:\n\n**Old:** {result['old_fact']}\n**New:** {result['new_fact']}"
                
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "old_fact": result['old_fact'],
                        "new_fact": result['new_fact'],
                        "updated_at": result['updated_at']
                    },
                    content=response,
                    reason="Successfully updated fact"
                )
            else:
                response = f"❌ {result['message']}"
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "error": result['message']
                    },
                    content=response,
                    reason="Update operation failed"
                )
                
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Update fact failed: {str(e)}"
            )
    
    async def _resolve_fact_conflicts(self, conflicts: List[Dict[str, Any]], new_fact: str, user_input: str) -> Optional[Command]:
        """Handle conflicts between new fact and existing facts"""
        try:
            if not conflicts:
                return None
            
            # Analyze the conflict
            primary_conflict = conflicts[0]
            existing_fact = primary_conflict.get("content", "")
            conflict_type = primary_conflict.get("conflict_type", "general_contradiction")
            
            response = f"⚠️ **Conflict Detected**\n\n"
            response += f"Your new information conflicts with what I currently know:\n\n"
            response += f"**Existing:** {existing_fact}\n"
            response += f"**New:** {new_fact}\n\n"
            
            if conflict_type == "temporal":
                response += "This looks like updated information. I'll replace the old fact with the new one."
                # Auto-resolve temporal conflicts
                await memory_agent.update_fact(existing_fact, new_fact)
                response += f"\n\n✅ Updated my records with the new information."
            elif conflict_type == "negation":
                response += "This appears to be a correction. I'll update my records."
                # Auto-resolve negations 
                await memory_agent.update_fact(existing_fact, new_fact)
                response += f"\n\n✅ Corrected my records."
            else:
                response += f"**Conflict Type:** {conflict_type.replace('_', ' ').title()}\n\n"
                response += "I'll save the new information and note the discrepancy for future reference."
                # Save the new fact but keep the old one with a note
                await memory_agent.save_fact(f"CONFLICT_NOTED: {new_fact} (conflicts with: {existing_fact})", "conflicts")
                response += f"\n\n✅ Saved the new information with conflict notation."
            
            return complete_command(
                state={
                    "learning_result": response,
                    "content": response,
                    "conflict_resolved": True,
                    "conflict_type": conflict_type,
                    "conflicts": conflicts
                },
                content=response,
                reason="Resolved fact conflict"
            )
            
        except Exception as e:
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Conflict resolution failed: {str(e)}"
            )

    async def _detect_implicit_learning(self, user_input: str, previous_output: str, session_id: str) -> Optional[Command]:
        """Detect implicit learning opportunities in natural conversation"""
        
        learning_opportunities = []
        
        # Check for relationship/collaboration mentions
        relationships = self._extract_relationship_mentions(user_input)
        if relationships:
            learning_opportunities.extend(relationships)
        
        # Check for company/project mentions
        work_context = self._extract_work_context(user_input)
        if work_context:
            learning_opportunities.extend(work_context)
        
        # Check for preference hints
        preferences = self._extract_preference_hints(user_input)
        if preferences:
            learning_opportunities.extend(preferences)
        
        if learning_opportunities:
            # Store all discovered information
            stored_facts = []
            for opportunity in learning_opportunities:
                try:
                    await memory_agent.save_fact(
                        opportunity["fact"], 
                        opportunity["category"]
                    )
                    stored_facts.append(opportunity["fact"])
                except Exception as e:
                    logger.warning(f"Failed to store implicit fact: {e}")
            
            if stored_facts:
                # Create a subtle acknowledgment (not overwhelming)
                response = "Got it! I'll keep that in mind for future conversations."
                
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "implicit_facts": stored_facts,
                        "learning_type": "implicit"
                    },
                    content=response,
                    reason=f"Detected and stored {len(stored_facts)} implicit facts"
                )
        
        return None

    async def _detect_behavioral_learning(self, user_input: str, previous_output: str, session_id: str) -> Optional[Command]:
        """Detect behavioral patterns and style preferences from user corrections"""
        
        # Check if this is a style correction
        style_correction = self._detect_style_correction(user_input, previous_output)
        if style_correction:
            try:
                # Store as a preference
                preference_fact = f"User prefers {style_correction['preference']} style for {style_correction['content_type']}"
                await memory_agent.save_preference(preference_fact, "style_preferences")
                
                response = f"✅ I'll remember that you prefer {style_correction['preference']} style."
                
                return complete_command(
                    state={
                        "learning_result": response,
                        "content": response,
                        "behavioral_learning": style_correction,
                        "learning_type": "behavioral"
                    },
                    content=response,
                    reason=f"Learned style preference: {style_correction['preference']}"
                )
            except Exception as e:
                logger.warning(f"Failed to store behavioral learning: {e}")
        
        return None

    def _extract_relationship_mentions(self, user_input: str) -> List[Dict[str, Any]]:
        """Extract relationship mentions from natural conversation"""
        relationships = []
        user_lower = user_input.lower()
        
        # Pattern: "Sarah and I are working on..."
        collab_patterns = [
            r"(.+?)\s+and\s+(?:i|we)\s+(?:are\s+)?(?:working\s+on|collaborating\s+on|building)",
            r"(?:i|we)\s+(?:work\s+with|collaborate\s+with)\s+(.+?)(?:\s+on|$)",
            r"(.+?)\s+is\s+(?:my|our)\s+(partner|colleague|teammate|collaborator)"
        ]
        
        for pattern in collab_patterns:
            matches = re.findall(pattern, user_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    person = match[0].strip().title()
                    relationship = match[1].strip() if len(match) > 1 else "colleague"
                else:
                    person = match.strip().title()
                    relationship = "colleague"
                
                if len(person) > 1 and person not in ["I", "We", "You"]:
                    relationships.append({
                        "fact": f"{person} is my {relationship}",
                        "category": "relationships",
                        "entities": {"person": person, "relationship": relationship}
                    })
        
        return relationships

    def _extract_work_context(self, user_input: str) -> List[Dict[str, Any]]:
        """Extract work context and project mentions"""
        work_context = []
        
        # Company mentions
        company_patterns = [
            r"(?:at|for)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
            r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:company|agency|studio|team)"
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, user_input)
            for match in matches:
                company = match.strip()
                if len(company) > 2:
                    work_context.append({
                        "fact": f"I work with {company}",
                        "category": "work_context",
                        "entities": {"company": company}
                    })
        
        # Project mentions
        project_patterns = [
            r"(?:working\s+on|building|developing)\s+(?:the\s+)?(.+?)(?:\s+project|$)",
            r"(?:project|app|website|platform)\s+(?:called\s+)?(.+?)(?:\s|$)"
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                project = match.strip()
                if len(project) > 2 and project.lower() not in ["it", "this", "that"]:
                    work_context.append({
                        "fact": f"I'm working on {project}",
                        "category": "projects",
                        "entities": {"project": project}
                    })
        
        return work_context

    def _extract_preference_hints(self, user_input: str) -> List[Dict[str, Any]]:
        """Extract subtle preference hints from user language"""
        preferences = []
        user_lower = user_input.lower()
        
        # Style preferences
        style_hints = [
            r"(?:please\s+)?(?:make\s+it|keep\s+it)\s+(more|less)\s+(.+)",
            r"i\s+(?:prefer|like|want)\s+(.+?)\s+(?:style|approach|way)",
            r"(?:always|usually|typically)\s+(?:make\s+it|keep\s+it)\s+(.+)"
        ]
        
        for pattern in style_hints:
            matches = re.findall(pattern, user_lower)
            for match in matches:
                if isinstance(match, tuple):
                    preference = " ".join(match).strip()
                else:
                    preference = match.strip()
                
                if len(preference) > 2:
                    preferences.append({
                        "fact": f"I prefer {preference} style",
                        "category": "style_preferences",
                        "entities": {"preference": preference}
                    })
        
        return preferences

    def _detect_style_correction(self, user_input: str, previous_output: str) -> Optional[Dict[str, Any]]:
        """Detect when user is making style corrections"""
        user_lower = user_input.lower()
        
        # Common style correction patterns
        correction_patterns = [
            (r"(?:make\s+it|make\s+that)\s+(more|less)\s+(.+)", "style_modification"),
            (r"(?:too|very)\s+(long|short|formal|casual|complex|simple)", "length_preference"),
            (r"(?:please\s+)?(?:shorten|lengthen|simplify|clarify)", "format_preference")
        ]
        
        for pattern, correction_type in correction_patterns:
            match = re.search(pattern, user_lower)
            if match:
                if correction_type == "style_modification":
                    modifier = match.group(1)  # more/less
                    style = match.group(2)     # casual/formal/etc
                    preference = f"{modifier} {style}"
                else:
                    preference = match.group(1) if match.groups() else match.group(0)
                
                return {
                    "preference": preference,
                    "correction_type": correction_type,
                    "content_type": self._infer_content_type(previous_output)
                }
        
        return None

    def _infer_content_type(self, content: str) -> str:
        """Infer what type of content this is for better preference learning"""
        if not content:
            return "general"
        
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["dear", "sincerely", "regards"]):
            return "email"
        elif any(word in content_lower for word in ["team", "update", "status"]):
            return "update"
        elif any(word in content_lower for word in ["meeting", "agenda", "discussion"]):
            return "meeting_content"
        else:
            return "general"

    async def _extract_enhanced_entities(self, content: str) -> Dict[str, Any]:
        """Extract enhanced entities with better name and organization detection"""
        entities = {}
        
        # Extract names (improved pattern)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        if names:
            # Filter out common false positives
            filtered_names = [name for name in names if name not in ["I", "The", "My", "This", "That"]]
            if filtered_names:
                entities["people"] = filtered_names
        
        # Extract organizations with better patterns
        org_patterns = [
            r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:Inc|LLC|Corp|Company|Agency|Studio|Labs?|Team)\b',
            r'\b(?:Folk\s+Devils|23andMe|Google|Apple|Microsoft|Meta)\b'  # Known companies
        ]
        
        organizations = []
        for pattern in org_patterns:
            orgs = re.findall(pattern, content, re.IGNORECASE)
            organizations.extend(orgs)
        
        if organizations:
            entities["organizations"] = organizations
        
        return entities

    def _determine_fact_category(self, entities: Dict[str, Any]) -> str:
        """Determine the appropriate category for a fact based on entities"""
        if entities.get("people"):
            return "relationships"
        elif entities.get("organizations"):
            return "work_context"
        elif "time" in str(entities).lower():
            return "time_tracking"
        else:
            return "general"

    def _create_natural_memory_response(self, fact_content: str, entities: Dict[str, Any]) -> str:
        """Create a natural response for memory storage"""
        if entities.get("people"):
            return f"✅ Got it! I'll remember that about {', '.join(entities['people'])}."
        elif entities.get("organizations"):
            return f"✅ I'll remember that information about {', '.join(entities['organizations'])}."
        else:
            return "✅ I'll remember that. I can recall this information in future conversations."

    def _extract_fact_content(self, user_input: str) -> str:
        """Extract fact content from user input using simple patterns"""
        user_lower = user_input.lower()
        
        # Remove common memory trigger words
        memory_triggers = ['remember', 'store', 'save', 'note that', 'keep in mind', 'fact:']
        
        fact_content = user_input
        for trigger in memory_triggers:
            if trigger in user_lower:
                # Remove the trigger and clean up
                fact_content = user_input.lower().replace(trigger, '').strip()
                # Capitalize first letter
                if fact_content:
                    fact_content = fact_content[0].upper() + fact_content[1:]
                break
        
        return fact_content or user_input
    
    async def _extract_time_info_semantically(self, user_input: str) -> tuple:
        """
        🚀 INTELLIGENT TIME EXTRACTION: Use LLM to understand time logging requests
        NO FALLBACKS - fail clearly if extraction fails.
        """
        try:
            # 🚀 INTELLIGENT MODEL SELECTION
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="LearningAgent", 
                context="time extraction"
            )
            
            logger.info(f"🔍 TIME EXTRACTION MODEL: {selected_model}")
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            extraction_prompt = f"""You are an intelligent time extractor. Analyze this user input to extract time logging information.

USER INPUT: "{user_input}"

Extract time information and respond with ONLY a JSON object:
{{
    "time_detected": true/false,
    "duration_hours": 0.0,
    "task": "what the time was spent on",
    "category": "development|meetings|planning|research|general",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

EXAMPLES:
- "log 2 hours on code review" → duration_hours: 2.0, task: "code review", category: "development"
- "I worked 1.5h on the API documentation" → duration_hours: 1.5, task: "API documentation", category: "development"
- "spent 3 hours in meetings" → duration_hours: 3.0, task: "meetings", category: "meetings"

USER INPUT: "{user_input}"
"""

            logger.info(f"🔍 EXTRACTING TIME INFO: '{user_input[:100]}...'")
            messages = [HumanMessage(content=extraction_prompt)]
            response = await llm.ainvoke(messages)
            
            logger.info(f"🔍 TIME EXTRACTION RAW: {response.content}")
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                logger.info(f"✅ TIME JSON PARSED: {result}")
                
                if not isinstance(result, dict) or not result.get("time_detected"):
                    logger.info(f"❌ NO TIME DETECTED: {result}")
                    return 0, "", "general"
                
                duration = result.get("duration_hours", 0)
                task = result.get("task", "")
                category = result.get("category", "general")
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                
                logger.info(f"✅ TIME EXTRACTED: {duration}h on {task} (category: {category}, confidence: {confidence})")
                logger.info(f"   Reasoning: {reasoning}")
                
                return duration, task, category
                
            except json.JSONDecodeError as e:
                error_msg = f"Time extraction LLM returned invalid JSON: {e}\nRaw response: {response.content}"
                logger.error(f"❌ TIME JSON ERROR: {error_msg}")
                raise ValueError(error_msg)
            
        except Exception as e:
            error_msg = f"Time extraction failed: {type(e).__name__}: {str(e)}"
            logger.error(f"❌ TIME EXTRACTION ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _extract_forget_query(self, user_input: str) -> str:
        """Extract what to forget from user input"""
        user_input_lower = user_input.lower()
        
        # Patterns to extract the content to forget
        forget_patterns = [
            r'forget\s+(?:that\s+)?(.+)',
            r'delete\s+(?:that\s+)?(.+)',
            r'remove\s+(?:that\s+)?(.+)',
            r'(?:i\s+)?don\'t\s+want\s+to\s+remember\s+(?:that\s+)?(.+)',
            r'(?:can\s+you\s+)?(?:please\s+)?(?:forget|delete|remove)\s+(.+)',
        ]
        
        for pattern in forget_patterns:
            match = re.search(pattern, user_input_lower, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                # Clean up common suffixes
                query = re.sub(r'\s+(?:please|thanks?|thank\s+you)$', '', query)
                return query
        
        # If no pattern matches, return the input minus command words
        clean_input = user_input_lower
        for word in ['forget', 'delete', 'remove', 'please', 'can', 'you']:
            clean_input = re.sub(rf'\b{word}\b', '', clean_input)
        
        return clean_input.strip()
    
    def _extract_update_info(self, user_input: str) -> tuple:
        """Extract old and new information from update requests"""
        user_input_lower = user_input.lower()
        
        # Pattern 1: "Update X to Y" or "Change X to Y"
        update_patterns = [
            r'(?:update|change)\s+(.+?)\s+to\s+(.+)',
            r'(?:correct|fix)\s+(.+?)\s+(?:to\s+)?(.+)',
            r'(?:actually|i\s+meant)\s+(.+?)\s+(?:not|instead\s+of)\s+(.+)',
            r'(?:actually|i\s+meant)\s+(.+)',  # Just the correction
        ]
        
        for pattern in update_patterns:
            match = re.search(pattern, user_input_lower, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    old_query = match.group(1).strip()
                    new_fact = match.group(2).strip()
                    return old_query, new_fact
                else:
                    # Single group - assume it's the correction
                    new_fact = match.group(1).strip()
                    return "", new_fact
        
        # Pattern 2: Look for "that" references that might need context resolution
        if 'that' in user_input_lower:
            # Simple extraction - could be enhanced with context resolution
            correction_match = re.search(r'(?:actually|i\s+meant|correct)\s+(.+)', user_input_lower)
            if correction_match:
                new_fact = correction_match.group(1).strip()
                return "that", new_fact
        
        # Fallback: assume entire input is the new fact
        clean_input = user_input
        for word in ['update', 'change', 'correct', 'fix', 'actually', 'i', 'meant', 'please']:
            clean_input = re.sub(rf'\b{word}\b', '', clean_input, flags=re.IGNORECASE)
        
        return "", clean_input.strip()
    
    async def _extract_task_info_semantically(self, user_input: str, conversation_context: str = "") -> tuple:
        """
        🚀 INTELLIGENT TASK EXTRACTION: Use LLM to understand task creation requests in natural conversation
        
        This replaces primitive regex pattern matching with semantic understanding.
        Understands priority, deadlines, and project context automatically.
        """
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on complexity
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="LearningAgent",
                context="task extraction"
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            extraction_prompt = f"""You are an intelligent task extractor. Analyze this conversation to understand what task the user wants to create or add to their to-do list.

CONVERSATION: "{user_input}"
CONTEXT: "{conversation_context}"

Extract task information and respond with ONLY a JSON object:
{{
    "task_detected": true/false,
    "task_description": "clear, actionable task description",
    "priority": "low|medium|high|urgent",
    "due_date": "YYYY-MM-DD or relative date like 'today', 'tomorrow', 'next week', or null",
    "project": "project or context this task belongs to (if mentioned)",
    "semantic_tags": ["tag1", "tag2", "tag3"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of extraction"
}}

EXAMPLES:
- "Add to my todo: Fix the authentication bug" → task: "Fix the authentication bug", priority: "medium"
- "Remind me to review the homepage design urgently" → task: "Review the homepage design", priority: "urgent"
- "I need to call the client tomorrow" → task: "Call the client", due_date: "tomorrow"
- "High priority: Update the API documentation" → task: "Update the API documentation", priority: "high"

Focus on:
1. Extracting a CLEAR, ACTIONABLE task description
2. Understanding urgency and priority from language
3. Detecting deadlines and time references
4. Inferring project context from conversation
5. Generating relevant tags for organization

USER INPUT: "{user_input}"
"""

            logger.info(f"🔍 EXTRACTING TASK INFO: '{user_input[:100]}...'")
            messages = [HumanMessage(content=extraction_prompt)]
            response = await llm.ainvoke(messages)
            logger.info(f"🔍 TASK EXTRACTION RESPONSE: {response.content}")
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                
                # Validate structure
                if not isinstance(result, dict) or not result.get("task_detected"):
                    logger.info(f"❌ NO TASK DETECTED: {result}")
                    return None, None, None, None, None
                
                task_description = result.get("task_description", "")
                priority = result.get("priority", "medium")
                due_date = result.get("due_date")
                project = result.get("project")
                semantic_tags = result.get("semantic_tags", [])
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                
                logger.info(f"✅ TASK EXTRACTED: {task_description} (priority: {priority}, confidence: {confidence})")
                logger.info(f"   Reasoning: {reasoning}")
                
                # Return enhanced task info including semantic tags
                return task_description, priority, due_date, project, semantic_tags
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ TASK EXTRACTION JSON ERROR: {e} - Response: {response.content}")
                return None, None, None, None, None
            
        except Exception as e:
            logger.error(f"❌ TASK EXTRACTION ERROR: {e}")
            return None, None, None, None, None

    
    async def _extract_file_info_semantically(self, user_input: str, conversation_context: str = "") -> tuple:
        """
        🚀 INTELLIGENT FILE EXTRACTION: Use LLM to understand file references in natural conversation
        
        This replaces primitive regex pattern matching with semantic understanding.
        Understands context, purpose, and relationships automatically.
        CRITICAL: Preserves original URLs exactly without transformation.
        """
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on complexity
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="LearningAgent",
                context="file reference extraction"
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            extraction_prompt = f"""You are an intelligent file reference extractor. Analyze this conversation to understand what files the user is referencing or working with.

CRITICAL RULE: PRESERVE ORIGINAL URLs EXACTLY - DO NOT TRANSFORM OR MODIFY URLs IN ANY WAY

CONVERSATION: "{user_input}"
CONTEXT: "{conversation_context}"

Extract file information and respond with ONLY a JSON object:
{{
    "file_detected": true/false,
    "file_path": "EXACT path/to/file.ext or EXACT URL as provided - DO NOT MODIFY",
    "file_type": "config|component|design|documentation|script|image|other",
    "purpose": "what this file is used for (e.g., 'homepage design', 'authentication setup', 'user interface')",
    "project": "project or context this file belongs to (if mentioned)",
    "semantic_tags": ["tag1", "tag2", "tag3"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of extraction"
}}

EXAMPLES:
- "I'm working on config.py for authentication" → file_path: "config.py", purpose: "authentication setup", file_type: "config"
- "Here's the Figma file https://www.figma.com/design/abc123/Project?node-id=1-2&t=xyz" → file_path: "https://www.figma.com/design/abc123/Project?node-id=1-2&t=xyz", purpose: "design file", file_type: "design"
- "The auth component needs updating" → file_path: "auth component", purpose: "user authentication", file_type: "component"

CRITICAL: 
1. DO NOT transform or modify URLs - copy them EXACTLY as provided
2. DO NOT replace URLs with generic placeholders
3. DO NOT remove query parameters or fragments from URLs
4. PRESERVE the complete original URL structure

Focus on:
1. Understanding the PURPOSE of the file (not just the name)
2. Extracting semantic meaning and context
3. Inferring project relationships from conversation
4. Generating relevant tags for retrieval
5. PRESERVING original URLs exactly

USER INPUT: "{user_input}"
"""

            logger.info(f"🔍 EXTRACTING FILE INFO: '{user_input[:100]}...'")
            messages = [HumanMessage(content=extraction_prompt)]
            response = await llm.ainvoke(messages)
            logger.info(f"🔍 FILE EXTRACTION RESPONSE: {response.content}")
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                
                # Validate structure
                if not isinstance(result, dict) or not result.get("file_detected"):
                    logger.info(f"❌ NO FILE DETECTED: {result}")
                    return None, None, None, None, None
                
                file_path = result.get("file_path", "")
                file_type = result.get("file_type", "other")
                purpose = result.get("purpose", "")
                project = result.get("project", "")
                semantic_tags = result.get("semantic_tags", [])
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                
                logger.info(f"✅ FILE EXTRACTED: {file_path} (purpose: {purpose}, confidence: {confidence})")
                logger.info(f"   Reasoning: {reasoning}")
                
                # 🚨 CRITICAL VALIDATION: Ensure URL preservation
                if "figma.com" in user_input.lower() and "figma.com" in file_path.lower():
                    # Extract original URL from user input to ensure it's preserved
                    import re
                    url_pattern = r'https?://[^\s]+'
                    original_urls = re.findall(url_pattern, user_input)
                    if original_urls:
                        original_url = original_urls[0]  # Take the first URL found
                        if original_url != file_path:
                            logger.warning(f"⚠️ URL MISMATCH DETECTED - Using original URL")
                            logger.warning(f"   Original: {original_url}")
                            logger.warning(f"   LLM returned: {file_path}")
                            file_path = original_url  # Use the original URL
                
                # Return enhanced file info including semantic tags
                return file_path, file_type, purpose, project, semantic_tags
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ FILE EXTRACTION JSON ERROR: {e} - Response: {response.content}")
                return None, None, None, None, None
            
        except Exception as e:
            logger.error(f"❌ FILE EXTRACTION ERROR: {e}")
            return None, None, None, None, None

    
    async def _extract_resource_info_semantically(self, user_input: str, conversation_context: str = "") -> tuple:
        """
        🚀 INTELLIGENT RESOURCE EXTRACTION: Use LLM to understand resource/link saving requests in natural conversation
        
        This replaces primitive regex pattern matching with semantic understanding.
        Understands purpose, categorization, and context automatically.
        """
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on complexity
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="LearningAgent",
                context="resource extraction"
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            extraction_prompt = f"""You are an intelligent resource extractor. Analyze this conversation to understand what URL/link/resource the user wants to save or bookmark.

CONVERSATION: "{user_input}"
CONTEXT: "{conversation_context}"

Extract resource information and respond with ONLY a JSON object:
{{
    "resource_detected": true/false,
    "url": "full URL of the resource",
    "title": "descriptive title for the resource (if mentioned or can be inferred)",
    "description": "what this resource is about or why it's useful",
    "category": "tool|reference|documentation|tutorial|inspiration|design|article|news|other",
    "semantic_tags": ["tag1", "tag2", "tag3"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of extraction"
}}

EXAMPLES:
- "Save this link: https://figma.com/..." → url: "https://figma.com/...", category: "design", description: "design file"
- "Bookmark https://grafana.com for later research" → url: "https://grafana.com", category: "tool", description: "for later research"
- "Here's a good tutorial: https://react.dev" → url: "https://react.dev", category: "tutorial", description: "React tutorial"

Focus on:
1. Extracting the COMPLETE URL accurately
2. Understanding the PURPOSE and context of saving
3. Categorizing based on domain and content type
4. Inferring title and description from context
5. Generating relevant tags for organization

USER INPUT: "{user_input}"
"""

            logger.info(f"🔍 EXTRACTING RESOURCE INFO: '{user_input[:100]}...'")
            messages = [HumanMessage(content=extraction_prompt)]
            response = await llm.ainvoke(messages)
            logger.info(f"🔍 RESOURCE EXTRACTION RESPONSE: {response.content}")
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                
                # Validate structure
                if not isinstance(result, dict) or not result.get("resource_detected"):
                    logger.info(f"❌ NO RESOURCE DETECTED: {result}")
                    return None, None, None, None, None
                
                url = result.get("url", "")
                title = result.get("title")
                description = result.get("description")
                category = result.get("category", "other")
                semantic_tags = result.get("semantic_tags", [])
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                
                logger.info(f"✅ RESOURCE EXTRACTED: {url} (category: {category}, confidence: {confidence})")
                logger.info(f"   Reasoning: {reasoning}")
                
                # Return enhanced resource info including semantic tags
                return url, title, description, category, semantic_tags
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ RESOURCE EXTRACTION JSON ERROR: {e} - Response: {response.content}")
                return None, None, None, None, None
            
        except Exception as e:
            logger.error(f"❌ RESOURCE EXTRACTION ERROR: {e}")
            return None, None, None, None, None


    async def _handle_semantic_query(self, session_id: str, user_input: str, classified_intent=None) -> Command:
        """
        🚀 SEMANTIC QUERY HANDLER: Understand natural language queries about files, tasks, and resources
        
        This replaces keyword-based searches with intelligent semantic understanding.
        Examples: "what file am I working on for homepage" → finds Figma design files
        """
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on complexity
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="LearningAgent",
                context="semantic query"
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            query_prompt = f"""You are an intelligent query router. Analyze this user query and determine what they're looking for.

USER QUERY: "{user_input}"

Classify the query and respond with ONLY a JSON object:
{{
    "query_type": "file_search|task_search|resource_search|general_search",
    "search_intent": "what the user is specifically looking for",
    "search_terms": "optimized search terms",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

QUERY TYPES:
- file_search: Looking for files, documents, designs, code files, etc.
- task_search: Looking for tasks, to-dos, what needs to be done
- resource_search: Looking for links, tools, resources, references
- general_search: General information, facts, preferences

EXAMPLES:
- "what Figma files am I working on" → file_search, intent: "Figma design files", terms: "Figma design files"
- "show me my design files for homepage" → file_search, intent: "homepage design files", terms: "homepage design files"
- "what file am I working on for homepage" → file_search, intent: "homepage files", terms: "homepage files"
- "what do I need to do today" → task_search, intent: "pending tasks", terms: "today pending"
- "show me design tools" → resource_search, intent: "design tools", terms: "design tools"
- "what did I say about Sarah" → general_search, intent: "facts about Sarah", terms: "Sarah"

USER INPUT: "{user_input}"
"""

            logger.info(f"🔍 QUERYING SEMANTIC SEARCH: '{user_input[:100]}...'")
            messages = [HumanMessage(content=query_prompt)]
            response = await llm.ainvoke(messages)
            logger.info(f"🔍 SEMANTIC SEARCH RESPONSE: {response.content}")
            
            # Parse JSON response
            import json
            try:
                classification = json.loads(response.content)
                
                query_type = classification.get("query_type", "general_search")
                search_intent = classification.get("search_intent", "")
                search_terms = classification.get("search_terms", user_input)
                confidence = classification.get("confidence", 0.0)
                reasoning = classification.get("reasoning", "")
                
                logger.info(f"✅ QUERY CLASSIFIED: {query_type} - {search_intent} (confidence: {confidence})")
                logger.info(f"   Reasoning: {reasoning}")
                
                # Route to appropriate semantic search method
                conversation_context = classified_intent.get("extracted_content", "") if classified_intent else ""
                
                if query_type == "file_search":
                    results = await memory_agent.find_files_by_intent(search_terms, conversation_context)
                    return await self._format_file_search_response(results, search_intent, user_input)
                
                elif query_type == "task_search":
                    results = await memory_agent.find_tasks_by_intent(search_terms, conversation_context)
                    return await self._format_task_search_response(results, search_intent, user_input)
                
                elif query_type == "resource_search":
                    results = await memory_agent.find_resources_by_intent(search_terms, conversation_context)
                    return await self._format_resource_search_response(results, search_intent, user_input)
                
                else:  # general_search
                    # Use existing fact search for general queries
                    results = await memory_agent.search_facts(search_terms)
                    return await self._format_general_search_response(results, search_intent, user_input)
                
            except json.JSONDecodeError as e:
                error_msg = f"Semantic query LLM returned invalid JSON: {e}\nRaw response: {response.content}"
                logger.error(f"❌ SEMANTIC QUERY JSON ERROR: {error_msg}")
                raise ValueError(error_msg)
            
        except Exception as e:
            logger.error(f"❌ SEMANTIC QUERY HANDLING ERROR: {e}")
            return create_command(
                CommandIntents.ERROR,
                state={"error": str(e)},
                reason=f"Semantic query failed: {str(e)}"
            )
    
    async def _format_file_search_response(self, results: List[Dict[str, Any]], intent: str, query: str) -> Command:
        """Format file search results into a natural response with enhanced design file support"""
        if not results:
            response = f"I couldn't find any files related to '{intent}'. Would you like me to help you save file references when you mention them?"
        else:
            response = f"Here are the files I found for '{intent}':\n\n"
            for i, file_info in enumerate(results, 1):
                file_path = file_info.get('file_path', '')
                
                # Check if it's a design file (Figma, etc.)
                if any(domain in file_path.lower() for domain in ['figma.com', 'sketch.com', 'adobe.com']):
                    response += f"{i}. 🎨 **Design File**: {file_path}"
                else:
                    response += f"{i}. 📄 **{file_path}**"
                
                if file_info.get('purpose'):
                    response += f"\n   Purpose: {file_info['purpose']}"
                if file_info.get('project'):
                    response += f"\n   Project: {file_info['project']}"
                if file_info.get('semantic_tags'):
                    response += f"\n   🏷️ Tags: {', '.join(file_info['semantic_tags'])}"
                if file_info.get('context'):
                    response += f"\n   📝 Context: {file_info['context']}"
                response += "\n\n"
        
        return complete_command(
            state={
                "learning_result": response,
                "content": response,
                "search_results": results,
                "search_type": "file_search",
                "is_search_result": True
            },
            content=response,
            reason=f"Found {len(results)} files for semantic query"
        )
    
    async def _format_task_search_response(self, results: List[Dict[str, Any]], intent: str, query: str) -> Command:
        """Format task search results into a natural response"""
        if not results:
            response = f"I couldn't find any tasks related to '{intent}'. Would you like me to help you create some tasks?"
        else:
            response = f"Here are the tasks I found for '{intent}':\n\n"
            for i, task_info in enumerate(results, 1):
                response += f"{i}. **{task_info['task']}**"
                if task_info.get('priority') != 'medium':
                    response += f" (Priority: {task_info['priority']})"
                if task_info.get('due_date'):
                    response += f" - Due: {task_info['due_date']}"
                if task_info.get('project'):
                    response += f" [{task_info['project']}]"
                if task_info.get('semantic_tags'):
                    response += f"\n   🏷️ Tags: {', '.join(task_info['semantic_tags'])}"
                response += "\n\n"
        
        return complete_command(
            state={
                "learning_result": response,
                "content": response,
                "search_results": results,
                "search_type": "task_search",
                "is_search_result": True
            },
            content=response,
            reason=f"Found {len(results)} tasks for semantic query"
        )
    
    async def _format_resource_search_response(self, results: List[Dict[str, Any]], intent: str, query: str) -> Command:
        """Format resource search results into a natural response"""
        if not results:
            response = f"I couldn't find any resources related to '{intent}'. Would you like me to help you save resources when you mention them?"
        else:
            response = f"Here are the resources I found for '{intent}':\n\n"
            for i, resource_info in enumerate(results, 1):
                response += f"{i}. **{resource_info['url']}**"
                if resource_info.get('title'):
                    response += f" - {resource_info['title']}"
                if resource_info.get('description'):
                    response += f"\n   📄 {resource_info['description']}"
                if resource_info.get('category') != 'general':
                    response += f" ({resource_info['category']})"
                if resource_info.get('semantic_tags'):
                    response += f"\n   🏷️ Tags: {', '.join(resource_info['semantic_tags'])}"
                response += "\n\n"
        
        return complete_command(
            state={
                "learning_result": response,
                "content": response,
                "search_results": results,
                "search_type": "resource_search",
                "is_search_result": True
            },
            content=response,
            reason=f"Found {len(results)} resources for semantic query"
        )
    
    async def _format_general_search_response(self, results: List[Dict[str, Any]], intent: str, query: str) -> Command:
        """Format general search results into a natural response"""
        if not results:
            response = f"I couldn't find any information about '{intent}' in my memory."
        else:
            response = f"Here's what I found about '{intent}':\n\n"
            for i, result in enumerate(results, 1):
                content = result.get('content', '')
                response += f"{i}. {content}\n\n"
        
        return complete_command(
            state={
                "learning_result": response,
                "content": response,
                "search_results": results,
                "search_type": "general_search",
                "is_search_result": True
            },
            content=response,
            reason=f"Found {len(results)} facts for semantic query"
        )

# Create instance for easy import
learning_agent = IntelligentLearningAgent() 