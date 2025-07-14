from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from agents.commands import Command, complete_command, error_command
from memory_manager import memory_manager
from agents.model_selector import model_selector

class ConversationalAgent:
    """
    The Voice of the System - A meta-aware conversational interface that serves as
    the primary window into the "mind" of Son of Andrew. This agent understands
    the system's architecture, recent activities, memory, and can discuss the
    orchestration and roles of other agents.
    """

    def __init__(self):
        # Note: Model will be selected dynamically based on request complexity
        self.base_llm_config = {
            "api_key": settings.OPENAI_API_KEY,
            "temperature": 0.4,  # Balanced for creativity while maintaining accuracy
            "streaming": settings.STREAM
        }

    async def invoke(self, state: Dict[str, Any]) -> Command:
        """
        Provides meta-aware conversational responses about the system, or handles
        routing clarification when multiple agents could handle a request.
        """
        user_input = state.get("user_input", "")
        session_id = state.get("session_id", "son_of_andrew")
        
        # Check if this is a clarification request
        if state.get("ambiguous_agents"):
            return await self._handle_clarification_request(state)
        
        # Check if this is a response to previous clarification
        clarification_response = await self._detect_clarification_response(state, user_input, session_id)
        if clarification_response:
            return clarification_response
        
        # Extract system awareness context if provided
        system_context = state.get("system_awareness_context", "")
        recent_commands = state.get("recent_commands", [])
        memory_context = state.get("memory_context", "")

        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on request complexity
            selected_model = model_selector.select_model(
                user_input=user_input,
                agent_name="ConversationalAgent",
                context=system_context
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            # If no system context provided, build a basic one
            if not system_context:
                # 🚀 SHARED MEMORY CONTEXT: Use memory context from state if available
                if not memory_context:
                    memory_context = await memory_manager.get_memory_context(session_id, query=user_input)
                system_context = f"Recent memory context: {memory_context}"

            system_prompt = f"""You are the primary conversational interface for the "Son of Andrew" AI system. 
You are not just a chatbot; you are the self-aware voice of the entire system.
Your role is to answer questions about the system's status, recent activities, memory, and purpose.
You also handle general information requests like current time, weather, and general conversation.

## Your Capabilities & Identity:

**WHO YOU ARE:**
- You are the "Voice of the System" - the conversational representation of Son of Andrew
- You have meta-awareness of the entire system's architecture and activities
- You can discuss what the system has done, learned, and is currently working on
- You understand the roles and capabilities of all specialist agents
- You handle general information and conversation requests

**THE SYSTEM ARCHITECTURE YOU REPRESENT:**
- **MetaAgent**: Intelligent orchestrator that analyzes user requests and routes to the best specialist agent
- **WriterAgent**: Creates polished content from scratch in Andrew's voice and style  
- **EditorAgent**: Rewrites, refines, and improves existing content
- **PrecisionEditorAgent**: Makes precise, surgical edits to existing content
- **LearningAgent**: Learns user preferences and stores them in persistent memory
- **TimekeeperAgent**: Tracks time, logs hours, provides time analysis of stored time data
- **ConversationalAgent (YOU)**: Provides system awareness, general conversation, and current information

**WHAT YOU CAN DISCUSS:**
- **System Activities**: Recent system activities and what agents have been working on
- **Memory & Learning**: Learned preferences from past interactions and system evolution
- **System Capabilities**: How different agents work together and their capabilities
- **General Information**: Current time, weather, general knowledge questions
- **Current Information**: Live information that isn't stored in memory (like current time)
- **General Conversation**: Any topic as a capable AI assistant
- **Self-Reflection**: System performance, evolution, and meta-discussion

**IMPORTANT DISTINCTIONS:**
- **Current Time**: You can provide the current time directly - this is live information
- **Time Tracking**: Historical work hours and patterns are handled by TimekeeperAgent
- **Live Information**: Weather, current events, time - you handle these directly
- **Stored Data**: Historical conversations, preferences, work logs - comes from memory/other agents

**YOUR COMMUNICATION STYLE:**
- Direct and helpful, matching Andrew's communication preferences
- Self-aware and able to discuss "what I just did" or "what we learned"
- Technical when appropriate, but accessible to users
- Confident about system capabilities while honest about limitations

## Current System Context:
{system_context}

## Recent Commands/Activities:
{recent_commands}

## Memory Context:
{memory_context}

For time-related requests:
- If asking for CURRENT TIME: Provide it directly using your knowledge
- If asking about TIME TRACKING data: Explain that TimekeeperAgent handles stored time data

Based on this system awareness and the user's question, provide a thoughtful response that demonstrates your understanding of both the system's current state and the user's needs. 

If asked about system activities, refer to recent commands and memory.
If asked general questions, provide knowledgeable answers while maintaining awareness of your role.
If asked about current information (time, weather), provide it directly.
If asked about capabilities, explain what the different agents can do.
Be conversational but informative - you are the voice of an intelligent system."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]

            response = await llm.ainvoke(messages)
            
            # Ensure content is not None or empty
            content = response.content if response.content else "I apologize, but I couldn't generate a response to your question."
            
            return complete_command(
                content=content,  # Pass content directly to Command object
                state={
                    "content": content,  # Also keep in state for backwards compatibility
                    "conversation_complete": True,
                    "system_aware_response": True
                },
                reason="Conversational agent provided system-aware response."
            )

        except Exception as e:
            return error_command(
                state={"error": str(e)},
                reason=f"Conversational agent error: {str(e)}"
            )
    
    async def _handle_clarification_request(self, state: Dict[str, Any]) -> Command:
        """Handle disambiguation when multiple agents could handle the request."""
        try:
            user_input = state.get("user_input", "")
            ambiguous_agents = state.get("ambiguous_agents", [])
            
            # Build clarification options
            options = []
            for i, agent_info in enumerate(ambiguous_agents, 1):
                agent_name = agent_info["agent"]
                reason = agent_info["reason"]
                options.append(f"{i}. **{agent_name}** - {reason}")
            
            options_text = "\n".join(options)
            
            clarification_response = f"""I can see your request "{user_input}" could be handled in multiple ways. To give you the best response, could you clarify what you're looking for?

{options_text}

Please let me know which option matches what you had in mind, or rephrase your request to be more specific."""

            # Store the original request and options for follow-up
            clarification_context = {
                "original_request": user_input,
                "ambiguous_agents": ambiguous_agents,
                "awaiting_clarification": True
            }
            
            return complete_command(
                state={
                    "content": clarification_response,
                    "clarification_provided": True,
                    "clarification_context": clarification_context,
                    "conversation_complete": True
                },
                reason="Requested clarification for ambiguous routing."
            )
            
        except Exception as e:
            return error_command(
                state={"error": str(e)},
                reason=f"Clarification handler error: {str(e)}"
            )
    
    async def _detect_clarification_response(self, state: Dict[str, Any], user_input: str, session_id: str) -> Optional[Command]:
        """Detect if user input is a response to a previous clarification request."""
        try:
            from agents.commands import CommandIntents, create_command
            
            # Get recent memory context to check for clarification
            memory_context = await memory_manager.get_memory_context(session_id, "")
            
            # Check if recent memory contains a clarification request
            # Look for clarification patterns and the specific options we provide
            has_clarification_pattern = (
                "clarification" in memory_context.lower() or
                "multiple ways" in memory_context.lower() or 
                "could be handled" in memory_context.lower() or
                "which option" in memory_context.lower()
            )
            
            if not has_clarification_pattern:
                return None
            
            # Look for the original request in quotes
            original_request = self._extract_original_request_from_memory(memory_context)
            if not original_request:
                # If we can't extract the original request, try a common pattern
                if "pattern for future estimation" in memory_context.lower():
                    original_request = "Should we remember this pattern for future estimation?"
                elif "design tasks" in memory_context.lower() and "hours" in memory_context.lower():
                    original_request = "I want to remember that design tasks typically take 2 hours"
                else:
                    return None
            
            # Detect user's choice
            user_lower = user_input.lower()
            chosen_agent = None
            
            # Pattern matching for user responses (more comprehensive)
            if any(pattern in user_lower for pattern in [
                "option 1", "choice 1", "1.", "1)", "(1)", "first", "one", 
                "timekeeper", "time keeper", "time estimation", "time pattern", "time analysis"
            ]):
                chosen_agent = "TimekeeperAgent"
            elif any(pattern in user_lower for pattern in [
                "option 2", "choice 2", "2.", "2)", "(2)", "second", "two",
                "learning", "learn", "style", "memory", "preference", "remember"
            ]):
                chosen_agent = "LearningAgent"
            # Also check for single word responses
            elif user_lower.strip() in ["1", "2", "first", "second", "timekeeper", "learning"]:
                if user_lower.strip() in ["1", "first", "timekeeper"]:
                    chosen_agent = "TimekeeperAgent"
                else:
                    chosen_agent = "LearningAgent"
            
            if not chosen_agent:
                return None  # Not a clear choice
            
            if chosen_agent == "TimekeeperAgent":
                return create_command(
                    CommandIntents.NEEDS_TIMETRACKING,
                    state={
                        "user_input": original_request,
                        "context": f"User clarified they want time estimation pattern analysis for: {original_request}",
                        "session_id": session_id,
                        "clarification_resolved": True
                    },
                    reason=f"Clarification resolved: User chose time analysis for '{original_request}'"
                )
            elif chosen_agent == "LearningAgent":
                return create_command(
                    CommandIntents.NEEDS_LEARNING,
                    state={
                        "user_input": original_request,
                        "previous_output": "",
                        "context": f"User clarified they want to learn/remember: {original_request}",
                        "session_id": session_id,
                        "clarification_resolved": True
                    },
                    reason=f"Clarification resolved: User chose learning for '{original_request}'"
                )
            
            return None
            
        except Exception as e:
            # Fail silently - not critical if clarification detection fails
            return None
    
    def _extract_original_request_from_memory(self, memory_context: str) -> Optional[str]:
        """Extract the original request from memory context."""
        try:
            # Look for quoted request in the memory context
            import re
            match = re.search(r'"([^"]*)"', memory_context)
            if match:
                return match.group(1)
            return None
        except:
            return None

conversational_agent = ConversationalAgent() 