from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig

from .tone_of_voice import get_andrew_voice
from .tools import AVAILABLE_TOOLS, analyze_image, get_memory_context, get_session_summary
from .commands import Command, CommandIntents, complete_command, error_command
from .shared_context import SharedContext
from .model_selector import model_selector
from config import settings
import logging

logger = logging.getLogger(__name__)

class WriterAgent(Runnable):
    """
    WriterAgent for creating new content in Andrew's voice.
    
    Now returns clean, structured outputs without metadata mixed in.
    Uses shared context for memory access instead of direct memory calls.
    """
    
    def __init__(self):
        # Note: Model will be selected dynamically based on request complexity
        self.base_llm_config = {
            "api_key": settings.OPENAI_API_KEY,
            "temperature": settings.WRITER_TEMP,
            "streaming": settings.STREAM
        }
        self.voice = get_andrew_voice()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for creating content."""
        voice = self.voice
        
        return f"""You are the Writer for Son of Andrew. Your job is to create content that sounds like Andrew wrote it.

ANDREW'S VOICE: {voice['core_voice']['tone']}
STYLE: {voice['core_voice']['style']}
ENERGY: {voice['core_voice']['energy']}

CORE PRINCIPLES:
{chr(10).join(f"- {principle}" for principle in voice['core_principles'])}

STYLE REQUIREMENTS:
{voice['writing_rules']['style_rules']}

FORBIDDEN ELEMENTS (never use these):
{chr(10).join(f"- {element}" for element in voice['writing_rules']['forbidden_elements'])}

WRITING APPROACH:
1. Be direct and punchy
2. Use simple, clear language
3. Avoid jargon and corporate speak
4. No fluff or unnecessary words
5. Sound human and authentic
6. Match Andrew's energy and tone

EXAMPLES OF ANDREW'S VOICE:
{chr(10).join(f"- {example}" for example in voice['example_responses'])}

CRITICAL: Return ONLY the content requested. Do not include:
- Headers or labels
- Explanations of your approach
- Metadata or system messages
- Analysis or commentary

Just return the clean content that fulfills the request."""

    async def ainvoke(
        self, 
        input: Dict[str, Any], 
        config: Optional[RunnableConfig] = None
    ) -> Command:
        """Main async entry point for the WriterAgent."""
        
        prompt = input.get("prompt", "")
        context = input.get("context", "")
        original_request = input.get("original_request", prompt)
        shared_context = input.get("shared_context")  # SharedContext if available
        
        if not prompt.strip():
            return error_command(
                state={"error": "No prompt provided for writing"},
                reason="Writer needs a prompt to create content"
            )
        
        # Build the writing prompt
        context_section = f"\nCONTEXT: {context}" if context else ""
        
        user_message = f"""Create content for this request:

REQUEST: {prompt}{context_section}

Remember: Return ONLY the content requested, nothing else."""

        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=user_message)
        ]
        
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on request complexity
            selected_model = model_selector.select_model(
                user_input=prompt,
                agent_name="WriterAgent",
                context=context
            )
            
            # Create LLM with selected model and bind tools
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            llm_with_tools = llm.bind_tools([
                analyze_image,
                get_memory_context,
                get_session_summary
            ])
            
            response = await llm_with_tools.ainvoke(messages)
            created_content = response.content.strip()
            
            # Return clean command with structured output
            return complete_command(
                state={
                    "created_content": created_content,
                    "prompt": prompt,
                    "context_used": context
                },
                reason="Content successfully created in Andrew's voice",
                content=created_content,  # Clean user-facing content
                metadata={
                    "request_type": "content_creation",
                    "prompt_used": prompt,
                    "content_length": len(created_content)
                }
            )
            
        except Exception as e:
            logger.error(f"Writer failed: {e}")
            return error_command(
                state={"error": str(e), "prompt": prompt},
                reason=f"Writer encountered an error: {str(e)}"
            )

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Command:
        """Legacy sync method - redirects to async implementation."""
        import asyncio
        try:
            return asyncio.run(self.ainvoke(input, config))
        except Exception as e:
            return error_command(
                state={"error": str(e)},
                reason="Writer agent sync method failed - use async ainvoke instead"
            )

# Create instance for easy import
writer_agent = WriterAgent() 