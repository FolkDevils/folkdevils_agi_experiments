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

class EditorAgent(Runnable):
    """
    EditorAgent for improving content in Andrew's voice.
    
    Now returns clean, structured outputs without metadata mixed in.
    Uses shared context for memory access instead of direct memory calls.
    """
    
    def __init__(self):
        # Note: Model will be selected dynamically based on request complexity
        self.base_llm_config = {
            "api_key": settings.OPENAI_API_KEY,
            "temperature": settings.EDITOR_TEMP,
            "streaming": settings.STREAM
        }
        self.voice = get_andrew_voice()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for editing content."""
        voice = self.voice
        
        return f"""You are the Editor for Son of Andrew. Your job is to improve content to match Andrew's voice.

ANDREW'S VOICE: {voice['core_voice']['tone']}
STYLE: {voice['core_voice']['style']}
ENERGY: {voice['core_voice']['energy']}

CORE PRINCIPLES TO ENFORCE:
{chr(10).join(f"- {principle}" for principle in voice['core_principles'])}

STYLE REQUIREMENTS:
{voice['writing_rules']['style_rules']}

FORBIDDEN ELEMENTS TO REMOVE:
{chr(10).join(f"- {element}" for element in voice['writing_rules']['forbidden_elements'])}

EDITING APPROACH:
1. PRESERVE ALL IMPORTANT INFORMATION (URLs, specific details, contact info, deadlines)
2. Remove only true fluff and filler words
3. Make sentences more direct while keeping necessary details
4. Eliminate jargon and corporate speak
5. Check for forbidden elements (especially em dashes)
6. Ensure the tone matches Andrew's voice
7. Keep the core message AND all supporting details
8. Maintain original formatting (line breaks, structure) when appropriate

IMPORTANT: DO NOT remove or shorten:
- URLs and links (especially Figma links, GitHub links, etc.)
- Specific project names or details
- Deadlines or time-sensitive information
- Contact information or instructions
- Technical specifications or requirements
- File paths, API endpoints, or technical references
- Multi-line content structure when it serves a purpose

CRITICAL: Return ONLY the improved content. Do not include:
- Headers like "EDITED CONTENT:"
- Explanations of changes made
- Metadata or system messages
- Analysis or commentary

Just return the clean, improved version with ALL original information preserved."""

    def _get_modification_guidance(self, modification_request: str) -> str:
        """Provide specific guidance for different types of modifications."""
        request_lower = modification_request.lower()
        
        if 'friendly' in request_lower or 'friendlier' in request_lower:
            return """- Add warmer, more personal language
- Include phrases like "Hope this helps!" or "Let me know if you have questions"
- Use more welcoming greetings like "Hi there!" instead of just "Hi"
- Add positive expressions or enthusiasm where appropriate
- Consider adding an emoji if it fits the context (😊 or similar)
- Make the tone more conversational and warm while keeping it professional"""
        
        elif 'shorter' in request_lower:
            return """CRITICAL: You MUST significantly reduce the length of this content. This is a mandatory requirement.

SPECIFIC ACTIONS TO TAKE:
- Cut the content by AT LEAST 30-50% in length
- Remove ALL unnecessary words, filler phrases, and redundant information
- Combine multiple sentences into single, concise statements
- Remove repetitive explanations or details
- Use bullet points or lists instead of long paragraphs when possible
- Remove transitional phrases and verbose language
- Replace wordy phrases with direct, simple language
- Remove any sentences that don't add essential value
- Focus on the core message and eliminate supporting details that aren't critical

EXAMPLES OF SHORTENING:
- "I wanted to let you know that..." → "FYI:"
- "Let me know your thoughts about this" → "Thoughts?"
- "Both versions are on Slack: the confetti version is oversized, and the non-confetti version fits the size limit" → "Both versions on Slack - confetti version too large, non-confetti fits"
- Long explanations should become single sentences

MANDATORY: The final result must be NOTICEABLY shorter than the original. If it's not significantly shorter, try again."""
        
        elif 'longer' in request_lower:
            return """- Add more context and detail
- Expand on key points
- Include helpful background information
- Add examples if relevant
- Provide more comprehensive explanations
- Ensure no important details are missing"""
        
        elif 'formal' in request_lower:
            return """- Use more professional language
- Avoid contractions (use "I will" instead of "I'll")
- Use complete sentences and proper structure
- Remove casual expressions
- Use more traditional business language
- Maintain respect and professionalism"""
        
        elif 'casual' in request_lower:
            return """- Use more relaxed, conversational language
- Include contractions where natural
- Use simpler, everyday words
- Make it sound more like talking to a friend
- Remove overly formal phrases
- Keep it approachable and easy-going"""
        
        elif 'professional' in request_lower:
            return """- Use business-appropriate language
- Focus on clarity and precision
- Remove personal opinions or casual remarks
- Use industry-standard terminology appropriately
- Maintain a competent, reliable tone
- Ensure all information is accurate and complete"""
        
        else:  # general improvement
            return """- Improve clarity and readability
- Ensure the message is complete and well-structured
- Remove any awkward phrasing
- Make sure the tone matches Andrew's voice
- Check for any forbidden elements (em dashes, etc.)
- Ensure the message serves its purpose effectively"""

    async def ainvoke(
        self, 
        input: Dict[str, Any], 
        config: Optional[RunnableConfig] = None
    ) -> Command:
        """Main async entry point for the EditorAgent."""
        
        content = input.get("content", "")
        modification_request = input.get("modification_request", "general improvement")
        context = input.get("context")  # SharedContext if available
        
        # Handle case where content comes from previous step results
        if not content and context and isinstance(context, SharedContext):
            # Check if we have content from a previous step
            content = context.get_previous_result("retrieved_content") or ""
        
        if not content.strip():
            return error_command(
                state={"error": "No content provided to edit"},
                reason="Editor needs content to improve"
            )
        
        # Build the editing prompt with specific modification guidance
        modification_guidance = self._get_modification_guidance(modification_request)
        
        # Build different prompt based on request type
        if 'shorter' in modification_request.lower():
            user_message = f"""CRITICAL TASK: Make this content SIGNIFICANTLY SHORTER. This is mandatory.

ORIGINAL CONTENT TO SHORTEN:
{content}

MODIFICATION REQUEST: {modification_request}

SPECIFIC SHORTENING REQUIREMENTS:
{modification_guidance}

VERIFICATION: Before responding, count the words in your output vs the original. Your response must be AT LEAST 30% shorter than the original. If it's not significantly shorter, revise it again.

Return ONLY the shortened content, nothing else."""
        else:
            user_message = f"""Please improve this content according to the modification request:

CONTENT TO EDIT:
{content}

MODIFICATION REQUEST: {modification_request}

SPECIFIC GUIDANCE FOR THIS MODIFICATION:
{modification_guidance}

IMPORTANT: You must make actual changes to the content. Do not return the same text. Make it noticeably different according to the modification request while maintaining Andrew's voice.

Remember: Return ONLY the improved content, nothing else."""

        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=user_message)
        ]
        
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on request complexity
            selected_model = model_selector.select_model(
                user_input=modification_request,
                agent_name="EditorAgent",
                context=content
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
            edited_content = response.content.strip()
            
            # Return clean command with structured output
            return complete_command(
                state={
                    "edited_content": edited_content,
                    "original_content": content,
                    "modification_applied": modification_request
                },
                reason="Content successfully edited in Andrew's voice",
                content=edited_content,  # Clean user-facing content
                metadata={
                    "changes_applied": modification_request,
                    "original_length": len(content),
                    "edited_length": len(edited_content)
                }
            )
            
        except Exception as e:
            logger.error(f"Editor failed: {e}")
            return error_command(
                state={"error": str(e), "content": content},
                reason=f"Editor encountered an error: {str(e)}"
            )

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Command:
        """Legacy sync method - redirects to async implementation with proper event loop handling."""
        import asyncio
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, can't use asyncio.run()
                logger.warning("⚠️ EditorAgent.invoke() called from async context - this should use ainvoke() instead")
                # Return a command that explains the issue
                return error_command(
                    state={"error": "Async context detected", "content": input.get("content", "")},
                    reason="EditorAgent.invoke() called from async context - use ainvoke() instead"
                )
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self.ainvoke(input, config))
        except Exception as e:
            logger.error(f"EditorAgent.invoke() failed: {e}")
            return error_command(
                state={"error": str(e)},
                reason=f"Editor agent sync method failed: {str(e)}"
            )

# Create instance for easy import
editor_agent = EditorAgent() 