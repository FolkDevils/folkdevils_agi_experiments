from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig

from .working_text_manager import working_text_manager, WorkingTextState
from .instruction_parser import instruction_parser, ParsedInstruction, InstructionType
from .edit_history_manager import edit_history_manager
from .semantic_matcher import semantic_matcher
from .confirmation_manager import confirmation_manager
from .tone_of_voice import get_andrew_voice
from .commands import Command, CommandIntents, complete_command, error_command
from .model_selector import model_selector
from config import settings
import logging

logger = logging.getLogger(__name__)

class PrecisionEditorAgent(Runnable):
    """
    Phase I Precision Editor Agent
    
    Performs reliable, instruction-aware editing with persistent working text.
    Handles: replace, remove, shorten, rewrite operations precisely.
    """
    
    def __init__(self):
        # Note: Model will be selected dynamically based on request complexity
        self.base_llm_config = {
            "api_key": settings.OPENAI_API_KEY,
            "temperature": 0.1,  # Low temperature for precision
            "streaming": settings.STREAM
        }
        self.voice = get_andrew_voice()
    
    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Command:
        """Sync entry point - delegates to async version"""
        import asyncio
        return asyncio.run(self.ainvoke(input, config))
    
    async def ainvoke(
        self, 
        input: Dict[str, Any], 
        config: Optional[RunnableConfig] = None
    ) -> Command:
        """Main entry point for precision editing"""
        
        session_id = input.get("session_id", "default")
        user_instruction = input.get("user_instruction", "")
        new_content = input.get("new_content", "")  # For setting initial working text
        resolved_content = input.get("resolved_content", "")  # Content resolved from references
        
        try:
            # Handle setting new working text
            if new_content and not user_instruction:
                return await self._set_working_text(session_id, new_content)
            
            # Handle edit instruction with potential resolved content
            if user_instruction:
                # If we have resolved content and no working text, set it first
                if resolved_content:
                    working_text = await working_text_manager.get_working_text(session_id)
                    if not working_text:
                        await working_text_manager.set_working_text(session_id, resolved_content)
                        logger.info(f"🔄 Set working text from resolved content for editing instruction")
                
                return await self._execute_edit_instruction(session_id, user_instruction)
            
            # Check if we have working text to show
            working_text = await working_text_manager.get_working_text(session_id)
            if working_text:
                return complete_command(
                    content=working_text.content,
                    state={
                        "content": working_text.content,
                        "working_text_version": working_text.version,
                        "has_working_text": True
                    },
                    reason="Showing current working text"
                )
            
            return error_command(
                state={"error": "No working text or instruction provided"},
                reason="Precision editor needs either working text or an edit instruction"
            )
            
        except Exception as e:
            logger.error(f"PrecisionEditorAgent error: {e}")
            return error_command(
                state={"error": str(e)},
                reason=f"Precision editor failed: {str(e)}"
            )
    
    async def _set_working_text(self, session_id: str, content: str) -> Command:
        """Set new working text for the session"""
        try:
            working_text = await working_text_manager.set_working_text(session_id, content)
            
            return complete_command(
                content=content,
                state={
                    "content": content,
                    "working_text_version": working_text.version,
                    "action": "set_working_text",
                    "has_working_text": True
                },
                reason=f"Working text set (version {working_text.version})"
            )
            
        except Exception as e:
            logger.error(f"Failed to set working text: {e}")
            return error_command(
                state={"error": str(e)},
                reason="Failed to set working text"
            )
    
    async def _execute_edit_instruction(self, session_id: str, instruction: str) -> Command:
        """Execute an edit instruction on the working text"""
        
        # Check for undo instruction first
        if self._is_undo_instruction(instruction):
            return await self._handle_undo(session_id, instruction)
        
        # Check for semantic confirmation responses (only if there's a pending confirmation)
        if confirmation_manager.has_pending_confirmation(session_id) and self._is_confirmation_response(instruction):
            return await self._handle_semantic_confirmation(session_id, instruction)
        
        # First check if instruction contains content to extract (like "rewrite this: [content]")
        extracted_content = self._extract_content_from_instruction(instruction)
        
        if extracted_content:
            # Set the extracted content as working text
            await working_text_manager.set_working_text(session_id, extracted_content)
            
            # Convert instruction to simple edit instruction
            simplified_instruction = self._simplify_instruction(instruction)
            
            # Parse the simplified instruction
            parsed = instruction_parser.parse_instruction(simplified_instruction)
        else:
            # Parse the instruction as-is
            parsed = instruction_parser.parse_instruction(instruction)
        
        if parsed.instruction_type == InstructionType.UNKNOWN:
            # Record failed edit in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=instruction,
                instruction_type="unknown",
                before_content="",
                after_content="",
                before_version=0,
                after_version=0,
                success=False,
                error_message="Unknown instruction type"
            )
            
            return error_command(
                state={
                    "error": "Unknown instruction type",
                    "instruction": instruction,
                    "parsed": parsed.to_dict()
                },
                reason=f"Could not understand instruction: {instruction}"
            )
        
        # Get current working text
        working_text = await working_text_manager.get_working_text(session_id)
        if not working_text:
            # Record failed edit in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=instruction,
                instruction_type=parsed.instruction_type.value,
                before_content="",
                after_content="",
                before_version=0,
                after_version=0,
                success=False,
                error_message="No working text available"
            )
            
            return error_command(
                state={"error": "No working text available"},
                reason="No working text to edit. Please provide content first."
            )
        
        # Apply the edit based on instruction type
        if parsed.instruction_type == InstructionType.REPLACE:
            return await self._apply_replace(session_id, working_text, parsed)
        
        elif parsed.instruction_type == InstructionType.REMOVE:
            return await self._apply_remove(session_id, working_text, parsed)
        
        elif parsed.instruction_type == InstructionType.SHORTEN:
            return await self._apply_shorten(session_id, working_text, parsed)
        
        elif parsed.instruction_type == InstructionType.REWRITE:
            return await self._apply_rewrite(session_id, working_text, parsed)
        
        else:
            return error_command(
                state={
                    "error": f"Unsupported instruction type: {parsed.instruction_type}",
                    "parsed": parsed.to_dict()
                },
                reason=f"Cannot handle instruction type: {parsed.instruction_type.value}"
            )
    
    async def _apply_replace(self, session_id: str, working_text: WorkingTextState, parsed: ParsedInstruction) -> Command:
        """Apply a replace instruction"""
        current_content = working_text.content
        target_text = parsed.target_text
        replacement_text = parsed.replacement_text
        
        if not target_text or replacement_text is None:
            # Record failed edit
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=parsed.raw_instruction,
                instruction_type="replace",
                before_content=current_content,
                after_content=current_content,
                before_version=working_text.version,
                after_version=working_text.version,
                success=False,
                error_message="Missing target or replacement text"
            )
            
            return error_command(
                state={"error": "Missing target or replacement text"},
                reason="Replace instruction needs both target and replacement text"
            )
        
        # Check if target text exists
        if target_text not in current_content:
            # Try semantic matching
            semantic_matches = semantic_matcher.find_semantic_matches(target_text, current_content)
            
            if semantic_matches:
                # Found semantic matches - request user confirmation
                suggestions = []
                for i, match in enumerate(semantic_matches):
                    confidence = "high" if match.similarity > 0.85 else "medium" if match.similarity > 0.7 else "low"
                    suggestions.append({
                        "option": i + 1,
                        "text": match.text,
                        "similarity": match.similarity,
                        "confidence": confidence
                    })
                
                # Store the pending confirmation
                confirmation_manager.create_replace_confirmation(
                    session_id=session_id,
                    original_target=target_text,
                    replacement_text=replacement_text,
                    suggestions=suggestions,
                    instruction=parsed.raw_instruction,
                    working_text_version=working_text.version
                )
                
                return complete_command(
                    content=f"I couldn't find exact text '{target_text}', but found similar options:\n\n" + 
                           "\n".join([f"**Option {s['option']}**: '{s['text']}' ({s['confidence']} confidence - {s['similarity']:.0%} similar)" for s in suggestions]) +
                           f"\n\nWould you like me to replace one of these? You can respond with:\n" +
                           "• 'yes' or 'option 1' for the first option\n" +
                           "• 'option 2' for the second option, etc.\n" +
                           "• 'no' or 'cancel' to skip this operation",
                    state={
                        "action": "semantic_confirmation_request",
                        "confirmation_type": "replace",
                        "original_target": target_text,
                        "replacement_text": replacement_text,
                        "suggestions": suggestions,
                        "instruction": parsed.raw_instruction,
                        "working_text_version": working_text.version,
                        "awaiting_user_confirmation": True
                    },
                    reason=f"Found {len(semantic_matches)} similar matches for '{target_text}' - waiting for user confirmation"
                )
            else:
                # No semantic matches found either
                await edit_history_manager.record_edit(
                    session_id=session_id,
                    instruction=parsed.raw_instruction,
                    instruction_type="replace",
                    before_content=current_content,
                    after_content=current_content,
                    before_version=working_text.version,
                    after_version=working_text.version,
                    success=False,
                    error_message=f"Target text '{target_text}' not found (no similar matches)"
                )
                
                return error_command(
                    state={
                        "error": "Target text not found",
                        "target_text": target_text,
                        "current_content": current_content[:200] + "..." if len(current_content) > 200 else current_content
                    },
                    reason=f"Could not find '{target_text}' or similar text in the working text"
                )
        
        # Perform the replacement
        new_content = current_content.replace(target_text, replacement_text)
        
        # Update working text
        updated_working_text = await working_text_manager.update_working_text(session_id, new_content)
        
        # Record successful edit in history
        await edit_history_manager.record_edit(
            session_id=session_id,
            instruction=parsed.raw_instruction,
            instruction_type="replace",
            before_content=current_content,
            after_content=new_content,
            before_version=working_text.version,
            after_version=updated_working_text.version,
            success=True
        )
        
        return complete_command(
            content=new_content,
            state={
                "content": new_content,
                "action": "replace",
                "target_text": target_text,
                "replacement_text": replacement_text,
                "working_text_version": updated_working_text.version,
                "instruction_summary": instruction_parser.get_instruction_summary(parsed)
            },
            reason=f"Replaced '{target_text}' with '{replacement_text}'"
        )
    
    async def _apply_remove(self, session_id: str, working_text: WorkingTextState, parsed: ParsedInstruction) -> Command:
        """Apply a remove instruction"""
        current_content = working_text.content
        target_text = parsed.target_text
        
        if not target_text:
            # Record failed edit
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=parsed.raw_instruction,
                instruction_type="remove",
                before_content=current_content,
                after_content=current_content,
                before_version=working_text.version,
                after_version=working_text.version,
                success=False,
                error_message="Missing target text"
            )
            
            return error_command(
                state={"error": "Missing target text"},
                reason="Remove instruction needs target text to remove"
            )
        
        # Check if target text exists
        if target_text not in current_content:
            # Try semantic matching
            semantic_matches = semantic_matcher.find_semantic_matches(target_text, current_content)
            
            if semantic_matches:
                # Found semantic matches - request user confirmation
                suggestions = []
                for i, match in enumerate(semantic_matches):
                    confidence = "high" if match.similarity > 0.85 else "medium" if match.similarity > 0.7 else "low"
                    suggestions.append({
                        "option": i + 1,
                        "text": match.text,
                        "similarity": match.similarity,
                        "confidence": confidence
                    })
                
                # Store the pending confirmation
                confirmation_manager.create_remove_confirmation(
                    session_id=session_id,
                    original_target=target_text,
                    suggestions=suggestions,
                    instruction=parsed.raw_instruction,
                    working_text_version=working_text.version
                )
                
                return complete_command(
                    content=f"I couldn't find exact text '{target_text}', but found similar options:\n\n" + 
                           "\n".join([f"**Option {s['option']}**: '{s['text']}' ({s['confidence']} confidence - {s['similarity']:.0%} similar)" for s in suggestions]) +
                           f"\n\nWould you like me to remove one of these? You can respond with:\n" +
                           "• 'yes' or 'option 1' for the first option\n" +
                           "• 'option 2' for the second option, etc.\n" +
                           "• 'no' or 'cancel' to skip this operation",
                    state={
                        "action": "semantic_confirmation_request",
                        "confirmation_type": "remove",
                        "original_target": target_text,
                        "suggestions": suggestions,
                        "instruction": parsed.raw_instruction,
                        "working_text_version": working_text.version,
                        "awaiting_user_confirmation": True
                    },
                    reason=f"Found {len(semantic_matches)} similar matches for '{target_text}' - waiting for user confirmation"
                )
            else:
                # No semantic matches found either
                await edit_history_manager.record_edit(
                    session_id=session_id,
                    instruction=parsed.raw_instruction,
                    instruction_type="remove",
                    before_content=current_content,
                    after_content=current_content,
                    before_version=working_text.version,
                    after_version=working_text.version,
                    success=False,
                    error_message=f"Target text '{target_text}' not found (no similar matches)"
                )
                
                return error_command(
                    state={
                        "error": "Target text not found",
                        "target_text": target_text,
                        "current_content": current_content[:200] + "..." if len(current_content) > 200 else current_content
                    },
                    reason=f"Could not find '{target_text}' or similar text in the working text"
                )
        
        # Remove the target text with smart cleanup
        new_content = self._smart_remove_text(current_content, target_text)
        
        # Update working text
        updated_working_text = await working_text_manager.update_working_text(session_id, new_content)
        
        # Record successful edit in history
        await edit_history_manager.record_edit(
            session_id=session_id,
            instruction=parsed.raw_instruction,
            instruction_type="remove",
            before_content=current_content,
            after_content=new_content,
            before_version=working_text.version,
            after_version=updated_working_text.version,
            success=True
        )
        
        return complete_command(
            content=new_content,
            state={
                "content": new_content,
                "action": "remove",
                "target_text": target_text,
                "working_text_version": updated_working_text.version,
                "instruction_summary": instruction_parser.get_instruction_summary(parsed)
            },
            reason=f"Removed '{target_text}'"
        )
    
    async def _apply_shorten(self, session_id: str, working_text: WorkingTextState, parsed: ParsedInstruction) -> Command:
        """Apply a shorten instruction using LLM"""
        current_content = working_text.content
        
        # Use LLM to shorten the content
        system_prompt = f"""You are a precise text editor. Your job is to make content significantly shorter while preserving all important information.

CRITICAL REQUIREMENTS:
- Reduce length by AT LEAST 30-50%
- Keep all essential information and meaning
- Remove filler words, redundant phrases, and unnecessary details
- Maintain the original tone and style
- Return ONLY the shortened content, no explanations

ORIGINAL CONTENT LENGTH: {len(current_content)} characters
TARGET: Reduce to approximately {int(len(current_content) * 0.6)} characters or less"""

        user_message = f"""Make this content significantly shorter:

{current_content}

Remember: Return ONLY the shortened content, nothing else."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on request complexity
            selected_model = model_selector.select_model(
                user_input=parsed.raw_instruction,
                agent_name="PrecisionEditorAgent",
                context=current_content
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            response = await llm.ainvoke(messages)
            shortened_content = response.content.strip()
            
            # Verify it's actually shorter
            if len(shortened_content) >= len(current_content) * 0.9:
                logger.warning(f"Content not significantly shortened: {len(shortened_content)} vs {len(current_content)}")
            
            # Update working text
            updated_working_text = await working_text_manager.update_working_text(session_id, shortened_content)
            
            # Record successful edit in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=parsed.raw_instruction,
                instruction_type="shorten",
                before_content=current_content,
                after_content=shortened_content,
                before_version=working_text.version,
                after_version=updated_working_text.version,
                success=True
            )
            
            return complete_command(
                content=shortened_content,
                state={
                    "content": shortened_content,
                    "action": "shorten",
                    "original_length": len(current_content),
                    "new_length": len(shortened_content),
                    "reduction_percentage": round((1 - len(shortened_content) / len(current_content)) * 100, 1),
                    "working_text_version": updated_working_text.version,
                    "instruction_summary": instruction_parser.get_instruction_summary(parsed)
                },
                reason=f"Shortened content from {len(current_content)} to {len(shortened_content)} characters"
            )
            
        except Exception as e:
            logger.error(f"Failed to shorten content: {e}")
            
            # Record failed edit in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=parsed.raw_instruction,
                instruction_type="shorten",
                before_content=current_content,
                after_content=current_content,
                before_version=working_text.version,
                after_version=working_text.version,
                success=False,
                error_message=str(e)
            )
            
            return error_command(
                state={"error": str(e)},
                reason="Failed to shorten content"
            )
    
    async def _apply_rewrite(self, session_id: str, working_text: WorkingTextState, parsed: ParsedInstruction) -> Command:
        """Apply a rewrite instruction using LLM"""
        current_content = working_text.content
        modification_request = parsed.modification_request or "general improvement"
        
        # Build system prompt with Andrew's voice
        voice = self.voice
        system_prompt = f"""You are a precise text editor working in Andrew's voice. Your job is to rewrite content according to specific instructions.

ANDREW'S VOICE: {voice['core_voice']['tone']}
STYLE: {voice['core_voice']['style']}
ENERGY: {voice['core_voice']['energy']}

CORE PRINCIPLES:
{chr(10).join(f"- {principle}" for principle in voice['core_principles'])}

REWRITE INSTRUCTION: {modification_request}

CRITICAL REQUIREMENTS:
- Preserve all important information and meaning
- Apply the specific modification requested
- Use Andrew's voice and style
- Return ONLY the rewritten content, no explanations
- Make meaningful improvements while keeping the core message intact"""

        user_message = f"""Rewrite this content to {modification_request}:

{current_content}

Remember: Return ONLY the rewritten content, nothing else."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        try:
            # 🚀 INTELLIGENT MODEL SELECTION: Select model based on request complexity
            selected_model = model_selector.select_model(
                user_input=parsed.raw_instruction,
                agent_name="PrecisionEditorAgent",
                context=current_content
            )
            
            # Create LLM with selected model
            llm = ChatOpenAI(
                model=selected_model,
                **self.base_llm_config
            )
            
            response = await llm.ainvoke(messages)
            rewritten_content = response.content.strip()
            
            # Update working text
            updated_working_text = await working_text_manager.update_working_text(session_id, rewritten_content)
            
            # Record successful edit in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=parsed.raw_instruction,
                instruction_type="rewrite",
                before_content=current_content,
                after_content=rewritten_content,
                before_version=working_text.version,
                after_version=updated_working_text.version,
                success=True
            )
            
            return complete_command(
                content=rewritten_content,
                state={
                    "content": rewritten_content,
                    "action": "rewrite",
                    "modification_request": modification_request,
                    "working_text_version": updated_working_text.version,
                    "instruction_summary": instruction_parser.get_instruction_summary(parsed)
                },
                reason=f"Rewrote content to {modification_request}"
            )
            
        except Exception as e:
            logger.error(f"Failed to rewrite content: {e}")
            
            # Record failed edit in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=parsed.raw_instruction,
                instruction_type="rewrite",
                before_content=current_content,
                after_content=current_content,
                before_version=working_text.version,
                after_version=working_text.version,
                success=False,
                error_message=str(e)
            )
            
            return error_command(
                state={"error": str(e)},
                reason="Failed to rewrite content"
            )
    
    def _clean_up_text(self, text: str) -> str:
        """Clean up text after removal operations"""
        # Remove double spaces
        text = ' '.join(text.split())
        
        # Remove orphaned punctuation at the start
        text = text.lstrip('.,!?;:')
        
        # Fix spacing around punctuation
        import re
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1 \2', text)  # Fix double punctuation
        
        return text.strip()
    
    def _smart_remove_text(self, content: str, text_to_remove: str) -> str:
        """Remove text with intelligent punctuation and spacing cleanup"""
        import re
        
        # Find the position of the text to remove
        start_pos = content.find(text_to_remove)
        if start_pos == -1:
            return content
        
        end_pos = start_pos + len(text_to_remove)
        
        # Get context before and after
        before = content[:start_pos]
        after = content[end_pos:]
        
        # Clean up the join point
        result = self._cleanup_sentence_boundary(before, after)
        
        return result
    
    def _cleanup_sentence_boundary(self, before: str, after: str) -> str:
        """Clean up the boundary when text is removed between two parts"""
        import re
        
        # Strip trailing whitespace from before
        before = before.rstrip()
        # Strip leading whitespace from after  
        after = after.lstrip()
        
        # Handle various punctuation scenarios
        
        # If before ends with punctuation and after starts with punctuation
        if before.endswith('.') and after.startswith('.'):
            # Remove duplicate period
            after = after[1:].lstrip()
        
        # If before ends with punctuation and after starts with a capital letter
        if before and before[-1] in '.!?' and after and after[0].isupper():
            # This is a proper sentence boundary, just add space
            return before + ' ' + after
        
        # If before ends with punctuation and after starts with lowercase
        if before and before[-1] in '.!?' and after and after[0].islower():
            # This is good, just add space
            return before + ' ' + after
        
        # If before doesn't end with punctuation but after starts with capital letter
        if before and before[-1] not in '.!?' and after and after[0].isupper():
            # We removed something in the middle of a sentence, need to connect properly
            # Add period to before if it seems like a complete thought
            if len(before.strip()) > 10:  # Reasonable sentence length
                return before + '. ' + after
            else:
                return before + ' ' + after.lower()
        
        # Default case: just join with space, handling any extra periods
        result = before + ' ' + after
        
        # Clean up any double periods or spaces
        result = re.sub(r'\.\.+', '.', result)  # Multiple periods -> single period
        result = re.sub(r'\s+', ' ', result)    # Multiple spaces -> single space
        result = result.strip()
        
        return result
    
    def _extract_content_from_instruction(self, instruction: str) -> Optional[str]:
        """Extract content from instructions like 'rewrite this: [content]'"""
        import re
        
        # Patterns for extracting content
        patterns = [
            r'(?:rewrite|edit|improve|polish|revise|update|fix)\s+(?:this|that):\s*(.+)',
            r'(?:please\s+)?(?:rewrite|edit|improve|polish|revise|update|fix)\s+(?:this|that):\s*(.+)',
            r'(?:please\s+)?(?:rewrite|edit|improve)\s*:\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, instruction, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content:
                    logger.info(f"Extracted content from instruction: {content[:50]}...")
                    return content
        
        return None
    
    def _simplify_instruction(self, instruction: str) -> str:
        """Convert content-containing instructions to simple edit instructions"""
        instruction_lower = instruction.lower()
        
        if 'rewrite' in instruction_lower:
            return 'rewrite it'
        elif 'edit' in instruction_lower:
            return 'rewrite it'
        elif 'improve' in instruction_lower:
            return 'rewrite it'
        elif 'polish' in instruction_lower:
            return 'rewrite it'
        elif 'revise' in instruction_lower:
            return 'rewrite it'
        elif 'update' in instruction_lower:
            return 'rewrite it'
        elif 'fix' in instruction_lower:
            return 'rewrite it'
        else:
            return 'rewrite it'
    
    def _is_undo_instruction(self, instruction: str) -> bool:
        """Check if instruction is an undo command"""
        instruction_lower = instruction.lower().strip()
        
        undo_patterns = [
            'undo',
            'undo that',
            'undo last edit',
            'undo the last change',
            'revert',
            'revert that',
            'go back',
            'undo previous',
            'undo the previous edit'
        ]
        
        return any(pattern in instruction_lower for pattern in undo_patterns)
    
    def _is_confirmation_response(self, instruction: str) -> bool:
        """Check if instruction is a response to a semantic confirmation request"""
        instruction_lower = instruction.lower().strip()
        
        confirmation_patterns = [
            'yes', 'y', 'ok', 'okay', 'sure', 'confirm',
            'option 1', 'option 2', 'option 3', '1', '2', '3',
            'no', 'n', 'cancel', 'skip', 'never mind', 'nevermind'
        ]
        
        return any(pattern in instruction_lower for pattern in confirmation_patterns)
    
    def _parse_confirmation_response(self, instruction: str) -> dict:
        """Parse user's confirmation response"""
        instruction_lower = instruction.lower().strip()
        
        # Check for rejection
        if any(word in instruction_lower for word in ['no', 'n', 'cancel', 'skip', 'never mind', 'nevermind']):
            return {"confirmed": False, "option": None}
        
        # Check for specific option
        if 'option 1' in instruction_lower or instruction_lower in ['1', 'yes', 'y', 'ok', 'okay', 'sure', 'confirm']:
            return {"confirmed": True, "option": 1}
        elif 'option 2' in instruction_lower or instruction_lower == '2':
            return {"confirmed": True, "option": 2}
        elif 'option 3' in instruction_lower or instruction_lower == '3':
            return {"confirmed": True, "option": 3}
        
        # Default to first option for general confirmations
        return {"confirmed": True, "option": 1}
    
    async def _handle_semantic_confirmation(self, session_id: str, instruction: str) -> Command:
        """Handle user confirmation of semantic matching suggestions"""
        try:
            # Parse the user's response
            response = self._parse_confirmation_response(instruction)
            
            if not response["confirmed"]:
                # Clear the pending confirmation and cancel
                confirmation_manager.clear_confirmation(session_id)
                return complete_command(
                    content="Operation cancelled. No changes made to the text.",
                    state={
                        "action": "semantic_confirmation_cancelled",
                        "reason": "User declined semantic substitution"
                    },
                    reason="User cancelled semantic substitution"
                )
            
            # Get the pending confirmation
            pending_confirmation = confirmation_manager.get_pending_confirmation(session_id)
            if not pending_confirmation:
                return error_command(
                    state={"error": "No pending confirmation found"},
                    reason="No pending semantic confirmation for this session"
                )
            
            # Get the option selected
            option_selected = response["option"]
            confirmed_text = confirmation_manager.get_confirmed_text(session_id, option_selected)
            
            if not confirmed_text:
                return error_command(
                    state={"error": f"Invalid option selected: {option_selected}"},
                    reason=f"Option {option_selected} not available"
                )
            
            # Execute the confirmed operation
            result = await self._execute_semantic_operation(session_id, pending_confirmation, confirmed_text)
            
            # Clear the pending confirmation
            confirmation_manager.clear_confirmation(session_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to handle semantic confirmation: {e}")
            confirmation_manager.clear_confirmation(session_id)
            return error_command(
                state={"error": str(e)},
                reason="Failed to process semantic confirmation"
            )
    
    async def _execute_semantic_operation(self, session_id: str, pending_confirmation, confirmed_text: str) -> Command:
        """Execute the confirmed semantic operation"""
        try:
            # Get current working text
            working_text = await working_text_manager.get_working_text(session_id)
            if not working_text:
                await edit_history_manager.record_edit(
                    session_id=session_id,
                    instruction=pending_confirmation.instruction,
                    instruction_type=pending_confirmation.confirmation_type.value,
                    before_content="",
                    after_content="",
                    before_version=0,
                    after_version=0,
                    success=False,
                    error_message="No working text available for semantic operation"
                )
                return error_command(
                    state={"error": "No working text available"},
                    reason="No working text to edit"
                )
            
            current_content = working_text.content
            
            # Execute the operation based on type
            if pending_confirmation.confirmation_type.value == "replace":
                # Replace the confirmed text
                new_content = current_content.replace(confirmed_text, pending_confirmation.replacement_text, 1)
                operation_desc = f"Replaced '{confirmed_text}' with '{pending_confirmation.replacement_text}'"
            
            elif pending_confirmation.confirmation_type.value == "remove":
                # Remove the confirmed text with smart cleanup
                new_content = self._smart_remove_text(current_content, confirmed_text)
                operation_desc = f"Removed '{confirmed_text}'"
            
            else:
                return error_command(
                    state={"error": f"Unknown confirmation type: {pending_confirmation.confirmation_type}"},
                    reason="Invalid confirmation type"
                )
            
            # Update working text
            updated_working_text = await working_text_manager.update_working_text(session_id, new_content)
            
            # Record successful edit in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=pending_confirmation.instruction,
                instruction_type=pending_confirmation.confirmation_type.value,
                before_content=current_content,
                after_content=new_content,
                before_version=working_text.version,
                after_version=updated_working_text.version,
                success=True,
                error_message=None
            )
            
            return complete_command(
                content=new_content,
                state={
                    "content": new_content,
                    "working_text_version": updated_working_text.version,
                    "action": f"semantic_{pending_confirmation.confirmation_type.value}",
                    "operation_performed": operation_desc,
                    "original_target": pending_confirmation.original_target,
                    "confirmed_text": confirmed_text
                },
                reason=f"Successfully executed semantic {pending_confirmation.confirmation_type.value}: {operation_desc}"
            )
            
        except Exception as e:
            logger.error(f"Failed to execute semantic operation: {e}")
            # Record failed edit in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=pending_confirmation.instruction,
                instruction_type=pending_confirmation.confirmation_type.value,
                before_content=current_content if 'current_content' in locals() else "",
                after_content=current_content if 'current_content' in locals() else "",
                before_version=working_text.version if 'working_text' in locals() else 0,
                after_version=working_text.version if 'working_text' in locals() else 0,
                success=False,
                error_message=str(e)
            )
            
            return error_command(
                state={"error": str(e)},
                reason="Failed to execute semantic operation"
            )
    
    async def _handle_undo(self, session_id: str, instruction: str) -> Command:
        """Handle undo instructions"""
        try:
            # Get the previous content from edit history
            previous_content = await edit_history_manager.undo_last_edit(session_id)
            
            if previous_content is None:
                return error_command(
                    state={"error": "No edits to undo"},
                    reason="No previous edits found to undo"
                )
            
            # Get current working text for history recording
            current_working_text = await working_text_manager.get_working_text(session_id)
            current_content = current_working_text.content if current_working_text else ""
            
            # Restore the previous content
            updated_working_text = await working_text_manager.update_working_text(session_id, previous_content)
            
            # Record the undo operation in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=instruction,
                instruction_type="undo",
                before_content=current_content,
                after_content=previous_content,
                before_version=current_working_text.version if current_working_text else 0,
                after_version=updated_working_text.version,
                success=True
            )
            
            return complete_command(
                content=previous_content,
                state={
                    "content": previous_content,
                    "action": "undo",
                    "working_text_version": updated_working_text.version,
                    "instruction_summary": "Undid last edit"
                },
                reason="Successfully undid the last edit"
            )
            
        except Exception as e:
            logger.error(f"Failed to undo edit: {e}")
            
            # Record failed undo in history
            await edit_history_manager.record_edit(
                session_id=session_id,
                instruction=instruction,
                instruction_type="undo",
                before_content="",
                after_content="",
                before_version=0,
                after_version=0,
                success=False,
                error_message=str(e)
            )
            
            return error_command(
                state={"error": str(e)},
                reason="Failed to undo edit"
            )

# Global instance
precision_editor_agent = PrecisionEditorAgent() 