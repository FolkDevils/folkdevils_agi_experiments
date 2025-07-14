from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class InstructionType(Enum):
    """Types of edit instructions we can handle"""
    REPLACE = "replace"
    REMOVE = "remove"
    SHORTEN = "shorten"
    REWRITE = "rewrite"
    UNKNOWN = "unknown"

@dataclass
class ParsedInstruction:
    """Represents a parsed edit instruction"""
    instruction_type: InstructionType
    target_text: Optional[str] = None  # Text to find/modify
    replacement_text: Optional[str] = None  # Text to replace with
    modification_request: Optional[str] = None  # General modification request
    confidence: float = 0.0  # Confidence in the parsing
    raw_instruction: str = ""  # Original instruction text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_type": self.instruction_type.value,
            "target_text": self.target_text,
            "replacement_text": self.replacement_text,
            "modification_request": self.modification_request,
            "confidence": self.confidence,
            "raw_instruction": self.raw_instruction
        }

class InstructionParser:
    """
    Parses user instructions into structured edit operations.
    Phase I supports: replace, remove, shorten, rewrite
    """
    
    def __init__(self):
        # Patterns for different instruction types
        self.replace_patterns = [
            # Handle double quotes (can contain apostrophes)
            r'replace\s+"([^"]+)"\s+with\s+"([^"]+)"',
            r'replace\s+"([^"]+)"\s+with\s+(.+)',
            r'change\s+"([^"]+)"\s+to\s+"([^"]+)"',
            r'change\s+"([^"]+)"\s+to\s+(.+)',
            r'update\s+"([^"]+)"\s+to\s+"([^"]+)"',
            r'update\s+"([^"]+)"\s+to\s+(.+)',
            r'substitute\s+"([^"]+)"\s+with\s+"([^"]+)"',
            # Handle single quotes
            r"replace\s+'([^']+)'\s+with\s+'([^']+)'",
            r"replace\s+'([^']+)'\s+with\s+(.+)",
            r"change\s+'([^']+)'\s+to\s+'([^']+)'",
            r"change\s+'([^']+)'\s+to\s+(.+)",
            r"update\s+'([^']+)'\s+to\s+'([^']+)'",
            r"update\s+'([^']+)'\s+to\s+(.+)",
            r"substitute\s+'([^']+)'\s+with\s+'([^']+)'",
            # Handle "instead of" patterns with double quotes
            r'say\s+"([^"]+)"\s+instead\s+of\s+"([^"]+)"',
            r'use\s+"([^"]+)"\s+instead\s+of\s+"([^"]+)"',
            # Handle "instead of" patterns with single quotes
            r"say\s+'([^']+)'\s+instead\s+of\s+'([^']+)'",
            r"use\s+'([^']+)'\s+instead\s+of\s+'([^']+)'",
            # Handle reference-based replacements with double quotes
            r'update\s+(?:that\s+)?to\s+say\s+"([^"]+)"\s+instead\s+of\s+"([^"]+)"',
            r'change\s+(?:that\s+)?to\s+say\s+"([^"]+)"\s+instead\s+of\s+"([^"]+)"',
            # Handle reference-based replacements with single quotes
            r"update\s+(?:that\s+)?to\s+say\s+'([^']+)'\s+instead\s+of\s+'([^']+)'",
            r"change\s+(?:that\s+)?to\s+say\s+'([^']+)'\s+instead\s+of\s+'([^']+)'",
            # Fallback patterns for mixed quotes (more complex regex)
            r'replace\s+["\']([^"\']*(?:["\'][^"\']*)*?)["\']\s+with\s+["\']([^"\']*(?:["\'][^"\']*)*?)["\']',
            r'change\s+["\']([^"\']*(?:["\'][^"\']*)*?)["\']\s+to\s+["\']([^"\']*(?:["\'][^"\']*)*?)["\']',
        ]
        
        self.remove_patterns = [
            # Handle double quotes (can contain apostrophes)
            r'remove\s+"([^"]+)"',
            r'delete\s+"([^"]+)"',
            r'take\s+out\s+"([^"]+)"',
            r'get\s+rid\s+of\s+"([^"]+)"',
            r'cut\s+"([^"]+)"',
            # Handle single quotes (can contain other quotes but more complex)
            r"remove\s+'([^']+)'",
            r"delete\s+'([^']+)'",
            r"take\s+out\s+'([^']+)'",
            r"get\s+rid\s+of\s+'([^']+)'",
            r"cut\s+'([^']+)'",
            # Handle mixed quotes (fallback to old behavior for backward compatibility)
            r'remove\s+["\']([^"\']*(?:["\'][^"\']*)*)["\']',
            r'delete\s+["\']([^"\']*(?:["\'][^"\']*)*)["\']',
            # Handle sentence/phrase removal with double quotes
            r'remove\s+the\s+(?:sentence|phrase|part|line)\s+"([^"]+)"',
            r'delete\s+the\s+(?:sentence|phrase|part|line)\s+"([^"]+)"',
            # Handle sentence/phrase removal with single quotes
            r"remove\s+the\s+(?:sentence|phrase|part|line)\s+'([^']+)'",
            r"delete\s+the\s+(?:sentence|phrase|part|line)\s+'([^']+)'",
            # Handle sentence/phrase removal without quotes
            r'remove\s+this\s+(?:sentence|phrase|part|line):\s*(.+?)(?:\.|$)',
            r'delete\s+this\s+(?:sentence|phrase|part|line):\s*(.+?)(?:\.|$)',
            r'update\s+to\s+remove\s+this\s+(?:sentence|phrase|part|line):\s*(.+?)(?:\.|$)',
            # Handle reference-based removal with double quotes
            r'update\s+(?:that\s+)?to\s+remove\s+(?:the\s+)?"([^"]+)"',
            r'change\s+(?:that\s+)?to\s+remove\s+(?:the\s+)?"([^"]+)"',
            # Handle reference-based removal with single quotes
            r"update\s+(?:that\s+)?to\s+remove\s+(?:the\s+)?'([^']+)'",
            r"change\s+(?:that\s+)?to\s+remove\s+(?:the\s+)?'([^']+)'",
            # Handle reference-based removal without quotes
            r'update\s+to\s+remove\s+(?:the\s+sentence\s+)?(.+?)(?:\.|$)',
            r'please\s+update\s+to\s+remove\s+(?:the\s+sentence\s+)?(.+?)(?:\.|$)',
        ]
        
        self.shorten_patterns = [
            r'make\s+(?:it|this|that)\s+shorter',
            r'shorten\s+(?:it|this|that)',
            r'make\s+(?:it|this|that)\s+more\s+concise',
            r'cut\s+(?:it|this|that)\s+down',
            r'reduce\s+the\s+length',
            r'make\s+(?:it|this|that)\s+briefer',
            # Update patterns for shortening
            r'update\s+(?:it|this|that)\s+to\s+be\s+shorter',
            r'update\s+(?:it|this|that)\s+to\s+make\s+(?:it|this|that)\s+shorter',
            r'change\s+(?:it|this|that)\s+to\s+be\s+shorter',
            r'change\s+(?:it|this|that)\s+to\s+make\s+(?:it|this|that)\s+shorter',
            # Please variants
            r'please\s+update\s+(?:it|this|that)\s+to\s+be\s+shorter',
            r'please\s+make\s+(?:it|this|that)\s+shorter',
            r'please\s+shorten\s+(?:it|this|that)',
        ]
        
        self.rewrite_patterns = [
            r'rewrite\s+(?:it|this|that)',
            r'rewrite\s+(?:it|this|that)\s+to\s+(.+)',
            r'redo\s+(?:it|this|that)',
            r'revise\s+(?:it|this|that)',
            r'improve\s+(?:it|this|that)',
            r'polish\s+(?:it|this|that)',
            r'make\s+(?:it|this|that)\s+better',
            r'make\s+(?:it|this|that)\s+more\s+(.+)',
            r'make\s+(?:it|this|that)\s+sound\s+(.+)',
            r'update\s+(?:it|this|that)\s+to\s+make\s+(?:it|this|that)\s+(.+)',
            r'update\s+to\s+make\s+(?:it|this|that)\s+(.+)',
            r'update\s+to\s+make\s+(?:it|this|that)\s+sound\s+(.+)',
            r'change\s+(?:it|this|that)\s+to\s+make\s+(?:it|this|that)\s+(.+)',
            r'change\s+to\s+make\s+(?:it|this|that)\s+(.+)',
            r'make\s+(?:it|this|that)\s+(?:more\s+)?(.+)',
        ]
    
    def parse_instruction(self, instruction: str) -> ParsedInstruction:
        """Parse a user instruction into a structured edit operation"""
        instruction = instruction.strip()
        instruction_lower = instruction.lower()
        
        # Try each instruction type
        result = self._try_parse_replace(instruction, instruction_lower)
        if result.instruction_type != InstructionType.UNKNOWN:
            return result
        
        result = self._try_parse_remove(instruction, instruction_lower)
        if result.instruction_type != InstructionType.UNKNOWN:
            return result
        
        result = self._try_parse_shorten(instruction, instruction_lower)
        if result.instruction_type != InstructionType.UNKNOWN:
            return result
        
        result = self._try_parse_rewrite(instruction, instruction_lower)
        if result.instruction_type != InstructionType.UNKNOWN:
            return result
        
        # Unknown instruction type
        return ParsedInstruction(
            instruction_type=InstructionType.UNKNOWN,
            raw_instruction=instruction,
            confidence=0.0
        )
    
    def _try_parse_replace(self, instruction: str, instruction_lower: str) -> ParsedInstruction:
        """Try to parse as a replace instruction"""
        for pattern in self.replace_patterns:
            # Search using the lowercase version for pattern matching
            match = re.search(pattern, instruction_lower, re.IGNORECASE)
            if match:
                # But extract the original case from the original instruction
                original_match = re.search(pattern, instruction, re.IGNORECASE)
                if original_match:
                    groups = original_match.groups()
                    
                    # Handle "instead of" patterns (replacement comes first)
                    if "instead of" in pattern:
                        replacement_text = groups[0].strip()
                        target_text = groups[1].strip()
                    else:
                        target_text = groups[0].strip()
                        replacement_text = groups[1].strip() if len(groups) > 1 else ""
                    
                    confidence = 0.9  # High confidence for explicit replace patterns
                    
                    return ParsedInstruction(
                        instruction_type=InstructionType.REPLACE,
                        target_text=target_text,
                        replacement_text=replacement_text,
                        raw_instruction=instruction,
                        confidence=confidence
                    )
        
        return ParsedInstruction(instruction_type=InstructionType.UNKNOWN)
    
    def _try_parse_remove(self, instruction: str, instruction_lower: str) -> ParsedInstruction:
        """Try to parse as a remove instruction"""
        for pattern in self.remove_patterns:
            # Search using the lowercase version for pattern matching
            match = re.search(pattern, instruction_lower, re.IGNORECASE)
            if match:
                # But extract the original case from the original instruction
                original_match = re.search(pattern, instruction, re.IGNORECASE)
                if original_match:
                    target_text = original_match.group(1).strip()
                    confidence = 0.9  # High confidence for explicit remove patterns
                    
                    return ParsedInstruction(
                        instruction_type=InstructionType.REMOVE,
                        target_text=target_text,
                        raw_instruction=instruction,
                        confidence=confidence
                    )
        
        return ParsedInstruction(instruction_type=InstructionType.UNKNOWN)
    
    def _try_parse_shorten(self, instruction: str, instruction_lower: str) -> ParsedInstruction:
        """Try to parse as a shorten instruction"""
        for pattern in self.shorten_patterns:
            if re.search(pattern, instruction_lower, re.IGNORECASE):
                confidence = 0.8  # Good confidence for shorten patterns
                
                return ParsedInstruction(
                    instruction_type=InstructionType.SHORTEN,
                    modification_request="make shorter",
                    raw_instruction=instruction,
                    confidence=confidence
                )
        
        return ParsedInstruction(instruction_type=InstructionType.UNKNOWN)
    
    def _try_parse_rewrite(self, instruction: str, instruction_lower: str) -> ParsedInstruction:
        """Try to parse as a rewrite instruction"""
        for pattern in self.rewrite_patterns:
            match = re.search(pattern, instruction_lower, re.IGNORECASE)
            if match:
                confidence = 0.7  # Moderate confidence for rewrite patterns
                
                # Extract specific modification request if available
                modification_request = "general improvement"
                if match.groups():
                    modification_request = match.group(1).strip()
                
                return ParsedInstruction(
                    instruction_type=InstructionType.REWRITE,
                    modification_request=modification_request,
                    raw_instruction=instruction,
                    confidence=confidence
                )
        
        return ParsedInstruction(instruction_type=InstructionType.UNKNOWN)
    
    def can_handle_instruction(self, instruction: str) -> bool:
        """Check if we can handle this instruction type"""
        parsed = self.parse_instruction(instruction)
        return parsed.instruction_type != InstructionType.UNKNOWN and parsed.confidence > 0.5
    
    def get_instruction_summary(self, parsed: ParsedInstruction) -> str:
        """Get a human-readable summary of the parsed instruction"""
        if parsed.instruction_type == InstructionType.REPLACE:
            return f"Replace '{parsed.target_text}' with '{parsed.replacement_text}'"
        elif parsed.instruction_type == InstructionType.REMOVE:
            return f"Remove '{parsed.target_text}'"
        elif parsed.instruction_type == InstructionType.SHORTEN:
            return "Make the text shorter"
        elif parsed.instruction_type == InstructionType.REWRITE:
            return f"Rewrite to {parsed.modification_request}"
        else:
            return "Unknown instruction"

# Global instance
instruction_parser = InstructionParser() 