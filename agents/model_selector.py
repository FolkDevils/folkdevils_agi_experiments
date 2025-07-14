"""
Intelligent Model Selector - Phase 1g

Determines the optimal LLM model based on request complexity to balance
speed and quality while reducing costs.

Models:
- gpt-4o-mini: Fast, cheap, good for simple operations
- gpt-4o: Slower, expensive, best for complex operations
"""

import re
import logging
import hashlib
import time
from typing import Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ModelComplexity(str, Enum):
    """Request complexity levels"""
    SIMPLE = "simple"      # gpt-4o-mini
    COMPLEX = "complex"    # gpt-4o


class ModelSelector:
    """
    Intelligent model selection based on request complexity.
    
    🚀 Phase 1g: Optimize speed and cost by using the right model for each task.
    """
    
    def __init__(self):
        # Model configuration
        self.models = {
            ModelComplexity.SIMPLE: "gpt-4o-mini",    # Fast & cheap
            ModelComplexity.COMPLEX: "gpt-4o"          # Quality & powerful
        }
        
        # Simple request patterns - these can use gpt-4o-mini
        self.simple_patterns = [
            # Time and basic queries
            r"what time",
            r"what date",
            r"when is",
            r"how are you",
            r"hello|hi|hey",
            r"thanks|thank you",
            r"yes|no|ok|okay",
            
            # Simple editing operations
            r"make it shorter",
            r"make this shorter", 
            r"shorten this",
            r"make it longer",
            r"make this longer",
            r"fix typos?",
            r"correct spelling",
            
            # Basic information
            r"what is \w+",
            r"who is \w+",
            r"where is \w+",
            r"how do i \w+",
            
            # Simple content requests
            r"write a (short|quick|brief)",
            r"send (a|an) (quick|short|brief)",
            r"create a (simple|basic|quick)",
            
            # Basic confirmations
            r"confirm that",
            r"verify that",
            r"check if",
        ]
        
        # Complex request patterns - these need gpt-4o
        self.complex_patterns = [
            # Analysis and reasoning
            r"analyze",
            r"evaluate",
            r"assess", 
            r"review and",
            r"compare",
            r"recommend",
            r"strategy",
            r"plan for",
            
            # Creative writing
            r"write (a|an) (detailed|comprehensive|thorough)",
            r"create (a|an) (detailed|comprehensive|thorough)",
            r"draft (a|an) (formal|professional|important)",
            
            # Complex editing
            r"rewrite (this|that) to be more",
            r"transform (this|that)",
            r"restructure",
            r"reorganize",
            r"completely rewrite",
            
            # Multi-step or conditional
            r"if .+ then",
            r"depending on",
            r"based on .+ (and|or)",
            r"taking into account",
            
            # Technical or specialized
            r"implement",
            r"develop",
            r"engineer",
            r"architect", 
            r"design a system",
            
            # Length indicators
            r"write (at least|more than) \d+",
            r"(long|lengthy|extensive|comprehensive) (essay|report|analysis)",
        ]
        
        # Context complexity indicators
        self.complexity_indicators = {
            "length": {
                "simple_threshold": 50,    # < 50 chars = likely simple
                "complex_threshold": 200   # > 200 chars = likely complex
            },
            "word_count": {
                "simple_threshold": 10,    # < 10 words = likely simple  
                "complex_threshold": 30    # > 30 words = likely complex
            },
            "sentence_count": {
                "complex_threshold": 3     # > 3 sentences = likely complex
            }
        }
        
        # Agent-specific model preferences
        self.agent_preferences = {
            "TimekeeperAgent": ModelComplexity.SIMPLE,        # Time queries are usually simple
            "ConversationalAgent": ModelComplexity.SIMPLE,   # General chat is usually simple
            "PrecisionEditorAgent": ModelComplexity.SIMPLE,  # Precision edits are focused
            "LearningAgent": ModelComplexity.SIMPLE,         # Memory operations are straightforward
            "WriterAgent": ModelComplexity.COMPLEX,          # Content creation benefits from quality
            "EditorAgent": ModelComplexity.COMPLEX,          # Content improvement benefits from quality
        }
        
        # Retry operations should use fast model
        self.retry_model = ModelComplexity.SIMPLE
        
        # 🚀 CACHING SYSTEM: Cache model selection decisions
        self.decision_cache: Dict[str, Tuple[str, float]] = {}  # cache_key -> (model, timestamp)
        self.cache_ttl = 3600  # 1 hour TTL for cache entries
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000  # Prevent unlimited cache growth
    
    def select_model(self, 
                    user_input: str, 
                    agent_name: Optional[str] = None,
                    context: Optional[str] = None,
                    is_retry: bool = False) -> str:
        """
        🚀 INTELLIGENT MODEL SELECTION: Select optimal model based on complexity with caching
        
        Args:
            user_input: The user's request
            agent_name: Name of the agent that will process this
            context: Additional context (content from previous steps, etc.)
            is_retry: True if this is a retry operation
            
        Returns:
            Model name (gpt-4o-mini or gpt-4o)
        """
        try:
            # Retry operations always use fast model for speed
            if is_retry:
                logger.info(f"🚀 MODEL SELECTION: Using {self.models[self.retry_model]} for retry operation")
                return self.models[self.retry_model]
            
            # 🚀 CACHING: Check cache first for performance optimization
            cache_key = self._generate_cache_key(user_input, agent_name, context)
            cached_model = self._get_cached_decision(cache_key)
            
            if cached_model:
                self.cache_hits += 1
                logger.info(f"🚀 MODEL SELECTION (CACHED): {user_input[:30]}... -> {cached_model} [cache hit {self.cache_hits}]")
                return cached_model
            
            # Cache miss - perform analysis
            self.cache_misses += 1
            complexity = self._classify_complexity(user_input, agent_name, context)
            selected_model = self.models[complexity]
            
            # Cache the decision for future use
            self._cache_decision(cache_key, selected_model)
            
            logger.info(f"🚀 MODEL SELECTION: {user_input[:30]}... -> {complexity.value} -> {selected_model} [cache miss {self.cache_misses}]")
            
            return selected_model
            
        except Exception as e:
            logger.error(f"❌ Model selection failed: {e}, defaulting to gpt-4o")
            return "gpt-4o"  # Safe fallback
    
    def _classify_complexity(self, 
                           user_input: str, 
                           agent_name: Optional[str] = None,
                           context: Optional[str] = None) -> ModelComplexity:
        """
        Classify request complexity using multiple signals
        """
        if not user_input:
            return ModelComplexity.SIMPLE
            
        user_lower = user_input.lower()
        
        # 1. Check explicit complex patterns first (high confidence)
        for pattern in self.complex_patterns:
            if re.search(pattern, user_lower):
                logger.debug(f"🎯 COMPLEX pattern matched: {pattern}")
                return ModelComplexity.COMPLEX
        
        # 2. Check explicit simple patterns (high confidence)
        for pattern in self.simple_patterns:
            if re.search(pattern, user_lower):
                logger.debug(f"🎯 SIMPLE pattern matched: {pattern}")
                return ModelComplexity.SIMPLE
        
        # 3. Use agent preferences as a signal
        if agent_name and agent_name in self.agent_preferences:
            agent_preference = self.agent_preferences[agent_name]
            logger.debug(f"🎯 AGENT preference: {agent_name} -> {agent_preference.value}")
            
            # If agent strongly prefers simple, trust it unless overruled by complexity
            if agent_preference == ModelComplexity.SIMPLE:
                if not self._has_complexity_indicators(user_input):
                    return ModelComplexity.SIMPLE
        
        # 4. Analyze structural complexity
        if self._has_complexity_indicators(user_input):
            logger.debug(f"🎯 COMPLEXITY indicators detected")
            return ModelComplexity.COMPLEX
        
        # 5. Default based on length and content
        if len(user_input) < self.complexity_indicators["length"]["simple_threshold"]:
            logger.debug(f"🎯 SHORT request ({len(user_input)} chars) -> simple")
            return ModelComplexity.SIMPLE
        elif len(user_input) > self.complexity_indicators["length"]["complex_threshold"]:
            logger.debug(f"🎯 LONG request ({len(user_input)} chars) -> complex")
            return ModelComplexity.COMPLEX
        
        # 6. Word count analysis
        word_count = len(user_input.split())
        if word_count < self.complexity_indicators["word_count"]["simple_threshold"]:
            logger.debug(f"🎯 FEW words ({word_count}) -> simple")
            return ModelComplexity.SIMPLE
        elif word_count > self.complexity_indicators["word_count"]["complex_threshold"]:
            logger.debug(f"🎯 MANY words ({word_count}) -> complex")
            return ModelComplexity.COMPLEX
        
        # 7. Default to simple for borderline cases (speed over perfection)
        logger.debug(f"🎯 BORDERLINE case -> defaulting to simple")
        return ModelComplexity.SIMPLE
    
    def _has_complexity_indicators(self, user_input: str) -> bool:
        """
        Check for structural complexity indicators
        """
        # Multiple sentences suggest complexity
        sentence_count = len([s for s in user_input.split('.') if s.strip()])
        if sentence_count > self.complexity_indicators["sentence_count"]["complex_threshold"]:
            return True
        
        # Multiple clauses (and/or/but)
        clause_connectors = len(re.findall(r'\b(and|or|but|however|although|because|since|while)\b', user_input.lower()))
        if clause_connectors > 2:
            return True
        
        # Conditional language
        if re.search(r'\b(if|when|unless|provided|assuming)\b', user_input.lower()):
            return True
        
        # Multiple requirements/steps
        if re.search(r'\b(first|then|next|also|additionally|furthermore)\b', user_input.lower()):
            return True
        
        return False
    
    def _generate_cache_key(self, user_input: str, agent_name: Optional[str], context: Optional[str]) -> str:
        """
        🚀 CACHING: Generate a cache key for the request
        
        Creates a hash based on input patterns and agent, not exact text
        to allow for similar requests to hit the cache
        """
        # Normalize the input to create pattern-based cache keys
        normalized_input = self._normalize_for_caching(user_input)
        
        # Include agent name as it affects model selection
        agent_part = agent_name or "unknown"
        
        # Include simplified context (length category)
        context_part = "with_context" if context and len(context) > 50 else "no_context"
        
        # Create cache key from normalized components
        cache_data = f"{normalized_input}|{agent_part}|{context_part}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _normalize_for_caching(self, user_input: str) -> str:
        """
        Normalize user input to create cache-friendly patterns
        """
        # Convert to lowercase and remove extra spaces
        normalized = re.sub(r'\s+', ' ', user_input.lower().strip())
        
        # Replace specific content with patterns for better cache hits
        # "write me an email about X" -> "write me an email about CONTENT"
        normalized = re.sub(r'\b(about|regarding|for|with)\s+\w+', r'\1 CONTENT', normalized)
        
        # Replace numbers with pattern
        normalized = re.sub(r'\b\d+\b', 'NUMBER', normalized)
        
        # Replace quoted content with pattern
        normalized = re.sub(r'"[^"]*"', 'QUOTED_CONTENT', normalized)
        
        # Truncate very long inputs to patterns
        if len(normalized) > 100:
            # Keep first and last parts for pattern matching
            normalized = normalized[:50] + "..." + normalized[-20:]
        
        return normalized
    
    def _get_cached_decision(self, cache_key: str) -> Optional[str]:
        """
        🚀 CACHING: Retrieve cached model decision if valid
        """
        current_time = time.time()
        
        if cache_key in self.decision_cache:
            model, timestamp = self.decision_cache[cache_key]
            
            # Check if cache entry is still valid
            if current_time - timestamp < self.cache_ttl:
                return model
            else:
                # Remove expired entry
                del self.decision_cache[cache_key]
        
        return None
    
    def _cache_decision(self, cache_key: str, model: str) -> None:
        """
        🚀 CACHING: Cache the model decision for future use
        """
        current_time = time.time()
        
        # Implement cache size limit to prevent memory issues
        if len(self.decision_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO eviction)
            oldest_key = min(self.decision_cache.keys(), 
                           key=lambda k: self.decision_cache[k][1])
            del self.decision_cache[oldest_key]
        
        # Cache the decision
        self.decision_cache[cache_key] = (model, current_time)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        🚀 CACHING: Get cache performance statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self.decision_cache),
            "max_cache_size": self.max_cache_size,
            "cache_ttl_seconds": self.cache_ttl
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models and current configuration"""
        return {
            "models": dict(self.models),
            "simple_patterns_count": len(self.simple_patterns),
            "complex_patterns_count": len(self.complex_patterns),
            "agent_preferences": dict(self.agent_preferences),
            "complexity_thresholds": self.complexity_indicators,
            "cache_stats": self.get_cache_stats()
        }


# Global instance
model_selector = ModelSelector() 