"""
Message Complexity Analyzer - Smart Processing Triggers

This module determines when messages warrant deep consciousness processing:
- Simple questions get fast responses
- Complex questions get full cognitive analysis
- Intelligent resource allocation based on query complexity
- Performance-first approach with cognitive depth when justified

This is the key to solving our timeout issues while maintaining intelligence.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ComplexityAssessment:
    """Assessment of message complexity and recommended processing"""
    complexity_score: float  # 0.0 to 1.0
    processing_recommendation: str  # 'minimal', 'standard', 'deep', 'comprehensive'
    reasoning: str
    estimated_processing_time: float  # seconds
    should_use_planning: bool
    should_use_coherence: bool
    should_use_metacognitive: bool
    should_use_deep_memory: bool
    complexity_factors: Dict[str, float]

class MessageComplexityAnalyzer:
    """
    Intelligent message complexity analysis for performance optimization
    
    This system prevents timeout issues by:
    - Quickly identifying simple vs complex queries
    - Recommending appropriate processing depth
    - Avoiding unnecessary LLM calls for simple questions
    - Enabling full consciousness for truly complex queries
    """
    
    def __init__(self):
        # Simple query patterns that don't need deep processing
        self.simple_patterns = [
            r'^(hello|hi|hey|good morning|good afternoon|good evening)',
            r'^(thanks?|thank you|thx)',
            r'^(yes|no|yeah|yep|nope|ok|okay)',
            r'^(how are you|how\'s it going)',
            r'^(what time|what date|what day)',
            r'^(status|show me|display|list)',
        ]
        
        # Complex query indicators that benefit from deep processing
        self.complexity_indicators = {
            'philosophical': {
                'patterns': [r'consciousness', r'meaning of', r'why do', r'what is the purpose', 
                           r'philosophy', r'philosophical', r'existence', r'reality', r'truth', r'wisdom',
                           r'self-awareness', r'recursive', r'implications', r'emerge'],
                'weight': 0.8
            },
            'strategic': {
                'patterns': [r'strategy', r'plan', r'approach', r'should i', r'what if', 
                           r'decide', r'choice', r'option', r'recommendation'],
                'weight': 0.7
            },
            'analytical': {
                'patterns': [r'analyze', r'explain', r'understand', r'compare', r'how does',
                           r'relationship', r'connection', r'pattern', r'trend', r'artificial intelligence'],
                'weight': 0.6
            },
            'creative': {
                'patterns': [r'create', r'design', r'imagine', r'brainstorm', r'innovative',
                           r'creative', r'generate', r'come up with'],
                'weight': 0.5
            },
            'reflective': {
                'patterns': [r'reflect', r'think about', r'consider', r'ponder', r'contemplate',
                           r'your thoughts', r'your opinion', r'how do you feel'],
                'weight': 0.6
            },
            'memory_intensive': {
                'patterns': [r'remember', r'recall', r'what did', r'before', r'previously',
                           r'history', r'past', r'earlier', r'connection', r'who is', r'what is',
                           r'tell me about', r'who was', r'what was', r'do you know'],
                'weight': 0.6
            }
        }
        
        # Length-based complexity factors
        self.length_thresholds = {
            'very_short': 20,    # "Hi" - minimal processing
            'short': 50,         # "How are you today?" - standard processing
            'medium': 150,       # Paragraph - may need deep processing
            'long': 400,         # Multiple paragraphs - likely complex
        }
        
        logger.info("🎯 Message complexity analyzer initialized - Ready for intelligent processing!")
    
    def analyze_complexity(self, message: str, conversation_context: List[Any] = None) -> ComplexityAssessment:
        """
        Analyze message complexity and recommend processing approach
        
        This is the performance optimization key - intelligent resource allocation
        """
        try:
            message_lower = message.lower().strip()
            factors = {}
            
            # 1. Quick simple pattern check (immediate optimization)
            if self._is_simple_query(message_lower):
                return ComplexityAssessment(
                    complexity_score=0.1,
                    processing_recommendation='minimal',
                    reasoning='Simple greeting or acknowledgment pattern detected',
                    estimated_processing_time=0.5,
                    should_use_planning=False,
                    should_use_coherence=False,
                    should_use_metacognitive=False,
                    should_use_deep_memory=False,
                    complexity_factors={'simple_pattern': 1.0}
                )
            
            # 2. Length-based complexity
            length_factor = self._calculate_length_complexity(message)
            factors['length'] = length_factor
            
            # 3. Semantic complexity indicators
            semantic_factors = self._analyze_semantic_complexity(message_lower)
            factors.update(semantic_factors)
            
            # 4. Question complexity
            question_factor = self._analyze_question_complexity(message_lower)
            factors['question_complexity'] = question_factor
            
            # 5. Context dependency
            context_factor = self._analyze_context_dependency(message_lower, conversation_context)
            factors['context_dependency'] = context_factor
            
            # 6. Calculate overall complexity score
            complexity_score = self._calculate_overall_complexity(factors)
            
            # 7. Generate processing recommendation
            recommendation = self._get_processing_recommendation(complexity_score, factors)
            
            # 8. Determine specific processing needs
            processing_needs = self._determine_processing_needs(complexity_score, factors)
            
            return ComplexityAssessment(
                complexity_score=complexity_score,
                processing_recommendation=recommendation['level'],
                reasoning=recommendation['reasoning'],
                estimated_processing_time=recommendation['time'],
                should_use_planning=processing_needs['planning'],
                should_use_coherence=processing_needs['coherence'],
                should_use_metacognitive=processing_needs['metacognitive'],
                should_use_deep_memory=processing_needs['deep_memory'],
                complexity_factors=factors
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing message complexity: {e}")
            # Safe fallback - standard processing
            return ComplexityAssessment(
                complexity_score=0.5,
                processing_recommendation='standard',
                reasoning=f'Error in complexity analysis: {e}',
                estimated_processing_time=2.0,
                should_use_planning=False,
                should_use_coherence=False,
                should_use_metacognitive=False,
                should_use_deep_memory=True,
                complexity_factors={'error': 1.0}
            )
    
    def _is_simple_query(self, message_lower: str) -> bool:
        """Quick check for simple patterns that don't need deep processing"""
        for pattern in self.simple_patterns:
            if re.search(pattern, message_lower):
                return True
        return False
    
    def _calculate_length_complexity(self, message: str) -> float:
        """Calculate complexity based on message length"""
        length = len(message)
        
        if length <= self.length_thresholds['very_short']:
            return 0.1
        elif length <= self.length_thresholds['short']:
            return 0.3
        elif length <= self.length_thresholds['medium']:
            return 0.6
        elif length <= self.length_thresholds['long']:
            return 0.8
        else:
            return 1.0
    
    def _analyze_semantic_complexity(self, message_lower: str) -> Dict[str, float]:
        """Analyze semantic indicators of complexity"""
        semantic_scores = {}
        
        for category, config in self.complexity_indicators.items():
            score = 0.0
            matches = 0
            
            for pattern in config['patterns']:
                if re.search(pattern, message_lower):
                    matches += 1
                    score += config['weight']
            
            # Score based on matches found, not total patterns
            if matches > 0:
                # Each match contributes the weight, up to the max weight
                semantic_scores[category] = min(score, config['weight'])
            else:
                semantic_scores[category] = 0.0
        
        return semantic_scores
    
    def _analyze_question_complexity(self, message_lower: str) -> float:
        """Analyze question complexity"""
        question_indicators = {
            'simple': [r'\?$', r'^what time', r'^what date', r'^how are'],
            'moderate': [r'^how do', r'^what is', r'^where', r'^when', r'^who'],
            'complex': [r'^why', r'^how would', r'^what if', r'^should i'],
            'philosophical': [r'^what does.*mean', r'^why do we', r'^what is the meaning']
        }
        
        for level, patterns in question_indicators.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    if level == 'simple':
                        return 0.2
                    elif level == 'moderate':
                        return 0.5
                    elif level == 'complex':
                        return 0.8
                    elif level == 'philosophical':
                        return 1.0
        
        # Check for multiple question marks (complex queries)
        question_marks = message_lower.count('?')
        if question_marks > 1:
            return min(0.6 + (question_marks * 0.1), 1.0)
        
        return 0.3  # Default for queries without obvious complexity indicators
    
    def _analyze_context_dependency(self, message_lower: str, conversation_context: List[Any]) -> float:
        """Analyze how much the query depends on conversation context"""
        context_indicators = [
            r'\bthis\b', r'\bthat\b', r'\bit\b', r'\bthey\b', r'\bthem\b',
            r'\bwe discussed\b', r'\bearlier\b', r'\bbefore\b', r'\babove\b',
            r'\bcontinue\b', r'\balso\b', r'\bfurther\b'
        ]
        
        context_score = 0.0
        for indicator in context_indicators:
            if re.search(indicator, message_lower):
                context_score += 0.2
        
        # Higher context dependency if conversation is ongoing
        if conversation_context and len(conversation_context) > 3:
            context_score += 0.3
        
        return min(context_score, 1.0)
    
    def _calculate_overall_complexity(self, factors: Dict[str, float]) -> float:
        """Calculate weighted overall complexity score"""
        weights = {
            'length': 0.2,
            'philosophical': 0.25,
            'strategic': 0.2,
            'analytical': 0.15,
            'reflective': 0.15,
            'creative': 0.1,
            'memory_intensive': 0.1,
            'question_complexity': 0.15,
            'context_dependency': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor, score in factors.items():
            if factor in weights:
                weighted_score += score * weights[factor]
                total_weight += weights[factor]
        
        # Normalize by actual weights used
        if total_weight > 0:
            return min(weighted_score / total_weight, 1.0)
        else:
            return 0.5  # Default fallback
    
    def _get_processing_recommendation(self, complexity_score: float, factors: Dict[str, float]) -> Dict[str, Any]:
        """Get processing level recommendation with reasoning"""
        if complexity_score <= 0.2:
            return {
                'level': 'minimal',
                'reasoning': 'Simple query with low complexity indicators',
                'time': 0.5
            }
        elif complexity_score <= 0.4:
            return {
                'level': 'standard',
                'reasoning': 'Moderate complexity requiring basic memory recall',
                'time': 1.5
            }
        elif complexity_score <= 0.7:
            return {
                'level': 'deep',
                'reasoning': 'Complex query benefiting from enhanced analysis',
                'time': 4.0
            }
        else:
            return {
                'level': 'comprehensive',
                'reasoning': 'Highly complex query requiring full consciousness capabilities',
                'time': 8.0
            }
    
    def _determine_processing_needs(self, complexity_score: float, factors: Dict[str, float]) -> Dict[str, bool]:
        """Determine which processing modules to activate - FIXED for memory recall"""
        return {
            'planning': (
                complexity_score > 0.8 and
                factors.get('strategic', 0) > 0.5
            ),
            'coherence': (
                complexity_score > 0.9 and 
                factors.get('philosophical', 0) > 0.6
            ),
            'metacognitive': (
                complexity_score > 0.8 and 
                factors.get('reflective', 0) > 0.5 and
                factors.get('philosophical', 0) > 0.5
            ),
            'deep_memory': (
                # CRITICAL FIX: Always enable memory for memory-intensive patterns
                factors.get('memory_intensive', 0) > 0.1 or
                complexity_score > 0.2 or 
                factors.get('context_dependency', 0) > 0.3
            )
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            'analyzer_type': 'MessageComplexityAnalyzer',
            'simple_patterns_count': len(self.simple_patterns),
            'complexity_categories': len(self.complexity_indicators),
            'optimization_focus': 'Performance-first with intelligent resource allocation'
        } 