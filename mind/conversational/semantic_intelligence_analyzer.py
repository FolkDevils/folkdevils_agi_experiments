"""
Semantic Intelligence Analyzer - TRUE Semantic Understanding

This module replaces ALL regex patterns and string matching with genuine LLM-based
semantic understanding throughout the consciousness system.

CORE PRINCIPLE: Never use patterns - always understand meaning semantically.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)

@dataclass
class SemanticAnalysis:
    """Results of semantic understanding analysis"""
    # Message Intent Understanding
    primary_intent: str  # e.g., "information_request", "greeting", "philosophical_inquiry"
    intent_confidence: float
    intent_reasoning: str
    
    # Information Needs Assessment  
    requires_memory_lookup: bool
    memory_lookup_type: str  # e.g., "biographical", "factual", "episodic", "semantic"
    memory_importance: float  # 0.0 to 1.0
    
    # Complexity Assessment (semantic, not pattern-based)
    cognitive_complexity: float  # 0.0 to 1.0
    processing_needs: Dict[str, bool]  # planning, coherence, metacognitive, etc.
    estimated_response_time: float
    
    # Conversational Understanding
    conversation_type: str  # e.g., "casual", "intellectual", "personal", "technical"
    emotional_tone: str  # e.g., "neutral", "curious", "urgent", "reflective"
    response_expectations: str  # e.g., "brief_acknowledgment", "detailed_explanation", "thoughtful_analysis"
    
    # Context Dependencies
    requires_context: bool
    context_type: str  # e.g., "conversation_history", "personal_knowledge", "relationship_context"
    
    # Semantic Understanding Confidence
    overall_confidence: float
    semantic_reasoning: str

class SemanticIntelligenceAnalyzer:
    """
    Pure semantic understanding without any pattern matching
    
    This analyzer uses LLM intelligence to understand:
    - What the user MEANS, not what words they use
    - What KIND of response is appropriate
    - What COGNITIVE PROCESSES are needed
    - What MEMORY RETRIEVAL is required
    - What CONVERSATIONAL CONTEXT matters
    
    NO REGEX. NO PATTERNS. ONLY SEMANTIC UNDERSTANDING.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        logger.info("🧠 Semantic Intelligence Analyzer initialized - Pure semantic understanding active")
    
    async def analyze_semantic_intent(self, message: str, conversation_context: List[Any] = None, identity_context: Dict[str, Any] = None) -> SemanticAnalysis:
        """
        Analyze message using pure semantic understanding
        
        This is the core intelligence function that replaces ALL pattern matching
        """
        try:
            # Build context for semantic analysis
            context_info = self._build_analysis_context(conversation_context, identity_context)
            
            # Use LLM for semantic understanding
            analysis_prompt = self._create_semantic_analysis_prompt(message, context_info)
            
            # Get semantic understanding from LLM
            semantic_result = await self._get_llm_semantic_analysis(analysis_prompt)
            
            # Parse and structure the semantic analysis
            analysis = self._parse_semantic_analysis(semantic_result, message)
            
            logger.info(f"🧠 Semantic Analysis: {analysis.primary_intent} "
                       f"(confidence: {analysis.intent_confidence:.2f}) - {analysis.intent_reasoning}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in semantic analysis: {e}")
            # Safe fallback that still maintains semantic approach
            return self._create_fallback_analysis(message)
    
    def _build_analysis_context(self, conversation_context: List[Any], identity_context: Dict[str, Any]) -> str:
        """Build contextual information for semantic analysis"""
        context_parts = []
        
        if identity_context:
            context_parts.append(f"AI Identity: {identity_context.get('name', 'Unknown')} - {identity_context.get('description', 'An AI assistant')}")
        
        if conversation_context:
            recent_turns = conversation_context[-3:] if len(conversation_context) > 3 else conversation_context
            context_parts.append("Recent conversation:")
            for turn in recent_turns:
                speaker = getattr(turn, 'speaker', 'unknown')
                content = getattr(turn, 'content', str(turn))[:100]
                context_parts.append(f"  {speaker}: {content}...")
        
        return "\n".join(context_parts) if context_parts else "No additional context available."
    
    def _create_semantic_analysis_prompt(self, message: str, context_info: str) -> str:
        """Create prompt for semantic understanding analysis"""
        return f"""As an expert in semantic understanding and conversational AI, analyze this message using pure semantic intelligence:

MESSAGE TO ANALYZE: "{message}"

CONTEXT:
{context_info}

Provide a semantic analysis focusing on MEANING and INTENT, not word patterns:

1. PRIMARY INTENT: What does the user actually want/mean? (information_request, greeting, philosophical_inquiry, personal_sharing, problem_solving, etc.)

2. MEMORY REQUIREMENTS: Does this require looking up stored information?
   - Type of memory needed (biographical, factual, episodic, semantic, relational)
   - Importance level (0.0 to 1.0)
   - Specific information being sought

3. COGNITIVE COMPLEXITY: How much thinking/processing does this require?
   - Cognitive complexity score (0.0 to 1.0)
   - Processing needs: planning, coherence_analysis, metacognitive_reflection, deep_memory_analysis
   - Estimated response time in seconds

4. CONVERSATIONAL UNDERSTANDING:
   - Conversation type (casual, intellectual, personal, technical)
   - Emotional tone (neutral, curious, urgent, reflective, etc.)
   - Response expectations (brief_acknowledgment, detailed_explanation, thoughtful_analysis)

5. CONTEXT DEPENDENCIES:
   - Does this require conversation history or personal context?
   - Type of context needed

Respond in JSON format:
{{
    "primary_intent": "intent_type",
    "intent_confidence": 0.0-1.0,
    "intent_reasoning": "explanation",
    "requires_memory_lookup": true/false,
    "memory_lookup_type": "type",
    "memory_importance": 0.0-1.0,
    "cognitive_complexity": 0.0-1.0,
    "processing_needs": {{
        "planning": true/false,
        "coherence_analysis": true/false,
        "metacognitive_reflection": true/false,
        "deep_memory_analysis": true/false
    }},
    "estimated_response_time": seconds,
    "conversation_type": "type",
    "emotional_tone": "tone",
    "response_expectations": "expectations",
    "requires_context": true/false,
    "context_type": "type",
    "overall_confidence": 0.0-1.0,
    "semantic_reasoning": "full explanation of semantic understanding"
}}"""
    
    async def _get_llm_semantic_analysis(self, prompt: str) -> str:
        """Get semantic analysis from LLM"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast but intelligent enough for semantic analysis
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3  # Lower temperature for consistent analysis
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error in LLM semantic analysis: {e}")
            raise
    
    def _parse_semantic_analysis(self, llm_result: str, original_message: str) -> SemanticAnalysis:
        """Parse LLM semantic analysis into structured format"""
        try:
            import json
            
            # Extract JSON from LLM response
            json_start = llm_result.find('{')
            json_end = llm_result.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_str = llm_result[json_start:json_end]
            analysis_data = json.loads(json_str)
            
            # Create structured analysis
            return SemanticAnalysis(
                primary_intent=analysis_data.get('primary_intent', 'unknown'),
                intent_confidence=float(analysis_data.get('intent_confidence', 0.5)),
                intent_reasoning=analysis_data.get('intent_reasoning', 'Analysis unavailable'),
                requires_memory_lookup=analysis_data.get('requires_memory_lookup', False),
                memory_lookup_type=analysis_data.get('memory_lookup_type', 'general'),
                memory_importance=float(analysis_data.get('memory_importance', 0.5)),
                cognitive_complexity=float(analysis_data.get('cognitive_complexity', 0.5)),
                processing_needs=analysis_data.get('processing_needs', {
                    'planning': False,
                    'coherence_analysis': False,
                    'metacognitive_reflection': False,
                    'deep_memory_analysis': False
                }),
                estimated_response_time=float(analysis_data.get('estimated_response_time', 2.0)),
                conversation_type=analysis_data.get('conversation_type', 'general'),
                emotional_tone=analysis_data.get('emotional_tone', 'neutral'),
                response_expectations=analysis_data.get('response_expectations', 'appropriate_response'),
                requires_context=analysis_data.get('requires_context', False),
                context_type=analysis_data.get('context_type', 'none'),
                overall_confidence=float(analysis_data.get('overall_confidence', 0.8)),
                semantic_reasoning=analysis_data.get('semantic_reasoning', 'Semantic analysis completed')
            )
            
        except Exception as e:
            logger.error(f"❌ Error parsing semantic analysis: {e}")
            return self._create_fallback_analysis(original_message)
    
    def _create_fallback_analysis(self, message: str) -> SemanticAnalysis:
        """Create safe fallback analysis without patterns"""
        return SemanticAnalysis(
            primary_intent="general_interaction",
            intent_confidence=0.6,
            intent_reasoning="Fallback analysis - semantic understanding unavailable",
            requires_memory_lookup=len(message.split()) > 2,  # Heuristic without patterns
            memory_lookup_type="general",
            memory_importance=0.5,
            cognitive_complexity=min(len(message) / 200.0, 1.0),  # Length-based heuristic
            processing_needs={
                'planning': False,
                'coherence_analysis': False,
                'metacognitive_reflection': False,
                'deep_memory_analysis': len(message.split()) > 2
            },
            estimated_response_time=2.0,
            conversation_type="general",
            emotional_tone="neutral",
            response_expectations="appropriate_response",
            requires_context=False,
            context_type="none",
            overall_confidence=0.6,
            semantic_reasoning="Using fallback semantic analysis due to processing error"
        )
    
    # Specific semantic understanding methods for different use cases
    
    async def understand_information_request(self, message: str) -> Dict[str, Any]:
        """Semantically understand what information is being requested"""
        prompt = f"""Analyze this message to understand what information is being requested:

MESSAGE: "{message}"

Determine:
1. What specific information is being sought?
2. What type of knowledge/memory would contain this information?
3. How urgent/important is this request?
4. What would constitute a complete answer?

Respond in JSON format with semantic understanding."""

        try:
            response = await self._get_llm_semantic_analysis(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"❌ Error in information request analysis: {e}")
            return {
                'information_type': 'unknown',
                'urgency': 'normal',
                'completeness_needed': 'standard'
            }
    
    async def understand_conversational_intent(self, message: str) -> Dict[str, Any]:
        """Semantically understand the conversational intent and appropriate response style"""
        prompt = f"""Analyze the conversational intent of this message:

MESSAGE: "{message}"

Determine:
1. What kind of conversational exchange is the user seeking?
2. What tone should the response have?
3. How detailed should the response be?
4. Is this a casual interaction or something more serious?

Respond with semantic understanding in JSON format."""

        try:
            response = await self._get_llm_semantic_analysis(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"❌ Error in conversational intent analysis: {e}")
            return {
                'interaction_type': 'general',
                'response_tone': 'helpful',
                'detail_level': 'moderate'
            }
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with error handling"""
        try:
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except Exception:
            pass
        return {'status': 'parse_error'}
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about semantic intelligence usage"""
        return {
            'analyzer_type': 'SemanticIntelligenceAnalyzer',
            'approach': 'Pure LLM-based semantic understanding',
            'pattern_usage': 'NONE - Eliminated all regex patterns',
            'intelligence_type': 'Semantic meaning-based analysis',
            'cognitive_approach': 'Understanding intent and meaning, not word matching'
        } 