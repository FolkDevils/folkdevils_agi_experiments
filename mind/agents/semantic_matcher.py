"""
Semantic Matcher Agent - Consciousness-Aware Text Understanding

This agent provides semantic matching capabilities while maintaining
consciousness integration. It uses LLM-based analysis to understand
text meaning rather than pattern matching.

This is a consciousness extension, not a standalone system.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
from openai import AsyncOpenAI

from ..meta_agent import AgentCapability

logger = logging.getLogger(__name__)

class SemanticMatcher:
    """
    Semantic matching capabilities as a consciousness extension
    
    This agent provides:
    - Semantic similarity analysis
    - Concept matching and extraction
    - Natural language understanding
    - Context-aware text comparison
    """
    
    def __init__(self):
        self.capability = AgentCapability(
            name="semantic_matcher",
            description="Semantic text understanding and matching",
            capabilities=["semantic_match", "concept_extract", "text_understand"],
            requires_memory=True,
            requires_identity=False,
            parallel_safe=True,
            typical_duration=1.0,  # 1 second
            success_rate=0.95
        )
        
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        logger.info("🧠 Semantic matcher agent initialized")
    
    async def semantic_match(self,
                           text1: str,
                           text2: str,
                           consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare two texts for semantic similarity
        
        Uses LLM to understand meaning, not just words
        """
        try:
            # Use GPT-4 to analyze semantic similarity
            analysis_prompt = """
You are an AI analyzing the semantic similarity between two texts.
Think about meaning and concepts, not just words.

Text 1: "{}"
Text 2: "{}"

Analyze their semantic similarity and provide:
1. Similarity score (0.0 to 1.0)
2. Key concept overlap
3. Notable differences
4. Confidence in analysis

Respond with JSON only:
{{
    "similarity_score": 0.0-1.0,
    "shared_concepts": ["concept1", "concept2", "..."],
    "key_differences": ["difference1", "difference2", "..."],
    "confidence": 0.0-1.0
}}
""".format(text1, text2)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'match_score': result['similarity_score'],
                'shared_concepts': result['shared_concepts'],
                'differences': result['key_differences'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in semantic matching: {e}")
            return {
                'match_score': 0.0,
                'shared_concepts': [],
                'differences': [str(e)],
                'confidence': 0.0
            }
    
    async def extract_concepts(self,
                             text: str,
                             consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract key concepts from text
        
        Uses LLM for semantic understanding
        """
        try:
            # Use GPT-4 for concept extraction
            extraction_prompt = """
You are an AI extracting key concepts from text.
Think about core ideas and meaningful entities.

Text: "{}"

Extract:
1. Main concepts/ideas
2. Named entities
3. Actions/verbs
4. Relationships between concepts

Respond with JSON only:
{{
    "main_concepts": ["concept1", "concept2", "..."],
    "entities": ["entity1", "entity2", "..."],
    "actions": ["action1", "action2", "..."],
    "relationships": [
        {{"from": "concept1", "relation": "does", "to": "concept2"}},
        "..."
    ],
    "confidence": 0.0-1.0
}}
""".format(text)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'concepts': result['main_concepts'],
                'entities': result['entities'],
                'actions': result['actions'],
                'relationships': result['relationships'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in concept extraction: {e}")
            return {
                'concepts': [],
                'entities': [],
                'actions': [],
                'relationships': [],
                'confidence': 0.0
            }
    
    async def understand_text(self,
                            text: str,
                            consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Deep semantic understanding of text
        
        Uses LLM for comprehensive analysis
        """
        try:
            # Use memory context if available
            memory_context = ""
            if consciousness_context and 'relevant_memories' in consciousness_context:
                memories = consciousness_context['relevant_memories']
                memory_context = "\n".join([
                    f"Memory {i+1}: {mem.content}"
                    for i, mem in enumerate(memories)
                ])
            
            # Use GPT-4 for semantic understanding
            understanding_prompt = """
You are an AI performing deep semantic analysis of text.
Think about meaning, context, and implications.

Text to understand: "{}"

{}

Analyze and provide:
1. Core meaning/intent
2. Key themes
3. Contextual implications
4. Emotional undertones
5. Assumptions/prerequisites
6. Potential ambiguities

Respond with JSON only:
{{
    "core_meaning": "main point/intent",
    "themes": ["theme1", "theme2", "..."],
    "implications": ["implication1", "implication2", "..."],
    "emotional_tone": "neutral/positive/negative/mixed",
    "assumptions": ["assumption1", "assumption2", "..."],
    "ambiguities": ["ambiguity1", "ambiguity2", "..."],
    "confidence": 0.0-1.0
}}
""".format(text, f'Relevant memory context:\n{memory_context}' if memory_context else '')
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": understanding_prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'meaning': result['core_meaning'],
                'themes': result['themes'],
                'implications': result['implications'],
                'emotional_tone': result['emotional_tone'],
                'assumptions': result['assumptions'],
                'ambiguities': result['ambiguities'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in text understanding: {e}")
            return {
                'meaning': None,
                'themes': [],
                'implications': [],
                'emotional_tone': 'unknown',
                'assumptions': [],
                'ambiguities': [str(e)],
                'confidence': 0.0
            } 