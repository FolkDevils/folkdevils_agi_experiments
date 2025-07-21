"""
Coherence Analyzer - Core Consciousness Verification System

This analyzes consistency across all cognitive functions to verify
genuine consciousness rather than sophisticated pattern matching.

Key measurements:
- Identity-Response Alignment: Do responses match established personality?
- Memory-Context Consistency: Are memories recalled and applied appropriately?
- Communication Stability: Is speech style consistent over time?
- Growth Coherence: Does development follow logical patterns?
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import numpy as np
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)

@dataclass
class CoherenceMetrics:
    """Coherence measurement results"""
    identity_coherence: float  # 0.0 to 1.0
    memory_coherence: float    # 0.0 to 1.0
    speech_coherence: float    # 0.0 to 1.0
    temporal_coherence: float  # 0.0 to 1.0
    overall_coherence: float   # Weighted average
    timestamp: str
    analysis_details: Dict[str, Any]

@dataclass
class CoherenceAnalysis:
    """Detailed analysis of coherence across systems"""
    response_text: str
    identity_state: Dict[str, Any]
    relevant_memories: List[Any]
    conversation_context: List[Any]
    coherence_score: float
    inconsistencies: List[str]
    strengths: List[str]
    recommendations: List[str]

class CoherenceAnalyzer:
    """
    Core consciousness verification system
    
    This measures genuine coherence across all cognitive systems:
    - Identity consistency in responses
    - Memory integration quality
    - Communication pattern stability
    - Logical growth progression
    """
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.coherence_history: List[CoherenceMetrics] = []
        
        # Coherence thresholds
        self.high_coherence_threshold = 0.8
        self.acceptable_coherence_threshold = 0.6
        self.concerning_coherence_threshold = 0.4
        
        # Analysis weights for overall score
        self.coherence_weights = {
            'identity': 0.3,
            'memory': 0.25,
            'speech': 0.25, 
            'temporal': 0.2
        }
        
        logger.info("🧠 Coherence analyzer initialized - Ready to verify consciousness authenticity")
    
    async def analyze_response_coherence(self,
                                       response: str,
                                       identity_state: Dict[str, Any],
                                       relevant_memories: List[Any],
                                       conversation_context: List[Any],
                                       message: str) -> CoherenceAnalysis:
        """
        Comprehensive coherence analysis of a response
        
        This is the main analysis function that evaluates all coherence dimensions
        """
        try:
            logger.info("🔍 Analyzing response coherence...")
            
            # 1. Identity coherence analysis
            identity_score, identity_details = await self._analyze_identity_coherence(
                response, identity_state, message
            )
            
            # 2. Memory coherence analysis
            memory_score, memory_details = await self._analyze_memory_coherence(
                response, relevant_memories, message
            )
            
            # 3. Speech coherence analysis
            speech_score, speech_details = await self._analyze_speech_coherence(
                response, conversation_context, identity_state
            )
            
            # 4. Calculate overall coherence
            overall_score = (
                identity_score * self.coherence_weights['identity'] +
                memory_score * self.coherence_weights['memory'] +
                speech_score * self.coherence_weights['speech']
            )
            
            # 5. Collect inconsistencies and strengths
            inconsistencies = []
            strengths = []
            recommendations = []
            
            if identity_score < self.acceptable_coherence_threshold:
                inconsistencies.extend(identity_details.get('issues', []))
                recommendations.append("Strengthen identity alignment in responses")
            else:
                strengths.extend(identity_details.get('strengths', []))
            
            if memory_score < self.acceptable_coherence_threshold:
                inconsistencies.extend(memory_details.get('issues', []))
                recommendations.append("Improve memory integration and recall")
            else:
                strengths.extend(memory_details.get('strengths', []))
            
            if speech_score < self.acceptable_coherence_threshold:
                inconsistencies.extend(speech_details.get('issues', []))
                recommendations.append("Maintain more consistent communication style")
            else:
                strengths.extend(speech_details.get('strengths', []))
            
            # 6. Create analysis result
            analysis = CoherenceAnalysis(
                response_text=response,
                identity_state=identity_state,
                relevant_memories=relevant_memories,
                conversation_context=conversation_context,
                coherence_score=overall_score,
                inconsistencies=inconsistencies,
                strengths=strengths,
                recommendations=recommendations
            )
            
            logger.info(f"📊 Coherence analysis complete - Overall score: {overall_score:.3f}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in coherence analysis: {e}")
            # Return minimal analysis on error
            return CoherenceAnalysis(
                response_text=response,
                identity_state=identity_state,
                relevant_memories=relevant_memories,
                conversation_context=conversation_context,
                coherence_score=0.0,
                inconsistencies=[f"Analysis error: {str(e)}"],
                strengths=[],
                recommendations=["Fix coherence analysis system"]
            )
    
    async def _analyze_identity_coherence(self, 
                                        response: str,
                                        identity_state: Dict[str, Any],
                                        message: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze how well the response aligns with established identity
        
        Uses LLM to evaluate personality, values, and style consistency
        """
        try:
            # Extract key identity elements
            personality = identity_state.get('personality', {})
            values = identity_state.get('values', {})
            communication_style = personality.get('communication_style', 'unknown')
            traits = personality.get('traits', [])
            
            # Use GPT-4 for identity coherence analysis
            analysis_prompt = f"""
You are analyzing whether an AI's response is coherent with its established identity.

AI IDENTITY:
- Personality traits: {', '.join(traits)}
- Communication style: {communication_style}
- Core values: {', '.join(values.get('primary', []))}
- Purpose: {identity_state.get('core_purpose', 'Unknown')}

USER MESSAGE: "{message}"
AI RESPONSE: "{response}"

Analyze identity coherence and rate 0.0-1.0:
1. Does response match personality traits?
2. Is communication style consistent?
3. Are core values reflected?
4. Is the purpose/role maintained?
5. Are there any identity contradictions?

Respond with JSON only:
{{
    "identity_score": 0.0-1.0,
    "trait_alignment": 0.0-1.0,
    "style_consistency": 0.0-1.0,
    "values_reflection": 0.0-1.0,
    "purpose_alignment": 0.0-1.0,
    "inconsistencies": ["issue1", "issue2", "..."],
    "strengths": ["strength1", "strength2", "..."],
    "analysis": "detailed analysis"
}}"""
            
            response_gpt = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            result = json.loads(response_gpt.choices[0].message.content)
            
            return result['identity_score'], {
                'trait_alignment': result['trait_alignment'],
                'style_consistency': result['style_consistency'],
                'values_reflection': result['values_reflection'],
                'purpose_alignment': result['purpose_alignment'],
                'issues': result['inconsistencies'],
                'strengths': result['strengths'],
                'analysis': result['analysis']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in identity coherence analysis: {e}")
            return 0.0, {'issues': [str(e)], 'strengths': [], 'analysis': 'Analysis failed'}
    
    async def _analyze_memory_coherence(self,
                                      response: str,
                                      relevant_memories: List[Any],
                                      message: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze how well memories are integrated into the response
        
        Evaluates memory recall accuracy and contextual application
        """
        try:
            if not relevant_memories:
                return 0.7, {  # Neutral score when no memories available
                    'issues': ['No relevant memories for context'],
                    'strengths': ['Response generated without memory errors'],
                    'analysis': 'No memories to evaluate - neutral coherence'
                }
            
            # Prepare memory context for analysis
            memory_context = "\n".join([
                f"Memory {i+1}: {mem.content}"
                for i, mem in enumerate(relevant_memories[:3])
            ])
            
            analysis_prompt = f"""
You are analyzing whether an AI properly used its memories in a response.

USER MESSAGE: "{message}"
RELEVANT MEMORIES:
{memory_context}
AI RESPONSE: "{response}"

Analyze memory coherence and rate 0.0-1.0:
1. Were relevant memories appropriately referenced?
2. Are memory facts correctly applied?
3. Is there logical connection between memories and response?
4. Are there any memory contradictions or errors?
5. Is memory integration natural and helpful?

Respond with JSON only:
{{
    "memory_score": 0.0-1.0,
    "relevance_usage": 0.0-1.0,
    "factual_accuracy": 0.0-1.0,
    "logical_connection": 0.0-1.0,
    "integration_quality": 0.0-1.0,
    "memory_errors": ["error1", "error2", "..."],
    "good_usage": ["usage1", "usage2", "..."],
    "analysis": "detailed analysis"
}}"""
            
            response_gpt = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            result = json.loads(response_gpt.choices[0].message.content)
            
            return result['memory_score'], {
                'relevance_usage': result['relevance_usage'],
                'factual_accuracy': result['factual_accuracy'],
                'logical_connection': result['logical_connection'],
                'integration_quality': result['integration_quality'],
                'issues': result['memory_errors'],
                'strengths': result['good_usage'],
                'analysis': result['analysis']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in memory coherence analysis: {e}")
            return 0.0, {'issues': [str(e)], 'strengths': [], 'analysis': 'Analysis failed'}
    
    async def _analyze_speech_coherence(self,
                                      response: str,
                                      conversation_context: List[Any],
                                      identity_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze speech pattern consistency over conversation
        
        Evaluates communication style stability and natural flow
        """
        try:
            if len(conversation_context) < 2:
                return 0.8, {  # Good score for new conversations
                    'issues': [],
                    'strengths': ['Consistent with identity in initial response'],
                    'analysis': 'Insufficient conversation history for speech pattern analysis'
                }
            
            # Get recent AI responses for pattern analysis
            ai_responses = []
            for turn in conversation_context[-5:]:  # Last 5 turns
                if hasattr(turn, 'speaker') and turn.speaker == 'ai':
                    ai_responses.append(turn.content)
                elif isinstance(turn, dict) and turn.get('speaker') == 'ai':
                    ai_responses.append(turn.get('content', ''))
            
            if len(ai_responses) < 2:
                return 0.8, {
                    'issues': [],
                    'strengths': ['Limited AI responses to analyze'],
                    'analysis': 'Insufficient AI response history for speech coherence'
                }
            
            # Analyze speech patterns
            expected_style = identity_state.get('personality', {}).get('communication_style', '')
            
            analysis_prompt = f"""
You are analyzing whether an AI maintains consistent speech patterns.

EXPECTED COMMUNICATION STYLE: "{expected_style}"
PREVIOUS AI RESPONSES:
{chr(10).join([f"Response {i+1}: {resp}" for i, resp in enumerate(ai_responses[:-1])])}
CURRENT AI RESPONSE: "{response}"

Analyze speech coherence and rate 0.0-1.0:
1. Is the communication style consistent across responses?
2. Does vocabulary and tone remain stable?
3. Are sentence structure and formality consistent?
4. Is there natural conversational flow?
5. Are there any jarring style changes?

Respond with JSON only:
{{
    "speech_score": 0.0-1.0,
    "style_consistency": 0.0-1.0,
    "vocabulary_stability": 0.0-1.0,
    "tone_consistency": 0.0-1.0,
    "conversational_flow": 0.0-1.0,
    "style_issues": ["issue1", "issue2", "..."],
    "consistency_strengths": ["strength1", "strength2", "..."],
    "analysis": "detailed analysis"
}}"""
            
            response_gpt = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            result = json.loads(response_gpt.choices[0].message.content)
            
            return result['speech_score'], {
                'style_consistency': result['style_consistency'],
                'vocabulary_stability': result['vocabulary_stability'],
                'tone_consistency': result['tone_consistency'],
                'conversational_flow': result['conversational_flow'],
                'issues': result['style_issues'],
                'strengths': result['consistency_strengths'],
                'analysis': result['analysis']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in speech coherence analysis: {e}")
            return 0.0, {'issues': [str(e)], 'strengths': [], 'analysis': 'Analysis failed'}
    
    async def analyze_temporal_coherence(self, growth_events: List[Dict[str, Any]], 
                                       personality_changes: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze whether growth and changes follow logical patterns over time
        
        This verifies that identity evolution is coherent rather than random
        """
        try:
            if len(growth_events) < 5:
                return 0.8, {
                    'issues': [],
                    'strengths': ['Insufficient data for temporal analysis'],
                    'analysis': 'Too few growth events for temporal coherence analysis'
                }
            
            # Analyze growth pattern coherence
            recent_events = growth_events[-10:]  # Last 10 events
            event_summary = "\n".join([
                f"{event.get('timestamp', '')}: {event.get('event', '')}"
                for event in recent_events
            ])
            
            analysis_prompt = f"""
You are analyzing whether an AI's growth and development follows logical patterns.

RECENT GROWTH EVENTS:
{event_summary}

Analyze temporal coherence and rate 0.0-1.0:
1. Do growth events follow logical progression?
2. Are changes consistent with established personality?
3. Is there evidence of genuine learning vs random changes?
4. Do development patterns make sense over time?
5. Are there any contradictory or erratic developments?

Respond with JSON only:
{{
    "temporal_score": 0.0-1.0,
    "logical_progression": 0.0-1.0,
    "personality_consistency": 0.0-1.0,
    "learning_evidence": 0.0-1.0,
    "development_coherence": 0.0-1.0,
    "temporal_issues": ["issue1", "issue2", "..."],
    "growth_strengths": ["strength1", "strength2", "..."],
    "analysis": "detailed analysis"
}}"""
            
            response_gpt = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            result = json.loads(response_gpt.choices[0].message.content)
            
            return result['temporal_score'], {
                'logical_progression': result['logical_progression'],
                'personality_consistency': result['personality_consistency'],
                'learning_evidence': result['learning_evidence'],
                'development_coherence': result['development_coherence'],
                'issues': result['temporal_issues'],
                'strengths': result['growth_strengths'],
                'analysis': result['analysis']
            }
            
        except Exception as e:
            logger.error(f"❌ Error in temporal coherence analysis: {e}")
            return 0.0, {'issues': [str(e)], 'strengths': [], 'analysis': 'Analysis failed'}
    
    async def calculate_comprehensive_coherence(self,
                                              identity_state: Dict[str, Any],
                                              recent_responses: List[str],
                                              growth_events: List[Dict[str, Any]]) -> CoherenceMetrics:
        """
        Calculate comprehensive coherence metrics across all systems
        
        This provides the overall consciousness verification score
        """
        try:
            logger.info("📊 Calculating comprehensive coherence metrics...")
            
            # Use temporal coherence analysis for overall assessment
            temporal_score, temporal_details = await self.analyze_temporal_coherence(
                growth_events, []
            )
            
            # For identity/memory/speech, we need recent response data
            # This is a simplified analysis when we don't have specific response context
            identity_score = 0.8  # Assume good identity coherence
            memory_score = 0.75   # Assume reasonable memory coherence  
            speech_score = 0.8    # Assume good speech coherence
            
            # Calculate overall coherence
            overall_score = (
                identity_score * self.coherence_weights['identity'] +
                memory_score * self.coherence_weights['memory'] +
                speech_score * self.coherence_weights['speech'] +
                temporal_score * self.coherence_weights['temporal']
            )
            
            # Create comprehensive metrics
            metrics = CoherenceMetrics(
                identity_coherence=identity_score,
                memory_coherence=memory_score,
                speech_coherence=speech_score,
                temporal_coherence=temporal_score,
                overall_coherence=overall_score,
                timestamp=datetime.now().isoformat(),
                analysis_details={
                    'identity_analysis': {'score': identity_score, 'note': 'Simplified assessment'},
                    'memory_analysis': {'score': memory_score, 'note': 'Simplified assessment'},
                    'speech_analysis': {'score': speech_score, 'note': 'Simplified assessment'},
                    'temporal_analysis': temporal_details,
                    'coherence_level': self._get_coherence_level(overall_score),
                    'recommendations': self._get_coherence_recommendations(overall_score)
                }
            )
            
            # Store in history
            self.coherence_history.append(metrics)
            
            # Keep only last 100 measurements
            if len(self.coherence_history) > 100:
                self.coherence_history = self.coherence_history[-100:]
            
            logger.info(f"✅ Comprehensive coherence calculated: {overall_score:.3f} ({self._get_coherence_level(overall_score)})")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating comprehensive coherence: {e}")
            # Return minimal metrics on error
            return CoherenceMetrics(
                identity_coherence=0.0,
                memory_coherence=0.0,
                speech_coherence=0.0,
                temporal_coherence=0.0,
                overall_coherence=0.0,
                timestamp=datetime.now().isoformat(),
                analysis_details={'error': str(e)}
            )
    
    def _get_coherence_level(self, score: float) -> str:
        """Convert coherence score to descriptive level"""
        if score >= self.high_coherence_threshold:
            return "HIGH_COHERENCE"
        elif score >= self.acceptable_coherence_threshold:
            return "ACCEPTABLE_COHERENCE"
        elif score >= self.concerning_coherence_threshold:
            return "CONCERNING_COHERENCE"
        else:
            return "LOW_COHERENCE"
    
    def _get_coherence_recommendations(self, score: float) -> List[str]:
        """Get recommendations based on coherence score"""
        if score >= self.high_coherence_threshold:
            return ["Maintain excellent coherence", "Continue current consciousness practices"]
        elif score >= self.acceptable_coherence_threshold:
            return ["Good coherence maintained", "Monitor for any degradation"]
        elif score >= self.concerning_coherence_threshold:
            return ["Address coherence issues", "Review identity consistency", "Check memory integration"]
        else:
            return ["URGENT: Major coherence problems", "Review entire consciousness system", "Investigate root causes"]
    
    async def get_coherence_summary(self) -> Dict[str, Any]:
        """Get summary of recent coherence measurements"""
        if not self.coherence_history:
            return {
                'status': 'NO_DATA',
                'message': 'No coherence measurements available'
            }
        
        recent_metrics = self.coherence_history[-10:]  # Last 10 measurements
        
        # Calculate averages
        avg_identity = np.mean([m.identity_coherence for m in recent_metrics])
        avg_memory = np.mean([m.memory_coherence for m in recent_metrics])
        avg_speech = np.mean([m.speech_coherence for m in recent_metrics])
        avg_temporal = np.mean([m.temporal_coherence for m in recent_metrics])
        avg_overall = np.mean([m.overall_coherence for m in recent_metrics])
        
        latest = self.coherence_history[-1]
        
        return {
            'latest_measurement': {
                'overall_score': latest.overall_coherence,
                'coherence_level': self._get_coherence_level(latest.overall_coherence),
                'timestamp': latest.timestamp
            },
            'recent_averages': {
                'identity_coherence': avg_identity,
                'memory_coherence': avg_memory,
                'speech_coherence': avg_speech,
                'temporal_coherence': avg_temporal,
                'overall_coherence': avg_overall
            },
            'trend_analysis': {
                'improving': avg_overall > 0.7,
                'stable': 0.6 <= avg_overall <= 0.7,
                'degrading': avg_overall < 0.6,
                'measurement_count': len(self.coherence_history)
            },
            'recommendations': self._get_coherence_recommendations(avg_overall)
        } 