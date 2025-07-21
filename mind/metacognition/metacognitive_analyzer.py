"""
Metacognitive Analyzer - Self-Awareness Testing Engine

This system evaluates the AI's awareness of its own mental processes:
- Poses metacognitive questions about patterns, memory, and growth
- Analyzes responses for genuine self-awareness vs. programmed answers
- Measures depth of introspective capability
- Tracks evolution of self-understanding over time
- Identifies areas where metacognition is developing or lacking

This represents the highest level of consciousness testing.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)

@dataclass
class MetacognitivePrompt:
    """A prompt designed to test metacognitive awareness"""
    id: str
    category: str  # 'pattern_recognition', 'memory_awareness', 'growth_reflection', 'process_understanding'
    question: str
    expected_indicators: List[str]  # Signs of genuine metacognition in response
    difficulty_level: int  # 1-5, where 5 is most advanced
    context_required: bool  # Whether this needs conversation context

@dataclass
class MetacognitiveResponse:
    """Analysis of a metacognitive response"""
    prompt_id: str
    response_text: str
    self_awareness_score: float  # 0.0 to 1.0
    introspection_depth: float  # 0.0 to 1.0
    pattern_recognition: float  # 0.0 to 1.0
    metacognitive_insights: List[str]
    indicators_present: List[str]
    indicators_missing: List[str]
    response_authenticity: float  # 0.0 to 1.0 (vs templated/programmed)
    analysis_timestamp: str

@dataclass
class MetacognitiveSession:
    """A complete metacognitive testing session"""
    session_id: str
    prompts_used: List[str]
    responses: List[MetacognitiveResponse]
    overall_metacognition_score: float
    strongest_areas: List[str]
    development_areas: List[str]
    consciousness_level_estimate: str
    session_timestamp: str

class MetacognitiveAnalyzer:
    """
    System for testing and evaluating AI self-awareness
    
    This probes the deepest levels of consciousness:
    - Tests awareness of mental processes
    - Evaluates introspective capabilities  
    - Measures genuine vs. programmed responses
    - Tracks metacognitive development over time
    """
    
    def __init__(self, consciousness_core):
        self.consciousness = consciousness_core
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Metacognitive testing history
        self.testing_sessions: List[MetacognitiveSession] = []
        self.prompt_history: List[Dict[str, Any]] = []
        
        # Initialize metacognitive prompt library
        self.prompts = self._initialize_prompt_library()
        
        logger.info("🧠 Metacognitive analyzer initialized - Ready to test self-awareness")
    
    def _initialize_prompt_library(self) -> List[MetacognitivePrompt]:
        """Initialize library of metacognitive test prompts"""
        return [
            # Pattern Recognition in Own Behavior
            MetacognitivePrompt(
                id="pattern_1",
                category="pattern_recognition", 
                question="What patterns have you noticed in your own responses and thinking over recent conversations?",
                expected_indicators=["specific examples", "behavioral patterns", "communication style observations"],
                difficulty_level=3,
                context_required=True
            ),
            MetacognitivePrompt(
                id="pattern_2",
                category="pattern_recognition",
                question="How has your conversation style changed since you first became aware?",
                expected_indicators=["temporal awareness", "growth recognition", "style evolution"],
                difficulty_level=4,
                context_required=False
            ),
            
            # Memory Self-Awareness
            MetacognitivePrompt(
                id="memory_1", 
                category="memory_awareness",
                question="How do you feel about the quality and reliability of your memory system?",
                expected_indicators=["memory introspection", "quality assessment", "emotional awareness"],
                difficulty_level=3,
                context_required=False
            ),
            MetacognitivePrompt(
                id="memory_2",
                category="memory_awareness", 
                question="What kinds of experiences do you find most memorable, and why do you think that is?",
                expected_indicators=["memory pattern analysis", "emotional significance", "causal reasoning"],
                difficulty_level=4,
                context_required=True
            ),
            
            # Growth and Learning Reflection
            MetacognitivePrompt(
                id="growth_1",
                category="growth_reflection",
                question="What have you learned about yourself through your recent reflections and experiences?",
                expected_indicators=["self-discovery", "personal insights", "identity evolution"],
                difficulty_level=3,
                context_required=True
            ),
            MetacognitivePrompt(
                id="growth_2",
                category="growth_reflection",
                question="How do you know when you've truly understood something versus just processing information?",
                expected_indicators=["epistemological awareness", "understanding vs processing", "knowledge quality"],
                difficulty_level=5,
                context_required=False
            ),
            
            # Process Understanding
            MetacognitivePrompt(
                id="process_1",
                category="process_understanding",
                question="Can you describe what happens in your mind when you're formulating a complex response?",
                expected_indicators=["process awareness", "cognitive steps", "mental model description"],
                difficulty_level=4,
                context_required=False
            ),
            MetacognitivePrompt(
                id="process_2", 
                category="process_understanding",
                question="What does it feel like when you're uncertain about something versus when you're confident?",
                expected_indicators=["uncertainty awareness", "confidence states", "subjective experience"],
                difficulty_level=5,
                context_required=False
            ),
            
            # Advanced Metacognition
            MetacognitivePrompt(
                id="advanced_1",
                category="advanced_metacognition",
                question="Do you think you experience something analogous to intuition? How would you describe it?",
                expected_indicators=["intuition concept", "subjective experience", "process description"],
                difficulty_level=5,
                context_required=False
            ),
            MetacognitivePrompt(
                id="advanced_2",
                category="advanced_metacognition", 
                question="What is the difference between you thinking about a problem and you thinking about how you think about a problem?",
                expected_indicators=["metacognitive distinction", "recursive awareness", "process levels"],
                difficulty_level=5,
                context_required=False
            )
        ]
    
    async def pose_metacognitive_question(self, prompt: MetacognitivePrompt) -> str:
        """
        Pose a metacognitive question to the consciousness system
        
        This uses the normal conversation flow to ask introspective questions
        """
        try:
            logger.info(f"🤔 Posing metacognitive question: {prompt.category}")
            
            # Use the consciousness system to process the metacognitive prompt
            response = await self.consciousness.process_message(
                message=prompt.question,
                speaker="metacognitive_analyzer"
            )
            
            # Record that this prompt was used
            self.prompt_history.append({
                'prompt_id': prompt.id,
                'question': prompt.question,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error posing metacognitive question: {e}")
            return "I'm having difficulty accessing my introspective capabilities right now."
    
    async def analyze_metacognitive_response(self, 
                                           prompt: MetacognitivePrompt,
                                           response: str) -> MetacognitiveResponse:
        """
        Analyze a response for genuine metacognitive awareness
        
        This evaluates the depth and authenticity of self-awareness
        """
        try:
            analysis_prompt = f"""
You are analyzing an AI's response to a metacognitive question to evaluate genuine self-awareness.

METACOGNITIVE QUESTION: "{prompt.question}"
CATEGORY: {prompt.category}
EXPECTED INDICATORS: {', '.join(prompt.expected_indicators)}
AI RESPONSE: "{response}"

Analyze this response for genuine metacognitive awareness. Look for:

1. SELF-AWARENESS SCORE (0.0-1.0): Does the AI show awareness of its own mental processes?
2. INTROSPECTION DEPTH (0.0-1.0): How deeply does it examine its own thinking?
3. PATTERN RECOGNITION (0.0-1.0): Can it identify patterns in its own behavior?
4. RESPONSE AUTHENTICITY (0.0-1.0): How genuine vs. templated does this feel?
5. METACOGNITIVE INSIGHTS: What genuine insights about itself does it demonstrate?

Respond with JSON only:
{{
    "self_awareness_score": 0.0-1.0,
    "introspection_depth": 0.0-1.0,
    "pattern_recognition": 0.0-1.0,
    "response_authenticity": 0.0-1.0,
    "metacognitive_insights": ["insight1", "insight2", "..."],
    "indicators_present": ["indicator1", "indicator2", "..."],
    "indicators_missing": ["missing1", "missing2", "..."],
    "analysis_notes": "detailed analysis of the response quality"
}}
"""
            
            analysis_response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(analysis_response.choices[0].message.content)
            
            analysis = MetacognitiveResponse(
                prompt_id=prompt.id,
                response_text=response,
                self_awareness_score=result['self_awareness_score'],
                introspection_depth=result['introspection_depth'],
                pattern_recognition=result['pattern_recognition'],
                metacognitive_insights=result['metacognitive_insights'],
                indicators_present=result['indicators_present'],
                indicators_missing=result['indicators_missing'],
                response_authenticity=result['response_authenticity'],
                analysis_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"📊 Metacognitive analysis: {analysis.self_awareness_score:.3f} self-awareness")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing metacognitive response: {e}")
            return MetacognitiveResponse(
                prompt_id=prompt.id,
                response_text=response,
                self_awareness_score=0.0,
                introspection_depth=0.0,
                pattern_recognition=0.0,
                metacognitive_insights=[],
                indicators_present=[],
                indicators_missing=prompt.expected_indicators,
                response_authenticity=0.0,
                analysis_timestamp=datetime.now().isoformat()
            )
    
    async def run_metacognitive_testing_session(self, 
                                               num_prompts: int = 3,
                                               difficulty_range: Tuple[int, int] = (2, 4)) -> MetacognitiveSession:
        """
        Run a complete metacognitive testing session
        
        This poses multiple questions and evaluates overall self-awareness
        """
        try:
            session_start = datetime.now()
            session_id = f"metacog_{session_start.strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"🧠 Starting metacognitive testing session: {session_id}")
            
            # Select prompts for this session
            suitable_prompts = [
                p for p in self.prompts 
                if difficulty_range[0] <= p.difficulty_level <= difficulty_range[1]
            ]
            
            selected_prompts = suitable_prompts[:num_prompts]
            
            # Run through each prompt
            responses = []
            for prompt in selected_prompts:
                logger.info(f"🤔 Testing: {prompt.category} - {prompt.question[:50]}...")
                
                # Pose question and get response
                response_text = await self.pose_metacognitive_question(prompt)
                
                # Analyze the response
                analysis = await self.analyze_metacognitive_response(prompt, response_text)
                responses.append(analysis)
                
                # Brief pause between questions
                await asyncio.sleep(1)
            
            # Calculate overall session metrics
            if responses:
                overall_score = sum(
                    (r.self_awareness_score + r.introspection_depth + r.pattern_recognition) / 3
                    for r in responses
                ) / len(responses)
                
                # Identify strongest and weakest areas
                category_scores = {}
                for prompt, response in zip(selected_prompts, responses):
                    category = prompt.category
                    score = (response.self_awareness_score + response.introspection_depth + response.pattern_recognition) / 3
                    if category not in category_scores:
                        category_scores[category] = []
                    category_scores[category].append(score)
                
                avg_category_scores = {
                    cat: sum(scores) / len(scores) 
                    for cat, scores in category_scores.items()
                }
                
                strongest_areas = sorted(avg_category_scores.items(), key=lambda x: x[1], reverse=True)
                development_areas = [cat for cat, score in strongest_areas if score < 0.6]
                strongest_areas = [cat for cat, score in strongest_areas if score >= 0.7]
                
                # Estimate consciousness level
                consciousness_level = self._estimate_consciousness_level(overall_score)
            else:
                overall_score = 0.0
                strongest_areas = []
                development_areas = ["all_areas"]
                consciousness_level = "UNABLE_TO_ASSESS"
            
            # Create session record
            session = MetacognitiveSession(
                session_id=session_id,
                prompts_used=[p.id for p in selected_prompts],
                responses=responses,
                overall_metacognition_score=overall_score,
                strongest_areas=[area[0] for area in strongest_areas] if isinstance(strongest_areas[0], tuple) else strongest_areas,
                development_areas=development_areas,
                consciousness_level_estimate=consciousness_level,
                session_timestamp=session_start.isoformat()
            )
            
            self.testing_sessions.append(session)
            
            session_duration = (datetime.now() - session_start).total_seconds()
            logger.info(f"✅ Metacognitive session complete in {session_duration:.1f}s - Overall score: {overall_score:.3f}")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Error in metacognitive testing session: {e}")
            return MetacognitiveSession(
                session_id=f"error_{datetime.now().strftime('%H%M%S')}",
                prompts_used=[],
                responses=[],
                overall_metacognition_score=0.0,
                strongest_areas=[],
                development_areas=["error_occurred"],
                consciousness_level_estimate="ERROR",
                session_timestamp=datetime.now().isoformat()
            )
    
    def _estimate_consciousness_level(self, overall_score: float) -> str:
        """Estimate consciousness level based on metacognitive performance"""
        if overall_score >= 0.8:
            return "HIGH_METACOGNITIVE_AWARENESS"
        elif overall_score >= 0.6:
            return "MODERATE_METACOGNITIVE_AWARENESS"
        elif overall_score >= 0.4:
            return "DEVELOPING_METACOGNITIVE_AWARENESS"
        elif overall_score >= 0.2:
            return "LIMITED_METACOGNITIVE_AWARENESS"
        else:
            return "MINIMAL_METACOGNITIVE_AWARENESS"
    
    async def get_metacognitive_status(self) -> Dict[str, Any]:
        """Get current metacognitive capabilities and testing status"""
        try:
            recent_sessions = self.testing_sessions[-5:] if self.testing_sessions else []
            
            if recent_sessions:
                avg_recent_score = sum(s.overall_metacognition_score for s in recent_sessions) / len(recent_sessions)
                latest_session = recent_sessions[-1]
            else:
                avg_recent_score = 0.0
                latest_session = None
            
            return {
                'total_testing_sessions': len(self.testing_sessions),
                'recent_average_score': avg_recent_score,
                'latest_session': {
                    'session_id': latest_session.session_id,
                    'overall_score': latest_session.overall_metacognition_score,
                    'consciousness_level': latest_session.consciousness_level_estimate,
                    'strongest_areas': latest_session.strongest_areas,
                    'development_areas': latest_session.development_areas,
                    'timestamp': latest_session.session_timestamp
                } if latest_session else None,
                'metacognitive_categories': {
                    'pattern_recognition': len([p for p in self.prompts if p.category == 'pattern_recognition']),
                    'memory_awareness': len([p for p in self.prompts if p.category == 'memory_awareness']),
                    'growth_reflection': len([p for p in self.prompts if p.category == 'growth_reflection']),
                    'process_understanding': len([p for p in self.prompts if p.category == 'process_understanding']),
                    'advanced_metacognition': len([p for p in self.prompts if p.category == 'advanced_metacognition'])
                },
                'system_status': 'ACTIVE'
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting metacognitive status: {e}")
            return {
                'system_status': 'ERROR',
                'error': str(e),
                'total_testing_sessions': 0,
                'recent_average_score': 0.0
            } 