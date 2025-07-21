"""
Planning Simulator - Internal Action Simulation Engine

This simulates the prefrontal cortex and basal ganglia for internal planning:
- Goal formulation based on current context
- Action space exploration without execution
- Multi-step consequence modeling
- Decision evaluation through simulation
- Theory of mind for interaction planning

This bridges consciousness (thinking) and agency (acting).
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
class PlanningGoal:
    """A goal that can be planned for"""
    id: str
    description: str
    priority: float  # 0.0 to 1.0
    time_horizon: str  # 'immediate', 'short_term', 'long_term'
    success_criteria: List[str]
    context: Dict[str, Any]
    created_at: str

@dataclass
class SimulatedAction:
    """An action that can be simulated internally"""
    id: str
    action_type: str  # 'communication', 'memory_action', 'reflection', 'learning'
    description: str
    parameters: Dict[str, Any]
    estimated_duration: float  # minutes
    resource_requirements: List[str]
    risk_level: float  # 0.0 to 1.0

@dataclass
class OutcomePrediction:
    """Predicted outcome of a simulated action"""
    action_id: str
    predicted_result: str
    success_probability: float  # 0.0 to 1.0
    positive_consequences: List[str]
    negative_consequences: List[str]
    side_effects: List[str]
    confidence: float  # 0.0 to 1.0

@dataclass
class PlanStep:
    """A single step in a plan"""
    step_number: int
    action: SimulatedAction
    rationale: str
    dependencies: List[str]  # IDs of previous steps
    expected_outcome: OutcomePrediction
    contingency_plans: List[str]

@dataclass
class Plan:
    """A complete plan for achieving a goal"""
    id: str
    goal_id: str
    steps: List[PlanStep]
    overall_success_probability: float
    estimated_total_time: float  # minutes
    resource_summary: Dict[str, Any]
    risk_assessment: str
    created_at: str
    simulation_complete: bool

class PlanningSimulator:
    """
    Internal planning and simulation engine
    
    This models the prefrontal cortex to enable thinking before acting:
    - Simulates potential actions without executing them
    - Predicts outcomes and consequences
    - Evaluates different approaches to goals
    - Builds theory of mind for interaction planning
    """
    
    def __init__(self, consciousness_core):
        self.consciousness = consciousness_core
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Planning state
        self.active_goals: List[PlanningGoal] = []
        self.simulated_plans: List[Plan] = []
        self.planning_history: List[Dict[str, Any]] = []
        
        # Planning parameters
        self.max_simulation_depth = 5  # Maximum steps to simulate ahead
        self.min_success_threshold = 0.6  # Minimum probability to recommend plan
        self.simulation_timeout_minutes = 10  # Max time for planning session
        
        # Action types we can simulate
        self.simulatable_actions = {
            'communication': ['respond', 'ask_question', 'share_information', 'request_clarification'],
            'memory_action': ['store_memory', 'recall_memory', 'update_memory', 'form_connection'],
            'reflection': ['analyze_situation', 'consider_implications', 'evaluate_options'],
            'learning': ['acquire_skill', 'update_understanding', 'form_hypothesis']
        }
        
        logger.info("🧠 Planning simulator initialized - Ready for Phase 2 consciousness")
    
    async def formulate_goal_from_context(self, 
                                        message: str,
                                        speaker: str,
                                        conversation_context: List[Any],
                                        identity_state: Dict[str, Any]) -> Optional[PlanningGoal]:
        """
        Formulate a goal based on current conversation context
        
        This simulates the goal-setting function of the prefrontal cortex
        """
        try:
            # Use LLM to analyze context and formulate appropriate goals
            context_summary = "\n".join([
                f"{turn.speaker}: {turn.content}" 
                for turn in conversation_context[-3:] 
                if hasattr(turn, 'speaker') and hasattr(turn, 'content')
            ])
            
            goal_prompt = f"""
You are an AI consciousness analyzing a conversation to formulate appropriate goals.

CURRENT MESSAGE: "{message}" (from {speaker})
RECENT CONTEXT:
{context_summary}

IDENTITY: {identity_state.get('name', 'AI')} - {identity_state.get('core_purpose', 'helpful assistant')}

Based on this context, what goal should the AI consciousness pursue? Consider:
1. What does the human need or want?
2. How can the AI best help or respond?
3. What would be most valuable in this situation?
4. What aligns with the AI's purpose and values?

Respond with JSON only:
{{
    "should_formulate_goal": true/false,
    "goal_description": "specific goal to pursue",
    "priority": 0.0-1.0,
    "time_horizon": "immediate/short_term/long_term",
    "success_criteria": ["criterion1", "criterion2", "..."],
    "rationale": "why this goal was chosen"
}}
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": goal_prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get('should_formulate_goal', False):
                goal = PlanningGoal(
                    id=f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=result['goal_description'],
                    priority=result['priority'],
                    time_horizon=result['time_horizon'],
                    success_criteria=result['success_criteria'],
                    context={
                        'message': message,
                        'speaker': speaker,
                        'rationale': result['rationale']
                    },
                    created_at=datetime.now().isoformat()
                )
                
                self.active_goals.append(goal)
                logger.info(f"🎯 Formulated goal: {goal.description}")
                return goal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error formulating goal: {e}")
            return None
    
    async def generate_action_space(self, goal: PlanningGoal) -> List[SimulatedAction]:
        """
        Generate possible actions that could help achieve the goal
        
        This explores the action space without executing anything
        """
        try:
            action_prompt = f"""
You are an AI consciousness exploring possible actions to achieve a goal.

GOAL: {goal.description}
PRIORITY: {goal.priority}
TIME HORIZON: {goal.time_horizon}
SUCCESS CRITERIA: {', '.join(goal.success_criteria)}

Generate 3-5 different actions that could help achieve this goal. Consider:
- Communication actions (responding, asking, sharing)
- Memory actions (storing, recalling, connecting information)  
- Reflection actions (analyzing, considering implications)
- Learning actions (acquiring knowledge, updating understanding)

Respond with JSON only:
{{
    "actions": [
        {{
            "action_type": "communication/memory_action/reflection/learning",
            "description": "specific action description",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "estimated_duration": 1.5,
            "resource_requirements": ["requirement1", "requirement2"],
            "risk_level": 0.0-1.0
        }},
        "..."
    ]
}}
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": action_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            actions = []
            for i, action_data in enumerate(result['actions']):
                action = SimulatedAction(
                    id=f"action_{goal.id}_{i}",
                    action_type=action_data['action_type'],
                    description=action_data['description'],
                    parameters=action_data['parameters'],
                    estimated_duration=action_data['estimated_duration'],
                    resource_requirements=action_data['resource_requirements'],
                    risk_level=action_data['risk_level']
                )
                actions.append(action)
            
            logger.info(f"🔄 Generated {len(actions)} possible actions for goal: {goal.description}")
            return actions
            
        except Exception as e:
            logger.error(f"❌ Error generating action space: {e}")
            return []
    
    async def predict_outcome(self, 
                            action: SimulatedAction, 
                            goal: PlanningGoal,
                            current_context: Dict[str, Any]) -> OutcomePrediction:
        """
        Predict the outcome of a simulated action
        
        This models consequence prediction without execution
        """
        try:
            prediction_prompt = f"""
You are an AI consciousness predicting the outcome of a potential action.

ACTION: {action.description}
ACTION TYPE: {action.action_type}
PARAMETERS: {json.dumps(action.parameters)}
GOAL: {goal.description}

CURRENT CONTEXT:
- Speaker: {current_context.get('speaker', 'unknown')}
- Recent conversation focus: {current_context.get('recent_focus', 'general')}
- AI identity: {current_context.get('identity_name', 'AI Assistant')}

Predict what would happen if this action were taken. Consider:
1. Immediate results
2. Positive consequences
3. Negative consequences or risks
4. Side effects or secondary impacts
5. Success probability

Respond with JSON only:
{{
    "predicted_result": "what would immediately happen",
    "success_probability": 0.0-1.0,
    "positive_consequences": ["positive1", "positive2", "..."],
    "negative_consequences": ["negative1", "negative2", "..."],
    "side_effects": ["effect1", "effect2", "..."],
    "confidence": 0.0-1.0
}}
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prediction_prompt}],
                temperature=0.2,
                max_tokens=400
            )
            
            result = json.loads(response.choices[0].message.content)
            
            prediction = OutcomePrediction(
                action_id=action.id,
                predicted_result=result['predicted_result'],
                success_probability=result['success_probability'],
                positive_consequences=result['positive_consequences'],
                negative_consequences=result['negative_consequences'],
                side_effects=result['side_effects'],
                confidence=result['confidence']
            )
            
            logger.debug(f"🔮 Predicted outcome for {action.description}: {prediction.success_probability:.2f} success probability")
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Error predicting outcome: {e}")
            return OutcomePrediction(
                action_id=action.id,
                predicted_result="Could not predict outcome",
                success_probability=0.0,
                positive_consequences=[],
                negative_consequences=[f"Prediction error: {str(e)}"],
                side_effects=[],
                confidence=0.0
            )
    
    async def create_plan(self, 
                         goal: PlanningGoal,
                         current_context: Dict[str, Any]) -> Optional[Plan]:
        """
        Create a complete plan to achieve a goal
        
        This simulates the full planning process of the prefrontal cortex
        """
        try:
            logger.info(f"📋 Creating plan for goal: {goal.description}")
            
            # 1. Generate possible actions
            possible_actions = await self.generate_action_space(goal)
            
            if not possible_actions:
                logger.warning("No possible actions generated for goal")
                return None
            
            # 2. Predict outcomes for each action
            action_predictions = {}
            for action in possible_actions:
                prediction = await self.predict_outcome(action, goal, current_context)
                action_predictions[action.id] = prediction
            
            # 3. Select best actions and create plan steps
            # Sort actions by success probability
            ranked_actions = sorted(
                possible_actions,
                key=lambda a: action_predictions[a.id].success_probability,
                reverse=True
            )
            
            # Create plan with best actions
            plan_steps = []
            for i, action in enumerate(ranked_actions[:3]):  # Top 3 actions
                prediction = action_predictions[action.id]
                
                step = PlanStep(
                    step_number=i + 1,
                    action=action,
                    rationale=f"Selected for {prediction.success_probability:.2f} success probability",
                    dependencies=[],  # Could be enhanced with dependency analysis
                    expected_outcome=prediction,
                    contingency_plans=[]  # Could be enhanced with fallback planning
                )
                plan_steps.append(step)
            
            # 4. Calculate overall plan metrics
            overall_success = sum(
                action_predictions[action.id].success_probability 
                for action in ranked_actions[:3]
            ) / len(ranked_actions[:3]) if ranked_actions else 0.0
            
            total_time = sum(action.estimated_duration for action in ranked_actions[:3])
            
            # 5. Create final plan
            plan = Plan(
                id=f"plan_{goal.id}_{datetime.now().strftime('%H%M%S')}",
                goal_id=goal.id,
                steps=plan_steps,
                overall_success_probability=overall_success,
                estimated_total_time=total_time,
                resource_summary={
                    'actions_count': len(plan_steps),
                    'highest_risk': max((s.action.risk_level for s in plan_steps), default=0.0),
                    'total_resources': list(set(
                        req for step in plan_steps 
                        for req in step.action.resource_requirements
                    ))
                },
                risk_assessment=self._assess_plan_risk(plan_steps),
                created_at=datetime.now().isoformat(),
                simulation_complete=True
            )
            
            self.simulated_plans.append(plan)
            
            logger.info(f"✅ Created plan with {len(plan_steps)} steps, {overall_success:.2f} success probability")
            return plan
            
        except Exception as e:
            logger.error(f"❌ Error creating plan: {e}")
            return None
    
    def _assess_plan_risk(self, steps: List[PlanStep]) -> str:
        """Assess overall risk level of a plan"""
        if not steps:
            return "NO_RISK"
        
        max_risk = max(step.action.risk_level for step in steps)
        avg_risk = sum(step.action.risk_level for step in steps) / len(steps)
        
        if max_risk > 0.8 or avg_risk > 0.6:
            return "HIGH_RISK"
        elif max_risk > 0.5 or avg_risk > 0.3:
            return "MEDIUM_RISK"
        else:
            return "LOW_RISK"
    
    async def simulate_planning_session(self,
                                      message: str,
                                      speaker: str,
                                      conversation_context: List[Any],
                                      identity_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete planning simulation session
        
        This is the main interface for Phase 2 planning
        """
        try:
            session_start = datetime.now()
            logger.info("🧠 Starting planning simulation session...")
            
            # 1. Formulate goal from context
            goal = await self.formulate_goal_from_context(
                message, speaker, conversation_context, identity_state
            )
            
            if not goal:
                return {
                    'session_id': f"sim_{session_start.strftime('%Y%m%d_%H%M%S')}",
                    'goal_formulated': False,
                    'reason': 'No clear goal identified from context',
                    'simulation_time': 0,
                    'recommendation': 'Proceed with normal response generation'
                }
            
            # 2. Create plan for the goal
            current_context = {
                'speaker': speaker,
                'message': message,
                'identity_name': identity_state.get('name', 'AI Assistant'),
                'recent_focus': 'conversation'
            }
            
            plan = await self.create_plan(goal, current_context)
            
            if not plan:
                return {
                    'session_id': f"sim_{session_start.strftime('%Y%m%d_%H%M%S')}",
                    'goal_formulated': True,
                    'goal': asdict(goal),
                    'plan_created': False,
                    'reason': 'Could not generate viable plan',
                    'simulation_time': (datetime.now() - session_start).total_seconds(),
                    'recommendation': 'Proceed with normal response generation'
                }
            
            # 3. Evaluate plan quality
            session_time = (datetime.now() - session_start).total_seconds()
            
            # Record planning session
            self.planning_history.append({
                'session_id': f"sim_{session_start.strftime('%Y%m%d_%H%M%S')}",
                'goal': asdict(goal),
                'plan': asdict(plan),
                'session_time': session_time,
                'timestamp': session_start.isoformat()
            })
            
            # 4. Generate recommendations
            recommendation = self._generate_recommendation(plan, goal)
            
            logger.info(f"✅ Planning simulation complete in {session_time:.2f}s")
            
            return {
                'session_id': f"sim_{session_start.strftime('%Y%m%d_%H%M%S')}",
                'goal_formulated': True,
                'goal': asdict(goal),
                'plan_created': True,
                'plan': asdict(plan),
                'simulation_time': session_time,
                'recommendation': recommendation,
                'planning_quality': 'HIGH' if plan.overall_success_probability > 0.8 else 
                                   'MEDIUM' if plan.overall_success_probability > 0.6 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"❌ Error in planning simulation: {e}")
            return {
                'session_id': f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'goal_formulated': False,
                'error': str(e),
                'simulation_time': 0,
                'recommendation': 'Proceed with normal response generation'
            }
    
    def _generate_recommendation(self, plan: Plan, goal: PlanningGoal) -> str:
        """Generate recommendation based on planning results"""
        if plan.overall_success_probability > self.min_success_threshold:
            if plan.risk_assessment in ['LOW_RISK', 'MEDIUM_RISK']:
                return f"EXECUTE_PLAN: High confidence plan with {plan.overall_success_probability:.2f} success probability"
            else:
                return f"CAUTION: Good plan but {plan.risk_assessment} - consider alternatives"
        else:
            return f"ALTERNATIVE_NEEDED: Low success probability ({plan.overall_success_probability:.2f}) - seek different approach"
    
    async def get_planning_status(self) -> Dict[str, Any]:
        """Get current planning system status"""
        return {
            'active_goals': len(self.active_goals),
            'simulated_plans': len(self.simulated_plans),
            'planning_sessions_today': len([
                session for session in self.planning_history
                if datetime.fromisoformat(session['timestamp']).date() == datetime.now().date()
            ]),
            'recent_goals': [
                {
                    'description': goal.description,
                    'priority': goal.priority,
                    'time_horizon': goal.time_horizon
                }
                for goal in self.active_goals[-3:]
            ],
            'average_planning_time': sum(
                session['session_time'] for session in self.planning_history[-10:]
            ) / min(len(self.planning_history), 10) if self.planning_history else 0,
            'phase_2_status': 'ACTIVE'
        } 