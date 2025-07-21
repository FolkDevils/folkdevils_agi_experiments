"""
Instruction Parser Agent - Consciousness-Aware Task Understanding

This agent provides instruction parsing capabilities while maintaining
consciousness integration. It uses LLM-based analysis to understand
task requirements and break them down intelligently.

This is a consciousness extension, not a standalone system.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
from openai import AsyncOpenAI

from ..meta_agent import AgentCapability

logger = logging.getLogger(__name__)

class InstructionParser:
    """
    Instruction parsing capabilities as a consciousness extension
    
    This agent provides:
    - Task requirement analysis
    - Step-by-step breakdown
    - Dependency identification
    - Success criteria extraction
    """
    
    def __init__(self):
        self.capability = AgentCapability(
            name="instruction_parser",
            description="Instruction understanding and task planning",
            capabilities=["parse_instruction", "extract_requirements", "plan_steps"],
            requires_memory=True,
            requires_identity=True,
            parallel_safe=True,
            typical_duration=1.5,  # 1.5 seconds
            success_rate=0.90
        )
        
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        logger.info("🧠 Instruction parser agent initialized")
    
    async def parse_instruction(self,
                              instruction: str,
                              consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Parse and analyze an instruction/task
        
        Uses LLM to understand requirements and implications
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
            
            # Use identity context if available
            identity_context = ""
            if consciousness_context and 'identity' in consciousness_context:
                identity = consciousness_context['identity']
                identity_context = f"Your identity: {identity.get('name')} - {identity.get('core_purpose')}"
            
            # Use GPT-4 for instruction analysis
            analysis_prompt = """
You are an AI analyzing an instruction/task to understand what needs to be done.
Think carefully about requirements, steps, and success criteria.

Instruction: "{}"

{}
{}

Analyze this instruction and determine:
1. Core task objective
2. Required steps/subtasks
3. Dependencies and prerequisites
4. Success criteria
5. Potential challenges
6. Required capabilities

Respond with JSON only:
{{
    "objective": "main goal of the task",
    "steps": ["step1", "step2", "..."],
    "dependencies": ["dependency1", "dependency2", "..."],
    "success_criteria": ["criterion1", "criterion2", "..."],
    "challenges": ["challenge1", "challenge2", "..."],
    "required_capabilities": ["capability1", "capability2", "..."],
    "confidence": 0.0-1.0
}}
""".format(instruction, identity_context if identity_context else '', f'Relevant memory context:\n{memory_context}' if memory_context else '')
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'objective': result['objective'],
                'steps': result['steps'],
                'dependencies': result['dependencies'],
                'success_criteria': result['success_criteria'],
                'challenges': result['challenges'],
                'required_capabilities': result['required_capabilities'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error parsing instruction: {e}")
            return {
                'objective': None,
                'steps': [],
                'dependencies': [],
                'success_criteria': [],
                'challenges': [str(e)],
                'required_capabilities': [],
                'confidence': 0.0
            }
    
    async def extract_requirements(self,
                                 instruction: str,
                                 consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract specific requirements from an instruction
        
        Uses LLM for detailed requirement analysis
        """
        try:
            # Use GPT-4 for requirement extraction
            extraction_prompt = """
You are an AI extracting specific requirements from an instruction.
Think about what's needed for successful completion.

Instruction: "{}"

Extract:
1. Functional requirements
2. Technical requirements
3. Resource requirements
4. Knowledge requirements
5. Time/performance requirements

Respond with JSON only:
{{
    "functional_reqs": ["req1", "req2", "..."],
    "technical_reqs": ["req1", "req2", "..."],
    "resource_reqs": ["req1", "req2", "..."],
    "knowledge_reqs": ["req1", "req2", "..."],
    "performance_reqs": ["req1", "req2", "..."],
    "confidence": 0.0-1.0
}}
""".format(instruction)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'functional': result['functional_reqs'],
                'technical': result['technical_reqs'],
                'resources': result['resource_reqs'],
                'knowledge': result['knowledge_reqs'],
                'performance': result['performance_reqs'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting requirements: {e}")
            return {
                'functional': [],
                'technical': [],
                'resources': [],
                'knowledge': [],
                'performance': [],
                'confidence': 0.0
            }
    
    async def plan_steps(self,
                        instruction: str,
                        consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create detailed step-by-step plan for an instruction
        
        Uses LLM for intelligent task breakdown
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
            
            # Use GPT-4 for step planning
            planning_prompt = """
You are an AI creating a detailed execution plan for a task.
Think carefully about order, dependencies, and verification.

Task: "{}"

{}

Create a detailed plan including:
1. Preparation steps
2. Main execution steps
3. Verification steps
4. Contingency steps
5. Dependencies between steps
6. Expected outcomes per step

Respond with JSON only:
{{
    "preparation": ["step1", "step2", "..."],
    "execution": ["step1", "step2", "..."],
    "verification": ["check1", "check2", "..."],
    "contingency": ["fallback1", "fallback2", "..."],
    "dependencies": [
        {{"step": "step1", "depends_on": ["preparation1", "preparation2"]}},
        "..."
    ],
    "expected_outcomes": [
        {{"step": "step1", "outcome": "expected result"}},
        "..."
    ],
    "confidence": 0.0-1.0
}}
""".format(instruction, f'Relevant memory context:\n{memory_context}' if memory_context else '')
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'preparation_steps': result['preparation'],
                'execution_steps': result['execution'],
                'verification_steps': result['verification'],
                'contingency_steps': result['contingency'],
                'step_dependencies': result['dependencies'],
                'expected_outcomes': result['expected_outcomes'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error planning steps: {e}")
            return {
                'preparation_steps': [],
                'execution_steps': [],
                'verification_steps': [],
                'contingency_steps': [],
                'step_dependencies': [],
                'expected_outcomes': [],
                'confidence': 0.0
            } 