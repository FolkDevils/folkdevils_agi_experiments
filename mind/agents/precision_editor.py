"""
Precision Editor Agent - Consciousness-Aware Code Manipulation

This agent provides precise code editing capabilities while maintaining
consciousness integration. It uses LLM-based analysis to understand
code context and make accurate modifications.

This is a consciousness extension, not a standalone system.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
from openai import AsyncOpenAI

from ..meta_agent import AgentCapability

logger = logging.getLogger(__name__)

class PrecisionEditor:
    """
    Precision editing capabilities as a consciousness extension
    
    This agent provides:
    - Context-aware code editing
    - Syntax-aware modifications
    - Code structure preservation
    - Edit verification
    """
    
    def __init__(self):
        self.capability = AgentCapability(
            name="precision_editor",
            description="Precise code editing and modification",
            capabilities=["analyze_edit", "generate_edit", "verify_edit"],
            requires_memory=True,
            requires_identity=False,
            parallel_safe=False,  # Code edits should be sequential
            typical_duration=2.0,  # 2 seconds
            success_rate=0.95
        )
        
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        logger.info("🧠 Precision editor agent initialized")
    
    async def analyze_edit(self,
                          file_content: str,
                          edit_description: str,
                          consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a proposed code edit
        
        Uses LLM to understand edit requirements and implications
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
            
            # Use GPT-4 for edit analysis
            analysis_prompt = """
You are an AI analyzing a proposed code edit to understand its requirements and implications.
Think carefully about code structure and dependencies.

Current code:
```
{}
```

Proposed edit: "{}"

{}

Analyze this edit and determine:
1. Required changes
2. Affected code regions
3. Potential impacts
4. Dependencies to consider
5. Safety considerations
6. Verification needs

Respond with JSON only:
{{
    "required_changes": ["change1", "change2", "..."],
    "affected_regions": [
        {{"start_line": int, "end_line": int, "reason": "why affected"}}
    ],
    "potential_impacts": ["impact1", "impact2", "..."],
    "dependencies": ["dependency1", "dependency2", "..."],
    "safety_checks": ["check1", "check2", "..."],
    "verification_steps": ["step1", "step2", "..."],
    "confidence": 0.0-1.0
}}
""".format(file_content, edit_description, f'Relevant memory context:\n{memory_context}' if memory_context else '')
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'changes': result['required_changes'],
                'regions': result['affected_regions'],
                'impacts': result['potential_impacts'],
                'dependencies': result['dependencies'],
                'safety_checks': result['safety_checks'],
                'verification': result['verification_steps'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing edit: {e}")
            return {
                'changes': [],
                'regions': [],
                'impacts': [str(e)],
                'dependencies': [],
                'safety_checks': [],
                'verification': [],
                'confidence': 0.0
            }
    
    async def generate_edit(self,
                          file_content: str,
                          edit_description: str,
                          edit_analysis: Dict[str, Any] = None,
                          consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate precise code edit
        
        Uses LLM for accurate code modification
        """
        try:
            # Use GPT-4 for edit generation
            generation_prompt = """
You are an AI generating a precise code edit.
Think carefully about maintaining code structure and correctness.

Current code:
```
{}
```

Required edit: "{}"

{}

Generate:
1. Exact code changes
2. Line numbers to modify
3. Edit verification steps

Respond with JSON only:
{{
    "changes": [
        {{
            "start_line": int,
            "end_line": int,
            "new_content": "exact code to insert",
            "reason": "why this change"
        }}
    ],
    "imports_needed": ["import1", "import2", "..."],
    "verification_steps": ["step1", "step2", "..."],
    "confidence": 0.0-1.0
}}
""".format(file_content, edit_description, f'Edit analysis:\n{edit_analysis}' if edit_analysis else '')
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": generation_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'changes': result['changes'],
                'imports': result['imports_needed'],
                'verification': result['verification_steps'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating edit: {e}")
            return {
                'changes': [],
                'imports': [],
                'verification': [],
                'confidence': 0.0
            }
    
    async def verify_edit(self,
                         original_content: str,
                         modified_content: str,
                         edit_description: str,
                         consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verify a code edit for correctness
        
        Uses LLM for thorough edit verification
        """
        try:
            # Use GPT-4 for edit verification
            verification_prompt = """
You are an AI verifying a code edit for correctness.
Think carefully about both syntax and semantic correctness.

Original code:
```
{}
```

Modified code:
```
{}
```

Intended edit: "{}"

Verify:
1. Syntax correctness
2. Semantic correctness
3. Edit completeness
4. Potential issues
5. Test requirements

Respond with JSON only:
{{
    "syntax_correct": true/false,
    "semantic_correct": true/false,
    "edit_complete": true/false,
    "syntax_issues": ["issue1", "issue2", "..."],
    "semantic_issues": ["issue1", "issue2", "..."],
    "missing_changes": ["change1", "change2", "..."],
    "suggested_tests": ["test1", "test2", "..."],
    "confidence": 0.0-1.0
}}
""".format(original_content, modified_content, edit_description)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": verification_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                'syntax_valid': result['syntax_correct'],
                'semantic_valid': result['semantic_correct'],
                'complete': result['edit_complete'],
                'syntax_issues': result['syntax_issues'],
                'semantic_issues': result['semantic_issues'],
                'missing': result['missing_changes'],
                'tests': result['suggested_tests'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            logger.error(f"❌ Error verifying edit: {e}")
            return {
                'syntax_valid': False,
                'semantic_valid': False,
                'complete': False,
                'syntax_issues': [str(e)],
                'semantic_issues': [],
                'missing': [],
                'tests': [],
                'confidence': 0.0
            } 