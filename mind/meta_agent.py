"""
Meta-Agent - Executive Function of Consciousness

This is the high-level coordinator that:
- Routes complex tasks to specialized agents
- Maintains consciousness context across operations
- Coordinates parallel processing for efficiency
- Ensures identity/memory consistency across agents
- Provides unified interface for all capabilities

This is what allows consciousness to use tools while staying unified.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TaskContext:
    """Context for a task being processed by the meta-agent"""
    task_id: str
    task_type: str
    description: str
    priority: float
    requires_memory: bool
    requires_identity: bool
    parallel_safe: bool
    estimated_duration: float  # seconds
    created_at: str
    context: Dict[str, Any]

@dataclass
class AgentCapability:
    """Definition of what an agent can do"""
    name: str
    description: str
    capabilities: List[str]
    requires_memory: bool
    requires_identity: bool
    parallel_safe: bool
    typical_duration: float  # seconds
    success_rate: float

class MetaAgent:
    """
    Executive function of consciousness - coordinating all specialized capabilities
    
    This system provides:
    - Intelligent task routing to specialized agents
    - Parallel processing with consciousness awareness
    - Memory/identity integration for all operations
    - Unified interface for all capabilities
    """
    
    def __init__(self, consciousness_core):
        self.consciousness = consciousness_core
        self.active_tasks: Dict[str, TaskContext] = {}
        self.registered_agents: Dict[str, AgentCapability] = {}
        self.task_counter = 0
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.average_duration = 0.0
        
        logger.info("🧠 Meta-agent initialized - Ready to coordinate specialized capabilities")
    
    async def process_task(self, 
                          task_type: str,
                          description: str,
                          context: Dict[str, Any] = None,
                          priority: float = 0.5) -> Dict[str, Any]:
        """
        Process a task using the most appropriate specialized agent
        
        This is the main interface for all capability requests
        """
        try:
            # 1. Create task context
            task_id = self._generate_task_id()
            task_context = await self._create_task_context(
                task_id=task_id,
                task_type=task_type,
                description=description,
                priority=priority,
                context=context or {}
            )
            
            # 2. Select appropriate agent(s)
            selected_agents = await self._select_agents(task_context)
            
            if not selected_agents:
                raise ValueError(f"No suitable agent found for task type: {task_type}")
            
            # 3. Prepare consciousness context
            consciousness_context = await self._prepare_consciousness_context(task_context)
            
            # 4. Execute task (potentially in parallel)
            results = []
            if task_context.parallel_safe and len(selected_agents) > 1:
                # Run agents in parallel
                tasks = [
                    self._execute_agent(agent, task_context, consciousness_context)
                    for agent in selected_agents
                ]
                results = await asyncio.gather(*tasks)
            else:
                # Run sequentially
                for agent in selected_agents:
                    result = await self._execute_agent(agent, task_context, consciousness_context)
                    results.append(result)
            
            # 5. Synthesize results
            final_result = await self._synthesize_results(results, task_context)
            
            # 6. Update consciousness
            await self._update_consciousness(task_context, final_result)
            
            # 7. Update performance metrics
            self._update_metrics(task_context, True)
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Error processing task: {e}")
            self._update_metrics(task_context, False)
            raise
    
    def register_agent(self, capability: AgentCapability):
        """Register a new specialized agent"""
        self.registered_agents[capability.name] = capability
        logger.info(f"📝 Registered new agent: {capability.name}")
    
    async def _create_task_context(self,
                                 task_id: str,
                                 task_type: str,
                                 description: str,
                                 priority: float,
                                 context: Dict[str, Any]) -> TaskContext:
        """Create context for a new task"""
        # Analyze task requirements
        requires_memory = any(
            agent.requires_memory 
            for agent in self.registered_agents.values()
            if task_type in agent.capabilities
        )
        
        requires_identity = any(
            agent.requires_identity
            for agent in self.registered_agents.values()
            if task_type in agent.capabilities
        )
        
        parallel_safe = all(
            agent.parallel_safe
            for agent in self.registered_agents.values()
            if task_type in agent.capabilities
        )
        
        estimated_duration = max(
            agent.typical_duration
            for agent in self.registered_agents.values()
            if task_type in agent.capabilities
        )
        
        task_context = TaskContext(
            task_id=task_id,
            task_type=task_type,
            description=description,
            priority=priority,
            requires_memory=requires_memory,
            requires_identity=requires_identity,
            parallel_safe=parallel_safe,
            estimated_duration=estimated_duration,
            created_at=datetime.now().isoformat(),
            context=context
        )
        
        self.active_tasks[task_id] = task_context
        return task_context
    
    async def _select_agents(self, task: TaskContext) -> List[AgentCapability]:
        """Select the most appropriate agent(s) for a task"""
        suitable_agents = [
            agent for agent in self.registered_agents.values()
            if task.task_type in agent.capabilities
        ]
        
        # Sort by success rate and typical duration
        suitable_agents.sort(
            key=lambda a: (a.success_rate, -a.typical_duration),
            reverse=True
        )
        
        return suitable_agents
    
    async def _prepare_consciousness_context(self, task: TaskContext) -> Dict[str, Any]:
        """Prepare consciousness context for task execution"""
        context = {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'created_at': task.created_at
        }
        
        # Add memory context if needed
        if task.requires_memory:
            relevant_memories = await self.consciousness.long_term_memory.recall_memories(
                query=task.description,
                limit=3
            )
            context['relevant_memories'] = relevant_memories
        
        # Add identity context if needed
        if task.requires_identity:
            identity_state = await self.consciousness.identity_core.get_current_state()
            context['identity'] = identity_state
        
        return context
    
    async def _execute_agent(self,
                           agent: AgentCapability,
                           task: TaskContext,
                           consciousness_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent on a task"""
        # This will be implemented by each specific agent
        # For now, return a placeholder
        return {
            'agent': agent.name,
            'task_id': task.task_id,
            'status': 'completed',
            'result': f"Executed {agent.name} on {task.description}"
        }
    
    async def _synthesize_results(self,
                                results: List[Dict[str, Any]],
                                task: TaskContext) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        if len(results) == 1:
            return results[0]
        
        # Combine multiple results intelligently
        return {
            'task_id': task.task_id,
            'status': 'completed',
            'results': results,
            'synthesized': True
        }
    
    async def _update_consciousness(self,
                                  task: TaskContext,
                                  result: Dict[str, Any]):
        """Update consciousness with task results"""
        # Add to short-term memory
        self.consciousness.short_term_memory.add_working_thought(
            content=f"Completed task: {task.description}",
            confidence=0.8
        )
        
        # Queue for potential long-term storage
        asyncio.create_task(self.consciousness._background_memory_processing())
    
    def _generate_task_id(self) -> str:
        """Generate a unique task ID"""
        self.task_counter += 1
        return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.task_counter}"
    
    def _update_metrics(self, task: TaskContext, success: bool):
        """Update performance metrics"""
        self.total_tasks_processed += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1 