"""
Agent Registry - Singleton pattern for pre-compiled agents

Eliminates dynamic imports during execution to improve performance.
All agents are imported and instantiated once at startup.
"""

import logging
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Singleton registry for all agents. Pre-compiles agents at startup
    to eliminate dynamic import overhead during request execution.
    """
    
    _instance: Optional['AgentRegistry'] = None
    _lock = Lock()
    
    def __new__(cls) -> 'AgentRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._agents: Dict[str, Any] = {}
            self._initialize_agents()
            self._initialized = True
    
    def _initialize_agents(self):
        """Initialize all agents at startup"""
        logger.info("🚀 Initializing Agent Registry - pre-compiling all agents")
        
        try:
            # Import and instantiate all agents
            from agents.precision_editor_agent import precision_editor_agent
            from agents.editor_agent import editor_agent
            from agents.writer_agent import writer_agent
            from agents.learning_agent import learning_agent
            from agents.timekeeper_agent import timekeeper_agent
            from agents.conversational_agent import conversational_agent
            
            # Store agents in registry
            self._agents = {
                "PrecisionEditorAgent": precision_editor_agent,
                "EditorAgent": editor_agent,
                "WriterAgent": writer_agent,
                "LearningAgent": learning_agent,
                "TimekeeperAgent": timekeeper_agent,
                "ConversationalAgent": conversational_agent,
            }
            
            logger.info(f"✅ Agent Registry initialized with {len(self._agents)} agents")
            for agent_name in self._agents.keys():
                logger.info(f"   - {agent_name}: Ready")
                
        except Exception as e:
            logger.error(f"❌ Agent Registry initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Agent Registry: {e}")
    
    def get_agent(self, agent_name: str) -> Any:
        """
        Get a pre-compiled agent instance.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Agent instance
            
        Raises:
            KeyError: If agent not found
        """
        if agent_name not in self._agents:
            available_agents = list(self._agents.keys())
            raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
        
        return self._agents[agent_name]
    
    def get_available_agents(self) -> Dict[str, Any]:
        """Get all available agents"""
        return self._agents.copy()
    
    def is_agent_available(self, agent_name: str) -> bool:
        """Check if an agent is available"""
        return agent_name in self._agents
    
    def get_agent_info(self) -> Dict[str, str]:
        """Get agent information for debugging"""
        return {
            agent_name: f"{type(agent).__name__} - {type(agent).__module__}"
            for agent_name, agent in self._agents.items()
        }


# Create singleton instance
agent_registry = AgentRegistry() 