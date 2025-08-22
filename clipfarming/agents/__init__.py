"""
Base Agent Class for Clipfarming System
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from loguru import logger
from clipfarming import ClipfarmingConfig


class BaseAgent(ABC):
    """Base class for all clipfarming agents"""
    
    def __init__(self, name: str, config: ClipfarmingConfig):
        self.name = name
        self.config = config
        self.logger = logger.bind(agent=name)
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the agent's main functionality"""
        pass
    
    async def setup(self):
        """Setup the agent (called before execution)"""
        self.logger.info(f"Setting up {self.name} agent")
    
    async def cleanup(self):
        """Cleanup the agent (called after execution)"""
        self.logger.info(f"Cleaning up {self.name} agent")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent"""
        return {
            "name": self.name,
            "status": "ready"
        }


class AgentOrchestrator:
    """Orchestrates multiple agents to work together"""
    
    def __init__(self, config: ClipfarmingConfig):
        self.config = config
        self.agents: List[BaseAgent] = []
        self.logger = logger.bind(component="orchestrator")
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the orchestrator"""
        self.agents.append(agent)
        self.logger.info(f"Added agent: {agent.name}")
    
    async def setup_all(self):
        """Setup all agents"""
        for agent in self.agents:
            await agent.setup()
    
    async def cleanup_all(self):
        """Cleanup all agents"""
        for agent in self.agents:
            await agent.cleanup()
    
    async def execute_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all agents in a pipeline"""
        self.logger.info("Starting agent pipeline execution")
        
        result = input_data
        
        for agent in self.agents:
            try:
                self.logger.info(f"Executing agent: {agent.name}")
                result = await agent.execute(result)
            except Exception as e:
                self.logger.error(f"Error in agent {agent.name}: {e}")
                raise
        
        self.logger.info("Pipeline execution completed")
        return result