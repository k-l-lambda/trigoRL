"""Agent registry for TrigoRL."""

from typing import Any, Dict, Type

from trigor.agents.base import BaseAgent, RandomAgent


# Agent registry mapping agent type names to classes
AGENTS: Dict[str, Type[BaseAgent]] = {
	"RandomAgent": RandomAgent,
}


def register_agent(name: str, agent_class: Type[BaseAgent]) -> None:
	"""
	Register a new agent type.

	Args:
	    name: Agent type name (e.g., 'PPO', 'DQN')
	    agent_class: Agent class (must inherit from BaseAgent)
	"""
	if not issubclass(agent_class, BaseAgent):
		raise ValueError(f"Agent class {agent_class} must inherit from BaseAgent")

	AGENTS[name] = agent_class
	print(f"Registered agent: {name}")


def make_agent(agent_type: str, observation_space, action_space, config: Dict[str, Any]) -> BaseAgent:
	"""
	Factory function to create an agent from configuration.

	Args:
	    agent_type: Agent type name (must be registered)
	    observation_space: Environment observation space
	    action_space: Environment action space
	    config: Agent-specific configuration

	Returns:
	    Instantiated agent

	Raises:
	    ValueError: If agent_type is not registered
	"""
	if agent_type not in AGENTS:
		available = ", ".join(AGENTS.keys())
		raise ValueError(f"Unknown agent type '{agent_type}'. Available: {available}")

	agent_class = AGENTS[agent_type]
	return agent_class(observation_space, action_space, config)


def list_agents() -> list:
	"""Return list of all registered agent types."""
	return list(AGENTS.keys())
