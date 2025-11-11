"""Agent module for TrigoRL."""

from trigor.agents.base import BaseAgent, RandomAgent
from trigor.agents.registry import list_agents, make_agent, register_agent

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "make_agent",
    "register_agent",
    "list_agents",
]
