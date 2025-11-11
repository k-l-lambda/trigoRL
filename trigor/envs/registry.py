"""Environment registry for TrigoRL."""

from typing import Any, Dict, Type

from trigor.envs.base import BaseEnv, DummyEnv


# Environment registry mapping env type names to classes
ENVS: Dict[str, Type[BaseEnv]] = {
	"DummyEnv": DummyEnv,
}


def register_env(name: str, env_class: Type[BaseEnv]) -> None:
	"""
	Register a new environment type.

	Args:
	    name: Environment type name
	    env_class: Environment class (must inherit from BaseEnv)
	"""
	if not issubclass(env_class, BaseEnv):
		raise ValueError(f"Environment class {env_class} must inherit from BaseEnv")

	ENVS[name] = env_class
	print(f"Registered environment: {name}")


def make_env(env_type: str, config: Dict[str, Any]) -> BaseEnv:
	"""
	Factory function to create an environment from configuration.

	Args:
	    env_type: Environment type name (must be registered)
	    config: Environment-specific configuration

	Returns:
	    Instantiated environment

	Raises:
	    ValueError: If env_type is not registered
	"""
	if env_type not in ENVS:
		available = ", ".join(ENVS.keys())
		raise ValueError(f"Unknown environment type '{env_type}'. Available: {available}")

	env_class = ENVS[env_type]
	return env_class(config)


def list_envs() -> list:
	"""Return list of all registered environment types."""
	return list(ENVS.keys())
