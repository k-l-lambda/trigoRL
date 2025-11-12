"""Model architectures for TrigoRL."""

from trigor.models.networks import MLP, PolicyValueNetwork

# Registry
from trigor.models.registry import list_models, make_model, register_model

# CausalLM models
from trigor.models.gpt2CausalLM import GPT2CausalLM
from trigor.models.llamaCausalLM import LlamaCausalLM
from trigor.models.rwkvCausalLM import RwkvCausalLM
from trigor.models.xlstmCausalLM import xLSTMCausalLM


__all__ = [
	# RL networks
	"MLP",
	"PolicyValueNetwork",
	# Model registry
	"register_model",
	"make_model",
	"list_models",
	# CausalLM models
	"GPT2CausalLM",
	"LlamaCausalLM",
	"RwkvCausalLM",
	"xLSTMCausalLM",
]
