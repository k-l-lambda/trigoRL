"""
Evaluation Language Model for ONNX Export.

This model wraps a trained ValueCausalLoss model to provide a clean interface
for value prediction inference. Designed specifically for ONNX export and
efficient inference in JavaScript.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .registry import register_model


@register_model("evaluation")
class EvaluationLM(nn.Module):
	"""
	Evaluation mode wrapper for value prediction inference.

	This model wraps a trained ValueCausalLoss model (base CausalLM + ValueHead)
	to provide a simple interface for game outcome prediction. It appends a VALUE
	token to the input sequence and returns a single scalar prediction.

	ONNX Input Signature:
	    - input_ids: [batch_size, seq_len] - Input token sequence

	ONNX Output:
	    - values: [batch_size] - Predicted game outcome values in range [-1, 1]

	Args:
	    base_model: Underlying CausalLM model (GPT2, LLaMA, RWKV, xLSTM)
	    value_head: ValueHead network for value prediction
	    value_id: Token ID for VALUE token (default: 3)
	"""

	def __init__(
		self,
		base_model: nn.Module,
		value_head: nn.Module,
		value_id: int = 3,
	):
		super().__init__()
		self.model = base_model
		self.value_head = value_head
		self.value_id = value_id


	def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass for value prediction.

		Args:
		    input_ids: Input token IDs [batch_size, seq_len]

		Returns:
		    values: Predicted values [batch_size] in range [-1, 1]
		"""
		batch_size, seq_len = input_ids.shape

		# Step 1: Append VALUE token to end of each sequence
		value_token = torch.full(
			(batch_size, 1),
			self.value_id,
			dtype=torch.long,
			device=input_ids.device
		)
		input_ids_with_value = torch.cat([input_ids, value_token], dim=1)
		# Shape: [batch_size, seq_len + 1]

		# Step 2: Get base model (unwrap if wrapped in loss module)
		if hasattr(self.model, 'model'):
			base = self.model.model
		else:
			base = self.model

		# Step 3: Forward through base model with output_hidden_states
		# Note: Standard causal mask is applied automatically by most models
		model_outputs = base(input_ids_with_value, output_hidden_states=True)

		# Step 4: Extract hidden states from last layer
		if hasattr(model_outputs, 'hidden_states'):
			hidden_states = model_outputs.hidden_states[-1]  # Last layer
		elif isinstance(model_outputs, dict) and 'hidden_states' in model_outputs:
			hidden_states = model_outputs['hidden_states'][-1]
		else:
			raise ValueError(
				f"Model output does not contain hidden_states. "
				f"Output type: {type(model_outputs)}, "
				f"Available attributes: {dir(model_outputs) if hasattr(model_outputs, '__dir__') else 'N/A'}"
			)
		# Shape: [batch_size, seq_len + 1, hidden_dim]

		# Step 5: Extract hidden state at VALUE token position (last position)
		value_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_dim]

		# Step 6: Pass through value_head to get value prediction
		values = self.value_head(value_hidden)  # [batch_size]

		return values


	@classmethod
	def from_value_causal_loss(
		cls,
		checkpoint_path: str,
		device: str = 'cpu'
	) -> 'EvaluationLM':
		"""
		Create EvaluationLM from ValueCausalLoss checkpoint.

		Args:
		    checkpoint_path: Path to .chkpt file
		    device: Device to load model on (default: 'cpu')

		Returns:
		    EvaluationLM instance with loaded weights

		Raises:
		    FileNotFoundError: If checkpoint or config file not found
		    ValueError: If checkpoint is not from ValueCausalLoss model
		"""
		checkpoint_path = Path(checkpoint_path)
		if not checkpoint_path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

		# Load checkpoint
		checkpoint = torch.load(str(checkpoint_path), map_location=device)
		state_dict = checkpoint['model_state_dict']

		# Load config from checkpoint or training directory
		if 'config' in checkpoint:
			config = checkpoint['config']
			# Handle both OmegaConf and dict configs
			if isinstance(config, dict):
				from omegaconf import OmegaConf
				config = OmegaConf.create(config)
		else:
			# Try to load config.yaml from training directory
			training_dir = checkpoint_path.parent.parent
			config_path = training_dir / 'config.yaml'
			if not config_path.exists():
				raise FileNotFoundError(
					f"Config not found in checkpoint or at {config_path}. "
					f"Cannot reconstruct model architecture."
				)
			config = OmegaConf.load(config_path)

		# Create temporary ValueCausalLoss to load weights
		from trigor.models.valueCausalLoss import ValueCausalLoss
		temp_model = ValueCausalLoss.from_config(config.model.config)
		temp_model.load_state_dict(state_dict)

		# Extract components
		base_model = temp_model.model
		value_head = temp_model.value_head
		value_id = temp_model.value_id

		# Create EvaluationLM
		eval_model = cls(
			base_model=base_model,
			value_head=value_head,
			value_id=value_id
		)

		eval_model.eval()
		eval_model.to(device)

		return eval_model


	@classmethod
	def from_state_dict(
		cls,
		state_dict: Dict,
		model_config: Dict,
		value_head_config: Dict,
		value_id: int = 3
	) -> 'EvaluationLM':
		"""
		Create EvaluationLM from state dict and configs.

		More explicit loading method when config is available separately.

		Args:
		    state_dict: Model state dictionary
		    model_config: Base model configuration
		    value_head_config: ValueHead configuration
		    value_id: VALUE token ID (default: 3)

		Returns:
		    EvaluationLM instance with loaded weights
		"""
		# Create base model
		from trigor.models import make_model
		base_model = make_model(model_config['type'], model_config['config'])

		# Create value head
		from trigor.models.valueHead import ValueHead
		value_head = ValueHead.from_config(value_head_config)

		# Create wrapper
		eval_model = cls(base_model, value_head, value_id)

		# Load weights (filter by prefix)
		base_state = {
			k.replace('model.', '', 1): v
			for k, v in state_dict.items()
			if k.startswith('model.')
		}
		value_state = {
			k.replace('value_head.', '', 1): v
			for k, v in state_dict.items()
			if k.startswith('value_head.')
		}

		eval_model.model.load_state_dict(base_state)
		eval_model.value_head.load_state_dict(value_state)

		return eval_model


	@classmethod
	def from_config(
		cls,
		config: Union[Dict, DictConfig],
		base_model: nn.Module
	) -> 'EvaluationLM':
		"""
		Create EvaluationLM with configuration.

		Args:
		    config: Configuration dict (may contain value_head_config, value_id)
		    base_model: Base CausalLM model instance

		Returns:
		    EvaluationLM instance
		"""
		# Create value head
		from trigor.models.valueHead import ValueHead
		value_head_config = config.get('value_head_config', {})
		value_head = ValueHead.from_config(value_head_config)

		# Get value_id
		value_id = config.get('value_id', 3)

		return cls(base_model, value_head, value_id)


	def get_model_info(self) -> Dict[str, any]:
		"""Get information about the wrapped models."""
		base_info = {}
		if hasattr(self.model, 'get_model_info'):
			base_info = self.model.get_model_info()

		value_info = {}
		if hasattr(self.value_head, 'get_model_info'):
			value_info = self.value_head.get_model_info()

		return {
			'model_class': 'EvaluationLM',
			'base_model': base_info.get('model_type', 'unknown'),
			'value_head': value_info,
			'value_id': self.value_id,
			'mode': 'evaluation',
			'onnx_compatible': True,
		}


	def __repr__(self) -> str:
		"""String representation."""
		base_repr = repr(self.model) if self.model else "None"
		value_repr = repr(self.value_head) if self.value_head else "None"
		return (
			f"EvaluationLM(\n"
			f"  base_model={base_repr},\n"
			f"  value_head={value_repr},\n"
			f"  value_id={self.value_id}\n"
			f")"
		)
