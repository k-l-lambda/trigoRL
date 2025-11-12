# TGN Dataset Factory

The dataset factory provides a configuration-driven approach to creating PyTorch datasets for TrigoRL, following the same registry pattern used for agents and environments.

## Features

- **Registry Pattern**: Register and discover dataset types dynamically
- **Config-Driven**: Create datasets from YAML configuration files
- **Hydra Integration**: Seamlessly integrates with Hydra/OmegaConf config system
- **Type Safety**: Factory validates dataset types and parameters
- **Extensible**: Easy to add new dataset types

## Quick Start

### 1. Create Dataset from Config Dictionary

```python
from trigor.data import make_dataset

config = {
    'type': 'TGNDataset',
    'data_dir': 'third_party/trigo/trigo-web/tools/output',
    'max_length': 512,
    'min_length': 10,
    'max_file_size': 10000,
}

dataset = make_dataset(dataset_type=config['type'], config=config)
print(f"Created dataset with {len(dataset)} samples")
```

### 2. Create Dataset from YAML Config

```python
from omegaconf import OmegaConf
from trigor.data import make_dataset

# Load config
cfg = OmegaConf.load('configs/dataset/tgn_default.yaml')

# Create dataset
dataset = make_dataset(dataset_type=cfg.type, config=cfg)
```

### 3. Use with Hydra

```python
import hydra
from omegaconf import DictConfig
from trigor.data import make_dataset
from torch.utils.data import DataLoader

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Create dataset from config
    dataset = make_dataset(dataset_type=cfg.dataset.type, config=cfg.dataset)

    # Create DataLoader
    from trigor.data import TGNDataset
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.dataloader.batch_size,
        shuffle=cfg.dataset.dataloader.shuffle,
        num_workers=cfg.dataset.dataloader.num_workers,
        collate_fn=TGNDataset.collate_batch,
    )

    # Training loop
    for batch in dataloader:
        # batch contains: input_ids, labels, attention_mask
        pass

if __name__ == "__main__":
    main()
```

## Configuration Files

Three preconfigured dataset configs are provided:

### configs/dataset/tgn_default.yaml
Standard configuration for general training:
- Max length: 2048 tokens
- Batch size: 8

### configs/dataset/tgn_small.yaml
Fast configuration for quick experiments:
- Max length: 512 tokens
- Batch size: 16

### configs/dataset/tgn_large.yaml
Full configuration for long sequences:
- Max length: 4096 tokens
- Batch size: 4

## YAML Config Structure

```yaml
# Dataset type (must be registered)
type: TGNDataset

# Path to data directory (supports Hydra interpolation)
data_dir: ${paths.root}/third_party/trigo/trigo-web/tools/output

# Tokenization settings
max_length: 2048
min_length: 10
max_file_size: 10000

# Optional tokenizer configuration
tokenizer_config: {}

# DataLoader settings (for reference)
dataloader:
  batch_size: 8
  shuffle: true
  num_workers: 4
  pin_memory: true
```

## Registry Functions

### list_datasets()
Returns list of all registered dataset types.

```python
from trigor.data import list_datasets

datasets = list_datasets()
print(f"Available datasets: {datasets}")
# Output: ['TGNDataset']
```

### register_dataset()
Register a custom dataset type. Can be used as a decorator or function.

**Usage as decorator (recommended):**
```python
from torch.utils.data import Dataset
from trigor.data import register_dataset

@register_dataset('CustomDataset')
class CustomDataset(Dataset):
    """Custom dataset with decorator registration."""

    def __init__(self, config):
        # Initialize custom dataset
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return {'data': torch.randn(10)}
```

**Usage as function:**
```python
class CustomDataset(Dataset):
    def __init__(self, config):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return {'data': torch.randn(10)}

# Register the dataset
register_dataset('CustomDataset', CustomDataset)
```

Both approaches are equivalent and result in the same registration.

### make_dataset()
Factory function to create dataset from config.

```python
from trigor.data import make_dataset

config = {
    'type': 'TGNDataset',
    'data_dir': 'data/tgn_games',
    'max_length': 1024,
}

dataset = make_dataset(dataset_type=config['type'], config=config)
```

**Arguments:**
- `dataset_type` (str): Name of registered dataset type
- `config` (dict): Configuration dictionary

**Returns:** Instantiated Dataset object

**Raises:** ValueError if dataset_type not registered

## TGNDataset Config Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | Required | Directory containing .tgn files |
| `max_length` | int | 2048 | Maximum sequence length |
| `min_length` | int | 10 | Minimum file size in bytes |
| `max_file_size` | int | 10000 | Maximum file size in bytes |
| `tokenizer_config` | dict | {} | Additional tokenizer settings |

## Using in Training Scripts

### Example: Integration with train.py

```python
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from trigor.data import TGNDataset, make_dataset
from trigor.agents import make_agent
from trigor.envs import make_env
from trigor.training.trainer import RLTrainer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    # Create dataset
    dataset = make_dataset(dataset_type=cfg.dataset.type, config=cfg.dataset)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.dataloader.batch_size,
        shuffle=cfg.dataset.dataloader.shuffle,
        num_workers=cfg.dataset.dataloader.num_workers,
        pin_memory=cfg.dataset.dataloader.pin_memory,
        collate_fn=TGNDataset.collate_batch,
    )

    # Create agent and environment
    env = make_env(cfg.env.type, cfg.env)
    agent = make_agent(cfg.agent.type, env.observation_space, env.action_space, cfg.agent)

    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        env=env,
        config=cfg.training,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    train()
```

### Command Line Usage

Switch dataset configs on the command line:

```bash
# Use default dataset config
python train.py

# Use small dataset for quick testing
python train.py dataset=tgn_small

# Use large dataset for full training
python train.py dataset=tgn_large

# Override specific parameters
python train.py dataset=tgn_default dataset.max_length=1024 dataset.dataloader.batch_size=16

# Use custom data directory
python train.py dataset.data_dir=/path/to/custom/data
```

## Extending with Custom Datasets

### Pattern 1: Using Decorator + from_config (Recommended)

The recommended approach is to use the decorator and implement a `from_config` classmethod. This makes your dataset fully self-contained:

```python
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from trigor.data import register_dataset

@register_dataset('MyCustomDataset')
class MyCustomDataset(Dataset):
    """Custom dataset for specific data format."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MyCustomDataset':
        """
        Create dataset from configuration dictionary.

        This method handles preprocessing and parameter extraction.
        """
        # Extract and process config
        data_dir = config['data_dir']
        custom_param = config.get('custom_param', 10)

        # Additional preprocessing
        if config.get('validate', False):
            # Perform validation
            pass

        return cls(data_dir=data_dir, custom_param=custom_param)

    def __init__(self, data_dir: str, custom_param: int = 10):
        self.data_dir = Path(data_dir)
        self.custom_param = custom_param
        # Load data
        self.data = self._load_data()

    def _load_data(self):
        # Custom loading logic
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'features': torch.randn(10),
            'label': torch.tensor(0),
        }
```

**Benefits:**
- Dataset handles its own construction logic
- Clean separation of concerns
- No special handling needed in factory
- `make_dataset()` automatically detects and uses `from_config()`
- Supports complex preprocessing and validation

### Pattern 2: Using Decorator Only (Simple Cases)

For simple datasets without complex config preprocessing:

```python
@register_dataset('SimpleDataset')
class SimpleDataset(Dataset):
    """Simple dataset - config passed directly to __init__."""

    def __init__(self, data_dir: str, size: int = 100):
        self.data_dir = data_dir
        self.size = size
        self.data = torch.randn(size, 10)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
```

**When to use:**
- Simple datasets with straightforward initialization
- Config parameters map directly to `__init__` parameters
- No preprocessing or validation needed

### Creating Config Files

Create `configs/dataset/my_custom.yaml`:

```yaml
type: MyCustomDataset

data_dir: ${paths.data}/my_custom_data
custom_param: 20

dataloader:
  batch_size: 32
  shuffle: true
  num_workers: 4
```

### Usage

```bash
python train.py dataset=my_custom
```

The `make_dataset()` function will:
1. Look up 'MyCustomDataset' in the registry
2. Check if it has a `from_config()` classmethod
3. If yes: call `MyCustomDataset.from_config(config)`
4. If no: call `MyCustomDataset(**config)`

No manual registration or factory modifications needed!

## Testing

Run the test suite to verify dataset factory:

```bash
python test_dataset_factory.py
```

This tests:
- Registry functionality
- Config-based dataset creation
- Dataset iteration
- DataLoader integration
- YAML config loading

## Architecture

The dataset factory follows the same pattern as agent and environment factories:

```
trigor/data/
├── __init__.py          # Exports: make_dataset, register_dataset, list_datasets
├── registry.py          # Factory and registry implementation
├── tgn_dataset.py       # TGNDataset implementation
└── tokenizer.py         # TGNByteTokenizer implementation

configs/dataset/
├── tgn_default.yaml     # Default configuration
├── tgn_small.yaml       # Small/fast configuration
└── tgn_large.yaml       # Large/full configuration
```

**Registry (DATASETS)**: Maps dataset type names to classes
**Factory (make_dataset)**: Creates instances from config
**Config**: Hydra YAML files for declarative dataset creation

## Best Practices

1. **Use Config Files**: Define datasets in YAML for reproducibility
2. **Hydra Interpolation**: Use `${paths.root}` for portable paths
3. **Static Collate**: Use `TGNDataset.collate_batch` for DataLoader
4. **Parameter Validation**: Factory validates config before instantiation
5. **Type Registration**: Register custom datasets before use
6. **Error Handling**: Factory provides clear error messages for invalid configs

## See Also

- [TGN Dataset Usage](tgn_dataset_usage.md) - Detailed TGNDataset documentation
- [Agent Registry](../trigor/agents/registry.py) - Similar pattern for agents
- [Environment Registry](../trigor/envs/registry.py) - Similar pattern for environments
- [Hydra Documentation](https://hydra.cc/) - Configuration framework
