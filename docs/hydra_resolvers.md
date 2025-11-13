# Hydra Resolvers Guide

## Overview

Hydra provides built-in resolvers that allow you to access runtime information in your configuration files. This project also adds **custom resolvers** for date/time functionality.

## Available Resolvers

### Custom Resolvers (Registered in train_lm.py)

#### Date - `${date:}`

Gets the current date in `yyyymmdd` format (e.g., `20251113`).

**Example:**
```yaml
training:
  # Use date in paths
  save_dir: ${paths.output}/checkpoints/${hydra:job.config_name}_${date:}
  # Result: ./outputs/checkpoints/trigo-gpt2_20251113

  wandb:
    # Use date in experiment name
    name: ${hydra:job.config_name}_${date:}
    # Result: trigo-gpt2_20251113

    tags:
      - ${hydra:job.config_name}
      - ${date:}
      # Results: ['trigo-gpt2', '20251113']
```

**Use cases:**
- Date-stamped checkpoint directories
- Experiment naming with dates
- Log file naming
- Daily experiment tracking

**Note:** The date is evaluated when the config is loaded, not when it's used. All uses of `${date:}` in a single run will have the same value.

### Built-in Hydra Resolvers

### 1. Config Name - `${hydra:job.config_name}`

Gets the configuration file name (without `.yaml` extension).

**Example:**
```yaml
# In trigo-gpt2.yaml
training:
  save_dir: ${paths.output}/checkpoints/${hydra:job.config_name}
  # Resolves to: ./outputs/checkpoints/trigo-gpt2

  wandb:
    name: ${hydra:job.config_name}
    # Resolves to: trigo-gpt2
```

**Use cases:**
- Automatic checkpoint directory naming
- Wandb run names
- Log file names
- Experiment tracking

### 2. Current Working Directory - `${hydra:runtime.cwd}`

Gets the current working directory where the script was launched.

**Example:**
```yaml
paths:
  data: ${hydra:runtime.cwd}/data
  # Resolves to: /home/user/project/data
```

**Use cases:**
- Absolute path construction
- Data loading paths
- Output directories

### 3. Hydra Output Directory - `${hydra:runtime.output_dir}`

Gets Hydra's managed output directory (timestamped by default).

**Example:**
```yaml
logging:
  log_file: ${hydra:runtime.output_dir}/train.log
  # Resolves to: ./outputs/2025-11-13/16-35-51/train.log
```

**Use cases:**
- Log files
- Temporary outputs
- Per-run artifacts

### 4. Override Directory Name - `${hydra:job.override_dirname}`

Gets a string representation of command-line overrides.

**Example:**
```yaml
# Run with: python train.py training.learning_rate=1e-3
experiment:
  name: ${hydra:job.config_name}_${hydra:job.override_dirname}
  # Resolves to: trigo-gpt2_training.learning_rate=1e-3
```

**Use cases:**
- Distinguishing runs with different parameters
- Experiment naming
- Sweep identification

### 5. Job Name - `${hydra:job.name}`

Gets the job name (usually "string" when running as a script, or the actual job name in multirun mode).

**Note:** For config file name, use `${hydra:job.config_name}` instead.

## Syntax Rules

### Basic Interpolation

```yaml
# Simple interpolation (no quotes needed)
name: ${hydra:job.config_name}
path: ${paths.output}/${hydra:job.config_name}
```

### In Strings

```yaml
# Inside strings (quotes required)
message: "Training config: ${hydra:job.config_name}"
```

### In Lists (Block Style - Recommended)

```yaml
# Block-style lists work directly
tags:
  - ${hydra:job.config_name}
  - baseline
  - experiment
```

### In Lists (Inline Style - Avoid)

```yaml
# Inline lists DON'T work without quotes
tags: [${hydra:job.config_name}, test]  # ✗ YAML parse error!

# Use block style instead
tags:
  - ${hydra:job.config_name}  # ✓ Works
  - test
```

## Practical Examples

### Example 1: Config-Based Checkpoint Directories

```yaml
# configs/training/trigo-gpt2.yaml
training:
  save_dir: ${paths.output}/checkpoints/${hydra:job.config_name}
  # Result: ./outputs/checkpoints/trigo-gpt2

# configs/training/trigo-llama.yaml
training:
  save_dir: ${paths.output}/checkpoints/${hydra:job.config_name}
  # Result: ./outputs/checkpoints/trigo-llama
```

**Benefit:** Each config automatically gets its own checkpoint directory without hardcoding names.

### Example 2: Wandb Integration

```yaml
training:
  wandb:
    enabled: true
    project: trigor
    name: ${hydra:job.config_name}  # Automatic run name
    tags:
      - ${hydra:job.config_name}
      - ${hydra:job.override_dirname}
```

**Benefit:** Wandb runs are automatically named after the config, making them easy to identify.

### Example 3: Experiment Tracking

```yaml
experiment:
  name: ${hydra:job.config_name}
  output_dir: ${paths.output}/${hydra:job.config_name}/${hydra:job.override_dirname}
  config_snapshot: ${hydra:runtime.output_dir}/config.yaml
```

**Benefit:** Complete experiment tracking with automatic naming and organization.

### Example 4: Logging Setup

```yaml
logging:
  log_dir: ${paths.output}/logs/${hydra:job.config_name}
  tensorboard_dir: ${paths.output}/tensorboard/${hydra:job.config_name}
  checkpoint_dir: ${paths.output}/checkpoints/${hydra:job.config_name}
```

**Benefit:** Organized output structure based on config name.

## Common Patterns

### Pattern 1: Per-Config Output Directories

```yaml
paths:
  root: .
  output: ${paths.root}/outputs

training:
  save_dir: ${paths.output}/checkpoints/${hydra:job.config_name}
  log_dir: ${paths.output}/logs/${hydra:job.config_name}
```

### Pattern 2: Timestamped Outputs

```yaml
training:
  save_dir: ${hydra:runtime.output_dir}/checkpoints
  # Result: ./outputs/2025-11-13/16-35-51/checkpoints
```

### Pattern 3: Combined Config + Override Naming

```yaml
experiment:
  name: ${hydra:job.config_name}_${hydra:job.override_dirname}
  # Example: trigo-gpt2_training.lr=1e-3_data.batch_size=16
```

## Accessing Resolvers in Python

If you need to access these values in Python code:

```python
import hydra
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    # Access Hydra config
    hydra_cfg = HydraConfig.get()

    config_name = hydra_cfg.job.config_name
    output_dir = hydra_cfg.runtime.output_dir
    cwd = hydra_cfg.runtime.cwd

    print(f"Config: {config_name}")
    print(f"Output: {output_dir}")
    print(f"CWD: {cwd}")
```

## Environment Variables

You can also use environment variables in configs:

```yaml
# Access environment variable
data_dir: ${oc.env:DATA_DIR}

# With default value
data_dir: ${oc.env:DATA_DIR,./data}
```

## OmegaConf Resolvers

OmegaConf also provides its own resolvers:

```yaml
# Select from config (with default)
value: ${oc.select:some.path,default_value}

# Decode from string
value: ${oc.decode:base64_string}
```

## Best Practices

1. **Use `config_name` for persistent artifacts**:
   - Checkpoints
   - Model outputs
   - Results

2. **Use `output_dir` for temporary artifacts**:
   - Logs
   - Debug files
   - Intermediate results

3. **Avoid inline lists with interpolations**:
   ```yaml
   # Don't do this
   tags: [${hydra:job.config_name}, test]

   # Do this instead
   tags:
     - ${hydra:job.config_name}
     - test
   ```

4. **Be explicit with quotes in strings**:
   ```yaml
   # Good practice
   message: "Config: ${hydra:job.config_name}"
   ```

5. **Don't hardcode what can be automatic**:
   ```yaml
   # Bad
   save_dir: ./outputs/checkpoints/trigo-gpt2

   # Good
   save_dir: ${paths.output}/checkpoints/${hydra:job.config_name}
   ```

## Common Mistakes

### Mistake 1: Wrong Resolver Name

```yaml
# ✗ Wrong
name: ${hydra:config_name}

# ✓ Correct
name: ${hydra:job.config_name}
```

### Mistake 2: Inline List Syntax

```yaml
# ✗ Parse error
tags: [${hydra:job.config_name}, test]

# ✓ Works
tags:
  - ${hydra:job.config_name}
  - test
```

### Mistake 3: Using job.name Instead of job.config_name

```yaml
# ✗ Returns "string" when running as script
name: ${hydra:job.name}

# ✓ Returns actual config name
name: ${hydra:job.config_name}
```

## Testing Resolvers

Test your resolver configuration:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/training", config_name="your-config")
def test(cfg: DictConfig):
    print(f"Config name: {cfg.training.wandb.name}")
    print(f"Save dir: {cfg.training.save_dir}")

if __name__ == '__main__':
    test()
```

## References

- [Hydra Documentation - OmegaConf](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Resolvers](https://hydra.cc/docs/advanced/override_grammar/extended/)

## Summary

**Key resolver for config file name:**
```yaml
${hydra:job.config_name}
```

**Common use cases:**
- `save_dir: ${paths.output}/checkpoints/${hydra:job.config_name}`
- `name: ${hydra:job.config_name}`
- `tags: [${hydra:job.config_name}]` (use block style)

**Result:** Automatic, maintainable configuration without hardcoded names!
