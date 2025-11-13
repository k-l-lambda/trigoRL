# Dotenv Configuration Implementation

## Summary

Added support for configuring wandb parameters via environment variables loaded from a `.env.local` file using python-dotenv. This provides a secure and convenient way to manage API keys and other sensitive configuration.

## Changes Made

### 1. Updated `train_lm.py`

**Added dotenv support:**
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(dotenv_path='.env.local')
```

**Added configuration override function:**
```python
def override_wandb_config_from_env(config: DictConfig):
    """Override wandb configuration with environment variables."""
    env_mappings = {
        'WANDB_API_KEY': None,  # Set as env var
        'WANDB_ENTITY': 'training.wandb.entity',
        'WANDB_PROJECT': 'training.wandb.project',
        'WANDB_ENABLED': 'training.wandb.enabled',
    }
    # ... implementation
```

**Called in main function:**
```python
def main(config: DictConfig):
    logger.info("Checking environment variables for wandb configuration:")
    override_wandb_config_from_env(config)
    # ... rest of training
```

### 2. Created Template File

**`.env.local.example`** - Template for users to copy:
```bash
# Weights & Biases Configuration
WANDB_API_KEY=your_wandb_api_key_here
# WANDB_PROJECT=trigor
# WANDB_ENTITY=your_username_or_team
# WANDB_ENABLED=true
```

### 3. Created Documentation

**`docs/dotenv_configuration.md`** - Comprehensive guide covering:
- Setup instructions
- Supported environment variables
- Priority order (config < env < CLI)
- Usage examples
- Security best practices
- Troubleshooting
- Team collaboration workflows
- CI/CD integration

### 4. Created Test

**`tests/test_dotenv.py`** - Verification script demonstrating:
- Loading environment variables
- Overriding config values
- API key masking in logs
- Boolean conversion for WANDB_ENABLED

**`.env.local.test`** - Test data file

## Supported Environment Variables

| Variable | Purpose | Config Path | Example |
|----------|---------|-------------|---------|
| `WANDB_API_KEY` | API authentication | (env var only) | `1234...` |
| `WANDB_ENTITY` | Team/username | `training.wandb.entity` | `my_team` |
| `WANDB_PROJECT` | Project name | `training.wandb.project` | `trigor` |
| `WANDB_ENABLED` | Enable logging | `training.wandb.enabled` | `true` |

## Priority Order

Configuration values are resolved in this order:

1. **Config file** (`configs/training/*.yaml`) - Base defaults
2. **Environment variables** (`.env.local`) - Override config
3. **CLI arguments** (`--override`) - Override both

Example:
```yaml
# Config: enabled=false
# .env.local: WANDB_ENABLED=true
# CLI: training.wandb.enabled=false
# Result: enabled=false (CLI wins)
```

## Usage Examples

### Basic Setup

1. Copy template:
```bash
cp .env.local.example .env.local
```

2. Edit `.env.local`:
```bash
WANDB_API_KEY=your_actual_key
WANDB_ENABLED=true
```

3. Run training:
```bash
python train_lm.py
```

### Output with Environment Variables

```
[2025-11-13 15:00:00][INFO] - Checking environment variables for wandb configuration:
[2025-11-13 15:00:00][INFO] -   WANDB_API_KEY: ****** (set from .env.local)
[2025-11-13 15:00:00][INFO] -   WANDB_ENABLED: true (from .env.local)
[2025-11-13 15:00:00][INFO] -   WANDB_PROJECT: trigor_test (from .env.local)
[2025-11-13 15:00:00][INFO] -   WANDB_ENTITY: my_team (from .env.local)
```

### Without Environment Variables

```
[2025-11-13 15:00:00][INFO] - Checking environment variables for wandb configuration:
```
(No overrides logged)

## Security Features

### ✅ Implemented

1. **API key masking** - Logged as `******` instead of actual value
2. **Gitignore protection** - `.env.local` matched by `*.local` pattern
3. **Template separation** - `.env.local.example` has no secrets
4. **Environment isolation** - API key set as env var, not stored in config

### 🔒 Best Practices

- Store sensitive values in `.env.local`
- Never commit `.env.local` to git
- Share `.env.local.example` with team
- Use different keys for dev/prod
- Rotate keys periodically

## Testing Results

Test script output:
```
Initial config:
training:
  wandb:
    enabled: false
    entity: null
    project: trigor

Loading .env.local.test...

Environment variables loaded:
  WANDB_API_KEY: Yes
  WANDB_ENABLED: true
  WANDB_PROJECT: trigor_test
  WANDB_ENTITY: test_user

Applying environment variable overrides:
  WANDB_API_KEY: ****** (masked)
  WANDB_ENTITY: test_user
  WANDB_PROJECT: trigor_test
  WANDB_ENABLED: True

Final config:
training:
  wandb:
    enabled: true
    entity: test_user
    project: trigor_test

✓ Test completed successfully!
```

## Files Created/Modified

**Created:**
- `.env.local.example` - Template file (commit to git)
- `docs/dotenv_configuration.md` - Comprehensive documentation
- `tests/test_dotenv.py` - Test script
- `.env.local.test` - Test data
- `docs/dotenv_summary.md` - This summary

**Modified:**
- `train_lm.py` - Added dotenv loading and override function

**Gitignored:**
- `.env.local` - Actual secrets (already covered by `*.local`)
- `.env.local.test` - Test file (already covered by `*.local`)

## Dependencies

Uses `python-dotenv` which is already in `requirements.txt`:
```
python-dotenv>=1.0.0
```

## Integration with Existing Features

### Compatible with CLI Overrides

```bash
# .env.local sets WANDB_ENABLED=true
# CLI can still override:
python train_lm.py training.wandb.enabled=false
```

### Compatible with Hydra Configs

```yaml
# Config sets default
training:
  wandb:
    project: trigor

# .env.local overrides
WANDB_PROJECT=my_project

# Final: my_project
```

### Compatible with Multiple Configs

```bash
# Works with any config file
python train_lm.py training=trigo-llama  # Uses .env.local
python train_lm.py training=trigo-rwkv   # Uses .env.local
```

## Future Enhancements

Potential additions:
- Support for multiple env files (`.env.dev`, `.env.prod`)
- More wandb config options (mode, notes, tags)
- Validation of API key format
- Auto-detection of wandb availability
- Environment-specific defaults

## Team Workflow

### For New Team Members

1. Clone repository
2. Copy template: `cp .env.local.example .env.local`
3. Get wandb API key from https://wandb.ai/authorize
4. Fill in `.env.local`
5. Run training with wandb enabled

### For CI/CD

Set environment variables in CI system:
```yaml
# GitHub Actions
env:
  WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  WANDB_ENABLED: true
```

No need for `.env.local` file in CI - environment variables are set directly.

## Backward Compatibility

✅ Fully backward compatible:
- Existing configs work without `.env.local`
- CLI overrides still work as before
- No changes to config file format
- Optional feature - doesn't break existing workflows
