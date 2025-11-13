# Environment Variables Configuration

## Overview

The training script supports configuring wandb parameters via environment variables loaded from a `.env.local` file. This is the recommended way to manage sensitive information like API keys without committing them to the repository.

## Setup

### 1. Create `.env.local` file

Copy the example file and fill in your values:

```bash
cp .env.local.example .env.local
```

Edit `.env.local` with your configuration:

```bash
# .env.local
WANDB_API_KEY=your_actual_api_key_here
WANDB_ENTITY=your_username
WANDB_PROJECT=trigor
WANDB_ENABLED=true
```

### 2. Get Your wandb API Key

1. Visit https://wandb.ai/authorize
2. Copy your API key
3. Paste it in `.env.local`

## Supported Environment Variables

### WANDB_API_KEY (Required for wandb)

Your wandb API key for authentication.

```bash
WANDB_API_KEY=1234567890abcdef1234567890abcdef12345678
```

**Security:** This value is sensitive and should never be committed to git. The `.env.local` file is already in `.gitignore`.

**How it works:** The API key is set as an environment variable that wandb automatically detects during initialization.

### WANDB_ENTITY (Optional)

Override the wandb entity (team or username).

```bash
WANDB_ENTITY=my_team_name
```

**Default:** Uses value from config file (usually `null`)

**CLI override:** You can still override via CLI:
```bash
python train_lm.py training.wandb.entity=other_team
```

### WANDB_PROJECT (Optional)

Override the wandb project name.

```bash
WANDB_PROJECT=my_custom_project
```

**Default:** Uses value from config file (default: `trigor`)

**CLI override:**
```bash
python train_lm.py training.wandb.project=other_project
```

### WANDB_ENABLED (Optional)

Enable or disable wandb logging.

```bash
WANDB_ENABLED=true
```

**Valid values:** `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off` (case-insensitive)

**Default:** Uses value from config file (default: `false`)

**CLI override:**
```bash
python train_lm.py training.wandb.enabled=true
```

## Priority Order

Configuration values are applied in this order (later overrides earlier):

1. **Config file** (`configs/training/trigo-gpt2.yaml`)
2. **Environment variables** (`.env.local`)
3. **CLI arguments** (`python train_lm.py training.wandb.enabled=true`)

### Example

**Config file:**
```yaml
training:
  wandb:
    enabled: false
    project: trigor
    entity: null
```

**.env.local:**
```bash
WANDB_ENABLED=true
WANDB_PROJECT=my_project
```

**CLI:**
```bash
python train_lm.py training.wandb.entity=my_team
```

**Final configuration:**
- `enabled`: `true` (from `.env.local`, overrides config)
- `project`: `my_project` (from `.env.local`, overrides config)
- `entity`: `my_team` (from CLI, overrides both)

## Usage Examples

### Basic Usage with API Key

1. Create `.env.local`:
```bash
WANDB_API_KEY=your_key_here
WANDB_ENABLED=true
```

2. Run training:
```bash
python train_lm.py
```

The script will:
- Load `.env.local`
- Enable wandb logging
- Authenticate with your API key
- Use default project name from config

### Custom Project and Entity

**.env.local:**
```bash
WANDB_API_KEY=your_key_here
WANDB_ENABLED=true
WANDB_PROJECT=trigo_experiments
WANDB_ENTITY=research_team
```

```bash
python train_lm.py
```

Result: Logs to `research_team/trigo_experiments` on wandb

### Temporarily Disable wandb

Even with API key in `.env.local`, you can disable wandb from CLI:

```bash
python train_lm.py training.wandb.enabled=false
```

### Different Project per Experiment

Keep API key in `.env.local`, override project via CLI:

**.env.local:**
```bash
WANDB_API_KEY=your_key_here
```

```bash
# Experiment 1
python train_lm.py training.wandb.enabled=true training.wandb.project=exp1

# Experiment 2
python train_lm.py training.wandb.enabled=true training.wandb.project=exp2
```

## Log Output

When the script loads environment variables, it logs them (with API key masked):

```
[2025-11-13 15:00:00][INFO] - Checking environment variables for wandb configuration:
[2025-11-13 15:00:00][INFO] -   WANDB_API_KEY: ****** (set from .env.local)
[2025-11-13 15:00:00][INFO] -   WANDB_ENABLED: true (from .env.local)
[2025-11-13 15:00:00][INFO] -   WANDB_PROJECT: trigo_experiments (from .env.local)
[2025-11-13 15:00:00][INFO] -   WANDB_ENTITY: my_team (from .env.local)
```

If no environment variables are set, you'll see:
```
[2025-11-13 15:00:00][INFO] - Checking environment variables for wandb configuration:
```
(no additional lines)

## Security Best Practices

### ✅ DO

- Store API keys in `.env.local`
- Add `.env.local` to `.gitignore` (already done)
- Share `.env.local.example` with your team (no secrets)
- Use different API keys for different environments (dev/prod)
- Rotate API keys periodically

### ❌ DON'T

- Commit `.env.local` to git
- Share your API key in chat/email
- Hardcode API keys in code or config files
- Use production API keys for local development
- Include API keys in screenshots or logs

## Troubleshooting

### wandb authentication fails

**Symptom:**
```
wandb: ERROR Unable to authenticate
```

**Solutions:**
1. Check API key is correct in `.env.local`
2. Verify `.env.local` exists in project root
3. Ensure no extra spaces in API key
4. Get a new API key from https://wandb.ai/authorize

### Environment variables not loading

**Symptom:** Changes to `.env.local` not reflected

**Solutions:**
1. Verify `.env.local` is in the same directory as `train_lm.py`
2. Check file name is exactly `.env.local` (not `.env.local.txt`)
3. Restart your terminal/IDE to pick up changes
4. Check for syntax errors in `.env.local`

### API key exposed in logs

**Expected behavior:** API key should be masked as `******` in logs

If you see the actual key:
1. Report as security issue
2. Rotate your API key immediately

### wandb still disabled despite WANDB_ENABLED=true

**Possible causes:**
1. CLI override: `training.wandb.enabled=false`
2. Typo in `.env.local`: `WANDB_ENABLE` (missing D)
3. Invalid value: Must be `true`, not `True` or `TRUE` (case-insensitive but check)

Check the final config in logs:
```yaml
training:
  wandb:
    enabled: true  # Should be true if working
```

## Team Collaboration

### Sharing Configuration

**Commit to git:**
- ✅ `.env.local.example` (template without secrets)
- ✅ Config files with defaults
- ❌ `.env.local` (contains secrets)

**Share with team:**
1. Team member clones repo
2. Copies `.env.local.example` to `.env.local`
3. Gets their own wandb API key
4. Fills in `.env.local`

### CI/CD Integration

For automated training runs, set environment variables in CI:

**GitHub Actions:**
```yaml
env:
  WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  WANDB_PROJECT: trigor_ci
  WANDB_ENABLED: true
```

**GitLab CI:**
```yaml
variables:
  WANDB_PROJECT: trigor_ci
  WANDB_ENABLED: "true"
  # WANDB_API_KEY set as secret variable in GitLab UI
```

**Docker:**
```bash
docker run -e WANDB_API_KEY=$WANDB_API_KEY \
           -e WANDB_ENABLED=true \
           training_image python train_lm.py
```

## Advanced Configuration

### Multiple Environment Files

Load different env files for different scenarios:

```python
# Development
load_dotenv('.env.local')

# Production
load_dotenv('.env.production')

# Testing
load_dotenv('.env.test')
```

Currently, the script only loads `.env.local` by default.

### Environment Variable Prefixes

All wandb-related variables use `WANDB_` prefix for clarity and to avoid conflicts.

### Adding New Environment Variables

To add support for more config options:

1. Add to `override_wandb_config_from_env()` in `train_lm.py`:
```python
env_mappings = {
    'WANDB_API_KEY': None,
    'WANDB_ENTITY': 'training.wandb.entity',
    'WANDB_PROJECT': 'training.wandb.project',
    'WANDB_ENABLED': 'training.wandb.enabled',
    'WANDB_NAME': 'training.wandb.name',  # New!
}
```

2. Add to `.env.local.example`:
```bash
# WANDB_NAME=my_run_name
```

3. Document in this file

## Files

- `.env.local.example` - Template with all supported variables
- `.env.local` - Your actual configuration (gitignored)
- `.gitignore` - Ensures `.env.local` isn't committed
- `train_lm.py` - Loads and applies environment variables
- `docs/dotenv_configuration.md` - This documentation
