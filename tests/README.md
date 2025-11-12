# Tests

This directory contains all test scripts for the TrigoRL project.

## Test Files

- `test_tgn_dataset.py` - Tests for TGN byte-level tokenizer and dataset loader
- `test_dataset_factory.py` - Tests for dataset factory and registry system
- `test_register_decorator.py` - Tests for `@register_dataset` decorator functionality
- `test_from_config.py` - Tests for `from_config()` pattern and dataset construction

## Running Tests

### Run All Tests

Use the Python test runner:

```bash
python tests/run_tests.py
```

Or use the shell script:

```bash
./tests/run_all_tests.sh
```

### Run Individual Tests

```bash
python tests/test_tgn_dataset.py
python tests/test_dataset_factory.py
python tests/test_register_decorator.py
python tests/test_from_config.py
```

## Test Coverage

All tests validate:

- ✅ **TGN Tokenizer**: Byte-level encoding/decoding
- ✅ **TGN Dataset**: File loading, batching, DataLoader integration
- ✅ **Dataset Registry**: Registration and factory creation
- ✅ **Decorator Pattern**: `@register_dataset` decorator functionality
- ✅ **from_config Pattern**: Self-contained dataset construction
- ✅ **Config-driven Creation**: YAML config loading and usage

## Test Results

Current status: **All tests passing** ✓

```
Total tests: 4
Passed: 4
Failed: 0
```
