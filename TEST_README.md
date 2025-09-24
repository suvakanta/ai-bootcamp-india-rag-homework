# Unit Tests for RAG Evaluation Tool

This document describes the unit tests implemented for the RAG evaluation tool (`eval.py`).

## Overview

The unit tests provide comprehensive coverage for all functions in the `eval.py` module, including:

- `test_dataset()` - Loading and processing the evaluation dataset
- `eval_dataset()` - Creating evaluation datasets from JSON data
- `validate_json()` - Validating JSON input format
- `run_ragas_eval()` - Running RAGAS evaluation metrics
- `process_file()` - Processing input files and orchestrating evaluation
- `main()` - Command-line interface functionality

## Test Coverage

The tests achieve **99% code coverage** of the `eval.py` module, testing:

- Normal operation scenarios
- Edge cases (empty data, missing files, etc.)
- Error conditions (invalid JSON, malformed data, etc.)
- Integration between functions

## Running the Tests

### Using unittest (built-in Python testing framework)

```bash
# Run all tests with verbose output
python -m unittest test_eval.py -v

# Run specific test class
python -m unittest test_eval.TestEvalModule -v

# Run specific test method
python -m unittest test_eval.TestEvalModule.test_validate_json_valid_data -v
```

### Using pytest (recommended)

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
python -m pytest test_eval.py -v

# Run tests with coverage report
python -m pytest test_eval.py --cov=eval --cov-report=term-missing

# Run tests with HTML coverage report
python -m pytest test_eval.py --cov=eval --cov-report=html
```

### Using coverage directly

```bash
# Run tests with coverage
python -m coverage run -m unittest test_eval.py

# Generate coverage report
python -m coverage report -m

# Generate HTML coverage report
python -m coverage html
```

## Test Structure

### Test Classes

1. **TestEvalModule** - Unit tests for individual functions
2. **TestIntegration** - Integration tests for function interactions

### Test Categories

- **Success cases**: Testing normal operation with valid inputs
- **Error cases**: Testing error handling with invalid inputs
- **Edge cases**: Testing boundary conditions (empty data, missing fields)
- **Mock tests**: Using mocks to isolate function behavior

## Key Testing Features

- **Mocking**: External dependencies (file I/O, dataset loading, evaluation) are mocked
- **Fixtures**: Test data is set up in `setUp()` methods
- **Isolation**: Each test is independent and doesn't affect others
- **Comprehensive**: Tests cover all code paths and error conditions

## Dependencies

The tests require the following packages (included in `requirements.txt`):

- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `unittest.mock` - Mocking functionality (built-in)

## Example Test Output

```
test_eval.py::TestEvalModule::test_eval_dataset_success PASSED
test_eval.py::TestEvalModule::test_validate_json_valid_data PASSED
test_eval.py::TestEvalModule::test_run_ragas_eval_success PASSED
...

======================= 21 passed in 1.66s =======================
```

## Contributing

When adding new functionality to `eval.py`, please:

1. Add corresponding unit tests in `test_eval.py`
2. Maintain test coverage above 95%
3. Test both success and failure scenarios
4. Use appropriate mocking for external dependencies