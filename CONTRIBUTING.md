# Contributing to efficient-plackett-luce

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/diogoribeiro7/efficient-plackett-luce.git
cd efficient-plackett-luce
```

1. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install development dependencies**

```bash
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=plackett_luce --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestPlackettLuceModel::test_fit_simple_data
```

### Code Style

We use **Black** for formatting and **flake8** for linting.

```bash
# Format code
black plackett_luce/ tests/ examples/

# Check formatting
black --check plackett_luce/ tests/ examples/

# Lint
flake8 plackett_luce/ tests/ examples/

# Sort imports
isort plackett_luce/ tests/ examples/
```

### Type Checking

```bash
mypy plackett_luce/
```

## Making Changes

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

1. **Make your changes**

  - Write clear, documented code
  - Add tests for new functionality
  - Update documentation as needed

2. **Ensure tests pass**

```bash
pytest
black --check .
flake8 .
```

1. **Commit your changes**

```bash
git add .
git commit -m "Add: brief description of changes"
```

Use conventional commit messages:

- `Add:` for new features
- `Fix:` for bug fixes
- `Docs:` for documentation
- `Test:` for tests
- `Refactor:` for code refactoring

- **Push and create a pull request**

```bash
git push origin feature/your-feature-name
```

## Pull Request Guidelines

- **Title**: Clear, concise description of changes
- **Description**: 

  - What changes were made
  - Why they were needed
  - Any breaking changes
  - Related issue numbers

- **Tests**: All tests must pass
- **Documentation**: Update docs if needed
- **Code style**: Must pass Black and flake8

## Areas for Contribution

### High Priority

- [ ] GPU acceleration with CuPy
- [ ] Sparse matrix optimization
- [ ] Additional validation metrics
- [ ] Performance benchmarks

### Documentation

- [ ] More examples
- [ ] Tutorial notebooks
- [ ] Video tutorials
- [ ] API reference improvements

### Testing

- [ ] Edge case tests
- [ ] Performance tests
- [ ] Integration tests
- [ ] Stress tests

### Features

- [ ] Time-varying models
- [ ] Bayesian uncertainty quantification
- [ ] scikit-learn integration
- [ ] Additional model variants

## Reporting Issues

When reporting issues, please include:

- **Python version**: `python --version`
- **Package version**: `pip show efficient-plackett-luce`
- **Operating system**
- **Minimal reproducible example**
- **Expected vs actual behavior**
- **Error messages** (full traceback)

## Questions?

- Open an issue with the `question` label
- Email: <dfr@ipp.ip.pt>
- Discussions: Use GitHub Discussions

## Code of Conduct

Be respectful, inclusive, and constructive. See <CODE_OF_CONDUCT.md> for details.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
