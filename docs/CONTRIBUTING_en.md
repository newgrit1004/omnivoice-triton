# Contributing Guide

Thank you for contributing to this project! Please follow the guidelines below.

## Development Environment Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd omnivoice-triton

# 2. Set up development environment
make setup

# 3. Pre-commit hooks are installed automatically by make setup
```

## Development Workflow

### 1. Branch Strategy

- `main`: Stable release branch
- `feature/*`: New feature development
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

```bash
# Create a new feature branch
git checkout -b feature/my-new-feature
```

### 2. Code Development

```bash
# Code formatting
make format

# Lint check
make lint

# Type check
make typecheck

# Run tests
make test
```

### 3. Commit Messages

Commit messages follow the `.gitmessage.txt` template:

```
[type] subject (50 chars or less)

body (wrap at 72 chars)
- What was changed
- Why it was changed

Related issue: #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation update
- `style`: Code formatting
- `refactor`: Refactoring
- `test`: Adding/updating tests
- `chore`: Build, config changes

### 4. Pull Request

1. Sync with the latest main branch
   ```bash
   git checkout main
   git pull origin main
   git checkout feature/my-new-feature
   git rebase main
   ```

2. Push your changes
   ```bash
   git push origin feature/my-new-feature
   ```

3. Create a Pull Request on GitHub

## Quality Standards

### Code Quality

- Ruff lint passes (`make lint`)
- Ruff format applied (`make format`)
- Ty type check passes (`make typecheck`)
- Test coverage >= 80% (`make test-cov`)

### Writing Tests

- Add tests for all new features
- Write regression tests for bug fixes
- Use `pytest`
- Test function naming: `test_<feature>_<scenario>`

Example:
```python
import pytest

def test_rms_norm_basic() -> None:
    """Basic behavior test."""
    result = triton_rms_norm(x, weight)
    assert torch.allclose(result, expected, atol=1e-5)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rms_norm_dtypes(dtype: torch.dtype) -> None:
    """Parametrized dtype test."""
    x = torch.randn(2, 512, 2048, dtype=dtype, device="cuda")
    # ...
```

### Documentation

- Write docstrings for all public functions/classes
- Use Google-style docstrings
- Update README.md when necessary

Example:
```python
def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization using a fused Triton kernel.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_size).
        weight: Normalization weight of shape (hidden_size,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor with the same shape as input.
    """
    ...
```

## Code Review

- All PRs require at least 1 approval
- Reviewers should respond within 24 hours
- Provide constructive feedback
- Verify CI passes

## Questions

For questions or suggestions, please create an issue.
