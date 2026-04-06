# Code Quality Tools Configuration Guide

This guide covers the configuration and usage of complexipy and skylos for code quality analysis.

## Overview

| Tool | Purpose | Language |
|------|---------|----------|
| **complexipy** | Cognitive complexity analysis | Rust |
| **skylos** | Dead code detection + security scanning | Python |

## Complexipy (Cognitive Complexity)

### What is Cognitive Complexity?

Cognitive complexity measures how hard code is to **understand**, not just how complex it is structurally. It accounts for:
- Nesting depth
- Logical operators
- Control flow structures
- Recursion

### Installation

```bash
uv add --dev complexipy
```

### Basic Usage

```bash
# Analyze entire project
uv run complexipy .

# Analyze specific directory
uv run complexipy src/

# With threshold
uv run complexipy . --max-complexity-allowed 15
```

### Threshold Guidelines

| Threshold | Level | Recommended For |
|-----------|-------|-----------------|
| **10** | Strict | New projects, clean codebases |
| **15** | Moderate (Default) | Existing codebases, gradual improvement |
| **20** | Lenient | Legacy code, initial adoption |

### Understanding Scores

| Score | Level | Action |
|-------|-------|--------|
| 1-10 | Low | Code is easy to understand |
| 11-15 | Medium | Consider simplifying |
| 16-20 | High | Should refactor |
| 21+ | Critical | Must refactor |

### Complexity Factors

| Factor | Complexity Added |
|--------|-----------------|
| `if`, `elif`, `else` | +1 (+ nesting bonus) |
| `for`, `while` | +1 (+ nesting bonus) |
| `try`, `except` | +1 |
| `and`, `or` | +1 each |
| Recursion | +1 |
| Nesting bonus | +1 per level |

### Output Formats

```bash
# Console (default)
uv run complexipy src/

# JSON
uv run complexipy src/ --output json

# CSV
uv run complexipy src/ --output csv
```

### Configuration

#### Environment Variable
```bash
export COMPLEXIPY_MAX_COMPLEXITY=15
```

#### pyproject.toml
```toml
[tool.complexipy]
max-complexity-allowed = 15
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/rohaquinlop/complexipy-pre-commit
    rev: v3.3.0
    hooks:
      - id: complexipy
        args: [--max-complexity-allowed, "15"]
```

---

## Skylos (Dead Code Detection)

### What is Dead Code?

Dead code refers to:
- Unused functions
- Unused classes
- Unused variables
- Unused imports

### Installation

```bash
uv add --dev skylos
```

### Basic Usage

```bash
# Find dead code
uv run skylos .

# With confidence threshold
uv run skylos . --confidence 60

# Gate mode (exit 1 if dead code found)
uv run skylos . --gate
```

### Confidence Scores

| Score | Meaning | Recommended Action |
|-------|---------|-------------------|
| **90-100** | Definitely unused | Safe to delete immediately |
| **70-89** | Highly likely unused | Brief review, then delete |
| **60-69** | Probably unused | Careful review required |
| **40-59** | Possibly unused | High false positive risk |
| **0-39** | Uncertain | Skip, likely used dynamically |

### Security Scanning

Skylos can also detect security issues:

```bash
# Enable security scanning
uv run skylos . --danger --secrets

# Security flags:
# --danger: Check for dangerous patterns
# --secrets: Check for hardcoded secrets
```

### What Security Scan Detects

| Type | Examples |
|------|----------|
| API Keys | `API_KEY = "sk-..."` |
| Passwords | `password = "..."` |
| Tokens | `token = "eyJ..."` |
| Private Keys | `-----BEGIN PRIVATE KEY-----` |

### Framework-Aware Analysis

Skylos automatically excludes common framework patterns:

| Framework | Auto-excluded |
|-----------|---------------|
| **Django** | Models, views, migrations, management commands |
| **Flask** | Route handlers, blueprints |
| **FastAPI** | Endpoint functions, dependencies |
| **Pytest** | Test functions, fixtures, conftest |

### Output Formats

```bash
# Console (default)
uv run skylos src/

# JSON (for CI/CD)
uv run skylos src/ --output json

# SARIF (GitHub integration)
uv run skylos src/ --output sarif

# CSV (spreadsheet analysis)
uv run skylos src/ --output csv
```

### Configuration

#### Environment Variables
```bash
export SKYLOS_CONFIDENCE=60
export SKYLOS_SECURITY=true  # Enable --danger --secrets
```

### Pre-commit Hook (Custom)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: skylos-dead-code
        name: Skylos Dead Code Check
        entry: uv run skylos
        args: ['.', '-c', '80', '--gate']
        language: system
        pass_filenames: false
        types: [python]
```

---

## Combined Usage

### Recommended Workflow

```
1. Run dead code detection first
   → Reduces codebase size before complexity analysis

2. Run complexity analysis
   → Focus on remaining complex functions

3. Fix issues in order:
   a. Security issues (immediate)
   b. Dead code (quick wins)
   c. Complex functions (refactoring)

4. Verify with tests
   → Ensure no regressions
```

### Makefile Targets

```makefile
# Individual tools
make complexity    # Run complexipy
make dead-code     # Run skylos

# Combined analysis
make code-quality  # Run both tools
```

### CI/CD Integration

```yaml
# .github/workflows/quality.yml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install UV
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Check complexity
        run: uv run complexipy src/ --max-complexity-allowed 15

      - name: Check dead code
        run: uv run skylos src/ --gate --confidence 80
```

---

## Troubleshooting

### Complexipy Issues

**Problem**: False positives on simple functions
**Solution**: Check if function has many early returns; this is actually good practice

**Problem**: Module-level complexity reported
**Solution**: Complexipy only analyzes functions; module-level code isn't scored

### Skylos Issues

**Problem**: Framework code reported as dead
**Solution**: Skylos auto-detects most frameworks; check if your framework is supported

**Problem**: Dynamic code reported as dead
**Solution**: Code using `getattr()`, `importlib`, or similar won't be detected as used; use lower confidence threshold

**Problem**: Test fixtures reported as dead
**Solution**: Ensure `conftest.py` files are in the scan path

---

## Quick Reference

### Complexipy Commands
```bash
uv run complexipy .                              # Basic scan
uv run complexipy src/ --max-complexity-allowed 10  # Strict
uv run complexipy . --output json                # JSON output
```

### Skylos Commands
```bash
uv run skylos .                    # Basic scan
uv run skylos . -c 80 --gate       # High confidence + gate
uv run skylos . --danger --secrets # Security scan
uv run skylos . --output json      # JSON output
```

### Both Tools
```bash
make code-quality                  # Run both
make pre-commit                    # Full pre-commit check
```
