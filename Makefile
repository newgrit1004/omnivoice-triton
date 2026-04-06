DIR?=.
SERVICE_NAME=python
VERSION=0.1

# 프로젝트 초기 설정
# UV: uv sync --all-extras --dev
# pip: pip install -r requirements.txt
.PHONY: setup
setup:
	git config commit.template .gitmessage.txt
	uv sync --all-extras --dev
	uv run pre-commit install

# 코드 포맷팅 (Black + isort 대체)
# UV: uv run ruff format
# pip: black . && isort .
.PHONY: format
format:
	uv run ruff format ${DIR}

# 코드 린팅 체크
# UV: uv run ruff check
# pip: ruff check . 또는 flake8 .
.PHONY: lint
lint:
	uv run ruff check ${DIR}

# 코드 린팅 + 자동 수정
# UV: uv run ruff check --fix
# pip: ruff check --fix . 또는 autopep8 --in-place --recursive .
.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix ${DIR}

# 타입 체킹 (mypy/pyright 대체)
# UV: uv run ty check
# pip: ty check 또는 mypy . 또는 pyright
.PHONY: typecheck
typecheck:
	uv run ty check

# 테스트 실행
# UV: uv run pytest
# pip: pytest
.PHONY: test
test:
	uv run pytest

# 테스트 + 커버리지 리포트
# UV: uv run pytest --cov
# pip: pytest --cov
.PHONY: test-cov
test-cov:
	uv run pytest --cov

# 커버리지 HTML 리포트 열기 (OS 자동 감지)
.PHONY: cov-report
cov-report:
	@if [ "$$(uname)" = "Darwin" ]; then \
		open htmlcov/index.html; \
	elif [ "$$(uname)" = "Linux" ]; then \
		xdg-open htmlcov/index.html 2>/dev/null || echo "브라우저로 htmlcov/index.html을 열어주세요"; \
	else \
		start htmlcov/index.html 2>/dev/null || echo "브라우저로 htmlcov/index.html을 열어주세요"; \
	fi

# 캐시 및 임시 파일 정리
.PHONY: clean
clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# 의존성만 설치 (프로덕션 + 개발)
# UV: uv sync
# pip: pip install -r requirements.txt
.PHONY: install
install:
	uv sync

# 의존성 업데이트 (lock 파일 갱신)
# UV: uv lock --upgrade
# pip: pip install --upgrade -r requirements.txt && pip freeze > requirements.txt
.PHONY: update
update:
	uv lock --upgrade

# pre-commit 전체 실행
# UV: uv run pre-commit run --all-files
# pip: pre-commit run --all-files
.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

# 코드 복잡도 검사 (complexipy)
# UV: uv run complexipy
.PHONY: complexity
complexity:
	uv run complexipy ${DIR} --max-complexity-allowed 15

# 데드 코드 검사 (skylos)
# UV: uv run skylos
.PHONY: dead-code
dead-code:
	uv run skylos ${DIR} --confidence 60

# 전체 코드 품질 검사 (complexity + dead-code)
.PHONY: code-quality
code-quality:
	@echo "=== Dead Code Analysis ==="
	uv run skylos ${DIR} --confidence 60 || true
	@echo ""
	@echo "=== Complexity Analysis ==="
	uv run complexipy ${DIR} --max-complexity-allowed 15 || true

# 전체 체크 (lint + typecheck + test)
.PHONY: check
check: lint typecheck test
