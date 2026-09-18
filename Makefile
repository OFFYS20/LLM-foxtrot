# Foxtrot — common tasks
.PHONY: help install install-backend install-frontend backend frontend dev test lint format build clean

help:
	@echo "Foxtrot"
	@echo "  make install     install backend + frontend dependencies"
	@echo "  make backend     run the API on :8000"
	@echo "  make frontend    run the UI on :3000"
	@echo "  make test        backend test suite"
	@echo "  make lint        ruff + tsc + next lint"
	@echo "  make build       production build of the frontend"
	@echo "  make clean       remove build artefacts and caches (keeps data/)"

install: install-backend install-frontend

install-backend:
	cd backend && python -m venv .venv && .venv/bin/pip install -U pip && \
		.venv/bin/pip install -r requirements-dev.txt

install-frontend:
	cd frontend && npm install

backend:
	cd backend && .venv/bin/python run.py

frontend:
	cd frontend && npm run dev

dev:
	@echo "Run 'make backend' and 'make frontend' in two terminals."

test:
	cd backend && .venv/bin/python -m pytest -q

lint:
	cd backend && .venv/bin/python -m ruff check app tests
	cd frontend && npm run typecheck && npm run lint

format:
	cd backend && .venv/bin/python -m ruff format app tests

build:
	cd frontend && npm run build

clean:
	rm -rf frontend/.next frontend/out backend/.pytest_cache backend/.ruff_cache
	find backend -name __pycache__ -type d -prune -exec rm -rf {} +
