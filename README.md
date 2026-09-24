# Webfic Checker

A consistency checker for Chinese web novels. It extracts character ages and story-time
statements from chapters and flags contradictions for the author to review.

Current stage: command-line prototype (age arithmetic checker).

## Setup

Requirements: Docker, Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env          # then set WEBFIC_LLM_API_KEY
docker compose up -d postgres
cd backend
uv sync
uv run alembic upgrade head
```

## Usage

```bash
uv run webfic ingest ../eval/golden/story01/text.txt   # split → extract → check
uv run webfic report <book-id>                          # contradiction report
uv run webfic usage <book-id>                           # token usage and cost
uv run webfic books                                     # list books
uv run webfic resume <book-id>                          # retry failed chapters
```

Book ids can be shortened to a unique prefix.

## Development

```bash
cd backend
uv run pytest          # unit + end-to-end tests (SQLite, fake LLM; no Docker or key needed)
uv run ruff check . && uv run ruff format --check .
```
