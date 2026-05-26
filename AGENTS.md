# Agent Guidelines

This file instructs AI coding agents (Claude Code, GitHub Copilot, Cursor, etc.) on how
to work correctly in this repository.

---

## Package management

- Use `uv` exclusively. Never generate `pip install`, `requirements.txt`, Poetry, or
  pipenv commands.
- Add dependencies with `uv add <package>` and commit `uv.lock` alongside every change.
- Sync environments with `uv sync` (production) or `uv sync --extra dev` (development).

## Python version

- Minimum: Python 3.11. Target: 3.12. Do not use APIs or syntax unavailable in 3.11.

## Formatting and linting

- Formatter: `ruff format`. Run via `make format`.
- Linter: `ruff check`. Run via `make lint`.
- Type checker: `mypy`. Run via `make type-check`.
- Line length: 100 characters.
- Run `make lint`, `make type-check`, and `make test` before marking any change complete.

## Commit style

This repository uses [release-please](https://github.com/googleapis/release-please) for
automated releases and changelog generation. All commits **must** follow
[Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | When to use |
|--------|-------------|
| `feat:` | A new capability or behavior visible to users |
| `fix:` | A bug fix |
| `chore:` | Maintenance, dependency bumps, tooling changes |
| `docs:` | Documentation-only changes |
| `ci:` | CI/CD pipeline changes |
| `refactor:` | Code restructuring with no behavior change |
| `test:` | Adding or correcting tests |
| `perf:` | Performance improvements |

- One concern per commit. Do not bundle unrelated changes.
- Commit messages in present tense, imperative mood (e.g. `feat: add panel-of-judges runner`).
- Breaking changes must include `BREAKING CHANGE:` in the commit footer or append `!`
  after the type (e.g. `feat!: remove legacy runner API`).

## Configuration files

- Format: YAML only. Two-space indentation. snake_case keys. No tabs.
- Registry files live under `config/registry/`.
- Production configs live under `prod_config/`.

## Data models

- Use Pydantic v2 for all models. Never use plain dataclasses or untyped dicts.

## Off-limits without explicit discussion

- `uv.lock` — never edit manually.
- Core abstractions (runner entrypoint, eval pipeline, judge logic) — ask before refactoring.

## Secrets

- Never hardcode API keys, tokens, or credentials.
- All secrets must be read from environment variables.
- Do not log raw prompts or model responses in production.

## Pull requests

- Rebase on `main` before opening or updating a PR.
- Use `git push --force-with-lease` after rebasing — never plain `--force`.
- PR title must follow Conventional Commits format (release-please reads it).
- Always include yourself under AI Tools Used (e.g. `Claude Code (claude-sonnet-4-6)`).
