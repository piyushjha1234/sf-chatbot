# sfrag — project context for Claude Code

Salesforce documentation RAG platform. The source of truth for scope is
`Salesforce_Universal_RAG_Implementation_Plan_v2.md`. `README.md` maps modules to plan phases. `HANDOVER.md` holds the
step-by-step continuation plan.

## Status

M1 core slice is built and tested with hermetic backends. It has not yet run against the real models or the Claude
API. That first real run is the first task on this machine.

## Decisions already made (do not re-litigate)

- **Models:** quality comes first. Embeddings use `BAAI/bge-m3` (dense + sparse); the reranker is
  `BAAI/bge-reranker-v2-m3`. The tokenizer is pinned to bge-m3.
- **LLM:** the Claude API. Answers use `claude-sonnet-5-5`. Query plans, condensing and low-confidence
  classification use `claude-haiku-5-5`. The judge is `claude-sonnet-5-5`. There are no sampling params; depth is set
  with `output_config.effort`.
- **Refusals:** answers stream through `client.beta.messages.stream` with `betas=["server-side-fallback-2026-07-01"]`
  and `fallbacks="default"`.
- **Vector store:** Qdrant, with named `dense` + `sparse` vectors and server-side RRF. Generations `chunks_vN` sit
  behind the alias `chunks_live`. Embedded mode locks its folder to one process; use `sfrag serve` (workers
  in-process) or a Qdrant server (`SFRAG_ENV=prod`).
- **Queue:** SQLite (WAL), lease-based, with idempotency keys and a DLQ. Every source type goes through
  `pipeline/orchestrator.py`.
- **Ranking and permissions:** persona, category and intent are soft ranking signals. The ACL is the only
  permission boundary.
- **Chunker:** must stay lossless. `HierarchicalChunker.reconstruct()` and its property test must keep passing.
- **Coverage:** the system is coverage-first. Unknown products and categories never block indexing. Every URL gets a
  terminal disposition with a reason.

## Conventions

- Python 3.12 with uv. Run `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src &&
  uv run pytest` before calling work done.
- Tests are hermetic and use `config/test.yaml`: a hashing embedder, a lexical reranker, simple tokenizer, no LLM and
  mocked HTTP. Never make tests depend on network, models or API keys.
- Bump `pipeline_version` in `config/base.yaml` whenever extraction, chunking or classification output changes. Then
  run `sfrag index rebuild` (blue/green).
- Tune only on the eval `dev` split. The `test` split requires `--confirm-holdout`, and every touch is logged.
- Secrets come from env vars only (`ANTHROPIC_API_KEY`, `SFRAG_API__API_KEYS`). Never commit `data/` or `.env`.
