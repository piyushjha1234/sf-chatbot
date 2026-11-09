# sfrag — Salesforce Universal Documentation & RAG Platform

M1 core slice of [the v2 plan](Salesforce_Universal_RAG_Implementation_Plan_v2.md): an end-to-end, Data 360-first
pipeline from source discovery to grounded, cited, persona-aware answers.

## What is built (M1 core)

| Area | Module | Notes |
|---|---|---|
| Config | `core/settings.py`, `config/*.yaml` | `base` → `{dev,prod,test}` → `SFRAG_*` env vars (`__` for nesting) |
| Source registry & policy | `config/sources.yaml`, `sources/registry.py` | per-host ToS/robots review fields, permitted flag, authority tier, scope regexes, locales |
| Discovery | `sources/discovery.py` | robots `Sitemap:` lines, nested sitemap indexes, gz, link following with depth limit, URL normalisation, external-reference review queue |
| Fetching | `sources/fetch.py` | per-host rate limit, robots.txt, conditional GET (ETag / Last-Modified), retries with backoff + jitter, `Retry-After`, size cap |
| Durable queue | `pipeline/queue.py` | SQLite WAL, leases + heartbeat, crash recovery, idempotency keys, retry backoff, DLQ |
| File ingestion | `sources/watcher.py` | watches `data/incoming/`; folder sets the ACL; debounced + stable-size check |
| Parsers | `pipeline/extract/` | HTML (structure-preserving, boilerplate removal, Trafilatura fallback, JS-shell detection), PDF (headings by font size, tables, code, page refs), Markdown, TXT, DOCX, PPTX, XLSX, CSV, JSON/YAML/OpenAPI, XML, source code |
| Dedup | `pipeline/dedup.py` | MinHash LSH, atomic reservation across workers, canonical selection (authority → en-US → incumbent) |
| Security | `pipeline/security.py`, ACLs | prompt-injection quarantine for untrusted sources; doc-level ACL filter in Qdrant |
| Classification | `enrich/classify.py`, `taxonomy/taxonomy.yaml` | tier 1 rules (URL, aliases, identifiers, release/API version/lifecycle); tier 2 LLM (Haiku 5.5) only below confidence threshold, cached |
| Chunking | `enrich/chunker.py` | structure-aware parent/child; sticky units; table header repeated per row group; code split at syntax boundaries; **lossless invariant** property-tested |
| Contextual headers | `enrich/context.py` | product › title › heading path \| release \| type, prepended for embedding only |
| Index | `index/` | bge-m3 dense + sparse, Qdrant named vectors, server-side RRF, identifier prefetch, blue/green generations behind an alias, rollback |
| Query understanding | `query/planner.py` | multi-turn condense; schema-validated plan (Haiku 5.5); trivial/identifier queries skip it; plan cache |
| Retrieval | `query/retriever.py` | plan ‖ raw recall in parallel, multi-query RRF, per-doc collapse, bge-reranker-v2-m3, metadata/release/lifecycle boosts, hard filters only when explicit + confident |
| Context | `query/context.py` | parent expansion (or child ± siblings, group reassembly), token-budget packing, sandwich ordering |
| Generation | `query/engine.py`, `query/prompts.py`, `llm/client.py` | Sonnet 5.5 streaming, cached system prompt, `fallbacks: "default"` on refusals, abstention, sufficiency gate |
| Verification | `query/verify.py` | strips citation ids not in context; uncited-sentence ratio |
| API | `api/app.py` | `/chat` (SSE or JSON), `/ingest/url`, `/ingest/file`, `/sources`, `/feedback`, `/coverage`, `/health`; API keys; rate limit |
| Eval | `eval/` | retrieval metrics (Recall@5/10, P@5, MRR, nDCG@10), answer mode with LLM judge, held-out guard + touch log, CI regression gate |
| Ops | `ops/coverage.py` | reconciliation (every URL has a disposition), exclusions/failures by reason, unclassified backlog |

Not built yet (planned M2+): Playwright fetcher for JS-rendered pages (help.salesforce.com lands in
`needs_render`), PII/secret scanning, embedding-kNN classifier tier, semantic answer cache, OpenTelemetry/Langfuse
tracing, dashboards/alerts, web UI.

## Run on the target machine

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), ≥16 GB RAM (bge-m3 + reranker ≈ 4.5 GB; a GPU or
Apple-silicon MPS speeds up ingestion considerably), and a Claude API key.

```bash
uv sync --extra models               # installs FlagEmbedding + torch for bge-m3 / bge-reranker-v2-m3
export ANTHROPIC_API_KEY=...          # or `ant auth login`
uv run sfrag init                     # dirs, DB migration, sources.yaml -> registry, first index generation
```

Before crawling, review `config/sources.yaml`: verify the seed URLs, complete each host's `review` block (ToS and
robots), and set `permitted`.

```bash
uv run sfrag discover                 # enqueue discovery for permitted web sources
uv run sfrag serve                    # API on :8000 with in-process workers + file watcher
# or, in separate terminals with a Qdrant server (docker compose up -d qdrant; SFRAG_ENV=prod):
uv run sfrag worker --concurrency 4
uv run sfrag watch
uv run sfrag serve --no-with-worker --no-with-watcher
```

Embedded Qdrant locks its folder to one process, so use `serve` (workers in-process) or a Qdrant server.

```bash
uv run sfrag ask "How do reconciliation rules work in identity resolution?" --persona data_engineer
curl -N localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question":"How do I create a data stream?","persona":"administrator"}'
uv run sfrag coverage                 # reconciliation report (also written to data/reports/)
uv run sfrag queue dlq                # dead letters with reasons; --retry to requeue
```

### User files and ACLs

Drop files under `data/incoming/`. The folder decides who can retrieve them:

| Folder | ACL |
|---|---|
| `incoming/public/` | everyone |
| `incoming/users/<name>/` | `user:<name>` |
| `incoming/groups/<name>/` | `group:<name>` |
| anything else | `group:internal` |

`/ingest/file` uploads always land in the uploader's private folder. Persona is only a ranking signal; the ACL is
the only permission boundary.

### Re-indexing (blue/green)

Changing the embedding model, chunking or classification means bumping `pipeline_version` and then running:

```bash
uv run sfrag index rebuild            # new generation → validate count → atomic alias swap (keeps N-1)
uv run sfrag index rollback           # instant revert
```

## Evaluation

The dataset format is documented in `src/sfrag/eval/runner.py`; see `eval/datasets/seed.example.jsonl` for
examples. Domain SMEs must write the real questions, gold URLs and answer points. The example rows are
placeholders.

```bash
uv run sfrag eval eval/datasets/seed.jsonl --split dev                      # retrieval only, no LLM cost
uv run sfrag eval eval/datasets/seed.jsonl --split smoke --mode answer \
     --baseline data/reports/baseline-smoke.json                          # exits 1 on regression
uv run sfrag eval eval/datasets/seed.jsonl --split test --confirm-holdout   # logged; milestone sign-off only
```

## Development

```bash
uv sync                    # no heavy models needed
SFRAG_ENV=test uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src
```

Tests are hermetic: a hashing embedder, a lexical reranker, mocked HTTP and a fake LLM. They cover:
- the lossless chunking property,
- crash recovery (an expired lease is re-leased),
- conditional refresh, tombstoning and near-duplicate canonicalisation,
- ACL isolation, blue/green rebuild and rollback,
- citation stripping, abstention, answer caching,
- API auth and rate limiting, and the eval harness.

The hashing and lexical backends exist only for tests. Production uses bge-m3 and bge-reranker-v2-m3.
