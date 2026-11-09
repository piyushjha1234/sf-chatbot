# Salesforce Universal Documentation & RAG Platform — Revised Plan (v2)

## Context
The user wrote `Salesforce_Universal_RAG_Implementation_Plan.md` (v1) and asked for optimisations that make it a production-ready solution. The repo holds only that plan, with no code yet. So this revision changes the plan itself: phasing, architecture, stack, quality gates and operations. The intended outcome is a plan that ships a measurable, working vertical slice early. It should then scale coverage safely and stop regressions from reaching users.

v1 is strong on taxonomy, chunking rules and coverage philosophy. Its main gaps:
1. Evaluation comes last (Phase 9), so phases 4–8 get built with no feedback signal.
2. It builds breadth-first: all 7 domains go through every phase before anything works end to end.
3. It plans two separate retrieval stores (Qdrant plus a separate BM25 index) and leaves the rerank and fusion plumbing custom.
4. Classification cost is unbounded: the LLM runs on every document and every chunk.
5. Three things are missing: JS-rendered page handling, near-duplicate handling across locales and versions, and Salesforce release-cycle awareness.
6. Operations are missing: SLOs, observability, security (prompt injection, ACLs on user files), blue/green re-indexing, CI quality gates and a user feedback loop.
7. The LLM provider, serving API, UI, multi-turn handling and abstention behaviour are not specified.

---

## Summary of optimisations (what changed and why)

| # | Change | Why |
|---|---|---|
| O1 | **Eval-first**: a 150-question gold "seed set" is built in Phase 0. The full 1,000-question set grows alongside the build. | Every later decision (chunk size, model, fusion) gets measured from day one. |
| O2 | **Vertical slice first**: one domain (Data 360) goes end to end before the others are added. | Pipeline flaws show up in weeks, not months. Other domains then reuse a proven path. |
| O3 | **Single hybrid store**: Qdrant holds dense and sparse vectors as named vectors, with server-side RRF through the Query API (`prefetch` + `fusion`). | One index, one consistency model, atomic updates. Fewer moving parts. The lexical adapter stays for swapping later. |
| O4 | **bge-m3** (dense + sparse in one pass, 8k context) for embeddings and **bge-reranker-v2-m3** for reranking, as the local baselines. | One model yields both vector types. Strong on technical text. Runs locally on CPU or GPU. |
| O5 | **Contextual chunk headers**: before embedding, each child chunk is prefixed with product › doc title › heading path. LLM-generated chunk context is optional and benchmark-gated. | Cheap, large recall gain for short, ambiguous chunks such as "Click Save". |
| O6 | **Tiered classification**: rules, URL paths and alias dictionaries run first, then an embedding kNN classifier. The LLM runs only on low-confidence cases. Results are cached by content hash. | Cuts LLM classification calls by about 80–95%. Results are deterministic and reproducible. |
| O7 | **Near-duplicate and canonicalisation**: SimHash/MinHash across locales, versions and mirrors, with canonical selection and duplicate-of links. Phase 1 indexes en-US only. | Salesforce docs repeat heavily. Duplicates waste the index and crowd out diverse evidence. |
| O8 | **Release awareness**: model the Salesforce release (Spring/Summer/Winter + API version) as first-class metadata. Current release is boosted; deprecated or retired content is down-ranked. | Wrong-release answers are the top failure mode for Salesforce questions. |
| O9 | **Rendering-aware fetcher**: plain HTTP with ETag/Last-Modified first. Playwright is the fallback for JS-rendered pages (help.salesforce.com, some developer docs). Per-host rate limits. | Many official pages do not render content without JS. Conditional requests make refreshes cheap. |
| O10 | **Fast-path query understanding**: a small, fast LLM builds the query plan. It runs in parallel with raw-query retrieval, and trivial queries skip it. Multi-turn follow-ups get a condense/rewrite step. | Keeps p95 latency within SLO. Low confidence never blocks retrieval. |
| O11 | **Token-budget context packing**: dedupe parents, order evidence by relevance (best first and last), and cap at a fixed token budget. | Prevents context bloat and "lost in the middle" degradation. |
| O12 | **Grounding enforcement**: per-claim citation IDs, post-generation citation verification, and explicit abstention when evidence is insufficient. | Faithfulness becomes enforced rather than hoped for. |
| O13 | **Security**: doc-level ACLs on user-uploaded files (filtered in Qdrant), PII/secret scanning at ingestion, and a prompt-injection quarantine for untrusted content. | User files are private, and ingested text is untrusted input to the LLM. |
| O14 | **Blue/green re-indexing** via Qdrant collection aliases. Every chunk records `pipeline_version` and `embedding_model_version`. | Model or chunking changes ship with zero downtime and instant rollback. |
| O15 | **Observability and SLOs**: OpenTelemetry traces across ingestion and query, LLM tracing (Langfuse or Phoenix, self-hosted), and quantitative SLOs. | "Production ready" needs measurable targets and debuggability. |
| O16 | **CI quality gate**: parser golden files, chunker property tests, and a 50-question smoke eval that blocks merges on regression. | Regressions caught before deploy. |
| O17 | **Feedback loop**: thumbs up/down plus a reason. Failed queries feed the dev eval set after review. | The eval set tracks real usage, not only synthetic data. |
| O18 | **Lean package layout** (`src/sfrag/`, uv, ruff, mypy, pytest). Data lives outside the repo. | Standard, typed, reproducible engineering baseline. |

**Assumptions** (adjust if wrong):
- The answer LLM sits behind a provider-agnostic interface (LiteLLM-style). The defaults are Claude Sonnet 5.5 for answers and Claude Haiku 5.5 for query planning and classification. A local Ollama model is a drop-in option for fully offline use.
- The serving API is FastAPI with SSE streaming, behind a thin web UI.
- At first this is a single-node deployment (Docker Compose), with a small internal user base.

---

## Target architecture

```text
INGESTION (async, independent)
Sources ─► Discovery/Registry ─► Fetch (HTTP+ETag │ Playwright fallback │ file watcher │ CLI/API)
        ─► Durable Queue (SQLite WAL, leases, retries, DLQ)
        ─► Extract/Normalize ─► Dedup/Canonicalize ─► Security scan (PII/secrets/injection)
        ─► Doc classification (rules → kNN → LLM) ─► Hierarchical chunking
        ─► Chunk classification ─► Contextual headers ─► bge-m3 dense+sparse
        ─► Qdrant (staging collection) ─► validate ─► alias swap

QUERY (sync, streaming)
Question (+history) ─► Condense ─┬─► Raw-query hybrid retrieval ──────────┐
                                 └─► Query plan (fast LLM, schema)───────►├─► Planned hybrid retrieval
                                                                           ▼
     RRF (server-side) ─► dedupe ─► rerank (+metadata/release boosts) ─► parent/sibling expansion
     ─► sufficiency check ─► token-budget packing ─► grounded generation ─► citation verification
     ─► stream answer + citations ─► feedback capture

CROSS-CUTTING: OTel traces · LLM tracing · coverage audit · eval harness · CI gates
```

---

## Phases and milestones

Each phase has **exit criteria**. Do not start M2 breadth until the M1 slice meets its gates.

### Phase 0: Scope, governance, eval seed (week 1–2)
- Source registry: authority tier, scope, permission/ToS status, retention, exclusion reason. Run a **ToS/robots review per host**. Trailhead and some content may restrict automated collection. Record that decision in the registry.
- Keep taxonomy (controlled vocabularies + alias tables) separate from ingestion rules. Feature lists are classification aids, never crawl filters.
- Provenance schema and coverage metric definitions.
- **Build the eval seed set**: 150 questions across 7 domains × personas × intents. Annotate expected sources and answer points (human-written, SME-reviewed).
- Define SLOs (see below) and the measurement methods behind them.
- **Exit**: registry and policy committed, seed set annotated, SLO doc agreed.

### Phase 1: Engineering foundation (week 2)
- Repo: `src/sfrag/` package, `pyproject.toml` (uv), ruff, mypy (strict on core), pytest, pre-commit, Docker Compose (app + Qdrant + Langfuse/Phoenix).
- Config: Pydantic Settings; one YAML per environment; all thresholds externalised.
- Canonical Pydantic models: `Source`, `Document`, `Section(parent)`, `Chunk(child)`, `QueryPlan`, `Answer`, `Citation`.
- Structured JSON logging plus OTel tracing baked in from the start.
- **Exit**: CI green (lint, type, test). `docker compose up` starts the stack.

### Phase 2: Discovery and coverage inventory
- Discover via robots.txt, nested sitemap indexes, documentation TOCs, All Docs/developer catalogs, internal and related links, PDFs, and supplementary official search.
- URL normalisation; keep meaningful locale/version/edition variants as linked variants (only en-US gets fetched in M1/M2).
- Document relationship graph (parent/child, related, supersedes, duplicate-of).
- Capture external references in a **review queue** before they become sources. Unknown resources stay eligible in the unclassified queue.
- Record `lastmod`, ETag and Last-Modified for incremental refresh.
- **Exit**: Data 360 URL inventory with reconciliation report (discovered vs fetched vs excluded, with reasons).

### Phase 3: Fetching, ingestion and raw storage
- Fetcher: httpx (HTTP/2) with conditional GET, per-host token-bucket rate limits, backoff with jitter, and a polite UA. Auto-detect empty or JS-shell pages, then re-fetch with **Playwright**.
- File ingestion: `watchfiles` on `incoming/`. The watcher only enqueues. A debounce plus a stable-size check prevents half-written files from being picked up.
- URL, CLI and API ingestion all go through the **same orchestrator**.
- Queue: SQLite in WAL mode, explicit job state machine (`pending → leased → done | failed → dlq`), lease timeouts with heartbeat, idempotency key = `source_id + content_hash + pipeline_version`, max retries, dead-letter queue with reason.
- Content-addressed raw store (`data/raw/<sha256>`). Raw content is retained only where permitted.
- Parser adapters behind one interface: HTML, PDF, MD, TXT, DOCX, PPTX, XLSX/CSV, JSON, XML, OpenAPI/YAML, code. Unsupported or malformed files get an explicit report, never a silent drop.
- **Exit**: Data 360 fully fetched. Kill-and-restart test proves no lost or duplicated jobs.

### Phase 4: Extraction, normalisation, dedup, security
- Formats become structured Markdown plus JSON sidecar metadata. Preserve headings, lists, tables, code, API schemas, prerequisites, warnings, permissions, limits, errors, links and PDF page refs.
- HTML: strip boilerplate with Trafilatura and BeautifulSoup site-specific selectors (nav/footer/feedback widgets). PDF: PyMuPDF (pymupdf4llm for Markdown with tables).
- Normalised content hash, distinct from the raw hash. A refresh only reprocesses when **meaningful** content changes, not boilerplate.
- **Near-duplicate detection** (MinHash LSH): choose a canonical by authority tier, recency and locale, then link the duplicates. Only canonicals are indexed.
- **Security scan**: PII/secret detection (e.g. Presidio + detect-secrets) on user uploads. Flag instruction-like text ("ignore previous instructions…") in untrusted sources. Quarantined chunks are indexed with a flag and wrapped as data in prompts.
- Raw and normalised versions are stored separately, so parsers can be re-run without re-downloading.
- **Exit**: parser golden-file tests pass for every format. Duplicate rate reported.

### Phase 5: Document intelligence and metadata
Fields (unchanged from v1, plus additions in **bold**): product/family, feature/subfeature, audience/persona, knowledge category, query intents, complexity, content type, technical tags, dependencies, edition/**release (e.g. `Winter '26`, API v65.0)**/locale, **lifecycle state (current/preview/deprecated/retired)**, authority/provenance, **ACL (public | owner/group for user files)**, **pipeline_version**.

Personas, knowledge categories and intents are unchanged from v1.

**Tiered classifier** (applied at doc level, then chunk level):
1. Deterministic: URL path patterns, breadcrumb, page metadata, alias dictionary matches (e.g. "CDP", "Data Cloud" → `data360`).
2. Embedding kNN against labelled exemplars (taxonomy descriptions + seed annotations).
3. LLM (fast model, JSON schema, batched) only when confidence < threshold. Cache by `content_hash + taxonomy_version`.

Document metadata stays broad; chunk metadata stays specific. Low or unknown confidence never blocks indexing. Persona is a ranking signal, never a permission boundary. **ACL is the only permission boundary.**
- **Exit**: classifier accuracy on 300 hand-labelled chunks is ≥ 90% for product and ≥ 80% for category. LLM call rate is < 20% of chunks.

### Phase 6: Structure-aware hierarchical chunking
Baseline unchanged from v1 (child ~600 tok, max 800; parent ~1,600 tok, max 2,200; ~60-token overlap only inside split sections; heading-first splitting; content-type rules for procedures, code, API, tables, troubleshooting, architecture and release notes).

Additions:
- **Tokenizer pinned** to the embedding model's tokenizer, so limits are exact.
- **Contextual header** prepended for embedding only (stored separately from display text): `[Product] › [Doc title] › [H1 › H2 › H3] | Release | Content type`.
- **Lossless invariant**: concatenating child chunks (minus overlap) reproduces the normalised section. This is enforced by property tests (Hypothesis).
- A `group_id` links split code, tables and API blocks so expansion can reassemble them.
- Small-section merge: sibling sections under ~150 tokens merge into one child, which prevents fragment chunks.
- **Exit**: property tests pass. No chunk exceeds the max. Seed-set Recall@10 baseline recorded.

### Phase 7: Embedding, storage and indexing
- **SQLite**: registry, versions, jobs, taxonomy, lineage, relationships, feedback. Migrations via Alembic or a simple versioned SQL.
- **Local files**: raw, normalised, parents, processing reports (content-addressed).
- **Qdrant**: one collection per index generation, with named vectors `dense` (bge-m3, 1024-d, cosine) + `sparse` (bge-m3 lexical weights; BM25 via fastembed as an alternative). Payload indexes on product, release, lifecycle, content_type, acl, doc_id, parent_id.
  - Use scalar quantisation (int8) with rescoring, sized for RAM.
  - Exact-identifier boost: a keyword payload index on extracted identifiers (API names, `__c` fields, endpoint paths, error codes) for exact-match prefetch.
- **Blue/green**: build `chunks_v{N}`, run the smoke eval, then swap the `chunks_live` alias. Keep N-1 for rollback.
- Stable chunk IDs = `hash(doc_canonical_id + section_path + ordinal)`. Upserts are idempotent. Tombstones handle removed docs.
- Embedding throughput: batch, GPU if available. A sizing estimate is recorded in Phase 2: docs × avg chunks → vectors → RAM/disk.
- **Exit**: full Data 360 indexed. Alias swap and rollback demonstrated.

### Phase 8: Query understanding
- **Condense step** for multi-turn: rewrite the follow-up into a standalone question using chat history.
- **Query plan** (fast LLM, JSON-schema constrained, validated by Pydantic, one retry on failure): products, features, categories, intents, persona, depth, release hint, cross-product flag, sub-questions, expected answer sections, confidence.
- **Raw-query retrieval starts in parallel**. The plan refines the run, and the result sets merge. Trivial or identifier-only queries (e.g. an error code) skip the plan.
- Hard filters apply only when explicit (the user names a product or release) **and** confidence is high. Everything else becomes a soft boost. Low confidence broadens retrieval.
- Cache plans by normalised question hash.

### Phase 9: Hybrid retrieval and answer generation
1. Qdrant Query API: `prefetch` dense (k=50) + sparse (k=50) + identifier-match, then `fusion: rrf`, for each sub-question.
2. Dedupe by chunk and collapse by canonical doc.
3. Cross-encoder rerank (bge-reranker-v2-m3) of the top ~40, plus metadata score adjustments: release-current boost, deprecated penalty, authority tier, soft persona/category/intent match. Weights are tuned on the dev set.
4. Parent and sibling expansion (reassemble `group_id` blocks).
5. **Sufficiency check**: if the rerank score of the top-k is below a threshold, broaden once (drop soft filters, use sub-questions), then abstain if still insufficient.
6. **Token-budget packing** (e.g. 8–12 evidence blocks, ≤ 12k tokens). Each block is tagged `[S1]…[Sn]` with title, heading, URL, page and release.
7. Grounded generation with a streamed response. Every factual sentence must cite `[Sn]`. Untrusted/quarantined content is wrapped and labelled as data.
8. **Citation verification**: strip or flag any citation ID that was not in the context. An optional NLI or LLM check verifies support for high-stakes answers.
9. Output: answer, citations (URL, title, heading, page, release), confidence, and "related docs".
- **Answer cache**: exact plus semantic (similarity ≥ 0.97, same persona/depth/ACL scope), invalidated on index alias swap.

### Phase 10: Persona-aware chatbot and API
- Personas: Auto, Business/Marketer, Functional Consultant, Administrator, Developer/Technical Consultant, Solution Architect, Data Engineer/Analyst, Support/Troubleshooter.
- Depth (beginner/standard/deep technical) stays orthogonal to persona. Persona shapes structure and emphasis only, never facts or access.
- Persona-specific answer templates. Examples: Admin = setup path + permissions; Developer = code + API version + limits; Support = symptom → cause → fix.
- FastAPI: `/chat` (SSE stream), `/ingest` (file/URL), `/sources`, `/feedback`, `/health`, `/metrics`. Auth (API key/OIDC) is required, because ACLs depend on identity.
- Thin UI with persona/depth selectors, citation cards with deep links, thumbs + reason feedback, and a "report outdated" action.
- Guardrails: out-of-scope detection (non-Salesforce questions get a polite refusal), per-user rate limits, max input length.

### Phase 11: Evaluation programme (continuous, not a final phase)
- **Seed (150)** in Phase 0 → **1,000** by end of M2. Sources: SME-written questions, synthetic questions generated from chunks (then filtered and human-reviewed), and real failed queries from feedback.
- Balanced and stratified by domain, persona, intent, difficulty and cross-product. **Split 700 dev / 300 held-out**, plus a fixed **50-question smoke set** for CI.
- Annotations: expected products, intents, categories, gold source docs and chunks, answer key points, expected abstention.
- Experiments: chunk size and overlap; hierarchical vs fixed/recursive/semantic; contextual headers on/off; embedding models; dense vs sparse vs hybrid; RRF vs weighted fusion; rerankers; metadata boost weights; candidate k and context budget.
- Metrics: Recall@5/10, Precision@5, MRR, nDCG@10. Answer correctness, faithfulness, completeness, citation precision/recall, abstention accuracy, persona alignment. p50/p95 latency, cost per query, memory, storage.
- LLM-as-judge **calibrated against human labels** (≥ 100 doubly-labelled items; report agreement).
- Rules: tune only on dev. Touch held-out only at milestone sign-off, and log every touch. Version the eval set.

### Phase 12: Coverage, monitoring and continuous refresh
- Scheduled re-discovery: sitemaps, TOCs, catalogs, release indexes, approved external references. Refresh cadence rises around the three Salesforce releases each year.
- Detect new, changed (via ETag/lastmod/normalised hash), redirected and removed pages. Removed pages get a tombstone plus a lifecycle state change, never a silent delete.
- Reprocess only affected docs when content changes. When pipeline or model versions change, do a blue/green rebuild.
- Lifecycle states: current, preview, deprecated, retired, archived.
- Dashboards: discovery reconciliation, processing, index coverage, unclassified backlog, DLQ by reason, exclusion by reason, freshness lag, query SLOs, feedback trends, top abstained questions (these become coverage gaps).
- Alerts: DLQ growth, freshness SLA breach, eval regression, p95 latency breach.

---

## Milestones

| Milestone | Scope | Gate |
|---|---|---|
| **M1: Vertical slice** | Data 360 only, phases 0–10 minimal | Seed-set Recall@10 ≥ 0.80, faithfulness ≥ 0.90, p95 ≤ 6s, kill/restart safe |
| **M2: Breadth** | All 7 domains + user files with ACLs | 1,000-question set complete; dev metrics ≥ M1; DLQ < 1% |
| **M3: Tuned** | Benchmark-driven tuning | Held-out results reported; chosen config beats v1 baseline with significance |
| **M4: Operated** | Refresh, dashboards, alerts, runbooks | Freshness SLA met for 2 consecutive weeks |

## SLOs (initial, revisit after M1)
- Answer p95 latency ≤ 6s to the full answer, and ≤ 1.5s to first token.
- Retrieval Recall@10 ≥ 0.85 (held-out). Faithfulness ≥ 0.92. Citation precision ≥ 0.95.
- Abstention: ≥ 90% correct on unanswerable items.
- Freshness: official-source changes indexed within 24h of detection.
- Ingestion: ≥ 99% of in-scope fetchable sources reach a terminal disposition, with any failure explained.

## Recommended stack
| Layer | Technology |
|---|---|
| Language / tooling | Python 3.12, uv, ruff, mypy, pytest + Hypothesis, pre-commit |
| Fetching | httpx (HTTP/2, conditional GET) + Playwright fallback |
| File watcher | watchfiles (debounced) |
| Queue / metadata | SQLite (WAL) with lease-based job table + DLQ; Postgres as the scale-out path |
| HTML / PDF / Office | Trafilatura + BeautifulSoup · PyMuPDF/pymupdf4llm · python-docx, python-pptx, openpyxl |
| Dedup | datasketch (MinHash LSH) |
| Security scan | Presidio, detect-secrets, injection heuristics |
| Schemas / config | Pydantic v2, Pydantic Settings |
| Vector + lexical | Qdrant (named dense + sparse vectors, server-side RRF, aliases, int8 quantisation) |
| Embeddings / reranker | bge-m3 / bge-reranker-v2-m3 (local, versioned, swappable) |
| LLM | Provider-agnostic client. Defaults: Claude Sonnet 5.5 (answers), Claude Haiku 5.5 (plans/classification); Ollama option |
| API / UI | FastAPI + SSE; thin web UI |
| Observability | OpenTelemetry, Langfuse or Arize Phoenix (self-hosted), Prometheus/Grafana |
| Eval | Custom harness + Ragas-style metrics, calibrated LLM judge |
| Deploy | Docker Compose (single node), path to Kubernetes |

## Project structure
```text
salesforce-rag/
  pyproject.toml  docker-compose.yml  config/{base,dev,prod}.yaml
  taxonomy/                # controlled vocabularies, alias tables (versioned)
  src/sfrag/
    core/                  # models, settings, ids, logging, tracing
    sources/               # registry, discovery, fetch, watcher
    pipeline/              # queue, orchestrator, extract, normalize, dedup, security
    enrich/                # classification, chunking, contextual headers
    index/                 # embeddings, qdrant, aliases
    query/                 # condense, plan, retrieve, rerank, expand, pack, generate, verify
    api/                   # FastAPI app, auth, UI
    eval/                  # datasets, runners, metrics, judge
    ops/                   # coverage audit, refresh scheduler, reports
  tests/{unit,golden,property,integration,eval_smoke}/
  data/  (gitignored: raw/, normalized/, parents/, sqlite/, reports/)
```

## Definition of done (v1 items kept, with measurable additions)
- Every configured or discovered in-scope source has a registry entry and a disposition with a reason.
- New and changed files and approved URLs automatically enter a recoverable, idempotent pipeline, proven by a kill/restart test.
- Unknown products, features and categories are retained and visible in the unclassified backlog.
- Structured content survives extraction and chunking, proven by golden-file and lossless property tests.
- Every chunk carries provenance, release, lifecycle, ACL, `pipeline_version` and `embedding_model_version`.
- Hybrid retrieval, rerank, parent expansion, sufficiency check and abstention all work.
- Structured query plans are produced, and answers carry verified citations.
- All 7 personas plus Auto are supported, and depth stays orthogonal to persona.
- SLOs are met and reported on dashboards. Alerts are wired.
- CI blocks regressions via the smoke eval. Re-indexing is blue/green with rollback.
- The 1,000-question benchmark (700/300) provides evidence for the final chunking and retrieval config.

**Coverage caveat (unchanged):** completeness is measurable only against the known, accessible and permitted source universe. It cannot be guaranteed for restricted, inaccessible, unlinked or undiscoverable material.

---

## Verification of this plan revision
- Every v1 requirement maps to a v2 phase. No requirement was dropped: objective, core decisions, 7 domains, taxonomy, chunking rules, hybrid + RRF + rerank + expansion, personas, 1,000-question eval, coverage caveat.
- After approval, write this to `Salesforce_Universal_RAG_Implementation_Plan_v2.md` next to the original. The original stays untouched.
- The implementation verifies itself through the milestone gates and SLOs above: CI smoke eval, kill/restart test, alias swap/rollback demo, and a held-out benchmark report.
