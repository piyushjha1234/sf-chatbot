# Salesforce Universal Documentation & RAG Platform
## Implementation Plan and Engineering Blueprint

## Objective
Build a local-first, coverage-first Salesforce knowledge platform that automatically ingests permitted documentation and user-provided files, classifies documents and chunks, applies structure-aware hierarchical chunking, indexes content for hybrid retrieval, and powers a persona-aware chatbot for functional, administration, developer, data, support and architecture questions.

## Core decisions
- Feature lists are classification aids, never crawl filters.
- New files and approved URLs enter one automatic, recoverable ingestion pipeline.
- Keep raw sources, normalized documents, parent sections, child chunks and metadata separately traceable.
- Store document-level and chunk-level metadata independently.
- Use structure-aware hierarchical parent-child chunking as the baseline.
- Use LLM query understanding to infer products, features, knowledge categories, query intents, persona, complexity and answer requirements.
- Use hybrid dense + lexical retrieval, RRF, reranking and parent-context expansion.
- Optimize chunking and retrieval later using approximately 1,000 representative questions and a held-out test set.

## Initial product domains
1. Marketing Cloud Engagement (`mce`)
2. Marketing Cloud Personalization (`mcp`)
3. Salesforce Personalization (`sf_personalization`)
4. Marketing Cloud Next (`mc_next`)
5. Data 360 (`data360`)
6. Marketing Cloud Intelligence (`mci`)
7. Salesforce APIs and Architecture (`sf_architecture`)

These are starting domains, not an exhaustive product boundary. Relevant additional products, platform documentation, APIs, SDKs and supporting technical resources must be discoverable through approved sources.

## Target architecture

```text
Sources -> Discovery/Registry -> Ingestion Queue -> Extraction/Normalization
        -> Document Classification -> Hierarchical Chunking
        -> Chunk Classification -> Embeddings + Lexical Index
        -> Query Understanding -> Hybrid Retrieval -> Reranking
        -> Parent Context Expansion -> Grounded Answer + Citations
        -> Coverage Audit + Incremental Refresh
```

Ingestion and inference are independent. Ingestion works without a chatbot session; the chatbot does not need to know how a source was collected.

## Phases

### Phase 0 — Scope, governance and source policy
- Configure official Salesforce Help, Developer, Architect, API, release notes, Trailhead and approved technical repositories.
- Define source authority, scope, source permissions, retention and exclusion reasons.
- Keep taxonomy separate from ingestion rules.
- Deliver source registry, policy, provenance schema and coverage metrics.

### Phase 1 — Universal discovery and coverage inventory
- Discover through robots/sitemaps, nested sitemap indexes, documentation TOCs, All Docs catalogs, developer catalogs, internal links, related links, PDFs and supplementary official search.
- Capture external references for review before adding sources.
- Normalize URLs, preserve meaningful locale/version/edition variants and build a document relationship graph.
- Maintain an unclassified queue; unknown resources remain eligible.
- Deliver URL registry, discovery graph and reconciliation report.

### Phase 2 — Automatic source ingestion and raw storage
- Watch `incoming/` using watchdog/watchfiles; watcher enqueues jobs only.
- Add URL ingestion and manual CLI/API ingestion through the same orchestrator.
- Use a durable SQLite-backed queue, content hashes, version tracking, retries and recovery.
- Support HTML, PDF, Markdown, TXT, DOCX, PPTX, JSON, XML, CSV, XLSX, OpenAPI/YAML and code through parser adapters.
- Retain raw source where permitted; explicitly report unsupported or malformed files.

### Phase 3 — Extraction, normalization and canonical document model
- Convert formats into structured Markdown plus JSON metadata.
- Preserve headings, lists, tables, code, API schemas, prerequisites, warnings, permissions, limitations, errors, links and PDF page references.
- Store raw and normalized versions separately so parsers can be improved without re-downloading.

### Phase 3.5 — Document intelligence and metadata
Use stable fields for:
- Product/product family
- Feature/subfeature
- Audience/persona
- Knowledge category
- Query intents supported
- Complexity
- Content type
- Technical tags
- Dependencies/relationships
- Edition/release/locale
- Source authority/provenance

Personas: Business/Marketer, Functional Consultant, Administrator, Developer/Technical Consultant, Solution Architect, Data Engineer/Analyst and Support/Troubleshooter.

Knowledge categories: Concept, Configuration, Implementation, API, Architecture, Security, Governance, Limits, Troubleshooting, Migration and Release/Change.

Query intents: Explain, Configure, Implement, Develop, Troubleshoot, Audit, Design, Compare, Optimize, Migrate, Evaluate and Administer.

Keep document-level metadata broad and chunk-level metadata specific. Use controlled vocabularies and alias normalization; low-confidence or unknown values do not block indexing. Persona is a ranking signal, not a permission boundary.

### Phase 4 — Structure-aware hierarchical chunking
**Recommended baseline:** structure-aware parent-child chunking with content-aware handling for code, APIs, tables, procedures, troubleshooting and release notes.

Initial settings:
- Child target: 500–700 tokens (aim near 600); usual maximum around 800.
- Parent target: 1,200–2,000 tokens (aim near 1,600); usual maximum around 2,200.
- Overlap: about 60 tokens, only when a long logical section must be split; no overlap across unrelated headings.
- Split at headings and structural boundaries first; semantic/token fallback only inside oversized sections.
- Retrieve child chunks, then expand to the relevant parent and selected siblings.

Rules:
- Keep prerequisites, ordered steps and warnings connected for procedures.
- Keep complete logical code examples/functions intact; split oversized code at syntax boundaries with shared group metadata.
- Keep API method, endpoint, auth, parameters, request/response, errors and examples connected where possible.
- Repeat table title/headers in every row group.
- Keep troubleshooting symptom, cause, diagnostics and resolution together.
- Preserve architecture relationships and diagram references.
- Keep release notes grouped by release and feature.

These are baseline settings, not claims of universal optimality. The benchmark phase will tune them.

### Phase 5 — Embedding, storage and indexing
- SQLite: source registry, versions, jobs, taxonomy, lineage and relationships.
- Local files: raw content, normalized documents, parents, processing reports.
- Qdrant: dense vector index and structured metadata payloads; keep lexical retrieval behind an adapter.
- BM25-compatible lexical search: exact Salesforce identifiers, field names, endpoint paths and error codes.
- Local embedding model and local reranker, both configurable and versioned.
- Index child chunks primarily; retain parent context in document storage.
- Use stable IDs and idempotent updates; keep a unified document registry across product collections.

### Phase 6 — LLM query understanding
The LLM creates a schema-validated query plan with:
- products and features
- knowledge categories
- one or more query intents
- selected/inferred persona
- complexity and desired depth
- cross-product flag
- expected answer sections
- confidence

Preserve the raw question. For multi-part questions, decompose into sub-questions. Low confidence broadens retrieval instead of applying restrictive filters.

### Phase 7 — Hybrid retrieval and answer generation
1. Dense semantic retrieval and BM25/lexical retrieval in parallel.
2. RRF rank fusion as the initial baseline.
3. Candidate deduplication.
4. Metadata-aware reranking.
5. Parent and relevant sibling context expansion.
6. Evidence/context sufficiency check.
7. Grounded response generation with source URL, title, heading and page reference where available.

Start with 30–50 retrieval candidates, rerank and provide roughly 8–12 high-quality candidates to context construction; tune later. Explicit product/version filters may be appropriate. Persona/category/intent should generally be soft ranking signals.

### Phase 8 — Persona-aware chatbot
- Auto
- Business/Marketer
- Functional Consultant
- Administrator
- Developer/Technical Consultant
- Solution Architect
- Data Engineer/Analyst
- Support/Troubleshooter

Keep persona separate from response depth: beginner, standard or deep technical. A developer can request a beginner explanation. Persona does not change source facts or authorize access.

### Phase 9 — 1,000-question evaluation (Milestone 2)
- Build about 1,000 balanced questions across products, roles, intents, difficulty and cross-product topics.
- Annotate expected products, intents, categories, sources, evidence chunks and answer points.
- Example split: 700 development / 300 held-out test, stratified by domain and intent.
- Compare chunk sizes, overlap, hierarchical vs fixed/recursive/semantic chunking, embedding models, hybrid vs single retrieval, rerankers and metadata ranking.
- Metrics: Recall@5/10, Precision@5, MRR, nDCG@10, answer correctness, faithfulness, completeness, citation precision/coverage, persona alignment, latency, memory and storage.
- Do not repeatedly tune against the held-out test set.

### Phase 10 — Coverage, monitoring and continuous refresh
- Revisit sitemaps, TOCs, catalogs, release indexes and approved external references.
- Detect new, changed, redirected and removed URLs.
- Reprocess only affected sources/chunks when content or pipeline versions change.
- Maintain current, historical, preview, deprecated and archived states.
- Report discovery reconciliation, processing coverage, index coverage, unclassified resources and all failure/exclusion states separately.

## Recommended baseline stack
| Layer | Technology |
|---|---|
| Language | Python |
| File watcher | watchdog/watchfiles |
| Job state | SQLite-backed queue initially |
| HTML | BeautifulSoup + Trafilatura |
| PDF | PyMuPDF |
| Office formats | python-docx / python-pptx and suitable spreadsheet parsers |
| Schema | Pydantic + JSON |
| Metadata | SQLite |
| Vector database | Qdrant |
| Lexical search | BM25-compatible index |
| Embeddings/reranker | Local, configurable models |
| Orchestration | Custom Python modules; framework utilities optional |

## Suggested project structure
```text
salesforce-rag/
  config/ taxonomy/ discovery/ ingestion/ extraction/
  normalization/ classification/ chunking/ indexing/
  query_understanding/ retrieval/ chatbot/ evaluation/
  coverage_audit/ storage/ database/ logs/ tests/ main.py
```

## Definition of done
- All configured and discovered in-scope sources have a traceable registry entry and disposition.
- New and changed files and approved URLs automatically enter a recoverable pipeline.
- Unknown products/features/categories are not discarded.
- Structured technical content survives extraction and chunking.
- Every chunk has source provenance and versioned metadata.
- Hybrid retrieval and parent-context expansion work.
- The LLM creates structured query plans and answers with citations.
- Functional, admin, developer, data, support and architect perspectives are supported.
- Coverage metrics expose unresolved gaps.
- The later 1,000-question benchmark provides evidence for tuning chunking and retrieval.

**Coverage caveat:** This design can measure completeness against the known, accessible and permitted source universe. It cannot guarantee collection of restricted, inaccessible, unlinked or undiscoverable Salesforce material.
