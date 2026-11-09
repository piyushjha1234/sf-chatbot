# Handover: continue sfrag on the new device

The order is:
1. Migrate the project.
2. Set up the machine.
3. Run the prompts below one at a time in Claude Code, opened in the project folder.

Each prompt is self-contained. Finish one, review its result, then start the next.

---

## Part A: Migrate (on the old machine)

Option 1 is git. It is recommended, because it gives you history and rollback.

```bash
cd ~/Desktop/assets/sf-chatbot
git add -A && git commit -m "M1 core slice"
git remote add origin <your-private-repo-url> && git push -u origin main
# new device:
git clone <your-private-repo-url> sf-chatbot && cd sf-chatbot
```

Option 2 is a plain copy. Leave out the venv and generated data.

```bash
cd ~/Desktop/assets
tar --exclude='sf-chatbot/.venv' --exclude='sf-chatbot/data' --exclude='*/__pycache__' \
    --exclude='*/.mypy_cache' --exclude='*/.ruff_cache' --exclude='*/.pytest_cache' --exclude='*/.hypothesis' \
    -czf sf-chatbot.tgz sf-chatbot
# copy sf-chatbot.tgz to the new device, then:  tar -xzf sf-chatbot.tgz && cd sf-chatbot
```

Nothing in `data/` needs to move. The new device builds its own index.

## Part B: Set up the new device

Requirements:
- Python 3.12 (uv installs it).
- 16 GB or more of RAM; 32 GB is comfortable.
- About 10 GB of free disk for models and the index.
- A GPU (CUDA) or Apple-silicon MPS is strongly recommended for ingestion speed.
- Optionally, Docker for a Qdrant server.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # if uv is missing
uv sync --extra models                               # installs torch + FlagEmbedding (bge-m3, bge-reranker-v2-m3)
export ANTHROPIC_API_KEY=sk-ant-...                  # or: ant auth login
export HF_TOKEN=hf_...                               # optional: faster, rate-limit-free model downloads
uv run pytest                                        # must be green before anything else
uv run sfrag init
```

The first real embedding call downloads about 2.3 GB for bge-m3 and about 2.3 GB for the reranker.

---

## Part C: Prompts for Claude Code (run in order)

### Prompt 1: Verify the environment and the real models

```
Read CLAUDE.md, README.md and HANDOVER.md first.

This repo was built and tested on another machine with hermetic test backends only. On this machine,
verify the production path end to end without crawling anything yet:
1. Run the full check suite (ruff, ruff format --check, mypy, pytest) and report results.
2. Confirm uv sync --extra models installed FlagEmbedding and torch, and report the device that
   index/embedder.py:_auto_device picks (cuda / mps / cpu).
3. Write a one-off script in a scratch location that loads BGEM3Embedder and BGEReranker through
   sfrag.container.Container with the dev config. Embed 3 sample Salesforce sentences, then check the
   dense dim is 1024 and the sparse vectors are non-empty. Rerank them against a query and check the
   scores fall in 0..1. Report load time, memory use and throughput (texts/sec).
4. Make one real call each to LLMClient.parse (fast model) and LLMClient.stream_answer (answer
   model). If any SDK parameter is rejected (output_config, fallbacks, betas), fix
   src/sfrag/llm/client.py against the current Anthropic Python SDK docs. Do not guess.
5. Ingest the single test file from data/incoming/public/ (create a short Markdown file about Data
   Cloud data streams), run `sfrag worker --drain`, then `sfrag ask` a question about it. Show the
   streamed answer and its citations.
Fix any defects you find, keeping all tests green. Summarise what you changed and why.
```

### Prompt 2: Verify sources and complete the policy review

```
Read CLAUDE.md. Then review config/sources.yaml against the live sites, read-only (no crawling yet):
1. For each start URL and sitemap, fetch it once with the project's Fetcher (same User-Agent). Report
   the HTTP status, final URL after redirects, content type, and whether the HTML has server-rendered
   content or is a JavaScript shell (use the HtmlParser / NeedsRender logic).
2. Fetch robots.txt for each host. Report the Sitemap: lines and whether our paths are allowed.
3. Find the real Data 360 (Data Cloud) documentation entry points and sitemaps on
   developer.salesforce.com and help.salesforce.com. Propose corrected start_urls, sitemaps and
   include/exclude regexes. Keep scope to en-US Data 360 for M1.
4. Summarise each host's terms-of-use position on automated access, with a link. Fill the `review`
   blocks with what you found, but leave `permitted` and the tos_checked flags for me to decide.
   List the decisions I need to make.
Do not change crawler code in this step unless a source is unreachable because of a bug.
```

### Prompt 3: First crawl and coverage triage

```
Read CLAUDE.md. I have approved the sources in config/sources.yaml (permitted: true where I want it).
Run the first Data 360 crawl and triage the results:
1. `uv run sfrag discover`, then `uv run sfrag worker --concurrency 4 --drain`. Report the duration,
   docs and chunks per minute, and any errors.
2. `uv run sfrag coverage`: present reconciliation per source, URL statuses, top exclusion/failure
   reasons, needs_render count, duplicate count, unclassified and low-confidence document counts, and
   documents by product.
3. Sample 15 indexed documents at random. For each, compare the normalised Markdown in
   data/normalized/ with the live page and judge whether headings, lists, tables, code, notes or
   warnings survived and whether boilerplate leaked in. Then inspect 30 random chunks (sizes, heading
   paths, contextual headers, classification).
4. Fix extraction or classification defects you find: site-specific selectors in
   pipeline/extract/html.py, aliases or URL patterns in taxonomy/taxonomy.yaml. Add a golden-file
   test for each fix, using a trimmed real page saved under tests/fixtures/. Bump pipeline_version,
   reprocess, and run `sfrag index rebuild`.
5. Report before/after numbers.
```

### Prompt 4: Answer quality review (needs the crawl from Prompt 3)

```
Read CLAUDE.md. Run a qualitative answer review on the live Data 360 index:
1. Write 25 realistic questions spread across all personas (business_marketer, functional_consultant,
   administrator, developer, solution_architect, data_engineer, support) and intents (explain,
   configure, troubleshoot, design, compare, develop). Include 3 multi-part, 3 identifier/error-code
   and 3 unanswerable questions.
2. Run each through `sfrag ask` with the matching persona. Record: the plan (products, confidence,
   sub-questions), top 5 retrieved docs with scores, the answer, citations, invalid citations,
   uncited ratio, abstention, and per-stage timings_ms.
3. Classify each failure as retrieval miss, ranking, context packing, generation or abstention, and
   name the root cause.
4. Propose and apply the smallest fixes (boost weights, thresholds, prompt wording, planner rules),
   with tests where it makes sense. Re-run the 25 questions and give a before/after table.
Save the 25 questions as eval/datasets/review-25.jsonl in the eval format (gold_urls where you are
confident, split "dev") so they seed the real eval set.
```

### Prompt 5: Eval baseline (run once SMEs have provided questions)

```
Read CLAUDE.md. SMEs have written questions in eval/datasets/seed.jsonl (format in
src/sfrag/eval/runner.py).
1. Validate the file: schema, duplicate ids, split counts (dev / test / smoke ≈ 50), and balance across
   persona, intent, difficulty and cross_product. Flag gold URLs that are not in the index and say why
   (not crawled, excluded, duplicate, needs_render).
2. Run `sfrag eval --split dev` (retrieval) and `--split smoke --mode answer`. Present the metrics
   against the M1 gates in the v2 plan (Recall@10 ≥ 0.80, faithfulness ≥ 0.90, p95 ≤ 6 s).
3. Save the smoke summary as data/reports/baseline-smoke.json, and show how to call the regression
   gate.
4. Check the LLM judge against my labels: pick 30 answers, show them to me for scoring, then report
   agreement before we rely on the judge.
Do not touch the test split.
```

### Prompt 6: Tune to meet the M1 gates (repeat as needed)

```
Read CLAUDE.md. Improve retrieval and answer metrics on the eval dev split only, toward the M1
gates. For each experiment, change one thing, run `sfrag eval --split dev` (and the smoke split in
answer mode for generation changes), then keep or revert. Candidates in priority order:
- the sufficiency/abstain thresholds and metadata boost weights in RetrievalSettings
- dense_k / sparse_k / rerank_k / final_k / context_token_budget
- child/parent chunk sizes and overlap (requires pipeline_version bump + rebuild)
- contextual header format
- answer_effort (low / medium / high) against quality and latency
Keep an experiment log table in data/reports/experiments.md (change, metrics, decision). Stop
when the gates are met or the gains flatten, then summarise the chosen config.
```

### Prompt 7: Playwright fetcher for JS-rendered pages

```
Read CLAUDE.md. help.salesforce.com pages render client-side and currently land in url status
`needs_render`. Add a headless-browser fetch path:
1. Add Playwright as an optional dependency (extra "render"), with install instructions in the README.
2. Add a renderer in sources/ that loads a URL, waits for the main article content (find robust
   selectors for help.salesforce.com articles), and returns the rendered HTML. Reuse the per-host
   rate limiter and robots checks, set a timeout, and reuse one browser across pages.
3. In the orchestrator, when HtmlParser raises NeedsRender and rendering is enabled for that source
   (new source config flag `render: true`), enqueue a `render_url` job that renders and then calls
   process_raw. Rendered HTML goes into the raw store with meta {rendered: true}.
4. Make link discovery work on rendered pages, because help articles link to each other.
5. Write hermetic tests with the browser mocked, plus one opt-in integration test marked
   `@pytest.mark.network`, skipped by default.
6. Re-crawl sf_help_data360 and report the needs_render count before and after, plus extraction quality
   on 10 sample articles.
```

### Prompt 8: Observability

```
Read CLAUDE.md. Add tracing and metrics:
1. OpenTelemetry spans for ingestion (discover, fetch, parse, classify, chunk, embed, upsert per
   job) and for queries (condense, plan, recall, rerank, expand, pack, generate, verify). Add
   attributes for doc_id, job kind, top score, token usage and stop_reason. Make the exporter
   configurable, off by default and off in tests.
2. Langfuse (self-hosted, added to docker-compose.yml) or Arize Phoenix tracing of each LLM call:
   prompt and model ids, usage including cache_read_input_tokens, and latency.
3. A Prometheus /metrics endpoint: request latency histograms, first-token latency, abstention
   rate, invalid-citation rate, queue depth by state, DLQ size, freshness lag.
4. Document how to read the traces, and show me one traced question end to end.
```

### Prompt 9: Security hardening for user files

```
Read CLAUDE.md. Harden the user-file path:
1. Run PII and secret scanning (Presidio + detect-secrets) at ingestion for source kind=file.
   Configurable policy: flag, redact or block. Record findings on the document and surface them in
   the coverage report.
2. Review the prompt-injection quarantine heuristics in pipeline/security.py, and add tests with
   realistic injected content.
3. Audit ACL enforcement end to end: every Qdrant query path must carry the ACL filter, and
   /ingest/file must write only to the caller's folder. Also cover the answer cache (keyed by
   principals) and parent expansion (it reads SQLite directly, so check it cannot leak sections of
   documents the caller cannot see). Add tests for each.
4. Run a security review of the API surface and report findings.
```

### Prompt 10: Web UI

```
Read CLAUDE.md. Build a thin web UI served by the FastAPI app at /:
- chat with SSE streaming from /chat (handle the plan, sources, delta, done and error events)
- persona and depth selectors, multi-turn history
- citation cards with title, section, release and a link, plus a confidence indicator and a clear
  abstention state
- thumbs up/down with reason, and "report outdated", posting to /feedback
- file upload to /ingest/file
- API key entry (stored in localStorage)
No build step: plain HTML/CSS/JS or a single CDN-pinned library. Accessible, works on mobile.
Add a smoke test that the page is served.
```

### Prompt 11: M2 breadth (after the M1 gates pass)

```
Read CLAUDE.md and the M2 milestone in the v2 plan. Expand beyond Data 360 one domain at a time,
in this order: mce, sf_architecture, mcp, sf_personalization, mc_next, mci. For each domain:
1. Research and add the sources (start URLs, sitemaps, scope regexes, review notes), then wait for my
   approval of `permitted` before crawling.
2. Crawl, run coverage triage, and fix extraction for any new site layouts (golden tests).
3. Refine taxonomy aliases and URL patterns using the low-confidence/unclassified reports.
4. Check cross-product questions still route correctly, and that dev-set metrics for earlier domains
   did not regress.
Also switch to a Qdrant server (docker compose, SFRAG_ENV=prod) with separate worker/API processes,
and implement the embedding-kNN classifier tier using labelled chunks from the eval work.
```

---

## Things only you can decide

- `permitted: true` for each host, after reading the terms-of-use summary from Prompt 2.
- Who the SMEs are, and the 150-question seed set (Prompt 5).
- API keys and user/group mapping for production (`SFRAG_API__API_KEYS`, `api.groups`).
- Monthly Claude API budget. Answer mode in eval spends real money; the prompts above ask before running large
  batches.
