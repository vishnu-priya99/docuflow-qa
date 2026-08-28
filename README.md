# Document Q&A System

Upload a mix of files into a chat session and ask questions about them.
Answers are grounded only in what you uploaded, with real citations
(filename, page, section) attached to every response. A LangGraph
pipeline decides, per question, whether to search documents semantically,
run generated SQL against spreadsheet data, or both.

- **Backend**: Python + FastAPI
- **Orchestration**: LangGraph
- **Structured data**: PostgreSQL (a real, hard requirement — not SQLite)
- **Semantic search**: Qdrant (a real server, no embedded/in-memory mode)
- **LLM + embeddings**: Ollama (fully local, no API key) or a mock provider for quick testing
- **Storage**: local disk

## 1. Get PostgreSQL running

With Docker:

```bash
docker compose up -d postgres qdrant
```

**No Docker** (e.g. Windows without Docker Desktop): `scripts/setup_postgres_windows.ps1`
installs a portable, no-admin-rights PostgreSQL 17 and creates the `docqa`
role/database plus a read-only `docqa_ro` role, all in one step. Safe to
re-run — it skips whatever's already set up.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_postgres_windows.ps1
```

This installs to `.pgsql-portable\` by default (git-ignored) — pass
`-InstallDir` for a different location, or `-Port` for a port other than
5432. It prints the exact `DATABASE_URL` lines to paste into `.env` when
done. Stop it later with the same command plus `-Stop`; it doesn't run as
a Windows service, so after a reboot you just run the script again (no
`-Stop`) to start it back up.

## 2. Configure and run the app

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux
```

The defaults in `.env.example` already match what the setup steps above
provision, so you likely don't need to change `DATABASE_URL`. A few
things worth knowing about the rest of `.env`:

- **`QDRANT_URL`** needs a real Qdrant server — grab a portable binary
  from the [Qdrant releases page](https://github.com/qdrant/qdrant/releases),
  run it, then point `QDRANT_URL` at `http://localhost:6333`. Its own
  dashboard is at `http://localhost:6333/dashboard`.
- **`LLM_PROVIDER=ollama`** gives you a fully local, offline setup — no
  API key, just `ollama pull qwen2.5:7b` (or any model you like) first.
  `LLM_PROVIDER=mock` needs zero setup and is handy for a quick look
  around, but its answers are canned, not real.
- **`EMBEDDING_PROVIDER=local`** is a dependency-free embedder, good
  enough for local testing. Switch to `EMBEDDING_PROVIDER=ollama`
  (`ollama pull nomic-embed-text`, and set `EMBEDDING_DIM=768`) for
  actually good semantic search.

Then run it:

```bash
uvicorn app.main:app --reload --reload-dir app
```

(`--reload-dir app` matters here — without it, `--reload` watches the
whole project, including `.venv`, and a dependency like `networkx`
touching its own files on disk ends up restarting your server every few
seconds.)

Open `http://localhost:8000`, type any name as your `user_id` to log in
(no password — that's intentional, not an oversight), start a chat,
upload files, and ask questions.

## Testing

```bash
pytest
```

Needs the same Postgres and Qdrant as the app itself (reads from `.env`,
same fallback defaults). Uses mocked LLM/embedding providers, so no API
key is required to run the suite. Qdrant test data lives in its own
collection, separate from anything real. A teardown step wipes everything
the suite created — by its test users' `user_`-prefix — so repeated runs
don't pile up debris in your database.

103 tests, covering: login and session lifecycle, session isolation and
cascading delete, ingestion + citation metadata for every supported file
type, Excel/CSV schema inference, SQL generation (COUNT/SUM/AVG/GROUP
BY/date filters), SQL injection and read-only enforcement, Qdrant search
isolation, and the SEMANTIC/STRUCTURED/HYBRID routing logic.

## Architecture

```
app/
  api/        FastAPI routes + dependency wiring
  core/       config, logging
  db/         SQLAlchemy engine/session, shared column types
  models/     ORM models (users, sessions, messages, files, excel_*)
  schemas/    Pydantic request/response models
  services/
    ingestion/   PDF/DOCX/PPTX/TXT parsers + chunker
    documents/   ingestion + upload orchestration
    excel/       schema inference, per-sheet table creation, ingestion
    retrieval/   Qdrant semantic search + sheet discovery for HYBRID
    sql/         schema description, SQL generation, validation, execution
    embeddings/  provider abstraction (local hashing / Ollama)
    llm/         provider abstraction (Ollama / mock)
    storage/     original-file storage abstraction
  graph/      LangGraph state, node implementations, workflow wiring
  vector/     Qdrant client + payload schemas
  prompts/    system/user prompt templates
frontend/     vanilla HTML/CSS/JS SPA
tests/        pytest suite + sample-file factories
```

### How a question gets answered

```
START -> load_session -> router
                            |-- SEMANTIC   -> semantic search over document chunks
                            |-- STRUCTURED -> pick relevant sheet(s) -> generate SQL
                            |                                          -> validate -> run
                            |-- HYBRID     -> find the relevant sheet via Qdrant first,
                                               then do both of the above -> combine
```

The router's only job is to classify the question — it never answers
anything itself. STRUCTURED and HYBRID share the same schema-selection
and SQL pipeline; the difference is that HYBRID narrows down *which*
sheet that pipeline should look at first, using a semantic search over
sheet summaries.

This is a deterministic pipeline, not autonomous agents — the control
flow above is fixed in code, not decided by an LLM at runtime. It's a
sensible-orchestration/modularity design, and it's honest about being
that rather than dressing plain function calls up as "agents." See
**Design decisions** below for the reasoning and the trade-off.

### Data isolation

Every row belonging to uploaded data carries `user_id` and `session_id`
(and `file_id` where relevant), and so does every Qdrant point. Every
search is filtered by them, and every API route checks a session
actually belongs to the user asking before touching it. Deleting a
session removes its structured-data tables, its Qdrant points, its
original files, and (via a database foreign key) its messages and file
records — all in one call, nothing left behind.

### SQL safety

Generated SQL is never trusted or run directly. It goes through
`services/sql/sql_validator.py` first, which parses it with `sqlglot` and
rejects anything that isn't exactly one `SELECT` statement, rejects any
write or schema-changing SQL anywhere in the parse tree, restricts every
table it touches to ones the current session actually owns, and caps the
row limit. On top of that, the query itself runs under a dedicated
read-only Postgres role (`docqa_ro`) with a timeout — so even a bug in
the validator can't turn into real damage, because the database
connection itself is physically incapable of writing anything.

### Citations

Every parser generates page numbers, DOCX heading-derived sections, slide
numbers/titles, or line ranges as it goes — deterministically, from the
file's real structure. None of this is invented or produced by an LLM;
a citation only ever gets attached when the parser can establish it for
certain.

## Configuration

See `.env.example` for the full list. The ones you'll actually touch:
`LLM_PROVIDER` / `LLM_MODEL`, `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` /
`EMBEDDING_DIM`, `OLLAMA_BASE_URL`, `CHUNK_SIZE_CHARS` /
`CHUNK_OVERLAP_CHARS`, `SQL_QUERY_TIMEOUT_SECONDS` / `SQL_MAX_ROWS`,
`QDRANT_URL`.

**If you swap in a different/smaller Ollama model**: `sql_prompt.py` and
`answer_prompt.py` both carry a few worked examples, added because small
models tend to say "I don't know" even when the answer is right in front
of them, on instructions alone. If you see that with a different model,
this is the likely reason.

## Design decisions & trade-offs

- **A fixed pipeline instead of autonomous agents.** Every step here is
  wired together in code ahead of time — the LLM fills in specific
  blanks (classify the question, write SQL, write the answer) but never
  decides what happens next. This is simpler, faster, and far more
  predictable than letting a model choose its own control flow, and for
  a task this well-defined, predictability matters more than flexibility.
  The honest cost: nothing here catches its own mistakes or decides to
  retry — see limitations below.
- **Reranking is always on.** A vector search alone tends to surface
  plausible-but-wrong passages alongside the real answer; a second
  pass specifically for relevance narrows that down before anything
  reaches the LLM.
- **SQL safety is layered, not single-point.** The validator and the
  read-only database role are independent of each other on purpose — if
  one has a bug, the other still holds.
- **Structured data gets real SQL, not a text dump.** Excel/CSV sheets
  become actual Postgres tables with inferred column types, so questions
  like "average X where Y" get computed correctly instead of an LLM
  eyeballing a table and guessing.

## Known limitations & future work

- **No OCR or image support.** Scanned/image-only PDFs and standalone
  images aren't readable — there's no vision/OCR step in the pipeline.
  This was a deliberate scope cut, not an oversight (see below).
- **No Markdown or code file support**, despite both being common
  document types — only PDF/DOCX/PPTX/TXT (documents) and XLSX/CSV
  (structured data) are handled today.
- **No conversation memory across turns.** Each question is answered
  independently — the app doesn't remember earlier questions in the
  same chat. (This was built and tested at one point, then deliberately
  removed for now to keep the system simple and predictable while it
  was still settling.)
- **No self-verification step.** The app doesn't check its own answer
  against the evidence before showing it to you — citations are attached
  because real source metadata was used to build the answer, but nothing
  double-checks the answer actually reflects that evidence correctly.
  Adding this — with a real retry loop when something looks
  ungrounded — is the most valuable next step, and the one place a
  genuinely autonomous decision (not just a fixed pipeline step) would
  add real value here.
- **Small local models occasionally misread a specific row** in a
  multi-row table lookup, even when the data given to them is completely
  unambiguous. Retrying doesn't reliably help — this is closer to a
  model-capability ceiling than a fixable bug, at least with a small
  model running fully local.
- **Not literally multi-agent.** The architecture is a clearly-modular,
  deterministic pipeline (see Design decisions), which is a defensible
  choice, but it's not the same thing as components making independent
  runtime decisions and collaborating — worth being upfront about rather
  than relabeling the same code with different names.
