# Document Q&A System

A multi-file document Q&A system: upload PDF/DOCX/PPTX/TXT/XLSX/CSV files
into a chat session and ask questions grounded only in what was uploaded.
LangGraph routes each question to a semantic (Qdrant), structured
(PostgreSQL SQL generation), or hybrid path.

- **Backend**: Python + FastAPI
- **Orchestration**: LangGraph
- **Structured data / metadata**: PostgreSQL (required)
- **Semantic search**: Qdrant (real server required - a portable no-Docker binary works, see below)
- **Storage**: local disk (pluggable)
- **LLM + embeddings**: configurable providers (Ollama/mock, local/Ollama)

## 1. Get PostgreSQL running

With Docker:

```bash
docker compose up -d postgres qdrant
```

**No Docker (e.g. Windows without Docker Desktop)**:
`scripts/setup_postgres_windows.ps1` sets up a **portable, no-installer,
no-admin-rights-required** PostgreSQL 17 (EDB's zip binary distribution -
no Windows service, no Program Files, just extracted binaries) and creates
the `docqa` role/database plus the `docqa_ro` read-only role in one go.
Idempotent - safe to re-run; it skips whatever's already done.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_postgres_windows.ps1
```

This installs into `.pgsql-portable\` inside the project (git-ignored) by
default - pass `-InstallDir` to put it elsewhere, or `-Port` to use a port
other than 5432. It prints the exact `DATABASE_URL`/`DATABASE_URL_READONLY`
lines to paste into `.env` when it finishes. Stop it later with the same
command plus `-Stop`. It does **not** run as a Windows service (needs
admin rights) and won't survive a reboot on its own - just re-run the
script (no `-Stop`) afterwards; it detects everything's already installed
and just starts it.

## 2. Configure and run the app

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

`.env.example`'s `DATABASE_URL`/`DATABASE_URL_READONLY` already match what
either setup path above provisions, so if you used the defaults you likely
don't need to change them. Otherwise edit `.env` to point at your Postgres.

Also in `.env`:
- `QDRANT_URL` must point at a real Qdrant server - no Docker needed, just
  a portable binary from https://github.com/qdrant/qdrant/releases
  (Windows: `qdrant-x86_64-pc-windows-msvc.zip`). Run the `qdrant.exe` it
  contains, then set `QDRANT_URL=http://localhost:6333`. Web dashboard:
  `http://localhost:6333/dashboard`. Must be running before starting this
  app and before running `pytest` (tests use a separate collection name
  on the same server).
- Set `LLM_PROVIDER=ollama` for a fully local/offline setup (no API key -
  `ollama pull qwen2.5:7b` first, or any model you have), or
  `LLM_PROVIDER=mock` to try the app with zero setup (deterministic
  canned-style routing/SQL - good for a quick look, not for real answer
  quality).
- `EMBEDDING_PROVIDER=local` (default) needs no API key - it's a
  dependency-free deterministic embedder good enough for local
  testing/demo. Set `EMBEDDING_PROVIDER=ollama` (`ollama pull
  nomic-embed-text` + `EMBEDDING_DIM=768`) for production-grade semantic
  retrieval.

Run it:

```bash
uvicorn app.main:app --reload --reload-dir app
```

`--reload-dir app` restricts the file-watcher to the `app/` source tree.
Without it, `--reload` watches the whole project root by default -
including `.venv` - so every file access inside an installed package
(seen with networkx, a langgraph dependency) gets treated as a source
change and restarts the server every few seconds, which would drop any
in-flight request mid-demo.

Open http://localhost:8000 - enter any `user_id` to "log in" (no password,
per spec), create a chat, upload files, ask questions.

## Testing

```bash
pytest
```

Requires the same running Postgres and Qdrant server as the app (reads
`DATABASE_URL`/`DATABASE_URL_READONLY`/`QDRANT_URL` from `.env`, falling
back to the same local defaults the app uses; Qdrant test data lands in a
separate collection, never the real one) - the LLM/embedding providers
stay mocked/local, so no API keys are needed. A session-scoped teardown
removes everything the suite created (by its `user_`-prefixed test user
ids, cascading through sessions/messages/files/excel_* rows, plus
dropping any dynamically-created `xlsx_*` tables and the whole test
Qdrant collection) so repeated runs don't leave debris. Coverage (79
tests): login/session lifecycle,
session isolation and cascading deletion, PDF/DOCX/PPTX/TXT ingestion and
citation metadata, Excel/CSV schema inference, COUNT DISTINCT/SUM/AVG/
GROUP BY/date-filtered SQL generation, SQL injection and read-only safety,
Qdrant session filtering, and SEMANTIC/STRUCTURED/HYBRID routing.

## Architecture

```
app/
  api/        FastAPI routes + dependency wiring
  core/       config, logging
  db/         SQLAlchemy engine/session, shared column types
  models/     ORM models (users, sessions, messages, files,
              excel_workbooks/sheets/schema)
  schemas/    Pydantic request/response models
  services/
    ingestion/   PDF/DOCX/PPTX/TXT parsers + deterministic chunker
    documents/   ingestion + upload orchestration
    excel/       schema inference, dynamic per-sheet table DDL, ingestion
    retrieval/   Qdrant semantic retrieval + hybrid sheet discovery
    sql/         schema description, SQL generation call, validator, executor
    embeddings/  provider abstraction (local hashing / Ollama)
    llm/         provider abstraction (Ollama / mock)
    storage/     original-file storage abstraction (local disk)
  graph/      LangGraph state, node implementations, workflow wiring
  vector/     Qdrant client + payload schemas
  prompts/    system/user prompt templates
frontend/     vanilla HTML/CSS/JS SPA (login, sidebar, upload, chat)
tests/        pytest suite + fixture factories for sample PDF/DOCX/PPTX/XLSX
```

### LangGraph workflow

```
START -> load_session -> router
                            |-- SEMANTIC   -> semantic_retrieval -----------\
                            |-- STRUCTURED -> schema_selection               \
                            |                   -> sql_generation             -> answer_generation -> END
                            |-- HYBRID     -> hybrid_discovery                /
                                                 -> schema_selection --------/
                                                      -> sql_generation
                                                           -> sql_validation -> sql_execution
```

The router only classifies (SEMANTIC/STRUCTURED/HYBRID) - it never answers.
STRUCTURED and HYBRID share the same schema-selection/SQL sub-pipeline;
HYBRID's `hybrid_discovery` step narrows which sheet(s) that sub-pipeline
sees using Qdrant before generating SQL.

### Data model & isolation

Every row that belongs to uploaded data carries `user_id` + `session_id`
(and `file_id` where applicable). Every Qdrant point payload carries the
same. Every retrieval/search is filtered by them; every API route checks
the session belongs to the requesting `user_id` before touching it.
Deleting a session cascades: dynamically-created structured-data tables are
dropped, Qdrant points for that session are deleted, original files are
removed from storage, and the session row's deletion cascades (FK
`ON DELETE CASCADE`) to its messages/files/excel_* rows.

### SQL safety

LLM-generated SQL is never executed directly. It goes through
`services/sql/sql_validator.py`, which: parses with `sqlglot` and rejects
anything that isn't exactly one `SELECT`; rejects any write/DDL node type
anywhere in the parse tree (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/
TRUNCATE/GRANT/...); restricts every referenced table to the current
session's own structured-data tables; and caps/injects a `LIMIT`. This is
paired with a dedicated read-only PostgreSQL role (`docqa_ro`) and a query
timeout (`SQL_QUERY_TIMEOUT_SECONDS`) as a second, DB-enforced layer -
independently confirmed by rejecting a raw `DELETE`/`CREATE TABLE`, not
just the validator.

### Metadata / citations

Parsers (`services/ingestion/*_parser.py`) generate all location metadata
programmatically - page numbers, DOCX heading-derived sections, slide
number/title, line/char ranges for TXT. Headings/sections are only ever
attached when a parser can deterministically establish them (Word's
"Heading *" styles; a conservative heading heuristic for PDF); nothing is
invented, and nothing is LLM-generated. Every chunk split from a larger
section inherits that section's metadata plus its own chunk_id/index.

## Configuration

See `.env.example` for the full list. Key knobs: `LLM_PROVIDER` (`ollama` /
`mock`) / `LLM_MODEL`, `EMBEDDING_PROVIDER` (`local` / `ollama`) /
`EMBEDDING_MODEL` / `EMBEDDING_DIM`, `OLLAMA_BASE_URL` (default
`http://localhost:11434`), `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS`,
`SQL_QUERY_TIMEOUT_SECONDS` / `SQL_MAX_ROWS`, `STORAGE_LOCAL_PATH`,
`QDRANT_URL` (a real server URL - no embedded/`:memory:` mode).

**A note on small local models via Ollama**: `app/prompts/sql_prompt.py`
and `app/prompts/answer_prompt.py` both include a couple of few-shot
examples specifically because smaller local models (tested against
`qwen2.5:7b`) tend to over-trigger the "I don't know" escape hatch on
purely instruction-based prompts, even when the answer is right there in
the evidence. The examples fixed it; if you use a different/smaller local
model and see it refusing answers it shouldn't, that prompt-sensitivity is
the likely cause.
