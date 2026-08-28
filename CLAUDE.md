# CLAUDE.md

Guidance for Claude Code sessions working in this repo. See `README.md` for
setup steps - this file covers non-obvious facts and hard-won gotchas that
setup steps don't explain.

## Stack facts that aren't obvious from the code

- **PostgreSQL is a hard requirement, not an option.** SQLite support was
  removed from the whole codebase. `pytest` needs a real running Postgres -
  there is no mocked-DB test path.
- **Postgres is a portable (no-installer) install outside the project
  folder**: binaries at `C:\Users\vishn\pgsql\pgsql`, data at
  `C:\Users\vishn\pgsql\data`. It does not survive a reboot (not a Windows
  service) - restart with
  `powershell -ExecutionPolicy Bypass -File scripts\setup_postgres_windows.ps1`
  (idempotent, also creates roles/db if missing).
- Two Postgres roles exist on purpose: `docqa`/`docqa` (full read/write, use
  for admin/cleanup) and `docqa_ro`/`docqa_ro` (genuinely blocked from any
  write, even `CREATE TABLE` - this is what makes LLM-generated SQL safe to
  execute).
- **Qdrant is a real portable server**, not embedded `:memory:` mode (that
  was deliberately removed). Dashboard at `http://localhost:6333/dashboard`.
- To delete a session's data, always use `DELETE /api/sessions/{id}`, never
  edit the DB directly - it's the only thing that cleans up the session's
  dynamically-created `xlsx_*` Postgres table and its Qdrant points. Direct
  DB edits leave orphaned `xlsx_*` tables behind.

## Model-behavior gotchas (all found via live reproduction, not guessing)

- `generate_sql()` runs Ollama at `temperature=0.0` specifically - SQL
  generation has one correct answer per question+schema, and the default
  0.8 temperature was proven to flip between a correct query and a
  refusal on an identical prompt.
- A local model can hedge on a multi-part question by writing a genuinely
  valid SQL query and then appending a `NO_QUERY` refusal marker onto the
  end anyway. `sql_generator.py::_clean_sql()` strips a trailing `NO_QUERY`
  when it trails real content, and only passes a standalone `NO_QUERY`
  through as a real refusal.
- **Prompt-wording fixes are unreliable on this model** - rewording a
  system-prompt rule has repeatedly had *zero effect* here (reproduced
  more than once). What worked instead: repositioning the same instruction
  into the user prompt, immediately before the generation cutoff (e.g.
  right before "SQL:"). If a prompt-only fix doesn't change output on a
  live test, don't assume a differently-worded version will - try moving
  it instead, or look for a deterministic code-level fix.
- **Known, accepted limitation**: the model can misread a row in a
  multi-row table lookup (e.g. picking the wrong lot's complaint count)
  even when the underlying data sent to it is completely clean and
  unambiguous. Tried the same prompt-rule fix that worked for a different,
  genuinely-ambiguous bug - zero effect here. Left as an open model-accuracy
  limitation, not a code bug - don't keep guessing at prompt fixes for this
  class of error without a concrete new lead.
- Tables embedded in PDF/DOCX/PPTX are rendered to text (pipe-delimited,
  blank cells as an explicit `(blank)` marker so the model doesn't
  hallucinate into them) and go through the SEMANTIC path like any other
  passage - there's no SQL engine for document-embedded tables, only for
  ingested Excel/CSV sheets. Same row-lookup risk above applies to them too.

## Working conventions

- Keep code comments precise and minimal - explain the non-obvious "why",
  not the "what" the code already says.
- Never declare a fix done from reasoning alone. Start a fresh `uvicorn`
  instance on an unused port, upload the real sample file(s), ask the exact
  failing question via curl, and check the real log output
  (`[SQL]`/`[RETRIEVAL]`/`[RERANK]`/`[ANSWER]`) before calling it fixed.
  Clean up the test instance and any test Postgres users/sessions
  afterward - never touch the user's own running server.
- Every fix should be checked for being generic (not document- or
  question-specific) before being called done - this has been asked
  explicitly, repeatedly.
- Sample files in `samples/` are deliberately medical-device/quality/
  regulatory-domain content (this project is a demo build for a BD AI
  Engineer interview) - keep any new sample content in that domain, not
  generic business filler.

## Where the fuller history lives

Claude's own long-term memory for this project (chronological bug/fix
history, proven demo question+answer sets per file-type combination,
Postgres cleanup patterns) is separate from this file and not part of the
repo. If you're a fresh Claude Code session without that memory, this file
plus `README.md` should be enough to work safely - ask the user if you need
the fuller history.
