-- Creates a read-only Postgres role used exclusively to execute
-- LLM-generated SQL against structured (Excel/CSV-derived) tables.
-- Applied automatically by docker-compose on first Postgres init.
-- For an existing database, run this manually as a superuser.

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'docqa_ro') THEN
      CREATE ROLE docqa_ro LOGIN PASSWORD 'docqa_ro';
   END IF;
END
$$;

GRANT CONNECT ON DATABASE docqa TO docqa_ro;
GRANT USAGE ON SCHEMA public TO docqa_ro;

-- Every table that exists at init time (structured-data tables are created
-- dynamically at runtime by the app's write-role connection, so we also
-- flip the default privileges for the app owner role going forward).
GRANT SELECT ON ALL TABLES IN SCHEMA public TO docqa_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE docqa IN SCHEMA public GRANT SELECT ON TABLES TO docqa_ro;

-- Safety net: never let the read-only role modify data even if a grant is
-- missed above.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM docqa_ro;

-- Cap how long any statement run by this role may execute.
ALTER ROLE docqa_ro SET statement_timeout = '5s';
ALTER ROLE docqa_ro SET default_transaction_read_only = on;
