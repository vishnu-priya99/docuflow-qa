"""Central application configuration.

All configuration is sourced from environment variables (see .env.example).
Nothing here should hardcode secrets or environment-specific values.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="http://localhost:8000", alias="CORS_ORIGINS")

    # --- PostgreSQL (required - see scripts/setup_postgres_windows.ps1 or
    # docker-compose.yml if you don't have one running) ---
    database_url: str = Field(
        default="postgresql+asyncpg://docqa:docqa@localhost:5432/docqa", alias="DATABASE_URL"
    )
    database_url_readonly: str | None = Field(default=None, alias="DATABASE_URL_READONLY")
    sql_query_timeout_seconds: int = Field(default=5, alias="SQL_QUERY_TIMEOUT_SECONDS")
    sql_max_rows: int = Field(default=500, alias="SQL_MAX_ROWS")

    # --- Qdrant (real server only, no embedded/:memory: mode - see
    # qdrant_client.py; .qdrant-portable/ has a no-Docker Windows binary) ---
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="document_chunks", alias="QDRANT_COLLECTION")

    # --- Storage ---
    storage_backend: Literal["local"] = Field(default="local", alias="STORAGE_BACKEND")
    storage_local_path: str = Field(default="./data/files", alias="STORAGE_LOCAL_PATH")

    # --- LLM ---
    # Only ollama (real, local) and mock (deterministic, used by the test
    # suite) are implemented.
    llm_provider: Literal["ollama", "mock"] = Field(default="mock", alias="LLM_PROVIDER")
    llm_model: str = Field(default="qwen2.5:7b", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")

    # --- Embeddings ---
    embedding_provider: Literal["ollama", "local"] = Field(
        default="local", alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    # --- Ollama (local LLM/embeddings, no API key) ---
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    # How long Ollama keeps a model loaded in memory after the last request
    # (Ollama's own "keep_alive" request field - a duration string like
    # "30m", or "-1" for indefinite). Ollama's own default is 5 minutes -
    # too short for a live demo/interview with natural pauses between
    # questions: the next request after a gap that long pays a full model
    # reload (seconds, not milliseconds) instead of just inference time.
    ollama_keep_alive: str = Field(default="30m", alias="OLLAMA_KEEP_ALIVE")

    # --- Semantic retrieval / reranking ---
    # Qdrant is searched for SEMANTIC_CANDIDATE_K nearest neighbors by raw
    # embedding similarity, then a reranking pass narrows that pool down to
    # the SEMANTIC_TOP_K most genuinely relevant chunks - embedding
    # similarity alone tends to surface plausible-but-unrelated chunks
    # alongside the real match; reranking is what actually filters those
    # out instead of just displaying every top-k neighbor as a source.
    #
    # RERANK_PROVIDER: "cross_encoder" (fast, purpose-trained, needs
    # `pip install sentence-transformers` - not installed by default, see
    # requirements.txt) | "llm" (reuses LLM_PROVIDER, judges relevance by
    # reasoning instead of pattern-matching, but a full generative call per
    # rerank is much slower than a cross-encoder pass).
    #
    # Model choice matters as much as provider choice here: an older/
    # smaller cross-encoder (e.g. ms-marco-MiniLM-L-6-v2, bge-reranker-
    # base) scores mostly by learned keyword-overlap, not real relevance -
    # verified live, a bare document title outranked the paragraph that
    # actually answered the question on both of those. mxbai-rerank-base-v1
    # (RERANK_MODEL's default) correctly ranked the real content above the
    # title on the same test - still a classification pass, not a
    # generative call, so no real speed cost over the older models.
    rerank_provider: Literal["cross_encoder", "llm"] = Field(
        default="llm", alias="RERANK_PROVIDER"
    )
    rerank_model: str = Field(default="mixedbread-ai/mxbai-rerank-base-v1", alias="RERANK_MODEL")
    # cross_encoder only: once RERANK_MODEL is confirmed already downloaded
    # (see sentence-transformers' cache, typically ~/.cache/huggingface),
    # this skips the Hub network check every startup still does by default
    # even though nothing needs downloading - measured live: ~5.3s with the
    # check vs ~1.3s without, and it also means the reranker no longer
    # needs internet access to start at all. Off by default because
    # forcing this before the model is ever cached (e.g. on a machine
    # running this for the first time) would make that first download fail
    # instead of proceeding normally.
    rerank_offline: bool = Field(default=False, alias="RERANK_OFFLINE")
    # llm rerank only: which Ollama model does the rerank judgment call.
    # Defaults to matching LLM_MODEL deliberately - a smaller dedicated
    # model here is tempting (reranking is closer to classification than
    # open-ended generation) but was measured to be a net loss: a
    # different model than LLM_MODEL forces Ollama to swap the loaded
    # model on every question, and that reload cost more time than the
    # smaller model saved. Only worth diverging if you have a hosted
    # rerank endpoint or enough RAM/VRAM for Ollama to keep both models
    # loaded at once - verify latency actually improves before relying on it.
    rerank_llm_model: str = Field(default="qwen2.5:7b", alias="RERANK_LLM_MODEL")
    # Absolute score floor, cross_encoder only: a candidate below this is
    # dropped even if it would otherwise make the top-N cut. None (default)
    # disables score filtering - always fill top-N by rank. Cross-encoder
    # raw scores are not reliably comparable across passages of different
    # length/style, so a fixed floor can silently drop a correct answer -
    # only set this after checking score distributions against your own
    # real documents and questions.
    rerank_min_score: float | None = Field(default=None, alias="RERANK_MIN_SCORE")
    # Relative cutoff, cross_encoder only: drop any candidate scoring more
    # than this far below the best score in the current batch. Adapts
    # per-query to whatever scale the model produces, unlike an absolute
    # floor. Does not guarantee detecting "nothing is relevant" - if
    # nothing in a batch is truly relevant, scores can still cluster
    # within one margin-width of each other.
    rerank_score_margin: float | None = Field(default=None, alias="RERANK_SCORE_MARGIN")
    semantic_candidate_k: int = Field(default=15, alias="SEMANTIC_CANDIDATE_K")
    semantic_top_k: int = Field(default=5, alias="SEMANTIC_TOP_K")

    # --- Chunking ---
    chunk_size_chars: int = Field(default=1200, alias="CHUNK_SIZE_CHARS")
    chunk_overlap_chars: int = Field(default=200, alias="CHUNK_OVERLAP_CHARS")

    # --- Uploads ---
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")

    @field_validator("cors_origins")
    @classmethod
    def _validate_cors(cls, v: str) -> str:
        return v

    @field_validator("database_url", "database_url_readonly")
    @classmethod
    def _require_postgres(cls, v: str | None) -> str | None:
        if v and not v.startswith("postgresql"):
            raise ValueError(
                f"Only PostgreSQL is supported (got {v!r}). Set DATABASE_URL to a "
                "postgresql+asyncpg:// URL - see scripts/setup_postgres_windows.ps1 "
                "or docker-compose.yml if you don't have Postgres running."
            )
        return v

    @field_validator("qdrant_url")
    @classmethod
    def _require_qdrant_server(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(
                f"QDRANT_URL must be a real server URL (got {v!r}) - no embedded/:memory: mode "
                "is supported. Run .qdrant-portable/qdrant.exe (or see "
                "https://github.com/qdrant/qdrant/releases for other platforms), then set "
                "QDRANT_URL=http://localhost:6333."
            )
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_readonly_database_url(self) -> str:
        return self.database_url_readonly or self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
