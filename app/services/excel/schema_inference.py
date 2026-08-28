"""Programmatic column-type and description inference for one sheet/table.

Nothing here calls an LLM - schema is derived entirely from pandas dtype
introspection plus simple statistics, per the spec's requirement that
structured-data schema be discovered deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnInference:
    column_name: str
    data_type: str  # TEXT | INTEGER | NUMERIC | BOOLEAN | DATE | TIMESTAMP
    is_numeric: bool
    is_date: bool
    is_text: bool
    is_categorical: bool
    description: str


def infer_column(series: pd.Series, column_name: str) -> ColumnInference:
    non_null = series.dropna()
    row_count = len(series)
    nunique = non_null.nunique() if len(non_null) else 0

    if pd.api.types.is_bool_dtype(series):
        data_type, is_numeric, is_date, is_text = "BOOLEAN", False, False, False
    elif pd.api.types.is_datetime64_any_dtype(series):
        data_type, is_numeric, is_date, is_text = "TIMESTAMP", False, True, False
    elif pd.api.types.is_integer_dtype(series):
        data_type, is_numeric, is_date, is_text = "INTEGER", True, False, False
    elif pd.api.types.is_float_dtype(series):
        data_type, is_numeric, is_date, is_text = "NUMERIC", True, False, False
    else:
        # Try to detect dates stored as text, otherwise treat as text/categorical.
        # pandas' date parser is very permissive - bare words like "January" or
        # "March" parse successfully (with a defaulted year/day), which would
        # misclassify month-name/weekday-name categorical columns as dates. Real
        # date representations virtually always contain at least one digit, so
        # require that before even attempting date parsing.
        looks_numeric_enough = (
            len(non_null) > 0 and non_null.astype(str).str.contains(r"\d", regex=True).mean() >= 0.9
        )
        if looks_numeric_enough:
            coerced = pd.to_datetime(non_null, errors="coerce", format="mixed")
            parse_rate = coerced.notna().mean()
        else:
            parse_rate = 0.0
        if looks_numeric_enough and parse_rate >= 0.9:
            data_type, is_numeric, is_date, is_text = "TIMESTAMP", False, True, False
        else:
            data_type, is_numeric, is_date, is_text = "TEXT", False, False, True

    # A text column is "categorical" if its vocabulary is small in absolute
    # terms (<=20 distinct values covers things like Department/Status/
    # Category even in a tiny sample) or small relative to the row count
    # for larger datasets - capped at 50 distinct values either way.
    is_categorical = bool(
        is_text and 0 < nunique <= 50 and (nunique <= 20 or nunique / max(row_count, 1) <= 0.5)
    )

    description = _describe(
        column_name=column_name,
        data_type=data_type,
        series=non_null,
        nunique=nunique,
        is_categorical=is_categorical,
    )

    return ColumnInference(
        column_name=column_name,
        data_type=data_type,
        is_numeric=is_numeric,
        is_date=is_date,
        is_text=is_text,
        is_categorical=is_categorical,
        description=description,
    )


def _describe(*, column_name: str, data_type: str, series: pd.Series, nunique: int, is_categorical: bool) -> str:
    if len(series) == 0:
        return f"{column_name}: {data_type} column (no non-null values)."
    if data_type in ("INTEGER", "NUMERIC"):
        try:
            return f"{column_name}: {data_type} column, range {series.min()}-{series.max()}."
        except (TypeError, ValueError):
            return f"{column_name}: {data_type} column."
    if data_type == "TIMESTAMP":
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            return f"{column_name}: date/time column, range {parsed.min()} to {parsed.max()}."
        except (TypeError, ValueError):
            return f"{column_name}: date/time column."
    if is_categorical:
        sample = ", ".join(str(v) for v in series.unique()[:8])
        return f"{column_name}: categorical text column with {nunique} distinct values (e.g. {sample})."
    return f"{column_name}: free-text column with {nunique} distinct values."
