"""Real-world spreadsheets frequently have a title row, a blank row, or a
merged banner above the actual column headers (e.g. "Q1 2024 Regional
Sales Report" in row 1, headers in row 4). Reading such a sheet with
pandas' default header=0 silently produces garbage: the title becomes the
column names and the real headers become a data row. This module scans the
raw grid for the row that actually looks like a header before handing off
to schema inference.
"""
from __future__ import annotations

import pandas as pd


def _looks_like_header_row(row: pd.Series, next_rows: list[pd.Series]) -> int:
    """Returns a score for how header-like this row is; 0/negative means
    "not a header" (blank, a single merged title cell, all-numeric, ...)."""
    non_null = row.notna()
    non_null_count = int(non_null.sum())
    if non_null_count < 2:
        return -1  # blank row, or a single-cell merged title banner

    non_null_values = row[non_null]
    all_stringy = non_null_values.map(lambda v: isinstance(v, str)).all()
    if not all_stringy:
        return -1  # a data row (numbers/dates), not a header

    score = non_null_count
    for candidate in next_rows:
        if candidate.notna().sum() >= non_null_count * 0.5:
            score += 5
            break
    return score


def detect_header_row(raw: pd.DataFrame, max_scan: int = 15) -> int:
    """``raw`` must be read with header=None. Returns the best-guess
    0-indexed row to use as the header; defaults to 0 (pandas' normal
    behavior) if nothing more convincing is found."""
    best_row = 0
    best_score = -1
    limit = min(max_scan, len(raw))
    for i in range(limit):
        next_rows = [raw.iloc[j] for j in range(i + 1, min(i + 4, len(raw)))]
        score = _looks_like_header_row(raw.iloc[i], next_rows)
        if score > best_score:
            best_score = score
            best_row = i
    return best_row


def _restore_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reading with header=None forces every column to accommodate its own
    (string) header value alongside the data, which collapses genuinely
    numeric columns to dtype "object" (real numbers as text). Re-attempt
    numeric conversion per column now that the header row is gone - but
    only adopt it if every previously-non-null value converts cleanly, so
    a real text column never gets silently corrupted into partial NaNs."""
    for col in df.columns:
        series = df[col]
        # Skip columns already in a proper numeric/datetime dtype - only
        # object/string-ish columns (pandas 3.0's default "str" dtype
        # included) are candidates for restoration.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().equals(series.notna()):
            df[col] = coerced
    return df


def normalize_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """``raw`` must be read with header=None (so title/banner rows above
    the real header are visible to the detector, not silently consumed as
    column names). Returns a proper DataFrame: real headers, data rows
    only, correct numeric dtypes restored, blank rows and trailing
    single-cell footnote rows dropped."""
    if raw.empty:
        return raw

    header_row = detect_header_row(raw)
    header = raw.iloc[header_row]
    body = raw.iloc[header_row + 1 :].reset_index(drop=True)
    body.columns = [str(c) if pd.notna(c) else f"column_{i}" for i, c in enumerate(header)]

    body = body.dropna(axis="index", how="all")

    # Trim trailing footnote-style rows (a single populated cell, e.g. "Note:
    # figures exclude returns") - but never touch rows in the middle, since
    # legitimately sparse data there is not this pipeline's business to drop.
    while len(body) and body.iloc[-1].notna().sum() <= 1:
        body = body.iloc[:-1]

    body = body.reset_index(drop=True)
    return _restore_numeric_dtypes(body)
