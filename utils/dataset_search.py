from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_dataset(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    else:
        raise ValueError("Only CSV, XLSX and XLS files are supported.")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    return dataframe


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def is_column_question(query: str) -> bool:
    words = set(normalize_text(query).split())
    return bool(words & {
        "column", "columns", "field", "fields", "feature", "features",
        "variable", "variables", "attribute", "attributes",
    })


def _column_terms(query: str) -> str:
    ignored = {
        "which", "what", "are", "is", "the", "related", "to", "column",
        "columns", "field", "fields", "feature", "features", "variable",
        "variables", "attribute", "attributes", "dataset", "data", "show",
        "find", "list",
    }
    return " ".join(
        word for word in normalize_text(query).split()
        if word not in ignored
    )


def find_related_columns(
    dataframe: pd.DataFrame,
    query: str,
    minimum_score: float = 0.28,
) -> list[dict[str, Any]]:
    columns = [str(column) for column in dataframe.columns]
    search_terms = _column_terms(query) or normalize_text(query)
    normalized_columns = [normalize_text(column) for column in columns]
    search_tokens = set(search_terms.split())
    results = []

    for column, normalized_column in zip(columns, normalized_columns):
        score = 0.0
        column_tokens = set(normalized_column.split())

        if search_terms == normalized_column:
            score = 1.0
        elif search_terms in normalized_column or normalized_column in search_terms:
            score = 0.92
        elif search_tokens and column_tokens:
            score = len(search_tokens & column_tokens) / len(search_tokens | column_tokens)

        if score >= minimum_score:
            results.append({
                "column": column,
                "score": float(score),
                "score_percentage": round(float(score) * 100, 2),
            })

    if results:
        return sorted(results, key=lambda item: item["score"], reverse=True)

    if not normalized_columns:
        return []

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(normalized_columns + [search_terms])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    for column, score in zip(columns, similarities):
        score = float(score)
        if score >= minimum_score:
            results.append({
                "column": column,
                "score": score,
                "score_percentage": round(score * 100, 2),
            })

    return sorted(results, key=lambda item: item["score"], reverse=True)


def dataframe_to_row_chunks(
    dataframe: pd.DataFrame,
    maximum_rows: int = 5000,
) -> list[str]:
    chunks = []

    for row_number, (_, row) in enumerate(
        dataframe.head(maximum_rows).iterrows(),
        start=1,
    ):
        values = []

        for column in dataframe.columns:
            value = row[column]
            if pd.isna(value):
                continue

            value_text = str(value).strip()
            if value_text:
                values.append(f"{column}: {value_text}")

        if values:
            chunks.append(f"Row {row_number} | " + " | ".join(values))

    return chunks


def search_dataset_rows(
    dataframe: pd.DataFrame,
    query: str,
    top_k: int = 8,
    minimum_score: float = 0.12,
) -> list[dict[str, Any]]:
    chunks = dataframe_to_row_chunks(dataframe)

    if not chunks:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    results = []

    for index in scores.argsort()[::-1]:
        score = float(scores[index])

        if score < minimum_score:
            continue

        results.append({
            "rank": len(results) + 1,
            "row_number": int(index) + 1,
            "chunk": chunks[index],
            "score": score,
            "score_percentage": round(score * 100, 2),
        })

        if len(results) >= top_k:
            break

    return results


def _research_field(query: str) -> str | None:
    normalized = normalize_text(query)

    mapping = {
        "objective": ["objective", "aim", "purpose"],
        "methodology": ["methodology", "method used"],
        "conclusion": ["conclusion"],
        "abstract": ["abstract"],
        "limitations": ["limitation", "limitations"],
        "future work": ["future work", "future scope"],
        "findings": ["main findings", "findings"],
        "research gap": ["research gap"],
    }

    for field, terms in mapping.items():
        if any(term in normalized for term in terms):
            return field

    return None


def search_dataset(
    filepath: str | Path,
    query: str,
    top_k: int = 8,
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    dataframe = load_dataset(filepath)
    normalized = normalize_text(query)

    if any(term in normalized for term in [
        "how many rows", "number of rows", "total rows", "how many records",
    ]):
        return {
            "mode": "structured_answer",
            "answer": f"The dataset contains {len(dataframe):,} rows.",
            "results": [],
            "available_columns": list(dataframe.columns),
        }

    if any(term in normalized for term in [
        "how many columns", "number of columns", "total columns",
    ]):
        return {
            "mode": "structured_answer",
            "answer": f"The dataset contains {len(dataframe.columns):,} columns.",
            "results": [],
            "available_columns": list(dataframe.columns),
        }

    if any(term in normalized for term in [
        "column names", "what are the columns", "list the columns",
        "show the columns", "available columns",
    ]):
        return {
            "mode": "structured_answer",
            "answer": "The dataset columns are: "
            + ", ".join(map(str, dataframe.columns))
            + ".",
            "results": [],
            "available_columns": list(dataframe.columns),
        }

    if "missing value" in normalized or "missing values" in normalized or "null values" in normalized:
        missing = dataframe.isna().sum()
        details = [
            {
                "column": str(column),
                "missing_values": int(value),
                "missing_percentage": round(
                    int(value) / max(len(dataframe), 1) * 100,
                    2,
                ),
            }
            for column, value in missing.items()
            if int(value) > 0
        ]
        total = int(missing.sum())
        return {
            "mode": "missing_values",
            "answer": (
                "The dataset contains no missing values."
                if total == 0
                else f"The dataset contains {total:,} missing values."
            ),
            "results": details,
            "available_columns": list(dataframe.columns),
        }

    if "duplicate" in normalized:
        count = int(dataframe.duplicated().sum())
        return {
            "mode": "structured_answer",
            "answer": f"The dataset contains {count:,} duplicate rows.",
            "results": [],
            "available_columns": list(dataframe.columns),
        }

    requested_field = _research_field(query)

    if requested_field:
        normalized_columns = {
            normalize_text(column): str(column)
            for column in dataframe.columns
        }
        requested_normalized = normalize_text(requested_field)

        if requested_normalized not in normalized_columns:
            return {
                "mode": "research_field_missing",
                "answer": (
                    f'This dataset does not contain an "{requested_field.title()}" '
                    "column. Select a PDF, DOCX, TXT or PPTX research document "
                    "for this question."
                ),
                "results": [],
                "available_columns": list(dataframe.columns),
            }

        actual_column = normalized_columns[requested_normalized]
        values = (
            dataframe[actual_column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        values = list(dict.fromkeys(values[values != ""].tolist()))[:10]

        if not values:
            return {
                "mode": "research_field_empty",
                "answer": f'The "{actual_column}" column contains no readable values.',
                "results": [],
                "available_columns": list(dataframe.columns),
            }

        return {
            "mode": "research_field_values",
            "answer": f'Values from "{actual_column}": ' + "; ".join(values),
            "results": [{"column": actual_column, "values": values}],
            "available_columns": list(dataframe.columns),
        }

    if is_column_question(query):
        results = find_related_columns(dataframe, query)

        return {
            "mode": "columns",
            "answer": (
                "Related column(s): "
                + ", ".join(item["column"] for item in results)
                + "."
                if results
                else "No related column was found."
            ),
            "results": results,
            "available_columns": list(dataframe.columns),
        }

    results = search_dataset_rows(dataframe, query, top_k=top_k)

    return {
        "mode": "rows",
        "answer": (
            f"{len(results)} relevant dataset record(s) found."
            if results
            else "No sufficiently relevant dataset records were found."
        ),
        "results": results,
        "available_columns": list(dataframe.columns),
    }
