from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_DATASET_TYPES = {".csv", ".xlsx", ".xls"}


def load_dataframe(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    extension = path.suffix.lower()

    if extension == ".csv":
        dataframe = pd.read_csv(path)
    elif extension in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    else:
        raise ValueError("Only CSV, XLSX and XLS files are supported.")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    return dataframe


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column(dataframe: pd.DataFrame, phrase: str) -> str | None:
    phrase_normalized = normalize(phrase)
    if not phrase_normalized:
        return None

    exact_map = {normalize(column): str(column) for column in dataframe.columns}

    if phrase_normalized in exact_map:
        return exact_map[phrase_normalized]

    for normalized_column, original_column in exact_map.items():
        if phrase_normalized in normalized_column or normalized_column in phrase_normalized:
            return original_column

    phrase_tokens = set(phrase_normalized.split())
    best_column = None
    best_score = 0.0

    for normalized_column, original_column in exact_map.items():
        column_tokens = set(normalized_column.split())
        union = phrase_tokens | column_tokens

        if not union:
            continue

        score = len(phrase_tokens & column_tokens) / len(union)

        if score > best_score:
            best_score = score
            best_column = original_column

    return best_column if best_score >= 0.25 else None


def extract_target_column(dataframe: pd.DataFrame, question: str) -> str | None:
    normalized_question = normalize(question)

    ignored = {
        "what", "which", "is", "are", "the", "a", "an", "of", "in", "for",
        "dataset", "data", "column", "columns", "field", "fields", "value",
        "values", "average", "mean", "median", "minimum", "maximum", "min",
        "max", "count", "number", "total", "show", "find", "calculate",
        "compare", "between", "by", "group", "unique", "most", "common",
        "percentage", "percent", "how", "many", "there", "missing",
        "duplicate", "duplicates", "rows", "records",
    }

    words = [word for word in normalized_question.split() if word not in ignored]

    for length in range(min(5, len(words)), 0, -1):
        for start in range(0, len(words) - length + 1):
            phrase = " ".join(words[start:start + length])
            column = find_column(dataframe, phrase)
            if column:
                return column

    return None


def _result(
    answer: str,
    *,
    status: str = "success",
    operation: str = "structured",
    details: Any = None,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "status": status,
        "operation": operation,
        "details": details,
    }


def answer_dataset_question(
    filepath: str | Path,
    question: str,
) -> dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    dataframe = load_dataframe(filepath)
    normalized_question = normalize(question)

    if any(term in normalized_question for term in [
        "how many rows", "number of rows", "total rows",
        "how many records", "number of records", "total records",
    ]):
        return _result(
            f"The dataset contains {len(dataframe):,} rows.",
            operation="row_count",
            details={"rows": len(dataframe)},
        )

    if any(term in normalized_question for term in [
        "how many columns", "number of columns", "total columns",
        "how many fields", "number of fields",
    ]):
        return _result(
            f"The dataset contains {len(dataframe.columns):,} columns.",
            operation="column_count",
            details={"columns": len(dataframe.columns)},
        )

    if any(term in normalized_question for term in [
        "column names", "what are the columns", "list the columns",
        "show the columns", "available columns", "dataset columns",
    ]):
        columns = list(dataframe.columns)
        return _result(
            "The dataset columns are: " + ", ".join(columns) + ".",
            operation="column_names",
            details={"columns": columns},
        )

    if any(term in normalized_question for term in [
        "missing value", "missing values", "null value", "null values", "nan",
    ]):
        missing = dataframe.isna().sum()
        total = int(missing.sum())
        details = {
            str(column): int(value)
            for column, value in missing.items()
            if int(value) > 0
        }

        if total == 0:
            return _result(
                "The dataset contains no missing values.",
                operation="missing_values",
                details={},
            )

        description = ", ".join(
            f"{column}: {value}" for column, value in details.items()
        )
        return _result(
            f"The dataset contains {total:,} missing values. {description}.",
            operation="missing_values",
            details=details,
        )

    if "duplicate" in normalized_question or "repeated rows" in normalized_question:
        count = int(dataframe.duplicated().sum())
        return _result(
            f"The dataset contains {count:,} duplicate rows.",
            operation="duplicate_rows",
            details={"duplicates": count},
        )

    group_match = re.search(
        r"(?:compare\s+)?(?:average|mean)\s+(.+?)\s+(?:by|between)\s+(.+)",
        normalized_question,
    )

    if group_match:
        value_phrase = group_match.group(1).strip()
        group_phrase = group_match.group(2).strip()

        value_column = find_column(dataframe, value_phrase)
        group_column = find_column(dataframe, group_phrase)

        if value_column and group_column:
            working = dataframe[[group_column, value_column]].copy()
            working[value_column] = pd.to_numeric(
                working[value_column],
                errors="coerce",
            )
            working = working.dropna(subset=[group_column, value_column])

            if working.empty:
                return _result(
                    f'No numeric values are available in "{value_column}" for comparison.',
                    status="not_found",
                    operation="group_mean",
                )

            comparison = (
                working.groupby(group_column)[value_column]
                .agg(["count", "mean", "median", "min", "max"])
                .reset_index()
                .round(2)
            )

            lines = [
                (
                    f"{row[group_column]}: mean {row['mean']}, "
                    f"median {row['median']}, count {int(row['count'])}"
                )
                for _, row in comparison.iterrows()
            ]

            return _result(
                f'Comparison of "{value_column}" by "{group_column}": '
                + "; ".join(lines)
                + ".",
                operation="group_mean",
                details=comparison.to_dict(orient="records"),
            )

    target_column = extract_target_column(dataframe, question)

    operations = [
        ("average", "mean"),
        ("mean", "mean"),
        ("median", "median"),
        ("minimum", "min"),
        (" min ", "min"),
        ("maximum", "max"),
        (" max ", "max"),
        ("sum", "sum"),
    ]

    for trigger, operation in operations:
        trigger_match = (
            trigger.strip() in normalized_question.split()
            if trigger.strip() in {"min", "max"}
            else trigger in normalized_question
        )

        if not trigger_match:
            continue

        if not target_column:
            return _result(
                "I could not identify the requested numeric column. "
                "Available columns: " + ", ".join(map(str, dataframe.columns)) + ".",
                status="not_found",
                operation=operation,
            )

        numeric = pd.to_numeric(dataframe[target_column], errors="coerce").dropna()

        if numeric.empty:
            return _result(
                f'The column "{target_column}" does not contain usable numeric values.',
                status="not_found",
                operation=operation,
            )

        value = getattr(numeric, operation)()

        return _result(
            f'The {operation} of "{target_column}" is {float(value):.2f}.',
            operation=operation,
            details={"column": target_column, "value": float(value)},
        )

    if "unique" in normalized_question:
        if not target_column:
            return _result(
                "I could not identify the requested column.",
                status="not_found",
                operation="unique_values",
            )

        values = (
            dataframe[target_column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        preview = values[:30]
        suffix = "" if len(values) <= 30 else f" (showing 30 of {len(values)})"

        return _result(
            f'Unique values in "{target_column}": '
            + ", ".join(preview)
            + suffix
            + ".",
            operation="unique_values",
            details={"column": target_column, "values": preview, "total": len(values)},
        )

    if "most common" in normalized_question or "mode" in normalized_question:
        if not target_column:
            return _result(
                "I could not identify the requested column.",
                status="not_found",
                operation="mode",
            )

        counts = dataframe[target_column].dropna().astype(str).value_counts()

        if counts.empty:
            return _result(
                f'The column "{target_column}" has no usable values.',
                status="not_found",
                operation="mode",
            )

        value = counts.index[0]
        count = int(counts.iloc[0])

        return _result(
            f'The most common value in "{target_column}" is "{value}" '
            f"with {count:,} occurrences.",
            operation="mode",
            details={"column": target_column, "value": value, "count": count},
        )

    if target_column and ("count" in normalized_question or "how many" in normalized_question):
        count = int(dataframe[target_column].notna().sum())
        return _result(
            f'The column "{target_column}" contains {count:,} non-missing values.',
            operation="non_missing_count",
            details={"column": target_column, "count": count},
        )

    return _result(
        "I could not map this question to a supported dataset calculation. "
        "Try asking about rows, columns, missing values, duplicates, "
        "mean, median, minimum, maximum, unique values, most common values, "
        "or a group comparison.",
        status="unsupported",
        operation="unsupported",
        details={"available_columns": list(dataframe.columns)},
    )
