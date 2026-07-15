from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from utils.embeddings import create_query_embedding
from utils.transformer_engine import answer_question
from utils.vector_store import search_documents, vector_store_info


TEXT_DOCUMENT_TYPES = {".pdf", ".docx", ".txt", ".pptx"}
DEFAULT_MINIMUM_SCORE = 0.25
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARACTERS = 6500


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _file_extension(filename: str) -> str:
    return Path(str(filename)).suffix.lower()


def _extract_direct_field(question: str, context: str) -> str | None:
    question_lower = question.lower()

    fields = {
        "title": ["title", "name of the paper", "name of the document"],
        "objective": ["objective", "aim", "purpose"],
        "methodology": ["methodology", "method used", "methods used"],
        "dataset": ["dataset", "data source"],
        "results": ["result", "results", "finding", "findings"],
        "conclusion": ["conclusion"],
        "limitations": ["limitation", "limitations"],
        "future work": ["future work", "future scope"],
    }

    requested_field = None

    for field, triggers in fields.items():
        if any(trigger in question_lower for trigger in triggers):
            requested_field = field
            break

    if not requested_field:
        return None

    label_map = {
        "title": [r"title"],
        "objective": [r"objective", r"research objective", r"aim", r"purpose"],
        "methodology": [r"methodology", r"methods?", r"research method"],
        "dataset": [r"dataset", r"data source"],
        "results": [r"results?", r"findings?"],
        "conclusion": [r"conclusion"],
        "limitations": [r"limitations?"],
        "future work": [r"future work", r"future scope"],
    }

    for label in label_map[requested_field]:
        match = re.search(
            rf"(?:^|\n)\s*{label}\s*:\s*(.+?)(?=\n|$)",
            context,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if not match:
            continue

        value = _normalize(match.group(1)).strip(" -:")

        if not value:
            continue

        if value[-1] not in ".!?":
            value += "."

        prefixes = {
            "title": "The title of the document is ",
            "objective": "The objective of the document is ",
            "methodology": "The methodology described is ",
            "dataset": "The dataset used is ",
            "results": "The document reports that ",
            "conclusion": "The conclusion is ",
            "limitations": "The document identifies this limitation: ",
            "future work": "The suggested future work is ",
        }

        return prefixes[requested_field] + value[0].lower() + value[1:]

    return None


def _build_context(results: list[dict[str, Any]]) -> str:
    parts = []
    total = 0

    for result in results:
        part = (
            f"Document: {result.get('filename', 'Unknown')}\n"
            f"Chunk: {result.get('chunk_id', 'Unknown')}\n"
            f"Content:\n{result.get('chunk', '')}"
        )

        if total + len(part) > MAX_CONTEXT_CHARACTERS:
            break

        parts.append(part)
        total += len(part)

    return "\n\n".join(parts)


def _confidence(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0

    scores = [
        max(0.0, min(float(item.get("score", 0.0)), 1.0))
        for item in results[:3]
    ]

    weights = [0.60, 0.25, 0.15][:len(scores)]
    weight_total = sum(weights)

    return round(
        sum(score * weight for score, weight in zip(scores, weights))
        / weight_total
        * 100,
        2,
    )


def ask_rag(
    question: str,
    *,
    selected_document: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
) -> dict[str, Any]:
    question = _normalize(question)

    if not question:
        raise ValueError("Question cannot be empty.")

    if vector_store_info().get("vectors", 0) <= 0:
        return {
            "answer": "No indexed research documents were found.",
            "confidence": 0.0,
            "confidence_label": "Very Low",
            "sources": [],
            "evidence": [],
            "status": "no_documents",
        }

    raw_results = search_documents(
        create_query_embedding(question),
        top_k=max(int(top_k) * 5, 25),
    )

    filtered = []

    for result in raw_results:
        filename = str(result.get("filename", ""))

        if _file_extension(filename) not in TEXT_DOCUMENT_TYPES:
            continue

        if selected_document and filename != selected_document:
            continue

        if float(result.get("score", 0.0)) < float(minimum_score):
            continue

        chunk = _normalize(result.get("chunk", ""))

        if not chunk:
            continue

        result = dict(result)
        result["chunk"] = chunk
        filtered.append(result)

    unique = []
    seen = set()

    for result in filtered:
        key = result["chunk"].lower()

        if key in seen:
            continue

        seen.add(key)
        unique.append(result)

    relevant = unique[:max(1, int(top_k))]

    if not relevant:
        return {
            "answer": "I could not find a reliable answer in the selected research document.",
            "confidence": 0.0,
            "confidence_label": "Very Low",
            "sources": [],
            "evidence": raw_results[:3],
            "status": "low_confidence",
        }

    context = _build_context(relevant)
    confidence = _confidence(relevant)
    sources = sorted({str(item.get("filename", "Unknown")) for item in relevant})

    direct_answer = _extract_direct_field(question, context)

    if direct_answer:
        return {
            "answer": direct_answer,
            "confidence": confidence,
            "confidence_label": (
                "High" if confidence >= 70
                else "Medium" if confidence >= 45
                else "Low"
            ),
            "sources": sources,
            "evidence": relevant,
            "status": "direct_extraction",
        }

    if confidence < 25:
        return {
            "answer": "The retrieved passages are not reliable enough to answer this question.",
            "confidence": confidence,
            "confidence_label": "Very Low",
            "sources": sources,
            "evidence": relevant,
            "status": "low_confidence",
        }

    generated = answer_question(question=question, context=context)
    answer = generated.get("answer", "") if isinstance(generated, dict) else str(generated)
    answer = _normalize(answer)

    if not answer:
        answer = "The selected document does not contain this information."

    return {
        "answer": answer,
        "confidence": confidence,
        "confidence_label": (
            "High" if confidence >= 70
            else "Medium" if confidence >= 45
            else "Low"
        ),
        "sources": sources,
        "evidence": relevant,
        "status": "generated",
    }
