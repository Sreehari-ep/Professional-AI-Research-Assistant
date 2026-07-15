from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.transformer_engine import generate_text


def _validate_text(text: str, field_name: str = "Text") -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field_name} cannot be empty.")

    return re.sub(r"\s+", " ", text).strip()


def _normalize_question(question: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def _parse_questions(raw_output: str, maximum: int) -> list[str]:
    candidates = re.split(
        r"\n+|(?<=\?)\s+",
        str(raw_output).replace("•", "\n"),
    )

    questions = []
    seen = set()

    for candidate in candidates:
        candidate = re.sub(
            r"^\s*(?:question\s*)?\d+\s*[\.\)\-:]\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(r"^[\-\*\s]+", "", candidate).strip()

        if not candidate:
            continue

        if "?" in candidate:
            candidate = candidate.split("?", 1)[0].strip() + "?"
        elif len(candidate.split()) >= 4:
            candidate = candidate.rstrip(".") + "?"
        else:
            continue

        normalized = _normalize_question(candidate)

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        questions.append(candidate)

        if len(questions) >= maximum:
            break

    return questions


def generate_research_questions(
    text: str,
    question_count: int = 10,
) -> str:
    text = _validate_text(text, "Document text")
    question_count = max(1, min(int(question_count), 20))

    raw_output = generate_text(
        prompt=(
            f"Generate exactly {question_count} distinct academic research "
            "questions based only on the document below.\n"
            "Number every question.\n"
            "Every item must end with a question mark.\n"
            "Cover objective, methodology, evidence, findings, limitations, "
            "research gap and future work when present.\n"
            "Do not repeat or invent information.\n\n"
            f"Document:\n{text[:6500]}\n\n"
            "Research questions:"
        ),
        max_new_tokens=max(240, question_count * 42),
        num_beams=4,
        repetition_penalty=1.5,
        no_repeat_ngram_size=4,
    )

    questions = _parse_questions(raw_output, question_count)

    fallbacks = [
        "What is the main objective of the document?",
        "What research problem does the document address?",
        "Which methodology is described in the document?",
        "What data or evidence is used?",
        "What are the principal findings?",
        "What conclusion is presented?",
        "What limitations are discussed?",
        "What research gap is identified?",
        "How does the approach compare with previous work?",
        "What future research directions are suggested?",
        "Which variables or concepts are most important?",
        "What evaluation methods are used?",
        "What practical applications are described?",
        "What assumptions are made?",
        "How reliable are the reported results?",
        "What challenges are identified?",
        "Which algorithms or techniques are applied?",
        "What ethical concerns are mentioned?",
        "How could the study be improved?",
        "What is the overall contribution of the research?",
    ]

    seen = {_normalize_question(question) for question in questions}

    for fallback in fallbacks:
        if len(questions) >= question_count:
            break

        normalized = _normalize_question(fallback)

        if normalized not in seen:
            questions.append(fallback)
            seen.add(normalized)

    return "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(questions[:question_count], start=1)
    )


def _validate_documents(
    documents: list[dict[str, Any]],
    minimum: int,
) -> list[dict[str, str]]:
    cleaned = []

    for index, document in enumerate(documents or [], start=1):
        if not isinstance(document, dict):
            continue

        text = str(document.get("text", "")).strip()

        if not text:
            continue

        cleaned.append({
            "filename": str(document.get("filename", f"Document {index}")),
            "text": text,
        })

    if len(cleaned) < minimum:
        raise ValueError(f"Select at least {minimum} readable document(s).")

    return cleaned


def generate_literature_review(
    documents: list[dict[str, Any]],
    topic: str = "",
) -> str:
    documents = _validate_documents(documents, minimum=1)

    combined = "\n\n".join(
        (
            f"PAPER {index}\n"
            f"Filename: {document['filename']}\n"
            f"Content:\n{document['text'][:3000]}"
        )
        for index, document in enumerate(documents, start=1)
    )

    return generate_text(
        prompt=(
            f"Write a structured academic literature review about "
            f"{topic.strip() or 'the selected topic'}.\n"
            "Use only the supplied documents.\n"
            "Include Introduction, Major Themes, Methodology Comparison, "
            "Key Findings, Limitations, Research Gaps, Future Directions, "
            "and Conclusion.\n"
            "Mention filenames when comparing documents.\n"
            "Do not invent authors, methods, results or citations.\n\n"
            f"{combined[:7500]}\n\n"
            "Literature review:"
        ),
        max_new_tokens=500,
        num_beams=4,
        repetition_penalty=1.4,
        no_repeat_ngram_size=3,
    )


def compare_papers(documents: list[dict[str, Any]]) -> str:
    documents = _validate_documents(documents, minimum=2)

    combined = "\n\n".join(
        (
            f"PAPER {index}\n"
            f"Filename: {document['filename']}\n"
            f"Content:\n{document['text'][:3000]}"
        )
        for index, document in enumerate(documents, start=1)
    )

    return generate_text(
        prompt=(
            "Compare the supplied research documents using only their content.\n"
            "Cover objective, methodology, dataset or evidence, algorithms, "
            "findings, strengths, limitations, similarities, differences and "
            "future work.\n"
            "Mention each filename and do not invent missing information.\n\n"
            f"{combined[:7500]}\n\n"
            "Paper comparison:"
        ),
        max_new_tokens=500,
        num_beams=4,
        repetition_penalty=1.4,
        no_repeat_ngram_size=3,
    )


def compare_similarity(first_text: str, second_text: str) -> float:
    first_text = _validate_text(first_text, "First document")
    second_text = _validate_text(second_text, "Second document")

    matrix = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    ).fit_transform([first_text, second_text])

    return round(
        float(cosine_similarity(matrix[0], matrix[1])[0][0]) * 100,
        2,
    )


def generate_citation(filename: str, style: str = "APA") -> str:
    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    year = datetime.now().year

    citations = {
        "APA": f"Unknown Author. ({year}). {title}.",
        "MLA": f'Unknown Author. "{title}." {year}.',
        "IEEE": f'[1] Unknown Author, "{title}," {year}.',
        "Harvard": f"Unknown Author ({year}) {title}.",
        "BibTeX": (
            f"@misc{{{Path(filename).stem},\n"
            f"  title={{{title}}},\n"
            f"  year={{{year}}}\n"
            f"}}"
        ),
    }

    if style not in citations:
        raise ValueError(f"Unsupported citation style: {style}")

    return citations[style]
