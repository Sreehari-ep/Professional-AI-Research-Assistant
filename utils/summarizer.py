from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from utils.transformer_engine import generate_text


STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "were", "have",
    "has", "using", "used", "study", "research", "paper", "results",
    "result", "method", "methods", "data", "analysis", "into", "their",
    "there", "which", "when", "where", "what", "while", "than", "then",
}


def split_document(text: str, maximum_characters: int = 3200) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document text cannot be empty.")

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks = []
    current = []
    length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if current and length + len(sentence) + 1 > maximum_characters:
            chunks.append(" ".join(current))
            current = []
            length = 0

        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks or [text[:maximum_characters]]


def generate_summary(text: str, summary_type: str = "Short") -> str:
    chunks = split_document(text)
    partial_summaries = []

    for index, chunk in enumerate(chunks[:10], start=1):
        partial = generate_text(
            prompt=(
                "Summarize the research-paper section below.\n"
                "Use only the supplied section.\n"
                "Preserve objectives, methodology, data, findings, "
                "limitations and conclusions when available.\n"
                "Do not repeat raw examples or long passages.\n\n"
                f"Section {index}:\n{chunk}\n\n"
                "Section summary:"
            ),
            max_new_tokens=180,
            num_beams=4,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
        )

        if partial:
            partial_summaries.append(partial)

    combined = "\n\n".join(partial_summaries)
    normalized_type = str(summary_type).strip().lower()

    if normalized_type == "detailed":
        instruction = (
            "Write a detailed academic summary under these headings: "
            "Objective, Background, Methodology, Data or Evidence, "
            "Main Findings, Limitations, and Conclusion."
        )
        token_limit = 460
    elif normalized_type in {"bullet", "bullets", "bullet points"}:
        instruction = (
            "Write a concise academic bullet-point summary covering "
            "objective, methodology, evidence, findings and conclusion."
        )
        token_limit = 320
    else:
        instruction = (
            "Write a short coherent academic summary covering the main topic, "
            "objective, method, findings and conclusion."
        )
        token_limit = 240

    return generate_text(
        prompt=(
            f"{instruction}\n"
            "Combine the section summaries below.\n"
            "Remove repeated ideas.\n"
            "Do not invent information.\n\n"
            f"{combined[:7000]}\n\n"
            "Final summary:"
        ),
        max_new_tokens=token_limit,
        num_beams=4,
        repetition_penalty=1.45,
        no_repeat_ngram_size=3,
    )


def generate_abstract(text: str) -> str:
    detailed_summary = generate_summary(text, "Detailed")

    return generate_text(
        prompt=(
            "Write a 150-250 word academic abstract from the summary below.\n"
            "Include background, objective, methodology, findings and conclusion.\n"
            "Do not invent information.\n\n"
            f"{detailed_summary}\n\nAbstract:"
        ),
        max_new_tokens=300,
        num_beams=4,
        repetition_penalty=1.4,
        no_repeat_ngram_size=3,
    )


def extract_keywords(
    text: str,
    keyword_count: int = 15,
) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document text cannot be empty.")

    keyword_count = max(1, min(int(keyword_count), 50))
    chunks = split_document(text, maximum_characters=1800)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=5000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(chunks)
    terms = vectorizer.get_feature_names_out()
    scores = matrix.mean(axis=0).A1

    ranked_indices = scores.argsort()[::-1]
    keywords = []
    normalized_seen = set()

    for index in ranked_indices:
        phrase = str(terms[index]).strip()
        normalized = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()

        if not normalized:
            continue

        words = normalized.split()

        if all(word in STOP_WORDS for word in words):
            continue

        if any(
            normalized in existing or existing in normalized
            for existing in normalized_seen
        ):
            continue

        normalized_seen.add(normalized)
        keywords.append(phrase.title())

        if len(keywords) >= keyword_count:
            break

    return keywords


def generate_keywords(text: str, keyword_count: int = 15) -> list[str]:
    return extract_keywords(text, keyword_count)


def generate_main_findings(text: str) -> str:
    return generate_text(
        prompt=(
            "Extract only the principal findings supported by this document.\n"
            "Return concise bullet points and do not invent findings.\n\n"
            f"{text[:6500]}"
        ),
        max_new_tokens=280,
        num_beams=4,
        repetition_penalty=1.4,
        no_repeat_ngram_size=3,
    )


def generate_limitations(text: str) -> str:
    return generate_text(
        prompt=(
            "Identify only limitations supported by this document.\n"
            "Return concise bullet points.\n\n"
            f"{text[:6500]}"
        ),
        max_new_tokens=260,
        num_beams=4,
        repetition_penalty=1.4,
        no_repeat_ngram_size=3,
    )


def generate_conclusion(text: str) -> str:
    return generate_text(
        prompt=(
            "Write a concise academic conclusion using only the document.\n\n"
            f"{text[:6500]}"
        ),
        max_new_tokens=240,
        num_beams=4,
        repetition_penalty=1.4,
        no_repeat_ngram_size=3,
    )
