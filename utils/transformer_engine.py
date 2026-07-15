from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_TOKENS = 512

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def load_model():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def _deduplicate_sentences(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen, cleaned = set(), []
    for sentence in sentences:
        key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            cleaned.append(sentence.strip())
    return " ".join(cleaned).strip()

def generate_text(
    prompt: str,
    max_new_tokens: int = 180,
    num_beams: int = 4,
    repetition_penalty: float = 1.35,
    no_repeat_ngram_size: int = 3,
) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    tokenizer, model, device = load_model()
    encoded = tokenizer(
        prompt.strip(),
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max(1, min(int(max_new_tokens), 512)),
            num_beams=max(1, int(num_beams)),
            do_sample=False,
            repetition_penalty=max(1.0, float(repetition_penalty)),
            no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
            early_stopping=True,
            length_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return _deduplicate_sentences(output)

def answer_question(question: str, context: str) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not context.strip():
        return {"answer": "No relevant document context was found.", "score": 0.0}

    prompt = (
        "You are an academic research assistant.\n"
        "Answer only from the context.\n"
        "Give one short, direct answer.\n"
        "Do not repeat words or phrases.\n"
        "If the answer is absent, reply exactly: "
        "\"The document does not contain this information.\"\n\n"
        f"Context:\n{context[:5000]}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Answer:"
    )
    answer = generate_text(
        prompt,
        max_new_tokens=60,
        num_beams=4,
        repetition_penalty=1.6,
        no_repeat_ngram_size=3,
    )
    return {
        "answer": answer or "The document does not contain this information.",
        "score": 0.0,
    }
