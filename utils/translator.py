from __future__ import annotations

import re
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_INPUT_TOKENS = 512

LANGUAGE_CODES = {
    "English": "eng_Latn",
    "Malayalam": "mal_Mlym",
    "Hindi": "hin_Deva",
    "Arabic": "arb_Arab",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Urdu": "urd_Arab",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Turkish": "tur_Latn",
    "Dutch": "nld_Latn",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
}

def supported_languages():
    return list(LANGUAGE_CODES.keys())

@lru_cache(maxsize=1)
def load_translation_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def _split_text(text: str, max_chars: int = 800) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks, current, length = [], [], 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and length + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks or [text[:max_chars]]

def translate_text(
    text: str,
    target_language: str,
    source_language: str = "English",
) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text cannot be empty.")
    if source_language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported source language: {source_language}")
    if target_language not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported target language: {target_language}")
    if source_language == target_language:
        return text.strip()

    tokenizer, model, device = load_translation_model()
    tokenizer.src_lang = LANGUAGE_CODES[source_language]
    target_code = LANGUAGE_CODES[target_language]
    target_id = tokenizer.convert_tokens_to_ids(target_code)

    if target_id is None or target_id == tokenizer.unk_token_id:
        raise ValueError(f"Unable to resolve target language token: {target_code}")

    outputs = []
    for chunk in _split_text(text.strip()):
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=512,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        translated = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if translated:
            outputs.append(translated)

    if not outputs:
        raise RuntimeError("The translation model returned no output.")
    return "\n\n".join(outputs)
