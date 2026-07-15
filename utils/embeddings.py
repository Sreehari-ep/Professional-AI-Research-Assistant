from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
@lru_cache(maxsize=1)
def model(): return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def create_embeddings(texts):
    return np.asarray(model().encode(texts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False),dtype="float32")
def create_query_embedding(query): return create_embeddings([query])
