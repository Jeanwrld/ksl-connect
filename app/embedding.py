from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: Optional[SentenceTransformer] = None


def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = load_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
