from typing import Optional

from fastapi import FastAPI, Header, HTTPException

from app.config import settings
from app.db import find_nearest, store_embedding
from app.embedding import embed_texts, load_model
from app.schemas import (
    DuplicateCheckRequest,
    DuplicateCheckResponse,
    DuplicateMatch,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
)

app = FastAPI(title="KSL Connect - Duplicate Detection Service")


@app.on_event("startup")
def startup() -> None:
    load_model()


def check_api_key(x_api_key: Optional[str]) -> None:
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok", model=settings.MODEL_NAME, embedding_dim=settings.EMBEDDING_DIM
    )


@app.post("/v1/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    vectors = embed_texts(req.texts)
    return EmbedResponse(
        model=settings.MODEL_NAME,
        embedding_dim=vectors.shape[1],
        embeddings=vectors.tolist(),
    )


@app.post("/v1/duplicates/check", response_model=DuplicateCheckResponse)
def duplicates_check(
    req: DuplicateCheckRequest, x_api_key: Optional[str] = Header(default=None)
):
    check_api_key(x_api_key)
    threshold = req.threshold if req.threshold is not None else settings.DUPLICATE_THRESHOLD
    top_k = req.top_k if req.top_k is not None else settings.TOP_K_DEFAULT

    vector = embed_texts([req.text])[0]

    embedding_stored = False
    if req.gloss_id is not None:
        embedding_stored = store_embedding(req.gloss_id, vector)
        if not embedding_stored:
            raise HTTPException(
                status_code=404, detail=f"gloss_id {req.gloss_id} not found"
            )

    rows = find_nearest(vector, top_k, exclude_id=req.gloss_id)
    matches = [
        DuplicateMatch(
            gloss_id=row[0],
            text=row[1],
            language=row[2],
            similarity=round(float(row[3]), 4),
            flagged=float(row[3]) >= threshold,
        )
        for row in rows
    ]

    return DuplicateCheckResponse(
        gloss_id=req.gloss_id,
        embedding_stored=embedding_stored,
        model=settings.MODEL_NAME,
        embedding_dim=settings.EMBEDDING_DIM,
        threshold_used=threshold,
        duplicate_flagged=any(m.flagged for m in matches),
        matches=matches,
    )
