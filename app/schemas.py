from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Language = Literal["en", "sw"]


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    model: str
    embedding_dim: int
    embeddings: List[List[float]]


class DuplicateCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    language: Language
    gloss_id: Optional[int] = Field(default=None, ge=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class DuplicateMatch(BaseModel):
    gloss_id: int
    text: str
    language: Language
    similarity: float
    flagged: bool


class DuplicateCheckResponse(BaseModel):
    gloss_id: Optional[int]
    embedding_stored: bool
    model: str
    embedding_dim: int
    threshold_used: float
    duplicate_flagged: bool
    matches: List[DuplicateMatch]


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_dim: int
