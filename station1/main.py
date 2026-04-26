"""
Station 1 — FastAPI application

Endpoints:
  POST /embed        – embed a single protein sequence
  POST /embed/batch  – embed up to 10 sequences in one call
  GET  /health       – liveness check

Input validation:
  • Rejects characters outside the ESM-2 amino-acid alphabet
  • Batch size capped at 10
"""

import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from embed import VALID_AMINO_ACIDS, embed_batch, embed_single, load_model

# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════

_AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJOUacdefghiklmnpqrstvwybxzjou\s]+$")


class SequenceRequest(BaseModel):
    sequence: str = Field(..., min_length=1, description="Protein sequence (IUPAC amino-acid characters only)")

    @field_validator("sequence")
    @classmethod
    def check_amino_acids(cls, v: str) -> str:
        stripped = v.strip().upper().replace("\n", "").replace("\r", "").replace(" ", "")
        if not stripped:
            raise ValueError("Sequence is empty after stripping whitespace")
        invalid = set(stripped) - VALID_AMINO_ACIDS
        if invalid:
            raise ValueError(
                f"Non-amino-acid characters: {sorted(invalid)}. "
                f"Allowed: {''.join(sorted(VALID_AMINO_ACIDS))}"
            )
        return v


class BatchRequest(BaseModel):
    sequences: list[SequenceRequest] = Field(..., min_length=1, max_length=10)


class EmbeddingResponse(BaseModel):
    embedding_dim: int
    embedding_preview: list[float]
    device_used: str
    sequence_length: int
    truncated: bool


class BatchResponse(BaseModel):
    results: list[EmbeddingResponse]
    batch_size: int


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ═══════════════════════════════════════════════════════════════════════════════
# App factory with startup model preload
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup so the first request isn't slow."""
    load_model()
    yield


app = FastAPI(
    title="BioShield AI — Station 1",
    description="ESM-2 protein embedding service",
    version="1.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health():
    from embed import _device, MODEL_NAME
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        device=str(_device) if _device else "not loaded",
    )


@app.post("/embed", response_model=EmbeddingResponse)
def embed_endpoint(req: SequenceRequest):
    try:
        result = embed_single(req.sequence)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return EmbeddingResponse(
        embedding_dim=result["embedding_dim"],
        embedding_preview=result["embedding_preview"],
        device_used=result["device_used"],
        sequence_length=result["sequence_length"],
        truncated=result["truncated"],
    )


@app.post("/embed/batch", response_model=BatchResponse)
def embed_batch_endpoint(req: BatchRequest):
    seqs = [item.sequence for item in req.sequences]
    try:
        results = embed_batch(seqs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    items = [
        EmbeddingResponse(
            embedding_dim=r["embedding_dim"],
            embedding_preview=r["embedding_preview"],
            device_used=r["device_used"],
            sequence_length=r["sequence_length"],
            truncated=r["truncated"],
        )
        for r in results
    ]
    return BatchResponse(results=items, batch_size=len(items))
