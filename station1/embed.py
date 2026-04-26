"""
Station 1 — ESM-2 Embedding Pipeline (Core Logic)

Model: facebook/esm2_t33_650M_UR50D
  - 650M parameters, 33 transformer layers, 1280-dim embeddings

Truncation behavior:
  ESM-2 tokenizes each amino acid as one token and prepends <cls> / appends <eos>,
  consuming 2 of the 1024-token context window. Sequences longer than 1022 amino
  acids are therefore truncated at the C-terminus. The 'truncated' flag in the
  return dict is set to True whenever this occurs.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 1024          # Total token budget (includes <cls> and <eos>)
EFFECTIVE_MAX_AA = 1022    # Usable amino-acid positions = MAX_LENGTH - 2
EMBEDDING_DIM = 1280
OUTPUT_DIR = Path("outputs")

# Standard 20 amino acids + B(Asx) Z(Glx) X(any) J(Xle) O(Pyl) U(Sec)
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJOU")

# ═══════════════════════════════════════════════════════════════════════════════
# Singleton — model and tokenizer are loaded exactly once on first use
# ═══════════════════════════════════════════════════════════════════════════════

_model: AutoModel | None = None
_tokenizer: AutoTokenizer | None = None
_device: torch.device | None = None


def get_device() -> torch.device:
    """Detect CUDA GPU; fall back to CPU. Prints device on first call."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            print(f"[Station 1] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            _device = torch.device("cpu")
            print("[Station 1] No CUDA GPU found — running on CPU")
    return _device


def load_model() -> tuple[AutoTokenizer, AutoModel]:
    """Load ESM-2 model + tokenizer once; return cached instances thereafter."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        device = get_device()
        print(f"[Station 1] Loading {MODEL_NAME} …")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        print(f"[Station 1] Model ready on {device}")
    return _tokenizer, _model


def validate_sequence(sequence: str) -> str:
    """
    Normalize and validate a protein sequence string.
    Strips whitespace/newlines (common in FASTA copy-paste), upper-cases,
    and rejects any characters outside the ESM-2 amino-acid alphabet.
    """
    cleaned = sequence.strip().upper().replace("\n", "").replace("\r", "").replace(" ", "")
    if not cleaned:
        raise ValueError("Empty sequence after stripping whitespace")
    invalid_chars = set(cleaned) - VALID_AMINO_ACIDS
    if invalid_chars:
        raise ValueError(
            f"Non-amino-acid characters found: {sorted(invalid_chars)}. "
            f"Accepted: {''.join(sorted(VALID_AMINO_ACIDS))}"
        )
    return cleaned


def embed_single(sequence: str, save_dir: str | Path = OUTPUT_DIR) -> dict:
    """
    Produce a 1280-dim mean-pooled embedding for one protein sequence.

    Pipeline:
      1. Validate → 2. Tokenize (truncate if >1022 AA)
      3. Forward pass (no_grad) → 4. Mean-pool token embeddings (skip cls/eos)
      5. Save .npy to disk

    Returns dict with embedding array plus metadata.
    """
    cleaned = validate_sequence(sequence)
    tokenizer, model = load_model()
    device = get_device()

    truncated = len(cleaned) > EFFECTIVE_MAX_AA

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # last_hidden_state: (1, seq_len, 1280)
    # Strip special tokens: position 0 = <cls>, position -1 = <eos>
    token_emb = outputs.last_hidden_state[0, 1:-1, :]   # (AA_count, 1280)
    embedding = token_emb.mean(dim=0).cpu().numpy()       # (1280,)

    # Persist to disk
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    seq_hash = hashlib.sha256(cleaned.encode()).hexdigest()[:12]
    npy_path = save_dir / f"emb_{seq_hash}.npy"
    np.save(npy_path, embedding)

    return {
        "embedding": embedding,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_preview": embedding[:5].tolist(),
        "device_used": str(device),
        "sequence_length": len(cleaned),
        "truncated": truncated,
        "npy_path": str(npy_path),
    }


def embed_batch(sequences: list[str], save_dir: str | Path = OUTPUT_DIR) -> list[dict]:
    """
    Embed multiple sequences. Processes each individually to avoid padding-token
    contamination in mean-pooling (padded positions would dilute the mean with
    meaningless vectors). Batch size is small (≤10) so the overhead is acceptable.
    """
    return [embed_single(seq, save_dir=save_dir) for seq in sequences]
