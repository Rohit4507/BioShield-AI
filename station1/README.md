# Station 1 — ESM-2 Embedding Pipeline

## What this does

Takes a raw protein sequence (amino-acid string), runs it through Meta's **ESM-2** protein language model (`facebook/esm2_t33_650M_UR50D`, 650 M parameters), and returns a **1280-dimensional embedding vector** that numerically represents the protein's biological properties. These embeddings are the input features for downstream threat-classification in Station 2.

**ESM-2 has never seen before?** — It was pre-trained on 250 million protein sequences from UniRef50, so it "understands" protein language the way GPT understands English.

---

## Hardware requirements

| Mode | RAM | VRAM | Disk (model cache) | Notes |
|------|-----|------|--------------------|-------|
| CPU  | 8 GB | — | ~2.5 GB | Works but slow (~5-10 s/sequence) |
| GPU  | 8 GB | **≥4 GB** | ~2.5 GB | Fast (~0.2 s/sequence). Any NVIDIA GPU with CUDA 11.8+ |

> ⚠️ **If you have <8 GB RAM**, the model may fail to load. Use a cloud VM or Colab.

---

## Step 1 — Environment setup (local, no Docker)

```bash
# Clone the repo and enter station1
cd BioShield-AI/station1

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
# GPU machine (CUDA 11.8+):
pip install -r requirements.txt

# CPU-only machine (smaller download, no CUDA libs):
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The first time you import the model, HuggingFace downloads ~2.5 GB of weights to `~/.cache/huggingface/`. This is a one-time cost.

---

## Step 2 — Run without Docker

### Start the API server

```bash
cd station1
uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait for the log line `[Station 1] Model ready on cpu` (or `cuda`). This takes 15-30 s on CPU.

### Test it

```bash
# Health check
curl http://localhost:8000/health

# Embed a single sequence (Ricin A-chain fragment)
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"sequence": "IFPKQYPIINFTTAGATVQSYTNFIRAVRGRLTTGADVRHEIPVLPNRVGLPINQRFIL"}'

# Batch embed (3 sequences)
curl -X POST http://localhost:8000/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"sequences": [{"sequence": "ACDEFGHIKLMNPQ"}, {"sequence": "RSTV WYACDEF"}, {"sequence": "MFVFLVLLPLVSSQ"}]}'
```

### Run the test suite

```bash
python test_station1.py
```

Expected output:
```
══════════════════════════════════════════════════════
BioShield AI — Station 1 Test Suite
══════════════════════════════════════════════════════
── Test 1: Single sequence (Ricin A-chain, 120 AA) ──
  PASS  shape is (1280,)
  PASS  embedding_dim == 1280
  ...
Results: 25 passed, 0 failed
══════════════════════════════════════════════════════
```

---

## Step 3 — Run with Docker

```bash
cd station1

# Build and start
docker compose up --build

# First run downloads the model (~2.5 GB) inside the container.
# Subsequent runs use the cached volume and start in ~20 s.
```

The API is available at `http://localhost:8000`. Use the same curl commands from Step 2.

To stop: `docker compose down` (model cache persists in the `hf_cache` Docker volume).

---

## Step 4 — Verify output

### Check the embedding file

After any embed call, a `.npy` file is saved in the `outputs/` directory:

```bash
# List saved embeddings
ls outputs/

# Load and inspect in Python
python -c "
import numpy as np
import glob
files = glob.glob('outputs/emb_*.npy')
for f in files:
    emb = np.load(f)
    print(f'{f}: shape={emb.shape}, mean={emb.mean():.4f}, std={emb.std():.4f}')
"
```

### Verify API response format

A successful `/embed` response looks like:
```json
{
  "embedding_dim": 1280,
  "embedding_preview": [0.0123, -0.0456, 0.0789, -0.0012, 0.0345],
  "device_used": "cpu",
  "sequence_length": 120,
  "truncated": false
}
```

- `embedding_dim` is always `1280` (fixed by the ESM-2 architecture).
- `embedding_preview` shows the first 5 float values for quick sanity checks.
- `truncated: true` means the input exceeded 1022 amino acids and was clipped at the C-terminus.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OutOfMemoryError` on GPU | Your GPU has <4 GB VRAM. Set `CUDA_VISIBLE_DEVICES=""` to force CPU mode. |
| `OSError: Can't load tokenizer` | Network issue during model download. Check internet and retry. Model is cached after first success. |
| `torch.cuda.is_available()` returns `False` on GPU machine | Install CUDA-enabled PyTorch: `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118` |
| Docker build fails on ARM Mac | Add `platform: linux/amd64` under the `station1` service in `docker-compose.yml`. |
