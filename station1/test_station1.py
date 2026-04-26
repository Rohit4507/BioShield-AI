"""
Station 1 — Tests

Uses real pathogen protein sequences:
  1. Single embed : Ricin A-chain fragment (Ricinus communis, select-agent toxin, 120 AA)
  2. Batch of 3   : Anthrax LF / SARS-CoV-2 N / Ebola VP40 fragments (~50-60 AA each)
  3. Truncation   : Full SARS-CoV-2 Spike protein (1273 AA — exceeds 1022 AA limit)
  4. Validation    : Non-amino-acid input must be rejected
  5. API tests     : /health, /embed, /embed/batch via FastAPI TestClient

Run:  python test_station1.py            (standalone)
      pytest test_station1.py -v         (pytest)
"""

import sys
import numpy as np

# ── Real pathogen sequences ──────────────────────────────────────────────────

# Ricin A-chain fragment — Ricinus communis (UniProt P02879, residues 1-120)
RICIN_A_CHAIN = (
    "IFPKQYPIINFTTAGATVQSYTNFIRAVRGRLTTGADVRHEIPVLPNRVGLPINQRFIL"
    "VELSNHAELSVTLALDVTNAYVVGYRAGNSAYFFHPDNQEDAEAITHLFTDVQNRYTFAF"
)

# Anthrax Lethal Factor fragment — Bacillus anthracis (UniProt P15917, residues 1-50)
ANTHRAX_LF = "AGGHGDVGMHVKEKEKNKDENKRKDEERNKTQEEHLKEIMKHIVKIEVKGEEAVKKEAAK"

# SARS-CoV-2 Nucleocapsid fragment (UniProt P0DTC9, residues 1-55)
SARS2_NUCLEO = "MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHG"

# Ebola VP40 fragment (UniProt Q05128, residues 1-50)
EBOLA_VP40 = "MRRVILPTAPPEYMEAIYPARSNSTATAILKEVQQMNEGLFNQNAPYAVTKEALDGLHHL"

# Full SARS-CoV-2 Spike protein (UniProt P0DTC2, 1273 AA) — will be truncated
SARS2_SPIKE_FULL = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS"
    "NVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIV"
    "NNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLE"
    "GKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQT"
    "LLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETK"
    "CTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISN"
    "CVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIAD"
    "YNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPC"
    "NGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVN"
    "FNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPG"
    "TNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYE"
    "CDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISV"
    "TTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVF"
    "AQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLG"
    "DIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMA"
    "YRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLV"
    "KQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASAN"
    "LAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICH"
    "DGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPE"
    "LDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGK"
    "YEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVL"
    "KGVKLHYT"
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name} -- {detail}")
        failed += 1


# ===========================================================================
# Test 1 -- Single sequence embedding (Ricin A-chain)
# ===========================================================================

def test_single():
    from embed import embed_single, EMBEDDING_DIM
    print("\n-- Test 1: Single sequence (Ricin A-chain, 120 AA) --")
    r = embed_single(RICIN_A_CHAIN)
    emb = r["embedding"]
    check("shape is (1280,)", emb.shape == (EMBEDDING_DIM,), f"got {emb.shape}")
    check("embedding_dim == 1280", r["embedding_dim"] == EMBEDDING_DIM)
    check("preview has 5 floats", len(r["embedding_preview"]) == 5)
    check("not truncated", r["truncated"] is False)
    check("sequence_length == 120", r["sequence_length"] == 120, f"got {r['sequence_length']}")
    check("npy file exists", __import__("pathlib").Path(r["npy_path"]).exists())
    # Verify saved file matches returned embedding
    loaded = np.load(r["npy_path"])
    check("saved .npy matches embedding", np.allclose(loaded, emb))


# ===========================================================================
# Test 2 -- Batch of 3 pathogen sequences
# ===========================================================================

def test_batch():
    from embed import embed_batch, EMBEDDING_DIM
    print("\n-- Test 2: Batch of 3 sequences --")
    seqs = [ANTHRAX_LF, SARS2_NUCLEO, EBOLA_VP40]
    results = embed_batch(seqs)
    check("returns 3 results", len(results) == 3, f"got {len(results)}")
    embeddings = np.array([r["embedding"] for r in results])
    check("batch shape is (3, 1280)", embeddings.shape == (3, EMBEDDING_DIM), f"got {embeddings.shape}")
    for i, r in enumerate(results):
        check(f"  result[{i}] dim==1280", r["embedding_dim"] == EMBEDDING_DIM)
        check(f"  result[{i}] not truncated", r["truncated"] is False)


# ===========================================================================
# Test 3 -- Truncation of long sequence (SARS-CoV-2 Spike, 1273 AA > 1022 limit)
# ===========================================================================

def test_truncation():
    from embed import embed_single, EMBEDDING_DIM, EFFECTIVE_MAX_AA
    print("\n-- Test 3: Truncation (SARS-CoV-2 Spike, 1273 AA) --")
    seq_len = len(SARS2_SPIKE_FULL.replace(" ", "").replace("\n", ""))
    print(f"   Input length: {seq_len} AA (limit: {EFFECTIVE_MAX_AA})")
    r = embed_single(SARS2_SPIKE_FULL)
    emb = r["embedding"]
    check("shape is (1280,)", emb.shape == (EMBEDDING_DIM,), f"got {emb.shape}")
    check("truncated == True", r["truncated"] is True)
    check("sequence_length reports full length", r["sequence_length"] == seq_len, f"got {r['sequence_length']}")


# ===========================================================================
# Test 4 -- Validation rejects bad input
# ===========================================================================

def test_validation():
    from embed import embed_single
    print("\n-- Test 4: Input validation --")
    try:
        embed_single("ACGT1234!!!")
        check("rejects non-AA chars", False, "no exception raised")
    except ValueError:
        check("rejects non-AA chars", True)
    try:
        embed_single("   ")
        check("rejects empty sequence", False, "no exception raised")
    except ValueError:
        check("rejects empty sequence", True)


# ===========================================================================
# Test 5 -- API endpoint tests via FastAPI TestClient
# ===========================================================================

def test_api():
    from fastapi.testclient import TestClient
    from main import app
    print("\n-- Test 5: API endpoints --")
    client = TestClient(app)

    # Health
    resp = client.get("/health")
    check("GET /health returns 200", resp.status_code == 200)
    check("health status is ok", resp.json()["status"] == "ok")

    # Single embed
    resp = client.post("/embed", json={"sequence": RICIN_A_CHAIN})
    check("POST /embed returns 200", resp.status_code == 200)
    body = resp.json()
    check("response has embedding_dim", body["embedding_dim"] == 1280)
    check("response has 5-element preview", len(body["embedding_preview"]) == 5)
    check("response has device_used", body["device_used"] in ("cpu", "cuda"))
    check("response has truncated flag", isinstance(body["truncated"], bool))

    # Batch embed
    batch_body = {"sequences": [{"sequence": s} for s in [ANTHRAX_LF, SARS2_NUCLEO, EBOLA_VP40]]}
    resp = client.post("/embed/batch", json=batch_body)
    check("POST /embed/batch returns 200", resp.status_code == 200)
    check("batch_size == 3", resp.json()["batch_size"] == 3)

    # Validation rejection
    resp = client.post("/embed", json={"sequence": "NOT_A_PROTEIN_123!!!"})
    check("rejects invalid input with 422", resp.status_code == 422)


# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BioShield AI -- Station 1 Test Suite")
    print("=" * 60)
    test_single()
    test_batch()
    test_truncation()
    test_validation()
    test_api()
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
