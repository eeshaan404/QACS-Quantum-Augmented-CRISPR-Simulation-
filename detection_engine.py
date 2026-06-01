"""
QACS — detection_engine.py

WHAT THIS FILE DOES IN PLAIN ENGLISH:
======================================
This is Part 1 of QACS — the detection engine.

A raw DNA or RNA sequence comes in with no label.
This file answers two questions:

  1. WHAT DISEASE IS THIS?
     Uses k-mer frequency analysis to compare the unknown
     sequence against reference disease signatures.
     Cross-validates with BLAST alignment.

  2. WHAT SHOULD CRISPR DO ABOUT IT?
     Once the disease is identified, automatically assigns
     the correct operation — delete, correct, or insert.
     This decision is based on published therapeutic research,
     not guesswork.

WHY TWO APPROACHES?
  K-mer analysis  — fast, works on sequences not in any database,
                    looks at sequence patterns as fingerprints
  BLAST alignment — slower, cross-validates against the entire
                    NCBI database, confirms the k-mer result

WHAT GOES IN:  Raw sequence from input_parser.py
WHAT COMES OUT:
  - predicted_disease  → most likely disease name
  - operation          → delete / correct / insert
  - confidence         → how sure we are (0.0 to 1.0)
  - kmer_scores        → ranked list of all disease matches
  - ready_for_part2    → True if confidence is high enough
  - repair_template    → known patch sequence (correct only)
  - mutation_pos       → position of mutation (correct only)
"""

from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from collections import Counter
import numpy as np
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context
Entrez.email = "your_email@example.com"

KMER_SIZE = 6

# ── OPERATION CONSTANTS ───────────────────────────────────────────────────────
DELETE  = "delete"
CORRECT = "correct"
INSERT  = "insert"


# ── REFERENCE DATABASE ────────────────────────────────────────────────────────

REFERENCE_DB = {
    "sickle cell disease": {
        "gene":           "HBB",
        "accession":      "NM_000518",
        "keywords":       ["HBB", "beta globin", "hemoglobin"],
        # OPERATION decided here based on biology:
        # Single point mutation — correction is the right strategy
        "operation":      CORRECT,
        "mutation_pos":   17,
        "repair_template":"CTGAGGAGAAG",  # correct sequence around codon 6
        "op_reason":      "E6V point mutation — correct the single wrong base"
    },
    "beta thalassemia": {
        "gene":           "HBB",
        "accession":      "NM_000518",
        "keywords":       ["HBB", "beta globin", "thalassemia"],
        "operation":      CORRECT,
        "mutation_pos":   None,   # varies by patient
        "repair_template":None,
        "op_reason":      "Various HBB mutations — correct strategy varies by patient"
    },
    "colorectal cancer": {
        "gene":           "KRAS",
        "accession":      "NM_004985",
        "keywords":       ["KRAS", "GTPase", "oncogene"],
        # Gain-of-function oncogene — delete it
        "operation":      DELETE,
        "mutation_pos":   34,
        "repair_template":None,
        "op_reason":      "KRAS G12D oncogene — delete to stop tumor growth"
    },
    "lung cancer": {
        "gene":           "EGFR",
        "accession":      "NM_005228",
        "keywords":       ["EGFR", "epidermal growth factor"],
        "operation":      DELETE,
        "mutation_pos":   None,
        "repair_template":None,
        "op_reason":      "Mutant EGFR drives cell division — delete selectively"
    },
    "hiv": {
        "gene":           "CCR5",
        "accession":      "NM_000579",
        "keywords":       ["CCR5", "chemokine receptor", "HIV"],
        # CCR5 knockout prevents HIV entry — delete is correct
        "operation":      DELETE,
        "mutation_pos":   None,
        "repair_template":None,
        "op_reason":      "CCR5 knockout prevents HIV from entering T-cells"
    },
    "cystic fibrosis": {
        "gene":           "CFTR",
        "accession":      "NM_000492",
        "keywords":       ["CFTR", "cystic fibrosis transmembrane"],
        "operation":      CORRECT,
        "mutation_pos":   1521,
        "repair_template":"ATCATTGGT",  # restores F508 codon
        "op_reason":      "F508del — correct the 3-nucleotide deletion"
    },
    "duchenne muscular dystrophy": {
        "gene":           "DMD",
        "accession":      "NM_004006",
        "keywords":       ["DMD", "dystrophin"],
        # Exon skipping strategy — delete the problem exon
        "operation":      DELETE,
        "mutation_pos":   None,
        "repair_template":None,
        "op_reason":      "Exon skipping — delete frameshift exon to restore reading frame"
    },
    "huntington disease": {
        "gene":           "HTT",
        "accession":      "NM_002111",
        "keywords":       ["HTT", "huntingtin", "CAG repeat"],
        "operation":      DELETE,
        "mutation_pos":   None,
        "repair_template":None,
        "op_reason":      "Silence mutant HTT allele — delete disrupts toxic expression"
    },
    "severe combined immunodeficiency": {
        "gene":           "ADA",
        "accession":      "NM_000022",
        "keywords":       ["ADA", "adenosine deaminase"],
        # Missing gene entirely — must insert a functional copy
        "operation":      INSERT,
        "mutation_pos":   None,
        "repair_template":None,
        "op_reason":      "ADA gene missing entirely — insert functional copy"
    }
}


# ── K-MER FUNCTIONS ───────────────────────────────────────────────────────────

def extract_kmers(sequence: str, k: int = KMER_SIZE) -> dict:
    """
    Chop sequence into overlapping k-length chunks.
    Count how often each chunk appears.
    Return as frequency dictionary (counts / total).
    """
    if len(sequence) < k:
        return {}
    kmers  = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
    counts = Counter(kmers)
    total  = sum(counts.values())
    return {kmer: count / total for kmer, count in counts.items()}


def cosine_similarity(profile_a: dict, profile_b: dict) -> float:
    """
    Measure how similar two k-mer profiles are.
    Returns 0.0 (completely different) to 1.0 (identical).
    Higher = more likely same disease.
    """
    all_kmers = set(profile_a.keys()) | set(profile_b.keys())
    if not all_kmers:
        return 0.0
    vec_a = np.array([profile_a.get(k, 0.0) for k in all_kmers])
    vec_b = np.array([profile_b.get(k, 0.0) for k in all_kmers])
    dot   = np.dot(vec_a, vec_b)
    mag_a = np.linalg.norm(vec_a)
    mag_b = np.linalg.norm(vec_b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return float(dot / (mag_a * mag_b))


def build_reference_profiles(sequences: dict) -> dict:
    return {disease: extract_kmers(seq) for disease, seq in sequences.items()}


def kmer_classify(unknown_sequence: str, reference_profiles: dict) -> list:
    unknown_profile = extract_kmers(unknown_sequence)
    if not unknown_profile:
        return []
    scores = [
        (disease, round(cosine_similarity(unknown_profile, ref), 4))
        for disease, ref in reference_profiles.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ── NCBI FETCH ────────────────────────────────────────────────────────────────

def fetch_reference_sequences(region_length: int = 500) -> dict:
    """Fetch reference sequences from NCBI for all diseases."""
    print("  Fetching reference sequences from NCBI...")
    sequences = {}
    for disease, entry in REFERENCE_DB.items():
        try:
            handle = Entrez.efetch(db="nucleotide", id=entry["accession"],
                                   rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            sequences[disease] = str(record.seq[:region_length]).upper()
            print(f"  ✓ {entry['gene']} ({disease})")
            time.sleep(0.4)
        except Exception as e:
            print(f"  ✗ Failed {entry['gene']}: {e}")
    return sequences


# ── BLAST VALIDATION ──────────────────────────────────────────────────────────

def blast_validate(sequence: str, top_disease: str) -> dict:
    """Cross-validate k-mer result using BLAST. Takes ~60 seconds."""
    print("  Running BLAST (~60 seconds)...")
    try:
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence,
                                        megablast=True, hitlist_size=5)
        blast_record  = next(NCBIXML.parse(result_handle))

        if not blast_record.alignments:
            return {"blast_match": "No hits", "identity_pct": 0.0,
                    "agrees_with_kmer": False, "confidence_boost": -0.15,
                    "error": None}

        top_hit      = blast_record.alignments[0]
        top_hsp      = top_hit.hsps[0]
        identity_pct = (top_hsp.identities / top_hsp.align_length) * 100
        description  = top_hit.title[:80]
        keywords     = REFERENCE_DB.get(top_disease, {}).get("keywords", [])
        agrees       = any(kw.lower() in description.lower() for kw in keywords)

        return {"blast_match": description, "identity_pct": round(identity_pct, 1),
                "agrees_with_kmer": agrees,
                "confidence_boost": 0.10 if agrees else -0.10, "error": None}

    except Exception as e:
        return {"blast_match": "", "identity_pct": 0.0,
                "agrees_with_kmer": False, "confidence_boost": 0.0,
                "error": f"BLAST failed: {str(e)}"}


# ── MAIN DETECTION FUNCTION ───────────────────────────────────────────────────

def detect_disease(unknown_sequence: str, reference_profiles: dict,
                   run_blast: bool = False) -> dict:
    """
    Detect disease AND determine the CRISPR operation.

    Parameters
    ----------
    unknown_sequence : str
        Raw patient sequence — no label.
    reference_profiles : dict
        Pre-built k-mer profiles from build_reference_profiles().
    run_blast : bool
        Cross-validate with BLAST (slow ~60s). Default False.

    Returns
    -------
    dict with keys:
        predicted_disease → disease name
        operation         → delete / correct / insert  ← NEW
        op_reason         → why this operation          ← NEW
        mutation_pos      → mutation position if known  ← NEW
        repair_template   → patch sequence if known     ← NEW
        confidence        → 0.0 to 1.0
        kmer_scores       → top 5 ranked matches
        blast_result      → BLAST result if run
        ready_for_part2   → True if confidence >= 0.6
        warning           → warning message if any
    """
    kmer_scores = kmer_classify(unknown_sequence, reference_profiles)

    if not kmer_scores:
        return {
            "predicted_disease": "unknown",
            "operation":         DELETE,
            "op_reason":         "Unknown disease — defaulting to delete",
            "mutation_pos":      None,
            "repair_template":   None,
            "confidence":        0.0,
            "kmer_scores":       [],
            "blast_result":      None,
            "ready_for_part2":   False,
            "warning":           "Sequence too short for k-mer analysis."
        }

    top_disease, top_score = kmer_scores[0]
    second_score = kmer_scores[1][1] if len(kmer_scores) > 1 else 0.0
    separation   = top_score - second_score
    confidence   = min(top_score + (separation * 0.5), 1.0)

    blast_result = None
    if run_blast:
        blast_result = blast_validate(unknown_sequence, top_disease)
        confidence   = min(max(confidence + blast_result["confidence_boost"], 0.0), 1.0)

    # ── Assign operation from disease database ────────────────────────────────
    # This is the key change — operation comes from biology, not the user
    db_entry       = REFERENCE_DB.get(top_disease, {})
    operation      = db_entry.get("operation",       DELETE)
    op_reason      = db_entry.get("op_reason",       "Default delete strategy")
    mutation_pos   = db_entry.get("mutation_pos",    None)
    repair_template= db_entry.get("repair_template", None)

    ready   = confidence >= 0.6
    warning = None
    if not ready:
        warning = (f"Low confidence ({confidence:.1%}). "
                   f"Manual review recommended.")
    elif confidence < 0.75:
        warning = (f"Moderate confidence ({confidence:.1%}). "
                   f"Verify before clinical use.")

    return {
        "predicted_disease": top_disease,
        "operation":         operation,
        "op_reason":         op_reason,
        "mutation_pos":      mutation_pos,
        "repair_template":   repair_template,
        "confidence":        round(confidence, 4),
        "kmer_scores":       kmer_scores[:5],
        "blast_result":      blast_result,
        "ready_for_part2":   ready,
        "warning":           warning
    }


# ── PRINT REPORT ──────────────────────────────────────────────────────────────

def print_detection_report(result: dict) -> None:
    op_symbols = {DELETE: "✂ DELETE", CORRECT: "✏ CORRECT", INSERT: "➕ INSERT"}
    print("\n" + "=" * 55)
    print("  QACS — PART 1 DETECTION REPORT")
    print("=" * 55)
    print(f"  Disease    : {result['predicted_disease'].title()}")
    print(f"  Operation  : {op_symbols.get(result['operation'], result['operation'])}")
    print(f"  Reason     : {result['op_reason']}")
    print(f"  Confidence : {result['confidence']:.1%}")
    print(f"  Part 2     : {'✓ YES' if result['ready_for_part2'] else '✗ NO'}")
    if result["mutation_pos"]:
        print(f"  Mutation @ : position {result['mutation_pos']}")
    if result["repair_template"]:
        print(f"  Repair seq : {result['repair_template']}")
    if result["warning"]:
        print(f"\n  ⚠ {result['warning']}")
    print(f"\n  Top k-mer matches:")
    for i, (disease, score) in enumerate(result["kmer_scores"], 1):
        bar = "█" * int(score * 20)
        print(f"  {i}. {disease:<35} {score:.3f} {bar}")
    print("=" * 55)


# ── RUN TESTS ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nQACS Part 1 — Detection Engine")
    print("Building reference profiles...\n")

    ref_sequences = fetch_reference_sequences()
    ref_profiles  = build_reference_profiles(ref_sequences)
    print(f"\n✓ {len(ref_profiles)} profiles loaded\n")

    tests = [
        ("HBB — expect: sickle cell + CORRECT",   "sickle cell disease"),
        ("CCR5 — expect: HIV + DELETE",            "hiv"),
        ("ADA — expect: SCID + INSERT",            "severe combined immunodeficiency"),
    ]

    for label, disease in tests:
        print(f"\n── {label} ──")
        seq    = ref_sequences.get(disease, "")[:400]
        result = detect_disease(seq, ref_profiles, run_blast=False)
        print_detection_report(result)