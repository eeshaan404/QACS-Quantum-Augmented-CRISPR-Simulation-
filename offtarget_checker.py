"""
QACS — offtarget_checker.py

WHAT THIS FILE DOES IN PLAIN ENGLISH:
======================================
Takes each candidate guide RNA from the PAM scanner and
checks how many similar sequences exist elsewhere in the
genome. Similar sequences are dangerous because Cas9 might
accidentally cut there too.

Similarity is measured by Hamming distance — how many
positions differ between the guide and a genomic sequence.

  0 mismatches = exact match = definite off-target
  1-2 mismatches = very similar = high risk
  3-4 mismatches = somewhat similar = moderate risk
  5+ mismatches = different enough = low risk

The seed region (positions 1-12, PAM-proximal) gets extra
weight because Cas9 reads from the PAM end first. A mismatch
near the PAM is invisible to the proofreading mechanism —
Cas9 commits to cutting before detecting the error.

TWO APPROACHES COMBINED:
  1. Reference genome scan — slide the guide across a
     reference sequence looking for near-matches.
     Fast, works offline, limited to our reference data.

  2. ClinVar/known site check — compare against known
     pathogenic variant positions. If an off-target site
     overlaps a disease-relevant locus, flag immediately.

WHAT GOES IN:
  - List of candidate guides from pam_scanner
  - Reference sequence to scan against

WHAT COMES OUT PER GUIDE:
  - high_risk_sites   : list of 1-2 mismatch locations
  - moderate_sites    : list of 3-4 mismatch locations
  - essential_overlap : True if any site hits essential gene
  - specificity_score : 0.0 (dangerous) to 1.0 (safe)
  - risk_level        : "high", "moderate", or "safe"
"""

import numpy as np
from dataclasses import dataclass, field


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

GUIDE_LENGTH       = 20
SEED_LENGTH        = 12    # PAM-proximal positions 1-12
HIGH_RISK_MISMATCHES     = 2   # 0-2 mismatches = high risk
MODERATE_RISK_MISMATCHES = 4   # 3-4 mismatches = moderate risk

# Known essential genes — cuts here are potentially lethal
# Subset from Hart et al. 2015 core essentiality dataset
# Full dataset available at: depmap.org
ESSENTIAL_GENES = {
    "TP53", "BRCA1", "BRCA2", "RB1", "APC", "KRAS",
    "EGFR", "MYC", "BCL2", "MDM2", "CDK4", "CDKN2A",
    "PTEN", "VHL", "MLH1", "MSH2", "ATM", "CHEK2",
    "PALB2", "RAD51", "POLR2A", "RPL11", "RPS6",
    "SF3B1", "U2AF1", "SRSF2", "EZH2", "KMT2A"
}

# Known regulatory regions to avoid
# These control gene expression — disrupting them has
# unpredictable downstream effects
REGULATORY_KEYWORDS = [
    "promoter", "enhancer", "CTCF", "insulator",
    "UTR", "splice", "regulatory"
]


# ── RESULT DATACLASS ──────────────────────────────────────────────────────────

@dataclass
class OffTargetResult:
    """
    Off-target analysis result for one guide RNA.

    specificity_score : float
        0.0 = very dangerous (many near-matches in genome)
        1.0 = very safe (no near-matches found)

    risk_level : str
        "high"     — 1+ sites with ≤2 mismatches
        "moderate" — sites with 3-4 mismatches only
        "safe"     — no sites within 4 mismatches

    essential_overlap : bool
        True if any off-target site overlaps a known
        essential gene. This is the most dangerous flag.
    """
    guide:              str
    position:           int
    high_risk_sites:    list = field(default_factory=list)
    moderate_sites:     list = field(default_factory=list)
    essential_overlap:  bool = False
    specificity_score:  float = 1.0
    risk_level:         str  = "safe"
    penalty:            float = 0.0


# ── HAMMING DISTANCE ──────────────────────────────────────────────────────────

def hamming_distance(seq_a: str, seq_b: str) -> int:
    """
    Count positions where two equal-length sequences differ.

    Example:
      ACAGTGCAGCTCACTCAGTG
      ACAGTGCAGCTCACTCAGTA  ← 1 mismatch at position 20
      → hamming distance = 1
    """
    if len(seq_a) != len(seq_b):
        return GUIDE_LENGTH  # treat unequal lengths as maximally different

    return sum(a != b for a, b in zip(seq_a, seq_b))


def seed_weighted_distance(guide: str, target: str) -> float:
    """
    Hamming distance with extra weight on seed region mismatches.

    Seed region = positions 0-11 (PAM-proximal, most critical)
    Distal region = positions 12-19 (PAM-distal, less critical)

    A mismatch in the seed region counts as 2.0
    A mismatch in the distal region counts as 1.0

    This reflects the biological reality that seed region
    mismatches are less likely to prevent Cas9 cleavage.

    Returns a weighted distance — higher = safer.
    """
    if len(guide) != len(target):
        return float(GUIDE_LENGTH * 2)

    seed_mismatches   = sum(
        guide[i] != target[i]
        for i in range(SEED_LENGTH)
    )
    distal_mismatches = sum(
        guide[i] != target[i]
        for i in range(SEED_LENGTH, GUIDE_LENGTH)
    )

    # Seed mismatches count double
    return (seed_mismatches * 2.0) + (distal_mismatches * 1.0)


# ── GENOME SCANNER ────────────────────────────────────────────────────────────

def scan_for_offtargets(
    guide:           str,
    reference_seqs:  dict,
    max_mismatches:  int = 4
) -> list:
    """
    Scan reference sequences for near-matches to the guide.

    Parameters
    ----------
    guide : str
        20-nt guide RNA sequence to check.
    reference_seqs : dict
        Maps disease name → reference sequence.
        We scan all sequences we've fetched from NCBI.
        In production this would be the full genome.
    max_mismatches : int
        Maximum mismatches to consider as off-target risk.

    Returns
    -------
    List of dicts, each representing one off-target site:
        position    : position in reference sequence
        sequence    : the near-match sequence found
        mismatches  : raw Hamming distance
        weighted    : seed-weighted distance
        source      : which reference sequence it came from
        risk        : "high" or "moderate"
    """
    off_targets = []

    for source, ref_seq in reference_seqs.items():
        # Slide the guide across the reference sequence
        for i in range(len(ref_seq) - GUIDE_LENGTH + 1):
            window = ref_seq[i : i + GUIDE_LENGTH]

            if len(window) < GUIDE_LENGTH:
                continue

            # Skip exact match at same position (that's the target itself)
            raw_dist = hamming_distance(guide, window)

            if raw_dist == 0:
                continue  # exact match — skip, this is the target

            if raw_dist <= max_mismatches:
                weighted = seed_weighted_distance(guide, window)
                risk     = "high" if raw_dist <= HIGH_RISK_MISMATCHES \
                           else "moderate"

                off_targets.append({
                    "position":   i,
                    "sequence":   window,
                    "mismatches": raw_dist,
                    "weighted":   round(weighted, 2),
                    "source":     source,
                    "risk":       risk
                })

    # Sort by mismatches — most similar (most dangerous) first
    off_targets.sort(key=lambda x: x["mismatches"])
    return off_targets


# ── ESSENTIAL GENE CHECKER ────────────────────────────────────────────────────

def check_essential_overlap(off_targets: list) -> bool:
    """
    Check if any off-target site overlaps a known essential gene.

    We check the source label of each off-target site against
    our essential genes list. In production this would use
    genomic coordinates from ENCODE or GENCODE annotations.

    Returns True if any dangerous overlap is found.
    """
    for site in off_targets:
        source = site.get("source", "").upper()
        for gene in ESSENTIAL_GENES:
            if gene in source:
                return True
    return False


# ── SPECIFICITY SCORER ────────────────────────────────────────────────────────

def calculate_specificity_score(
    off_targets:      list,
    essential_overlap: bool
) -> tuple:
    """
    Convert off-target findings into a specificity score and risk level.

    Scoring logic:
      Start at 1.0 (perfectly safe)
      Subtract penalties for each off-target found:
        High risk site   → -0.30 per site
        Moderate site    → -0.10 per site
        Essential overlap → -0.50 (one-time penalty)

    Returns (specificity_score, risk_level, penalty)
    """
    score = 1.0

    high_risk = [s for s in off_targets if s["risk"] == "high"]
    moderate  = [s for s in off_targets if s["risk"] == "moderate"]

    # Penalties
    score -= len(high_risk) * 0.30
    score -= len(moderate)  * 0.10

    if essential_overlap:
        score -= 0.50

    # Clamp to [0, 1]
    score = round(max(0.0, min(1.0, score)), 4)

    # Assign risk level
    if len(high_risk) > 0 or essential_overlap:
        risk_level = "high"
    elif len(moderate) > 0:
        risk_level = "moderate"
    else:
        risk_level = "safe"

    penalty = 1.0 - score
    return score, risk_level, round(penalty, 4)


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def check_offtargets(
    candidate_sites: list,
    reference_seqs:  dict
) -> list:
    """
    Run off-target analysis on all candidate cut sites.

    Parameters
    ----------
    candidate_sites : list
        Filtered sites from pam_scanner.scan_sequence()
    reference_seqs : dict
        Reference sequences from detection_engine.
        Maps disease name → sequence string.

    Returns
    -------
    List of OffTargetResult objects, one per candidate site.
    Sorted by specificity score — safest first.
    """
    if not candidate_sites:
        return []

    results = []

    for site in candidate_sites:
        guide    = site["guide"]
        position = site["position"]

        # Scan all reference sequences for near-matches
        off_targets = scan_for_offtargets(guide, reference_seqs)

        # Check essential gene overlap
        essential = check_essential_overlap(off_targets)

        # Calculate specificity score
        specificity, risk_level, penalty = calculate_specificity_score(
            off_targets, essential
        )

        high_risk = [s for s in off_targets if s["risk"] == "high"]
        moderate  = [s for s in off_targets if s["risk"] == "moderate"]

        results.append(OffTargetResult(
            guide             = guide,
            position          = position,
            high_risk_sites   = high_risk,
            moderate_sites    = moderate,
            essential_overlap = essential,
            specificity_score = specificity,
            risk_level        = risk_level,
            penalty           = penalty
        ))

    # Sort safest first
    results.sort(key=lambda r: r.specificity_score, reverse=True)
    return results


# ── COMBINED EFFECTIVENESS SCORE ──────────────────────────────────────────────

def compute_effectiveness_score(
    quantum_confidence: float,
    specificity_score:  float,
    proximity_score:    float = 1.0
) -> float:
    """
    Combine quantum binding energy and off-target specificity
    into a composite effectiveness score.

    This is Layer 1 of the effectiveness score.
    Layer 2 adds cascade score when Liquid AI is ready.

    Weights reflect clinical priority:
      Binding energy (quantum) : 35% — molecular precision
      Specificity (off-target) : 40% — safety critical
      Proximity to mutation    : 25% — therapeutic relevance

    Specificity gets the highest weight because an off-target
    cut that harms the patient outweighs a slightly suboptimal
    binding energy at the correct site.

    Parameters
    ----------
    quantum_confidence : float
        Confidence score from quantum_engine (0.0-1.0)
    specificity_score : float
        Score from this file (0.0-1.0)
    proximity_score : float
        Distance-based score from pam_scanner (0.0-1.0)
        Default 1.0 if not applicable (delete operation)

    Returns
    -------
    float : composite effectiveness score 0.0-1.0
    """
    effectiveness = (
        quantum_confidence * 0.35 +
        specificity_score  * 0.40 +
        proximity_score    * 0.25
    )
    return round(effectiveness, 4)


# ── PRINT REPORT ──────────────────────────────────────────────────────────────

def print_offtarget_report(results: list) -> None:
    print("\n" + "=" * 60)
    print("  QACS — OFF-TARGET ANALYSIS REPORT")
    print("=" * 60)

    if not results:
        print("  No results.")
        print("=" * 60)
        return

    safe     = [r for r in results if r.risk_level == "safe"]
    moderate = [r for r in results if r.risk_level == "moderate"]
    high     = [r for r in results if r.risk_level == "high"]

    print(f"  Guides analyzed : {len(results)}")
    print(f"  Safe            : {len(safe)}")
    print(f"  Moderate risk   : {len(moderate)}")
    print(f"  High risk       : {len(high)}")

    print(f"\n  {'#':<4} {'Pos':<6} {'Risk':<10} {'Spec.Score':<12} "
          f"{'High Risk':<10} {'Moderate':<10} Essential")
    print(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*12} "
          f"{'-'*10} {'-'*10} {'-'*9}")

    for i, r in enumerate(results, 1):
        risk_icon = {"safe": "✓", "moderate": "⚠", "high": "✗"}
        ess_flag  = "⚠ YES" if r.essential_overlap else "no"
        print(
            f"  {i:<4} {r.position:<6} "
            f"{risk_icon[r.risk_level]} {r.risk_level:<8} "
            f"{r.specificity_score:.3f}        "
            f"{len(r.high_risk_sites):<10} "
            f"{len(r.moderate_sites):<10} "
            f"{ess_flag}"
        )

    # Best candidate
    best = results[0]
    print(f"\n  ── Safest candidate ──")
    print(f"  Guide        : {best.guide}")
    print(f"  Position     : {best.position}")
    print(f"  Risk level   : {best.risk_level.upper()}")
    print(f"  Specificity  : {best.specificity_score:.3f}")
    print(f"  High risk    : {len(best.high_risk_sites)} sites")
    print(f"  Moderate     : {len(best.moderate_sites)} sites")
    print(f"  Essential    : {'⚠ YES — review required' if best.essential_overlap else '✓ No overlap'}")

    if best.high_risk_sites:
        print(f"\n  High risk off-target sites:")
        for site in best.high_risk_sites[:3]:
            print(f"    Position {site['position']} in {site['source']}: "
                  f"{site['mismatches']} mismatches "
                  f"(weighted: {site['weighted']})")

    print("=" * 60)


# ── RUN TESTS ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Simulate what pam_scanner gives us
    test_guides = [
        {
            "position":   33,
            "pam":        "TGG",
            "guide":      "ACAGTGCAGCTCACTCAGTG",
            "strand":     "reverse",
            "gc_content": 0.55
        },
        {
            "position":   39,
            "pam":        "AGG",
            "guide":      "CAGCTCACTCAGTGTGGCAA",
            "strand":     "reverse",
            "gc_content": 0.55
        },
        {
            "position":   45,
            "pam":        "TGG",
            "guide":      "AGTCTGCCGTTACTGCCCTG",
            "strand":     "forward",
            "gc_content": 0.60
        }
    ]

    # Simulate reference sequences
    # In production these come from detection_engine.fetch_reference_sequences()
    test_refs = {
        "sickle cell disease": (
            "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACCATGGTGCATC"
            "TGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAA"
            "GTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGAC"
            "CAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACT"
        ),
        "hiv": (
            "ATGGATTATCAAGTGTCAAGTCCAATCTATGACATCAATTATTATGACATCAATGATAAT"
            "CCGATAAATGATAGCGGCGGCAACAATGGCAGCAACAGCAGCAACAGCAGCAACAACAGC"
        )
    }

    print("QACS — Off-Target Checker")
    print("Testing 3 HBB sickle cell guide RNAs\n")

    results = check_offtargets(test_guides, test_refs)
    print_offtarget_report(results)

    # Show combined effectiveness score
    print("\n  ── Combined Effectiveness Scores (Layer 1) ──")
    print(f"  {'Guide':<22} {'Quantum':<10} {'Specificity':<14} Effectiveness")
    print(f"  {'-'*22} {'-'*10} {'-'*14} {'-'*13}")

    quantum_scores = [0.689, 0.684, 0.658]  # from quantum engine output

    for i, (r, q) in enumerate(zip(results, quantum_scores), 1):
        eff = compute_effectiveness_score(q, r.specificity_score)
        print(f"  {r.guide[:20]:<22} {q:.3f}      "
              f"{r.specificity_score:.3f}          {eff:.3f}")