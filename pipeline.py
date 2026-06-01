"""
QACS — pipeline.py

WHAT THIS FILE DOES IN PLAIN ENGLISH:
======================================
This is the conductor. It connects every module in order
and runs the full QACS analysis from a single function call.

The flow is:
  1. input_parser      — validate the sequence
  2. detection_engine  — identify disease + assign operation
  3. pam_scanner       — find and filter cut sites
  4. output            — print the complete report

Nothing in this file does any biology itself.
It just calls the other files in the right order and
passes results between them cleanly.

WHAT GOES IN:  A raw DNA or RNA sequence string
WHAT COMES OUT: A complete analysis report printed to screen
                and returned as a dictionary for the app to use
"""

from input_parser      import parse_input, print_parse_report
from detection_engine  import (detect_disease, build_reference_profiles,
                                fetch_reference_sequences, print_detection_report)
from pam_scanner       import scan_sequence, print_pam_report


# ── GLOBAL REFERENCE PROFILES ────────────────────────────────────────────────
# We build these once when the pipeline starts.
# Building them fetches sequences from NCBI — takes ~10 seconds.
# After that they stay in memory and every analysis reuses them.
# This is why the app loads fast after the first startup.

_ref_profiles  = None
_ref_sequences = None


def initialize(verbose: bool = True) -> bool:
    """
    Load reference profiles from NCBI.
    Must be called once before running any analysis.

    Returns True if successful, False if something failed.
    """
    global _ref_profiles, _ref_sequences

    if _ref_profiles is not None:
        return True  # already loaded

    if verbose:
        print("Initializing QACS — loading reference profiles from NCBI...")

    try:
        _ref_sequences = fetch_reference_sequences()
        _ref_profiles  = build_reference_profiles(_ref_sequences)
        if verbose:
            print(f"✓ Ready — {len(_ref_profiles)} disease profiles loaded\n")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


# ── MAIN PIPELINE FUNCTION ────────────────────────────────────────────────────

def run_analysis(raw_sequence: str, verbose: bool = True) -> dict:
    """
    Run the full QACS pipeline on a raw sequence.

    Parameters
    ----------
    raw_sequence : str
        Raw DNA or RNA sequence from the user.
        Can have spaces, lowercase, newlines — we clean it.
    verbose : bool
        If True, print reports at each stage.

    Returns
    -------
    dict with keys:
        success          → True if pipeline completed
        parse_result     → output of input_parser
        detection_result → output of detection_engine
        scan_result      → output of pam_scanner
        error            → error message if something failed
    """

    # Make sure references are loaded
    if _ref_profiles is None:
        success = initialize(verbose=verbose)
        if not success:
            return {"success": False, "error": "Failed to load reference profiles."}

    # ── Stage 1: Validate sequence ────────────────────────────────────────────
    if verbose:
        print("\n" + "─" * 55)
        print("  STAGE 1 — Input Validation")
        print("─" * 55)

    parse_result = parse_input(raw_sequence)

    if verbose:
        print_parse_report(parse_result)

    if not parse_result.is_valid:
        return {
            "success":          False,
            "parse_result":     parse_result,
            "detection_result": None,
            "scan_result":      None,
            "error":            f"Validation failed: {parse_result.errors}"
        }

    # ── Stage 2: Detect disease + assign operation ────────────────────────────
    if verbose:
        print("\n" + "─" * 55)
        print("  STAGE 2 — Disease Detection + Operation Assignment")
        print("─" * 55)

    detection_result = detect_disease(
        parse_result.sequence,
        _ref_profiles,
        run_blast=False
    )

    if verbose:
        print_detection_report(detection_result)

    if not detection_result["ready_for_part2"]:
        return {
            "success":          False,
            "parse_result":     parse_result,
            "detection_result": detection_result,
            "scan_result":      None,
            "error":            f"Low confidence: {detection_result['confidence']:.1%}"
        }

    # ── Stage 3: PAM scan + filter by operation ───────────────────────────────
    if verbose:
        print("\n" + "─" * 55)
        print("  STAGE 3 — Cut Site Analysis")
        print("─" * 55)

    scan_result = scan_sequence(parse_result, detection_result)

    if verbose:
        print_pam_report(scan_result)

    # ── Final summary ─────────────────────────────────────────────────────────
    if verbose:
        _print_summary(detection_result, scan_result)

    return {
        "success":          True,
        "parse_result":     parse_result,
        "detection_result": detection_result,
        "scan_result":      scan_result,
        "error":            None
    }


def _print_summary(detection: dict, scan: dict) -> None:
    """Print the final one-page summary of the analysis."""

    op_symbols = {
        "delete":  "✂  DELETE  — cut and disrupt",
        "correct": "✏  CORRECT — cut and patch",
        "insert":  "➕  INSERT  — cut and add new gene"
    }

    filtered = scan.get("filtered_sites", [])
    top_site = filtered[0] if filtered else None

    print("\n" + "═" * 55)
    print("  QACS — FINAL SUMMARY")
    print("═" * 55)
    print(f"  Disease    : {detection['predicted_disease'].title()}")
    print(f"  Confidence : {detection['confidence']:.1%}")
    print(f"  Operation  : {op_symbols.get(detection['operation'], detection['operation'])}")
    print(f"  Reason     : {detection['op_reason']}")

    if detection.get("mutation_pos"):
        print(f"  Mutation @ : position {detection['mutation_pos']}")
    if detection.get("repair_template"):
        print(f"  Repair seq : {detection['repair_template']}")

    print(f"\n  PAM sites found    : {scan['total_sites']}")
    print(f"  Relevant for op    : {len(filtered)}")

    if top_site:
        print(f"\n  ── Top recommended cut site ──")
        if detection["operation"] == "insert":
            partner = top_site.get("pair_site", {})
            print(f"  Cut A      : pos {top_site['position']} — {top_site['guide']}")
            print(f"  Cut B      : pos {partner.get('position','?')} — {partner.get('guide','?')}")
            print(f"  Gap        : {top_site.get('gap', '?')} bases")
            print(f"  Avg GC     : {top_site.get('avg_gc', 0):.1%}")
        else:
            print(f"  Guide RNA  : {top_site['guide']}")
            print(f"  PAM        : {top_site['pam']}")
            print(f"  Position   : {top_site['position']}")
            print(f"  Strand     : {top_site['strand']}")
            print(f"  GC content : {top_site['gc_content']:.1%}")

    print(f"\n  ⚡ Quantum VQE ranking     — next build")
    print(f"  ⚡ Liquid AI cascade model — next build")
    print("═" * 55)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Test sequences
    HBB = ("ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACCATGGTGCATC"
           "TGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAA"
           "GTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGAC"
           "CAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACT")

    CCR5 = ("ATGGATTATCAAGTGTCAAGTCCAATCTATGACATCAATTATTATGACATCAATGATAAT"
            "CCGATAAATGATAGCGGCGGCAACAATGGCAGCAACAGCAGCAACAGCAGCAACAACAGC"
            "AGCAGCAGCAACAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGC"
            "AGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGC")

    for label, seq in [("HBB — Sickle Cell", HBB), ("CCR5 — HIV", CCR5)]:
        print(f"\n{'═'*55}")
        print(f"  TEST: {label}")
        print(f"{'═'*55}")
        run_analysis(seq)