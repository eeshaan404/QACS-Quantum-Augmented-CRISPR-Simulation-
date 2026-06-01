"""
QACS — pam_scanner.py

WHAT THIS FILE DOES IN PLAIN ENGLISH:
======================================
Finds every location CRISPR could physically cut in the sequence,
then filters them based on the operation the detection engine decided.

Cas9 cannot cut just anywhere. It needs to find NGG — any base
followed by two guanines — right next to the target. These are
called PAM sites. Without NGG, Cas9 cannot land.

This file scans both strands of the DNA for every NGG, then
filters the results based on what the detection engine said:

  DELETE  → all sites ranked by GC quality
  CORRECT → only sites within 10 bases of the mutation position
  INSERT  → pairs of sites with 20-50 base gap between them

The operation and mutation position come from detection_engine.py.
This file does not decide anything — it only finds and filters.

WHAT GOES IN:
  - ParseResult from input_parser.py
  - Detection result from detection_engine.py

WHAT COMES OUT:
  - All PAM sites found
  - Filtered sites appropriate for the operation
"""

from input_parser import parse_input

DELETE  = "delete"
CORRECT = "correct"
INSERT  = "insert"

GUIDE_LENGTH    = 20
MIN_INSERT_GAP  = 20
MAX_INSERT_GAP  = 50
MAX_CORRECT_DIST = 10


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement strand of a DNA sequence."""
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(complement.get(base, base) for base in reversed(sequence))


def find_pam_sites(sequence: str, strand: str = "forward") -> list:
    """
    Walk through the sequence looking for NGG.
    At each NGG, extract the 20 bases before it as the guide RNA.
    Return a list of all candidate cut sites found.
    """
    sites = []
    for i in range(GUIDE_LENGTH, len(sequence) - 2):
        if sequence[i + 1] == "G" and sequence[i + 2] == "G":
            pam   = sequence[i : i + 3]
            guide = sequence[i - GUIDE_LENGTH : i]
            gc    = (guide.count("G") + guide.count("C")) / GUIDE_LENGTH
            sites.append({
                "position":   i,
                "pam":        pam,
                "guide":      guide,
                "strand":     strand,
                "gc_content": round(gc, 3)
            })
    return sites


def filter_by_operation(all_sites: list, operation: str,
                         mutation_pos: int = None) -> list:
    """
    Filter cut sites based on the operation from detection_engine.py.

    DELETE  → return all sites sorted by GC quality
    CORRECT → return only sites near the mutation position
    INSERT  → find pairs of sites with correct gap
    """
    if operation == DELETE:
        return sorted(all_sites, key=lambda s: abs(s["gc_content"] - 0.55))

    elif operation == CORRECT:
        if mutation_pos is None:
            return sorted(all_sites, key=lambda s: abs(s["gc_content"] - 0.55))
        nearby = [s for s in all_sites
                  if abs(s["position"] - mutation_pos) <= MAX_CORRECT_DIST]
        if not nearby:
            # Widen search if nothing found within 10 bases
            nearby = [s for s in all_sites
                      if abs(s["position"] - mutation_pos) <= 30]
        nearby.sort(key=lambda s: abs(s["position"] - mutation_pos))
        return nearby

    elif operation == INSERT:
        pairs = []
        for i, site_a in enumerate(all_sites):
            for site_b in all_sites[i + 1:]:
                gap = abs(site_b["position"] - site_a["position"])
                if MIN_INSERT_GAP <= gap <= MAX_INSERT_GAP:
                    avg_gc = (site_a["gc_content"] + site_b["gc_content"]) / 2
                    pairs.append({
                        "position":   site_a["position"],
                        "pam":        site_a["pam"],
                        "guide":      site_a["guide"],
                        "strand":     site_a["strand"],
                        "gc_content": site_a["gc_content"],
                        "pair_site":  site_b,
                        "gap":        gap,
                        "avg_gc":     round(avg_gc, 3),
                        "gc_score":   abs(avg_gc - 0.55)
                    })
        pairs.sort(key=lambda p: p["gc_score"])
        return pairs

    return all_sites


def scan_sequence(parse_result, detection_result: dict = None) -> dict:
    """
    Scan both strands for PAM sites and filter by operation.

    Parameters
    ----------
    parse_result : ParseResult
        Output of input_parser.parse_input()
    detection_result : dict
        Output of detection_engine.detect_disease()
        Contains operation, mutation_pos, etc.
        If None, defaults to delete with no filtering.
    """
    # Handle dataclass or dict input
    if hasattr(parse_result, "is_valid"):
        is_valid = parse_result.is_valid
        sequence = parse_result.sequence
    else:
        is_valid = parse_result["is_valid"]
        sequence = parse_result["sequence"]

    if not is_valid:
        return {
            "sequence":       "",
            "operation":      DELETE,
            "total_sites":    0,
            "forward_sites":  [],
            "reverse_sites":  [],
            "all_sites":      [],
            "filtered_sites": [],
            "error": "Cannot scan an invalid sequence."
        }

    # Get operation and mutation position from detection result
    if detection_result:
        operation    = detection_result.get("operation",    DELETE)
        mutation_pos = detection_result.get("mutation_pos", None)
    else:
        operation    = DELETE
        mutation_pos = None

    # Scan both strands
    forward_sites = find_pam_sites(sequence, strand="forward")
    reverse_sites = find_pam_sites(reverse_complement(sequence), strand="reverse")
    all_sites     = forward_sites + reverse_sites

    # Filter based on operation from detection engine
    filtered_sites = filter_by_operation(all_sites, operation, mutation_pos)

    return {
        "sequence":       sequence,
        "operation":      operation,
        "total_sites":    len(all_sites),
        "forward_sites":  forward_sites,
        "reverse_sites":  reverse_sites,
        "all_sites":      all_sites,
        "filtered_sites": filtered_sites,
        "error":          None
    }


def print_pam_report(scan_result: dict, top_n: int = 5) -> None:
    print("\n" + "=" * 55)
    print("  QACS — PAM SCAN REPORT")
    print("=" * 55)

    if scan_result["error"]:
        print(f"  ✗ {scan_result['error']}")
        print("=" * 55)
        return

    op = scan_result["operation"]
    print(f"  Operation      : {op.upper()}")
    print(f"  Total PAM sites: {scan_result['total_sites']}")
    print(f"  Forward strand : {len(scan_result['forward_sites'])} sites")
    print(f"  Reverse strand : {len(scan_result['reverse_sites'])} sites")
    print(f"  Filtered sites : {len(scan_result['filtered_sites'])} for {op}")

    filtered = scan_result["filtered_sites"]
    if not filtered:
        print(f"\n  ⚠ No sites match {op} criteria.")
        print("=" * 55)
        return

    print(f"\n  Top {min(top_n, len(filtered))} candidates:")

    if op == INSERT:
        print(f"  {'#':<4} {'Pos A':<8} {'Pos B':<8} {'Gap':<6} {'Avg GC':<8} Guide A")
        print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*22}")
        for idx, site in enumerate(filtered[:top_n], 1):
            partner = site.get("pair_site", {})
            print(f"  {idx:<4} {site['position']:<8} "
                  f"{partner.get('position','?'):<8} {site['gap']:<6} "
                  f"{site['avg_gc']:.1%}   {site['guide']}")
    else:
        print(f"  {'#':<4} {'Strand':<10} {'Pos':<6} {'PAM':<5} {'GC%':<8} Guide")
        print(f"  {'-'*4} {'-'*10} {'-'*6} {'-'*5} {'-'*8} {'-'*22}")
        for idx, site in enumerate(filtered[:top_n], 1):
            print(f"  {idx:<4} {site['strand']:<10} {site['position']:<6} "
                  f"{site['pam']:<5} {site['gc_content']:.1%}    {site['guide']}")

    print("=" * 55)


if __name__ == "__main__":
    HBB = ("ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAAC"
           "GTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTT"
           "TAAGGAGACCAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAG")

    # Simulate what detection engine returns
    delete_detection  = {"operation": DELETE,  "mutation_pos": None}
    correct_detection = {"operation": CORRECT, "mutation_pos": 17}
    insert_detection  = {"operation": INSERT,  "mutation_pos": None}

    for label, detection in [
        ("DELETE",  delete_detection),
        ("CORRECT (mutation @ pos 17)", correct_detection),
        ("INSERT",  insert_detection)
    ]:
        print(f"\n── TEST: {label} ──")
        parsed = parse_input(HBB + HBB)
        scan   = scan_sequence(parsed, detection)
        print_pam_report(scan)