"""
QACS — input_parser.py

WHAT THIS FILE DOES IN PLAIN ENGLISH:
======================================
This is the front door of QACS. Its only job is to check
that the sequence is real and usable before anything else
touches it.

It does NOT decide what operation to perform — that is
the detection engine's job. It only answers:
  - Is this a valid DNA or RNA sequence?
  - How long is it?
  - What is its GC content?

Think of it like a security guard who checks your ID.
The guard doesn't decide what you do inside the building.
They just verify you are who you say you are.

WHAT GOES IN:  A raw DNA or RNA string from the user
WHAT COMES OUT:
  - is_valid    → True or False
  - sequence    → cleaned uppercase sequence
  - seq_type    → "DNA" or "RNA"
  - length      → number of bases
  - gc_content  → fraction of G and C (0.0 to 1.0)
  - window_gc   → GC per 20-base sliding window
  - errors      → problems that stop the pipeline
  - warnings    → issues that reduce confidence
"""

from dataclasses import dataclass, field

VALID_DNA_BASES = set("ATGC")
VALID_RNA_BASES = set("AUGC")
AMBIGUOUS_BASES = set("NRYSWKMBDHV")
MIN_LENGTH      = 20
WINDOW_SIZE     = 20


@dataclass
class ParseResult:
    """
    Bundles everything the parser learned about a sequence.
    Every downstream module reads from this object by name.
    Example: result.sequence, result.gc_content, result.is_valid
    """
    is_valid:   bool
    sequence:   str
    seq_type:   str
    length:     int
    gc_content: float
    window_gc:  list = field(default_factory=list)
    errors:     list = field(default_factory=list)
    warnings:   list = field(default_factory=list)


def detect_sequence_type(sequence: str) -> str:
    """
    DNA uses T. RNA uses U. Never both.
    Look for which one is present and return the label.
    """
    has_t = "T" in sequence
    has_u = "U" in sequence
    if has_t and not has_u:
        return "DNA"
    elif has_u and not has_t:
        return "RNA"
    else:
        return "UNKNOWN"


def parse_input(raw_sequence: str) -> ParseResult:
    """
    Validate a raw DNA or RNA sequence.

    Parameters
    ----------
    raw_sequence : str
        Whatever the user pasted in. Can have spaces,
        lowercase, newlines — we clean it first.

    Returns
    -------
    ParseResult object. Check .is_valid first.
    If False, read .errors. If True, everything is ready.
    """
    errors   = []
    warnings = []

    # Clean: strip whitespace, uppercase, remove spaces/newlines
    sequence = raw_sequence.strip().upper()
    sequence = sequence.replace(" ", "").replace("\n", "").replace("\r", "")

    # Empty check
    if not sequence:
        return ParseResult(is_valid=False, sequence="", seq_type="UNKNOWN",
                           length=0, gc_content=0.0,
                           errors=["Input sequence is empty."])

    # Detect DNA vs RNA
    seq_type = detect_sequence_type(sequence)
    if seq_type == "UNKNOWN":
        errors.append(
            "Cannot determine if this is DNA or RNA. "
            "DNA must contain T (not U). RNA must contain U (not T)."
        )

    # Character audit
    # Sort every character into: valid, ambiguous, or invalid
    valid_set       = VALID_DNA_BASES if seq_type == "DNA" else VALID_RNA_BASES
    invalid_chars   = []
    ambiguous_found = []

    for char in sequence:
        if char in valid_set:
            continue
        elif char in AMBIGUOUS_BASES:
            if char not in ambiguous_found:
                ambiguous_found.append(char)
        else:
            if char not in invalid_chars:
                invalid_chars.append(char)

    if invalid_chars:
        errors.append(
            f"Invalid characters found: {', '.join(invalid_chars)}. "
            f"Only standard nucleotide letters are allowed."
        )

    if ambiguous_found:
        warnings.append(
            f"Ambiguous bases detected: {', '.join(ambiguous_found)}. "
            f"Lower confidence near these positions."
        )

    # Length check — need at least 20 bases for one guide RNA
    if len(sequence) < MIN_LENGTH:
        errors.append(
            f"Sequence too short: {len(sequence)} bases. "
            f"Minimum {MIN_LENGTH} bases required."
        )

    # Stop here if errors found
    if errors:
        return ParseResult(is_valid=False, sequence=sequence,
                           seq_type=seq_type, length=len(sequence),
                           gc_content=0.0, errors=errors, warnings=warnings)

    # GC content
    gc_content = (sequence.count("G") + sequence.count("C")) / len(sequence)

    if gc_content < 0.4:
        warnings.append(f"Low GC content ({gc_content:.1%}). Ideal: 40-70%.")
    elif gc_content > 0.7:
        warnings.append(f"High GC content ({gc_content:.1%}). Ideal: 40-70%.")

    # Sliding window GC
    window_gc = []
    for i in range(len(sequence) - WINDOW_SIZE + 1):
        window = sequence[i : i + WINDOW_SIZE]
        window_gc.append(
            (window.count("G") + window.count("C")) / WINDOW_SIZE
        )

    return ParseResult(is_valid=True, sequence=sequence, seq_type=seq_type,
                       length=len(sequence), gc_content=gc_content,
                       window_gc=window_gc, errors=[], warnings=warnings)


# Backwards compatible alias so existing files don't break
def validate_dna(raw_sequence: str) -> ParseResult:
    return parse_input(raw_sequence)


def print_parse_report(result: ParseResult) -> None:
    print("\n" + "=" * 55)
    print("  QACS — INPUT PARSER REPORT")
    print("=" * 55)
    print(f"  Status     : {'✓ VALID' if result.is_valid else '✗ INVALID'}")
    print(f"  Type       : {result.seq_type}")
    print(f"  Length     : {result.length} bases")
    if result.is_valid:
        print(f"  GC content : {result.gc_content:.1%}")
        print(f"  Windows    : {len(result.window_gc)} x {WINDOW_SIZE}-nt")
    if result.errors:
        print("\n  ERRORS:")
        for e in result.errors:
            print(f"    ✗ {e}")
    if result.warnings:
        print("\n  WARNINGS:")
        for w in result.warnings:
            print(f"    ⚠ {w}")
    print("=" * 55)


if __name__ == "__main__":
    tests = [
        ("Valid DNA — HBB sickle cell",
         "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAAC"),
        ("Valid RNA sequence",
         "AUGGUGCACCUGACUCCUGAGGAGAAGUCUGCCGUUACUGCCCUGUGGGGCAAGGUGAAC"),
        ("Too short",
         "ATGCAT"),
        ("Invalid characters",
         "ATGCXYZ123ATGCATGCATGC"),
        ("Empty input",
         ""),
    ]

    for label, seq in tests:
        print(f"\n▶  {label}")
        result = parse_input(seq)
        print_parse_report(result)