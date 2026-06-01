"""
QACS — app.py

WHAT THIS FILE DOES IN PLAIN ENGLISH:
======================================
Opens a desktop window where a user can paste a DNA or RNA
sequence and run the full QACS pipeline with one click.

The window has two panels:
  TOP    — paste your sequence here
  BOTTOM — results appear here after clicking Run

On startup it loads reference profiles from NCBI in the
background so the user doesn't wait when they click Run.

The analysis runs in a background thread so the window
never freezes during processing.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from pipeline import initialize, run_analysis, _ref_profiles

BG_DARK    = "#1E1E1E"
BG_PANEL   = "#2C2C2C"
BG_INPUT   = "#3A3A3A"
TEAL       = "#1D9E75"
TEAL_DARK  = "#0F6E56"
TEXT_WHITE = "#F0EEE8"
TEXT_GRAY  = "#9A9890"
TEXT_GREEN = "#4CAF50"
TEXT_RED   = "#EF5350"
TEXT_AMBER = "#EF9F27"

FONT_TITLE  = ("Arial", 18, "bold")
FONT_LABEL  = ("Arial", 11)
FONT_MONO   = ("Courier New", 11)
FONT_RESULT = ("Courier New", 10)
FONT_SMALL  = ("Arial", 9)

# Example sequences for the dropdown
EXAMPLES = {
    "HBB — Sickle Cell": (
    "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACCATGGTGCATC"
    "TGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAA"
    "GTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGAC"
    "CAATAGAAACTGGGCATGTGGAGACAGAGAAGACTCTTGGGTTTCTGATAGGCACTGACT"
    "CTCTCTGCCTATTGGTCTATTTTCCCACCCTAGGCTGCTGGTGGTCTACCCTTGGACCC"
    "AGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCT"
    "AAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCT"
    "GGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACG"
    ),
    "CCR5 — HIV": (
        "ATGGATTATCAAGTGTCAAGTCCAATCTATGACATCAATTATTATGACATCAATGATAAT"
        "CCGATAAATGATAGCGGCGGCAACAATGGCAGCAACAGCAGCAACAGCAGCAACAACAGC"
        "AGCAGCAGCAACAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGC"
        "AGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGC"
    ),
    "KRAS — Colorectal Cancer": (
        "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGAC"
        "GATACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCT"
        "ACAGGAAGCAAGTAGTAATTGATGGAGAAACCTGTCTCTTGGATATTCTCGACACAGCA"
        "GGTCAAGAGGAGTACAGTGCAATGAGGGACCAGTACATGAGGACTGGGGAGGGCTTTCT"
    ),
    "Random (should fail)": (
        "TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
        "CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
    )
}


class QACSApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("QACS — Quantum CRISPR Analysis System")
        self.root.geometry("860x720")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        self._build_header()
        self._build_input_panel()
        self._build_buttons()
        self._build_output_panel()
        self._build_status_bar()
        self._load_references_async()


    def _build_header(self):
        header = tk.Frame(self.root, bg=TEAL_DARK, pady=12)
        header.pack(fill=tk.X)
        tk.Label(header, text="QACS", font=("Arial", 22, "bold"),
                 fg=TEXT_WHITE, bg=TEAL_DARK).pack(side=tk.LEFT, padx=20)
        tk.Label(header, text="Quantum-Augmented CRISPR Simulation System",
                 font=FONT_LABEL, fg="#A8D8C8", bg=TEAL_DARK).pack(side=tk.LEFT, padx=4)
        tk.Label(header, text="v0.2  |  Detection + Operation Assignment",
                 font=FONT_SMALL, fg="#A8D8C8", bg=TEAL_DARK).pack(side=tk.RIGHT, padx=20)


    def _build_input_panel(self):
        frame = tk.Frame(self.root, bg=BG_DARK, pady=12)
        frame.pack(fill=tk.X, padx=20)
        tk.Label(frame, text="Paste DNA or RNA sequence:",
                 font=FONT_LABEL, fg=TEXT_GRAY, bg=BG_DARK).pack(anchor=tk.W)
        self.seq_input = tk.Text(
            frame, height=6, font=FONT_MONO, bg=BG_INPUT, fg=TEXT_WHITE,
            insertbackground=TEAL, relief=tk.FLAT, padx=10, pady=8, wrap=tk.WORD
        )
        self.seq_input.pack(fill=tk.X, pady=(6, 0))
        self._set_placeholder()


    def _build_buttons(self):
        frame = tk.Frame(self.root, bg=BG_DARK, pady=10)
        frame.pack(fill=tk.X, padx=20)

        self.run_btn = tk.Button(
            frame, text="▶  Run QACS Analysis",
            font=("Arial", 11, "bold"), bg=TEAL, fg=TEXT_WHITE,
            activebackground=TEAL_DARK, activeforeground=TEXT_WHITE,
            relief=tk.FLAT, padx=20, pady=8, cursor="hand2",
            command=self._run_analysis
        )
        self.run_btn.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            frame, text="✕  Clear", font=FONT_LABEL,
            bg=BG_PANEL, fg=TEXT_GRAY, activebackground=BG_INPUT,
            activeforeground=TEXT_WHITE, relief=tk.FLAT,
            padx=16, pady=8, cursor="hand2", command=self._clear
        ).pack(side=tk.LEFT)

        tk.Label(frame, text="Try an example:", font=FONT_SMALL,
                 fg=TEXT_GRAY, bg=BG_DARK).pack(side=tk.LEFT, padx=(20, 6))

        self.example_var = tk.StringVar(value="Select...")
        example_menu = ttk.Combobox(
            frame, textvariable=self.example_var,
            values=list(EXAMPLES.keys()), width=28, state="readonly"
        )
        example_menu.pack(side=tk.LEFT)
        example_menu.bind("<<ComboboxSelected>>", self._load_example)


    def _build_output_panel(self):
        tk.Label(self.root, text="RESULTS", font=("Arial", 10, "bold"),
                 fg=TEXT_GRAY, bg=BG_DARK).pack(anchor=tk.W, padx=20)
        tk.Frame(self.root, bg=BG_PANEL, height=1).pack(fill=tk.X, padx=20)

        self.output = scrolledtext.ScrolledText(
            self.root, font=FONT_RESULT, bg=BG_PANEL, fg=TEXT_WHITE,
            insertbackground=TEAL, relief=tk.FLAT,
            padx=14, pady=10, wrap=tk.WORD, state=tk.DISABLED
        )
        self.output.pack(fill=tk.BOTH, expand=True, padx=20, pady=(4, 0))

        self.output.tag_configure("title",   foreground=TEAL, font=("Courier New", 11, "bold"))
        self.output.tag_configure("success", foreground=TEXT_GREEN)
        self.output.tag_configure("warning", foreground=TEXT_AMBER)
        self.output.tag_configure("error",   foreground=TEXT_RED)
        self.output.tag_configure("label",   foreground=TEXT_GRAY)
        self.output.tag_configure("value",   foreground=TEXT_WHITE)
        self.output.tag_configure("mono",    font=("Courier New", 10))


    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Loading reference profiles...")
        bar = tk.Frame(self.root, bg=BG_PANEL, pady=4)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(bar, textvariable=self.status_var, font=FONT_SMALL,
                 fg=TEXT_GRAY, bg=BG_PANEL).pack(side=tk.LEFT, padx=14)


    def _set_placeholder(self):
        self.seq_input.insert("1.0", "Paste your DNA or RNA sequence here...")
        self.seq_input.configure(fg="#666660")
        self.seq_input.bind("<FocusIn>",  self._clear_placeholder)
        self.seq_input.bind("<FocusOut>", self._restore_placeholder)

    def _clear_placeholder(self, event=None):
        if self.seq_input.get("1.0", tk.END).strip() == \
                "Paste your DNA or RNA sequence here...":
            self.seq_input.delete("1.0", tk.END)
            self.seq_input.configure(fg=TEXT_WHITE)

    def _restore_placeholder(self, event=None):
        if not self.seq_input.get("1.0", tk.END).strip():
            self.seq_input.insert("1.0", "Paste your DNA or RNA sequence here...")
            self.seq_input.configure(fg="#666660")


    def _load_example(self, event=None):
        selected = self.example_var.get()
        sequence = EXAMPLES.get(selected, "")
        if sequence:
            self.seq_input.configure(fg=TEXT_WHITE)
            self.seq_input.delete("1.0", tk.END)
            self.seq_input.insert("1.0", sequence)


    def _load_references_async(self):
        def load():
            self._set_status("Loading reference profiles from NCBI...")
            success = initialize(verbose=False)
            if success:
                from pipeline import _ref_profiles
                self._set_status(
                    f"✓ Ready — {len(_ref_profiles)} disease profiles loaded"
                )
                self._write("QACS ready. Paste a sequence and click Run.\n\n",
                            "label")
                self._write(
                    "What changed in v0.2:\n"
                    "  ✓ Operation (delete/correct/insert) is now determined\n"
                    "    automatically by the detection engine based on biology\n"
                    "  ✓ Cut sites are filtered based on the operation\n"
                    "  ✓ Correct → sites near mutation only\n"
                    "  ✓ Insert  → pairs of sites with correct gap\n"
                    "  ✓ Delete  → all sites ranked by GC quality\n\n",
                    "label"
                )
            else:
                self._set_status("✗ Failed to load profiles. Check internet connection.")

        threading.Thread(target=load, daemon=True).start()


    def _run_analysis(self):
        raw = self.seq_input.get("1.0", tk.END).strip()
        if not raw or raw == "Paste your DNA or RNA sequence here...":
            messagebox.showwarning("No Input", "Please paste a sequence first.")
            return

        self.run_btn.configure(state=tk.DISABLED, text="Analyzing...")
        self._clear_output()
        self._set_status("Running analysis...")

        def run():
            try:
                self._run_pipeline(raw)
            finally:
                self.run_btn.configure(state=tk.NORMAL,
                                       text="▶  Run QACS Analysis")
                self._set_status("Analysis complete.")

        threading.Thread(target=run, daemon=True).start()


    def _run_pipeline(self, raw_sequence: str):
        from pipeline import _ref_profiles
        from input_parser     import parse_input
        from detection_engine import detect_disease
        from pam_scanner      import scan_sequence

        self._write("═" * 52 + "\n", "title")
        self._write("  QACS ANALYSIS REPORT\n", "title")
        self._write("═" * 52 + "\n", "title")

        # Stage 1
        self._write("\n● Stage 1 — Sequence Validation\n", "label")
        self._set_status("Stage 1: Validating...")
        parsed = parse_input(raw_sequence)

        if not parsed.is_valid:
            self._write("  ✗ Validation failed:\n", "error")
            for e in parsed.errors:
                self._write(f"    {e}\n", "error")
            return

        self._write(f"  ✓ Type     : {parsed.seq_type}\n", "success")
        self._write(f"  ✓ Length   : {parsed.length} bases\n", "success")
        self._write(f"  ✓ GC       : {parsed.gc_content:.1%}\n", "success")
        for w in parsed.warnings:
            self._write(f"  ⚠ {w}\n", "warning")

        # Stage 2
        self._write("\n● Stage 2 — Disease Detection + Operation\n", "label")
        self._set_status("Stage 2: Detecting disease...")
        detection = detect_disease(parsed.sequence, _ref_profiles, run_blast=False)

        op_symbols = {"delete": "✂ DELETE", "correct": "✏ CORRECT",
                      "insert": "➕ INSERT"}
        conf_tag = ("success" if detection["confidence"] >= 0.75
                    else "warning" if detection["confidence"] >= 0.6
                    else "error")

        self._write(f"  Disease    : {detection['predicted_disease'].title()}\n", "value")
        self._write(f"  Operation  : {op_symbols.get(detection['operation'])}\n", "value")
        self._write(f"  Reason     : {detection['op_reason']}\n", "label")
        self._write(f"  Confidence : {detection['confidence']:.1%}\n", conf_tag)

        if detection.get("mutation_pos"):
            self._write(f"  Mutation @ : position {detection['mutation_pos']}\n", "value")
        if detection.get("repair_template"):
            self._write(f"  Repair seq : {detection['repair_template']}\n", "value")
        if detection["warning"]:
            self._write(f"  ⚠ {detection['warning']}\n", "warning")

        if not detection["ready_for_part2"]:
            self._write("\n  ✗ Confidence too low to proceed.\n", "error")
            return

        self._write("\n  Top matches:\n", "label")
        for i, (disease, score) in enumerate(detection["kmer_scores"], 1):
            bar = "█" * int(score * 20)
            self._write(f"  {i}. {disease:<35} {score:.3f} {bar}\n", "mono")

        # Stage 3
        self._write("\n● Stage 3 — Cut Site Analysis\n", "label")
        self._set_status("Stage 3: Scanning for PAM sites...")
        scan = scan_sequence(parsed, detection)

        self._write(f"  Total PAM sites   : {scan['total_sites']}\n", "value")
        self._write(f"  Relevant for op   : {len(scan['filtered_sites'])}\n", "value")

        filtered = scan["filtered_sites"]
        self._write(f"\n  Top 5 candidates for {detection['operation'].upper()}:\n",
                    "label")

        if detection["operation"] == "insert":
            self._write(
                f"  {'#':<4} {'Pos A':<8} {'Pos B':<8} {'Gap':<6} {'Avg GC':<8} Guide A\n",
                "mono"
            )
            self._write("  " + "-" * 50 + "\n", "mono")
            for idx, site in enumerate(filtered[:5], 1):
                partner = site.get("pair_site", {})
                self._write(
                    f"  {idx:<4} {site['position']:<8} "
                    f"{partner.get('position','?'):<8} {site['gap']:<6} "
                    f"{site['avg_gc']:.1%}   {site['guide']}\n", "mono"
                )
        else:
            self._write(
                f"  {'#':<4} {'Strand':<10} {'Pos':<6} {'PAM':<5} {'GC%':<8} Guide\n",
                "mono"
            )
            self._write("  " + "-" * 50 + "\n", "mono")
            for idx, site in enumerate(filtered[:5], 1):
                self._write(
                    f"  {idx:<4} {site['strand']:<10} {site['position']:<6} "
                    f"{site['pam']:<5} {site['gc_content']:.1%}    "
                    f"{site['guide']}\n", "mono"
                )

        # Summary
        top = filtered[0] if filtered else None
        self._write("\n" + "═" * 52 + "\n", "title")
        self._write("  SUMMARY\n", "title")
        self._write("═" * 52 + "\n", "title")
        self._write(f"  Disease    : {detection['predicted_disease'].title()}\n", "value")
        self._write(f"  Operation  : {op_symbols.get(detection['operation'])}\n", "value")
        self._write(f"  Confidence : {detection['confidence']:.1%}\n", conf_tag)

        if top:
            if detection["operation"] == "insert":
                partner = top.get("pair_site", {})
                self._write(f"  Cut A      : {top['guide']}\n", "success")
                self._write(f"  Cut B      : {partner.get('guide','?')}\n", "success")
                self._write(f"  Gap        : {top.get('gap','?')} bases\n", "success")
            else:
                self._write(f"  Best guide : {top['guide']}\n", "success")
                self._write(f"  PAM        : {top['pam']}\n", "success")
                self._write(f"  Position   : {top['position']}\n", "success")
                self._write(f"  GC content : {top['gc_content']:.1%}\n", "success")

        self._write("\n  ⚡ Quantum VQE ranking     — next build\n", "warning")
        self._write("  ⚡ Liquid AI cascade model — next build\n", "warning")
        self._write("═" * 52 + "\n", "title")


    def _write(self, text: str, tag: str = "value"):
        def w():
            self.output.configure(state=tk.NORMAL)
            self.output.insert(tk.END, text, tag)
            self.output.see(tk.END)
            self.output.configure(state=tk.DISABLED)
        self.root.after(0, w)

    def _clear_output(self):
        def c():
            self.output.configure(state=tk.NORMAL)
            self.output.delete("1.0", tk.END)
            self.output.configure(state=tk.DISABLED)
        self.root.after(0, c)

    def _clear(self):
        self.seq_input.delete("1.0", tk.END)
        self._set_placeholder()
        self._clear_output()

    def _set_status(self, message: str):
        self.root.after(0, lambda: self.status_var.set(message))


if __name__ == "__main__":
    root = tk.Tk()
    app  = QACSApp(root)
    root.mainloop()