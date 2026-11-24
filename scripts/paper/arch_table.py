"""arch_table - the paper's long table of the ARCH-COMP26 times: every tool of the reports and CORA.cpp

One row per ARCH-COMP26 instance that CORA.cpp runs, grouped by category (AFF, NLN, AINNCS); one
column per tool with a time in a category report, then CORA.cpp (CORA++ in the paper) and the
speedup = (the fastest counted time of another tool) / (CORA.cpp's time), two significant digits.
A time counts as the README's best-tool column does: the expected verdict, on the report's
problem. The fastest counted time of a row is bold.

Syntax:   python scripts/paper/arch_table.py [--check]
Reads:    examples/benchmarks/README.md (CORA.cpp's verdicts and times, the main tables of the
          sections AFF, NLN and AINNCS results; never copied by hand) and
          examples/benchmarks/data/arch26_tool_times.csv (the other tools' times, as the reports
          print them, with the report table each comes from and why a time does not count)
Writes:   paper/generated/arch-comp-table.tex (copied into the paper repository as it is)
Checks:   --check renders the table and compares it with the committed file, writes nothing
Exit code: 0 if the file is written or up to date, else 1 with what is wrong.

Authors:       Claude Opus 5.5
Written:       24-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed file, write nothing")
    args = parser.parse_args()
    readme = parse_readme((REPO / README).read_text(encoding="utf-8"))
    times = load_times(REPO / TIMES)
    problems = consistency(readme, times)
    if problems:
        print("\n".join(problems))
        return 1
    rows = [table_row(category, label, readme[category][name], times.get(label, {}))
            for category, label, name, _ in ROWS]
    text = render(rows, readme)
    # the medians, next to the README's summary, so that a reader of the log sees both
    for category in CATEGORIES:
        values = [r["speedup"] for r in rows if r["category"] == category and r["speedup"] and not r["indicative"]]
        print(f"{category}: median speedup {fmt_speedup(statistics.median(values))} over {len(values)} "
              f"instances (README summary: {readme['summary'].get(category, '?')})")
    out = REPO / OUTPUT
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"  {OUTPUT} is stale; run python3 scripts/paper/arch_table.py")
            return 1
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"arch_table: wrote {OUTPUT}")
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

REPO = Path(__file__).resolve().parents[2]

README = "examples/benchmarks/README.md"

TIMES = "examples/benchmarks/data/arch26_tool_times.csv"

OUTPUT = "paper/generated/arch-comp-table.tex"

CATEGORIES = ("AFF", "NLN", "AINNCS")

TITLES = {"AFF": "Linear continuous and hybrid dynamics (AFF)",
          "NLN": "Nonlinear continuous and hybrid dynamics (NLN)",
          "AINNCS": "Neural-network control systems (AINNCS)"}

# the tool columns: the name in the CSV and the header (the caption explains the short ones)
TOOLS = (("CORA", "CORA"), ("JuliaReach", "JR"), ("JuliaReach discrete-time", r"JR\textsubscript{d}"),
         ("Ariadne", "Ari"), ("DynIbex", "DI"), ("KeYmaera X", "KX"), ("PRoTECT", "PR"),
         ("CROWN-Reach", "CR"), ("immrax", "imx"))

# why a time does not count, as the CSV's `note` says it, and its mark in the table
MARKS = {"other-problem": r"\dag", "code-differs": r"\S", "verdict": r"\ddag"}

# the rows: category, label in the table (and in the CSV), the row of the README table, the note letters
ROWS = ([("AFF", name.replace("_", "-"), name, note) for name, note in (
    ("HEAT01", ""), ("HEAT02", ""), ("HEAT03", ""), ("HEAT04", "a"), ("CBC01", ""), ("CBC02", ""), ("CBC03", ""),
    ("CBF01", ""), ("CBF02", ""), ("CBF03", ""), ("RND01", "b"), ("RND02", "b"), ("RND03", "b"), ("RND04", "b"),
    ("RND05", "b"), ("RND06", "b"), ("SRNA01", ""), ("SRA01", ""), ("SRA02", ""), ("SRA03", ""), ("SRA04", ""),
    ("SRA05", ""), ("SRA06", ""), ("SRA07", ""), ("SRA08", ""), ("SRU01", ""), ("SRU02", ""), ("DTN01", ""),
    ("DTN02", ""), ("DTN03", ""), ("DTN04", ""), ("DTN05", ""), ("DTN06", ""), ("PLAA01_BND42", ""),
    ("PLAA01_BND50", ""), ("PLAD01_BND30", ""), ("PLAD01_BND42", ""), ("PLAN01", "c"), ("GRBX01", ""),
    ("GRBX02", ""), ("BRKDC01", ""), ("BRKNC01", ""), ("BRKNP01", "c"))]
    # NLN: the README's names
    + [("NLN", name, name, note) for name, note in (
        ("TRAF22", ""), ("SPRE22", ""), ("ROBE25 (1)", ""), ("ROBE25 (2)", ""), ("ROBE25 (3)", ""), ("CVDP23", "i"),
        ("LALO20 W001", ""), ("LALO20 W005", ""), ("LALO20 W01", ""), ("TSPS25", ""), ("LOVO25", "d"))]
    # AINNCS: the report's instances; where the README has two rows, the one of the report's problem
    + [("AINNCS", label, name, note) for label, name, note in (
        ("Unicycle", "Unicycle reach, with the report's disturbance w", "e"),
        ("ACC", "ACC safe-distance", ""),
        ("TORA remain", "TORA remain", ""),
        ("TORA reach-tanh", "TORA reach-tanh", ""),
        ("TORA reach-sigmoid", "TORA reach-sigmoid", ""),
        ("Single Pendulum", "Single Pendulum reach", ""),
        ("DP less-robust", "Double Pendulum less-robust", "fi"),
        ("DP more-robust", "Double Pendulum more-robust", ""),
        ("Airplane", "Airplane continuous", ""),
        *[(f"VCAS {v}", f"VCAS {v}", "") for v in (
            "middle-19.5", "middle-22.5", "middle-25.5", "middle-28.5",
            "worst-19.5", "worst-22.5", "worst-25.5", "worst-28.5")],
        ("Attitude Control", "Attitude Control avoid", ""),
        ("QUAD", "QUAD reach", "i"),
        ("Docking", "Docking constraint", "g"),
        ("NAV standard", "NAV standard", "i"),
        ("NAV robust", "NAV robust", ""),
        ("CartPole*", "CartPole* reach", "hi"))])

# the rows of the README tables that are not rows of the paper's table, and why
EXCLUDED = {
    "ISSC01_ISS02": "not in ARCH26", "ISSC01_ISU02": "not in ARCH26", "ISSF01_ISS01": "not in ARCH26",
    "ISSF01_ISU01": "not in ARCH26", "RAND01": "not in ARCH26", "RAND02": "not in ARCH26",
    "Unicycle reach": "the script without the report's disturbance (note e)",
    "Double Pendulum less-robust, the script (no splitting)": "note f",
    "Docking constraint, polytope and splitting": "note g",
    "CartPole reach, the report's set": "note h",
}

# README rows quoted by the notes (their CORA.cpp times)
QUOTED = {"e": "Unicycle reach", "f": "Double Pendulum less-robust, the script (no splitting)",
          "g": "Docking constraint, polytope and splitting", "h": "CartPole reach, the report's set"}

# the notes on single rows; {t} is the CORA.cpp time of the README row that QUOTED names
NOTES = {
    "a": "HEAT04: ARCH26 ran a new Krylov verification of CORA, not in CORA v2026.1.0; CORA++ runs the reach of "
         "CORA's ARCH23 script with a time step of 0.001.",
    "b": "RND01--06: the report's instances are generated on the competition server; CORA++ runs the files of "
         "CORA's repeatability package (RND06: generated by us in their style), so the speedups (in parentheses) "
         "are indicative and left out of the medians.",
    "c": "PLAN01, BRKNP01: the report verifies them with another setup than CORA's ARCH23 script, which does not "
         "verify them, neither in MATLAB nor in CORA++ (\\ddag).",
    "d": "LOVO25: CORA++ gives a sound verdict; CORA's script sets the result to true without a test.",
    # the rows of the report's problem whose README variants the note quotes
    "e": "Unicycle: CORA++ with the report's disturbance $w$; with $w = 0$, as CORA's script, {t}\\,s.",
    "f": "DP less-robust: CORA++ splits the initial set (depth 14); without splitting, as CORA's "
         "script, the verdict is UNKNOWN ({t}\\,s).",
    "g": "Docking: UNKNOWN for CORA, JuliaReach and CORA++ on the whole initial set; CORA++ verifies it with a "
         "polytope inside the constraint and a splitting of the initial set in {t}\\,s, which no other tool does.",
    "h": "CartPole*: the initial set has 0.4 times the radius of the report's, as CORA ran it; CORA++ verifies "
         "the report's set in {t}\\,s, which no tool ran.",
    "i": "CORA++ runs the 13 boxes of the initial set of CVDP23 and the pieces of a split initial set (DP "
         "less-robust, QUAD, NAV standard, CartPole*) on 8 threads.",
}

MARK_NOTES = (
    r"\dag~Not counted: the report states another problem (JuliaReach discrete-time: time points only; "
    r"immrax: continuous feedback instead of sample-and-hold; KeYmaera X and PRoTECT: a simplified or "
    r"another property).",
    r"\S~Not counted: the tool's ARCH-COMP26 repeatability code solves another problem than the report states "
    r"(JuliaReach: DTN, TRAF22, LALO20, LOVO25; CROWN-Reach: Single Pendulum, Attitude Control).",
    r"\ddag~Not counted: the verdict is UNKNOWN or not the expected one.",
)


def parse_readme(text):
    """The main table of each results section of the README, as {category: {instance: {column: cell}}},
    and the README's median speedups against the best tool, as {'summary': {category: text}}."""
    out = {"summary": {}}
    for category in CATEGORIES:
        section = text.split(f"## {category} results", 1)[1].split("\n## ", 1)[0]
        lines = [ln for ln in section.splitlines() if ln.startswith("| ")]
        start = next(i for i, ln in enumerate(lines) if ln.startswith("| instance | expected (ARCH26) |"))
        header = cells(lines[start])
        table = {}
        for ln in lines[start + 2:]:
            row = cells(ln)
            if len(row) != len(header):
                break
            table[row[0]] = dict(zip(header, row, strict=True))
        out[category] = table
    # the summary: category, ..., speedup vs best ARCH26 tool (median / geo. mean)
    for ln in text.split("## Summary", 1)[1].split("\n## ", 1)[0].splitlines():
        row = cells(ln) if ln.startswith("| ") else []
        if row and row[0] in CATEGORIES:
            out["summary"][row[0]] = row[4].split("/")[0].strip()
    return out


def cells(line):
    """The cells of a Markdown table row."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def load_times(path):
    """The other tools' times by instance and tool: {instance: {tool: (time text, note)}}."""
    out = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["instance"], {})[row["tool"]] = (row["time"], row["note"])
    return out


def consistency(readme, times):
    """What does not fit: a README row that is neither a row nor excluded, a row or a CSV instance
    that is missing, an unknown tool or note."""
    problems = []
    used = {(c, n) for c, _, n, _ in ROWS}
    for category in CATEGORIES:
        for name in readme[category]:
            if (category, name) not in used and name not in EXCLUDED:
                problems.append(f"  {README}: the {category} row '{name}' is neither in ROWS nor in EXCLUDED")
    for category, _, name, _ in ROWS:
        if name not in readme[category]:
            problems.append(f"  {README}: no {category} row '{name}'")
    # every CSV entry belongs to a row, a known tool and a known note, and is a number
    labels = {label for _, label, _, _ in ROWS}
    tools = {t for t, _ in TOOLS}
    for instance, entries in times.items():
        if instance not in labels:
            problems.append(f"  {TIMES}: '{instance}' is no row of the table")
        for tool, (value, note) in entries.items():
            if tool not in tools or (note and note not in MARKS):
                problems.append(f"  {TIMES}: '{instance}', '{tool}': unknown tool or note '{note}'")
            if not re.fullmatch(r"\d+(\.\d+)?", value):
                problems.append(f"  {TIMES}: '{instance}', '{tool}': '{value}' is not a time")
    return problems


def first_token(cell):
    """The verdict of a cell without its comment or footnote: '1 (script: 0)' -> '1'."""
    return cell.split()[0] if cell.split() else ""


def number(cell):
    """A time cell of the README without its footnote mark: '0.102 [b]' -> '0.102'."""
    return re.sub(r"\s*\[\w+\]", "", cell).strip()


def table_row(category, label, readme_row, entries):
    """One row: the tools' cells, CORA.cpp's, the bold one and the speedup."""
    cpp_time = number(readme_row["CORA.cpp [s]"])
    expected = first_token(readme_row["expected (ARCH26)"])
    verdict = first_token(readme_row["CORA.cpp"])
    cpp_counts = verdict == expected and verdict not in ("U", "-")
    counted = [float(t) for t, note in entries.values() if not note]
    best_other = min(counted) if counted else None
    fastest = min(counted + ([float(cpp_time)] if cpp_counts else []), default=None)
    speedup = best_other / float(cpp_time) if cpp_counts and best_other is not None else None
    tool_cells = []
    for tool, _ in TOOLS:
        if tool not in entries:
            tool_cells.append("--")
            continue
        value, note = entries[tool]
        tool_cells.append(cell(value, MARKS.get(note, ""), not note and float(value) == fastest))
    return {"category": category, "label": label, "tools": tool_cells, "speedup": speedup,
            "cpp": cell(cpp_time, "" if cpp_counts else MARKS["verdict"],
                        cpp_counts and float(cpp_time) == fastest),
            "indicative": label.startswith("RND")}


def cell(value, mark, bold):
    """A time as printed, bold if it is the fastest, with its mark."""
    text = rf"\textbf{{{value}}}" if bold else value
    return text + (rf"\textsuperscript{{{mark}}}" if mark else "")


def fmt_speedup(x):
    """Two significant digits: 159 -> 160, 2.26 -> 2.3, 0.0324 -> 0.032, 0.997 -> 1.0."""
    digits = 1 - math.floor(math.log10(x))
    if round(x, digits) >= 10 ** (2 - digits):
        digits -= 1
    rounded = round(x, digits)
    return f"{rounded:.{digits}f}" if digits > 0 else f"{int(rounded)}"


def render(rows, readme):
    """The LaTeX of the table: a longtable in booktabs style, the notes below it."""
    ncols = len(TOOLS) + 3
    header = ("Instance & " + " & ".join(h for _, h in TOOLS) + r" & CORA++ & Speedup \\")
    notes = {letter for _, _, _, letters in ROWS for letter in letters}
    out = [
        f"% Generated by {Path(__file__).relative_to(REPO).as_posix()} from {README} and {TIMES};",
        "% do not edit. Needs longtable, booktabs and pdflscape (the table is too wide for a portrait",
        "% page at \\footnotesize), and the bibliography keys althoff2026arch, geretti2026arch and",
        "% sasaki2026arch (the ARCH-COMP26 category reports).",
        r"\begin{landscape}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{@{}l" + "r" * (ncols - 1) + "@{}}",
        # the caption with the legend, then the header of the first and of every other page and the feet
        r"\caption{ARCH-COMP26 results: the computation times in seconds of the tools in the category "
        r"reports~\cite{althoff2026arch,geretti2026arch,sasaki2026arch} and of CORA++, one row per instance; "
        r"`--' where a tool has no time. The fastest counted time is bold. Speedup: the fastest counted time of "
        r"another tool divided by CORA++'s. Tools: JR~JuliaReach, JR\textsubscript{d}~JuliaReach discrete-time, "
        r"Ari~Ariadne, DI~DynIbex, KX~KeYmaera~X, PR~PRoTECT, CR~CROWN-Reach, imx~immrax; DP~Double Pendulum. "
        r"CORA++: single runs, "
        r"16 CPUs of a shared machine, the same machine as the reports' CORA and JuliaReach times of AFF and NLN. "
        r"Notes and marks below the table.}",
        r"\label{tab:arch-comp-long}\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{ncols}}}{{r@{{}}}}{{\textit{{continued on the next page}}}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    # the rows, a rule and the title of the category before each group
    for i, category in enumerate(CATEGORIES):
        if i:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{TITLES[category]}}}}} \\")
        for (_, _, _, letter), row in zip(ROWS, rows, strict=True):
            if row["category"] != category:
                continue
            label = row["label"] + (rf"\textsuperscript{{{','.join(letter)}}}" if letter else "")
            speedup = "--" if row["speedup"] is None else fmt_speedup(row["speedup"])
            if row["indicative"] and row["speedup"] is not None:
                speedup = f"({speedup})"
            out.append(" & ".join([label, *row["tools"], row["cpp"], speedup]) + r" \\")
    # the notes and the marks below the table, with the quoted CORA.cpp times
    out.append(r"\end{longtable}")
    out.append(r"\par\smallskip\noindent")
    lines = []
    for letter in sorted(notes):
        text = NOTES[letter]
        if letter in QUOTED:
            category = next(c for c in CATEGORIES if QUOTED[letter] in readme[c])
            text = text.replace("{t}", number(readme[category][QUOTED[letter]]["CORA.cpp [s]"]))
        lines.append(rf"\textsuperscript{{{letter}}}~{text}")
    out.append((" \\\\\n").join([*lines, *MARK_NOTES]))
    out.append(r"\par")
    out.append(r"\endgroup")
    out.append(r"\end{landscape}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
