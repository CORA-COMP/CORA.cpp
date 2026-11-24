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
    texts = {OUTPUT: render(rows, readme), SUMMARY: render_summary(rows)}
    # the medians, next to the README's summary, so that a reader of the log sees both
    for category in CATEGORIES:
        values = [r["speedup"] for r in rows if r["category"] == category and r["speedup"] and not r["indicative"]]
        print(f"{category}: median speedup {fmt_speedup(statistics.median(values))} over {len(values)} "
              f"instances (README summary: {readme['summary'].get(category, '?')})")
    # compare with the committed files, or write them
    stale = [name for name, text in texts.items()
             if not (REPO / name).exists() or (REPO / name).read_text(encoding="utf-8") != text]
    if args.check:
        for name in stale:
            print(f"  {name} is stale; run python3 scripts/paper/arch_table.py")
        return 1 if stale else 0
    for name, text in texts.items():
        (REPO / name).parent.mkdir(parents=True, exist_ok=True)
        (REPO / name).write_text(text, encoding="utf-8")
        print(f"arch_table: wrote {name}")
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

REPO = Path(__file__).resolve().parents[2]

README = "examples/benchmarks/README.md"

TIMES = "examples/benchmarks/data/arch26_tool_times.csv"

OUTPUT = "paper/generated/arch-comp-table.tex"

SUMMARY = "paper/generated/arch-comp-summary.tex"

CATEGORIES = ("AFF", "NLN", "AINNCS")

TITLES = {"AFF": "Linear continuous and hybrid dynamics (AFF)",
          "NLN": "Nonlinear continuous and hybrid dynamics (NLN)",
          "AINNCS": "Neural-network control systems (AINNCS)"}

# every tool of the CSV: its name there and a short header
ALL_TOOLS = (("CORA", "CORA"), ("JuliaReach", "JR"), ("JuliaReach discrete-time", r"JR\textsubscript{d}"),
         ("Ariadne", "Ari"), ("DynIbex", "DI"), ("KeYmaera X", "KX"), ("PRoTECT", "PR"),
         ("CROWN-Reach", "CR"), ("immrax", "imx"))

# the tool columns of the table (the user's choice, 2026-11-24): the bold time and the speedup count these only
TOOLS = (("CORA", "CORA"), ("JuliaReach", "JuliaReach"), ("CROWN-Reach", "CROWN-Reach"))

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
        ("CartPole*", "CartPole reach, the report's set", "hi"))])

# the rows of the README tables that are not rows of the paper's table, and why
EXCLUDED = {
    "ISSC01_ISS02": "not in ARCH26", "ISSC01_ISU02": "not in ARCH26", "ISSF01_ISS01": "not in ARCH26",
    "ISSF01_ISU01": "not in ARCH26", "RAND01": "not in ARCH26", "RAND02": "not in ARCH26",
    # variants that are not the problem of the specification (the user, 2026-11-24: only the specification's)
    "Unicycle reach": "the script without the specification's disturbance",
    "Double Pendulum less-robust, the script (no splitting)": "the same problem; the row splits",
    "Docking constraint, polytope and splitting": "another constraint set than the specification's",
    "CartPole* reach": "the script's smaller initial set, not the specification's",
}

# README rows quoted by the notes (their CORA.cpp times)
QUOTED = {}

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
    # the rows where the specification's problem needs a word
    "e": "Unicycle: with the disturbance $w$ of the specification.",
    "f": "DP less-robust: CORA++ splits the initial set (depth 14).",
    "g": "Docking: UNKNOWN for CORA, JuliaReach and CORA++.",
    "h": "CartPole*: the initial set of the specification; CORA's time is for 0.4 times its radius (\\dag).",
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
    tools = {t for t, _ in ALL_TOOLS}
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
    # a verdict counts if it is the expected one, or a verified one where no tool has a verdict
    cpp_counts = verdict not in ("U", "-") and (verdict == expected or (expected == "-" and verdict in ("V", "1")))
    # only the shown tools take part in the bold time and the speedup
    entries = {tool: entries[tool] for tool, _ in TOOLS if tool in entries}
    counted = [float(t) for t, note in entries.values() if not note]
    best_other = min(counted) if counted else None
    fastest = min(counted + ([float(cpp_time)] if cpp_counts else []), default=None)
    speedup = best_other / float(cpp_time) if cpp_counts and best_other is not None else None
    # each tool's cell, bold where it is the fastest counted time
    tool_cells = []
    for tool, _ in TOOLS:
        if tool not in entries:
            tool_cells.append("--")
            continue
        value, note = entries[tool]
        tool_cells.append(cell(value, MARKS.get(note, ""), not note and float(value) == fastest))
    # the tools with the fastest counted time, for the summary table
    winners = [tool for tool, _ in TOOLS if tool in entries and not entries[tool][1]
               and float(entries[tool][0]) == fastest]
    if cpp_counts and float(cpp_time) == fastest:
        winners.append("CORA++")
    return {"category": category, "label": label, "tools": tool_cells, "speedup": speedup, "winners": winners,
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
    # CORA++ spans its time and speedup below a partial rule; the bold headers sit on their row
    header = (" & " * len(TOOLS) + r" & \multicolumn{2}{c@{}}{\textbf{CORA++}} \\"
              + "\n" + rf"\cmidrule(l){{{ncols - 1}-{ncols}}}" + "\n"
              + r"\textbf{Instance} & " + " & ".join(rf"\textbf{{{h}}}" for _, h in TOOLS) + r" & Time & Speedup \\")
    notes = {letter for _, _, _, letters in ROWS for letter in letters}
    out = [
        f"% Generated by {Path(__file__).relative_to(REPO).as_posix()} from {README} and {TIMES};",
        "% do not edit. Needs longtable and booktabs, and the bibliography keys althoff2026arch,",
        "% geretti2026arch and sasaki2026arch (the ARCH-COMP26 category reports).",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{@{}l" + "r" * (ncols - 1) + "@{}}",
        # the caption with the legend, then the header of the first and of every other page and the feet
        r"\caption{ARCH-COMP26 results: the computation times in seconds of CORA, JuliaReach and CROWN-Reach in "
        r"the category reports~\cite{althoff2026arch,geretti2026arch,sasaki2026arch} and of CORA++, one row per "
        r"instance; `--' where a tool has no time. The fastest counted time of these four is bold. Speedup: the "
        r"fastest counted time of CORA, JuliaReach and CROWN-Reach divided by CORA++'s. DP~Double Pendulum. "
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
            speedup = "--" if row["speedup"] is None else r"$\times$" + fmt_speedup(row["speedup"])
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
    used = {m for row in rows for m in MARKS.values() if m in " ".join([*row["tools"], row["cpp"]])}
    marks = [text for text in MARK_NOTES if text.split("~")[0] in used]
    out.append((" \\\\\n").join([*lines, *marks]))
    out.append(r"\par")
    out.append(r"\endgroup")
    return "\n".join(out) + "\n"


def render_summary(rows):
    """The table of the main body, laid out as the long table: per category how often each tool is the
    fastest, as fastest/instances, and for CORA++ also the median speedup with its quartiles (the instances
    with a speedup, RND left out as in the long table)."""
    tools = [t for t, _ in TOOLS]
    ncols = len(tools) + 3
    out = [
        f"% Generated by {Path(__file__).relative_to(REPO).as_posix()} from {README} and {TIMES};",
        "% do not edit. Needs booktabs; the long table is tab:arch-comp-long.",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}l" + "c" * (len(tools) + 2) + "@{}}",
        r"\toprule",
        # CORA++ spans its two columns, the other headers on the row below, as in the long table
        " & " * len(tools) + r" & \multicolumn{2}{c@{}}{\textbf{CORA++}} \\",
        rf"\cmidrule(l){{{ncols - 1}-{ncols}}}",
        r"\textbf{Category} & " + " & ".join(rf"\textbf{{{h}}}" for _, h in TOOLS) + r" & Fastest & Speedup \\",
        r"\midrule",
    ]
    # one row per category: the counts, then the median speedup and its quartiles
    for category in CATEGORIES:
        mine = [r for r in rows if r["category"] == category]
        wins = [sum(name in r["winners"] for r in mine) for name in [*tools, "CORA++"]]
        values = sorted(r["speedup"] for r in mine if r["speedup"] and not r["indicative"])
        q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
        speedup = rf"$\times${fmt_speedup(median)} [{fmt_speedup(q1)}, {fmt_speedup(q3)}]"
        # the tool with the most fastest instances in bold, as the fastest time in the long table
        counts = [rf"\textbf{{{w}/{len(mine)}}}" if w == max(wins) else f"{w}/{len(mine)}" for w in wins]
        # a tool without a time in the category did not take part in it (CROWN-Reach in AFF, NLN): `--'
        took_part = [any(r["tools"][i] != "--" for r in mine) for i in range(len(tools))] + [True]
        counts = [c if part else "--" for c, part in zip(counts, took_part, strict=True)]
        out.append(" & ".join([category, *counts, speedup]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
