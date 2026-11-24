"""vnn_table - the paper's tables of the preliminary VNN-COMP results: solved instances per tool

The appendix table has one row per benchmark and, per tool, the verified, falsified and unknown
instances (unsat, sat, the rest: unknown, timeout, error); the main-body table has a selection of
the benchmarks and the solved (verified + falsified) of the instances, the most solved of a row bold.
An incorrect result (one that another tool contradicts) is never solved; it is marked. Tools without
rows show `--`: today only CORA++ and alpha-beta-CROWN have results on the dev sets.

Syntax:   python scripts/paper/vnn_table.py [--check]
Reads:    examples/vnncomp/results/dev_results.csv (one row per benchmark, instance and tool)
Writes:   paper/generated/vnn-comp-table.tex and paper/generated/vnn-comp-summary.tex (copied into
          the paper repository as they are)
Checks:   --check renders both tables and compares them with the committed files, writes nothing
Exit code: 0 if the files are written or up to date, else 1 with what is wrong.

Authors:       Claude Opus 5.5
Written:       24-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import collections
import csv
import re
import sys
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed files, write nothing")
    args = parser.parse_args()
    rows, problems = load(REPO / RESULTS)
    if problems:
        print("\n".join(problems))
        return 1
    table = tally(rows)
    texts = {OUTPUT: render(table), SUMMARY: render_summary(table)}
    # the totals in the log, so that a reader sees the state without the PDF
    for tool, _ in TOOLS:
        ran = [b for b in table if table[b][tool]["run"]]
        print(f"{tool}: {sum(table[b][tool]['solved'] for b in ran)} solved on {len(ran)} benchmarks")
    # compare with the committed files, or write them
    stale = [name for name, text in texts.items()
             if not (REPO / name).exists() or (REPO / name).read_text(encoding="utf-8") != text]
    if args.check:
        for name in stale:
            print(f"  {name} is stale; run python3 scripts/paper/vnn_table.py")
        return 1 if stale else 0
    for name, text in texts.items():
        (REPO / name).parent.mkdir(parents=True, exist_ok=True)
        (REPO / name).write_text(text, encoding="utf-8")
        print(f"vnn_table: wrote {name}")
    return 0


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

REPO = Path(__file__).resolve().parents[2]
# the input and the two outputs
RESULTS = "examples/vnncomp/results/dev_results.csv"
OUTPUT = "paper/generated/vnn-comp-table.tex"
SUMMARY = "paper/generated/vnn-comp-summary.tex"
# the columns: the top tools of VNN-COMP, then CORA++ (the CSV's name, the header)
TOOLS = (("alpha-beta-CROWN", r"$\alpha,\!\beta$-CROWN"), ("vibecheck", "vibecheck"), ("PyRAT", "PyRAT"),
         ("CORA", "CORA"), ("CORA++", "CORA++"))
# the benchmarks of the main body: small and large networks, verified and falsified
MAIN = ("acasxu_2023", "tllverifybench_2023", "metaroom_2023", "lsnc_relu", "nn4sys", "soundnessbench_2026",
        "traffic_signs_recognition_2023", "cifar100_2024", "vit_2023")
# the short names of the long benchmark names, so that both tables fit the line width
SHORT = {"challenging_certified_training_2026": "cert_training", "traffic_signs_recognition_2023": "traffic_signs"}
RESULTS_KNOWN = {"unsat", "sat", "unknown", "timeout", "error", "not run"}
# the marks of the appendix: a tool that has not run all instances of a benchmark, an incorrect result
PARTIAL, WRONG = r"\dag", r"\ddag"


def load(path):
    """The rows of the CSV, and what is wrong with them (unknown tools or results)."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = {t for t, _ in TOOLS}
    problems = [f"vnn_table: {path.name}: unknown tool {r['tool']!r}" for r in rows if r["tool"] not in names]
    problems += [f"vnn_table: {path.name}: unknown result {r['result']!r}" for r in rows
                 if r["result"] not in RESULTS_KNOWN]
    missing = [b for b in MAIN if b not in {r["benchmark"] for r in rows}]
    problems += [f"vnn_table: the main-body benchmark {b} has no rows" for b in missing]
    return rows, problems


def tally(rows):
    """Per benchmark and tool: verified, falsified, unknown, incorrect, not run, solved, and the instances."""
    out = collections.OrderedDict()
    for b in sorted({r["benchmark"] for r in rows}, key=str.lower):
        mine = [r for r in rows if r["benchmark"] == b]
        instances = len({(r["onnx"], r["vnnlib"]) for r in mine if r["result"] != "not run" or r["onnx"]})
        out[b] = {"instances": instances}
        for tool, _ in TOOLS:
            c = collections.Counter(r["result"] for r in mine if r["tool"] == tool)
            wrong = sum(r["incorrect"] == "yes" for r in mine if r["tool"] == tool)
            run = sum(c.values()) - c["not run"]
            out[b][tool] = {"verified": c["unsat"], "falsified": c["sat"],
                            "unknown": c["unknown"] + c["timeout"] + c["error"], "incorrect": wrong,
                            "run": run, "not run": c["not run"],
                            "solved": c["unsat"] + c["sat"] - wrong}
    return out


def label(benchmark):
    """The benchmark's short name or its name without the year, underscores escaped."""
    return SHORT.get(benchmark, re.sub(r"_?20\d\d$", "", benchmark)).replace("_", r"\_")


def header(first):
    """The header row: the bold tool names, CORA++ last."""
    return rf"\textbf{{{first}}} & " + " & ".join(rf"\textbf{{{h}}}" for _, h in TOOLS) + r" \\"


def render(table):
    """The appendix table: a longtable in booktabs style, verified/falsified/unknown per tool."""
    ncols = len(TOOLS) + 2
    # the header has the instances after the benchmark; the caption is the legend
    head = header("Benchmark").replace(r"\textbf{Benchmark} & ", r"\textbf{Benchmark} & \textbf{$n$} & ")
    out = [
        f"% Generated by {Path(__file__).relative_to(REPO).as_posix()} from {RESULTS}; do not edit.",
        "% Needs longtable and booktabs.",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{@{}lr" + "c" * len(TOOLS) + "@{}}",
        # the caption with the legend and the setup of the runs
        r"\caption{Preliminary VNN-COMP results on the development instances: per benchmark the sampled "
        r"instances~$n$ and, per tool, the verified/falsified/unknown instances (unknown includes timeouts and "
        r"errors); `--' where a tool has no run yet. The most solved (verified + falsified) of a row are bold. "
        r"$\alpha,\!\beta$-CROWN 0.8.0 with its VNN-COMP'26 configurations and CORA++ ran on the same machine "
        r"(an A100 and one CPU thread per instance), one instance at a time, with the benchmarks' timeouts. "
        r"There is no ground truth on the development instances; every falsification of CORA++ is a checked "
        r"counterexample, and no result of one tool is contradicted by another.}",
        # the label, then the header of the first and of every other page and the feet
        r"\label{tab:vnn-comp-long}\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\toprule", head, r"\midrule", r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{ncols}}}{{r@{{}}}}{{\textit{{continued on the next page}}}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    used = set()
    # one row per benchmark; a benchmark that no tool has run says so across the tools
    for b, entry in table.items():
        if not any(entry[t]["run"] for t, _ in TOOLS):
            out.append(rf"{label(b)} & -- & \multicolumn{{{len(TOOLS)}}}{{c@{{}}}}{{\textit{{not run yet}}}} \\")
            continue
        best = max(entry[t]["solved"] for t, _ in TOOLS)
        cells = []
        for tool, _ in TOOLS:
            e = entry[tool]
            if not e["run"]:
                cells.append("--")
                continue
            text = f"{e['verified']}/{e['falsified']}/{e['unknown']}"
            text = rf"\textbf{{{text}}}" if e["solved"] == best else text
            marks = [m for m, on in ((PARTIAL, e["not run"]), (WRONG, e["incorrect"])) if on]
            used.update(marks)
            cells.append(text + (rf"\textsuperscript{{{','.join(marks)}}}" if marks else ""))
        out.append(" & ".join([label(b), str(entry["instances"]), *cells]) + r" \\")
    out.append(r"\end{longtable}")
    # the marks that occur, below the table
    notes = {PARTIAL: rf"\textsuperscript{{{PARTIAL}}}~{partial_note(table)}",
             WRONG: rf"\textsuperscript{{{WRONG}}}~incorrect results (contradicted by another tool) "
                    r"included, not counted as solved."}
    lines = [notes[m] for m in (PARTIAL, WRONG) if m in used]
    if lines:
        out += [r"\par\smallskip\noindent", " \\\\\n".join(lines), r"\par"]
    out.append(r"\endgroup")
    return "\n".join(out) + "\n"


def partial_note(table):
    """The note of the partial mark: which tool has not run how many instances of which benchmark."""
    parts = [f"{tool} {e['not run']} of {table[b]['instances']} on {label(b)}"
             for b in table for tool, _ in TOOLS
             for e in [table[b][tool]] if e["run"] and e["not run"]]
    return "not all instances run yet: " + "; ".join(parts) + "."


def render_summary(table):
    """The main-body table: solved/instances of a selection of benchmarks, and the total of the run ones."""
    out = [
        f"% Generated by {Path(__file__).relative_to(REPO).as_posix()} from {RESULTS}; do not edit.",
        "% Needs booktabs; the full table is tab:vnn-comp-long.",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "c" * len(TOOLS) + "@{}}",
        r"\toprule",
        header("Benchmark"),
        r"\midrule",
    ]
    # one row per selected benchmark, then the total over every benchmark that CORA++ has run
    for b in MAIN:
        out.append(summary_row(label(b), [table[b]]))
    ran = [table[b] for b in table if table[b]["CORA++"]["run"]]
    out += [r"\midrule", summary_row(rf"All ({len(ran)} benchmarks)", ran), r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def summary_row(name, entries):
    """A row of the main-body table: per tool solved/instances, `--' without a run, the most solved bold."""
    total = sum(e["instances"] for e in entries)
    solved = {t: sum(e[t]["solved"] for e in entries) for t, _ in TOOLS}
    run = {t: any(e[t]["run"] for e in entries) for t, _ in TOOLS}
    if not any(run.values()):
        return rf"{name} & \multicolumn{{{len(TOOLS)}}}{{c@{{}}}}{{\textit{{not run yet}}}} \\"
    best = max(solved[t] for t, _ in TOOLS if run[t])
    cells = ["--" if not run[t] else (rf"\textbf{{{solved[t]}/{total}}}" if solved[t] == best
                                      else f"{solved[t]}/{total}") for t, _ in TOOLS]
    # a tool that has not run every instance yet: the mark of the appendix
    cells = [c + (rf"\textsuperscript{{{PARTIAL}}}" if run[t] and any(e[t]["not run"] for e in entries) else "")
             for c, (t, _) in zip(cells, TOOLS, strict=True)]
    return " & ".join([name, *cells]) + r" \\"


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
