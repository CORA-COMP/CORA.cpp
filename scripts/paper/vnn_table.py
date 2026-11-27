"""vnn_table - the paper's tables of the preliminary VNN-COMP results: solved instances per tool

One row per benchmark, grouped by the size n of its largest input (small, medium, large), and per
tool the verified, falsified and unknown instances (unsat, sat, the rest), bold where (tied) best of
the row, then the sums over every benchmark; the summary sums each group. Rows marked `projected`
(official shares scaled to the sample) are in italics, their tools starred. An incorrect result
(contradicted by another tool) is never solved; it is marked. Tools without rows show `--`.

Syntax:   python scripts/paper/vnn_table.py [--check] [--results CSV] [--output TEX]
Reads:    examples/vnncomp/results/dev_results.csv, or --results (one row per benchmark, instance and
          tool, optionally a `projected` column), and examples/vnncomp/results/input_dims.csv (each
          benchmark's input dimensions, the n of the paper)
Writes:   paper/generated/vnn-comp-table.tex and vnn-comp-summary.tex, or --output and the summary beside it
Checks:   --check renders the table and compares it with the committed file, writes nothing
Updates:  --input-dims DIR writes input_dims.csv from the vnnlib files of DIR/<benchmark>/1.0/ (else 2.0/)
Exit code: 0 if the files are written or up to date, else 1 with what is wrong.

Authors:       Claude Opus 5.5
Written:       24-November-2026
Last update:   27-November-2026
Reviewed:      ---
"""
import argparse
import collections
import csv
import gzip
import math
import re
import sys
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed files, write nothing")
    parser.add_argument("--input-dims", metavar="DIR", help="write input_dims.csv from the benchmarks in DIR")
    parser.add_argument("--results", metavar="CSV", help="the results to tabulate (default: dev_results.csv)")
    parser.add_argument("--output", metavar="TEX", help="where to write the table (default: paper/generated)")
    args = parser.parse_args()
    if args.input_dims:
        return write_input_dims(Path(args.input_dims))
    rows, problems = load(Path(args.results) if args.results else REPO / RESULTS)
    # a CORA++ column only if the results name it (the dev results have one, the final results one per machine)
    global TOOLS
    TOOLS = tuple(t for t in TOOLS if t[0] not in MACHINES or any(r["tool"] == t[0] for r in rows))
    dims = load_dims(REPO / DIMS)
    problems += [f"vnn_table: {DIMS} has no row of {b}"
                 for b in sorted({r["benchmark"] for r in rows}) if b not in dims]
    if problems:
        print("\n".join(problems))
        return 1
    table = tally(rows)
    source = Path(args.results).name if args.results else RESULTS
    output = Path(args.output) if args.output else Path(OUTPUT)
    texts = {output: render(table, dims, source),
             output.with_name(Path(SUMMARY).name): render_summary(table, dims, source)}
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
DIMS = "examples/vnncomp/results/input_dims.csv"
OUTPUT = "paper/generated/vnn-comp-table.tex"
SUMMARY = "paper/generated/vnn-comp-summary.tex"
# a tabular wider than the line shrinks to it (graphicx), a narrower one keeps its size
FIT = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
# the summary's rows: a benchmark by its largest input n, below each bound (the last without one)
SIZES = ((r"Small ($n < 100$)", 100), (r"Medium ($100 \le n < 1\,000$)", 1000), (r"Large ($n \ge 1\,000$)", None))
# the columns (the CSV's name, the header): the top VNN-COMP tools, then CORA++ per machine
TOOLS = (*sorted((("alpha-beta-CROWN", r"$\alpha,\!\beta$-CROWN"), ("CORA", "CORA"), ("PyRAT", "PyRAT"),
                  ("vibecheck", "vibecheck")), key=lambda tool: tool[0].lower()),
         ("CORA++", "CORA++"), ("CORA++ A100", "CORA++ (A100)"),
         ("CORA++ laptop", "CORA++ (laptop)"))  # the others by name; CORA++ is always last
# the CORA++ columns: shown only if the results name them
MACHINES = {"CORA++", "CORA++ A100", "CORA++ laptop"}
# the machines' columns, under one CORA++ heading
SUB = {"CORA++ A100": "A100", "CORA++ laptop": "laptop"}
# the short names of the long benchmark names, so that the table fits the line width
SHORT = {"challenging_certified_training_2026": "cert_training", "traffic_signs_recognition_2023": "traffic_signs",
         "adaptive_cruise_control_non_linear_2026": "acc_non_linear", "collins_aerospace_benchmark": "collins_aero",
         "smart_turn_multimodal_2026": "smart_turn"}
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
    return rows, problems


def load_dims(path):
    """`{benchmark: (fewest, most)}` input dimensions of the committed CSV; empty if it is absent."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["benchmark"]: (int(r["min"]), int(r["max"])) for r in csv.DictReader(f)}


def write_input_dims(root):
    """Writes input_dims.csv: per benchmark the fewest and most inputs of a property of version 1.0
    (of 2.0 for a benchmark that has no 1.0 instances)."""
    out = []
    for bench in sorted(p for p in root.iterdir() if p.is_dir()):
        versions = [bench / v / "instances.csv" for v in ("1.0", "2.0")]
        inst = next((v for v in versions if v.exists() and v.stat().st_size), None)
        if inst is None:
            continue
        with inst.open(newline="", encoding="utf-8") as f:
            names = {row[1].strip() for row in csv.reader(f) if len(row) > 1}
        found = [n for n in (count_inputs(inst.parent / name) for name in sorted(names)) if n]
        if found:
            out.append((inst.parent.parent.name, min(found), max(found)))
    # one row per benchmark, sorted, so that a rerun on the same benchmarks changes nothing
    with (REPO / DIMS).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["benchmark", "min", "max"])
        writer.writerows(out)
    print(f"vnn_table: wrote {DIMS} ({len(out)} benchmarks)")
    return 0


def count_inputs(path):
    """The inputs that a vnnlib property declares (plain or gzipped), or None without the file: the X_i
    of version 1.0, or the entries of the first network's `declare-input` tensors of version 2.0."""
    for p in (path, path.with_name(path.name + ".gz")):
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            seen, entries, networks = set(), 0, 0
            with opener(p, "rt", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"\s*\(declare-const X_(\d+)", line)
                    shape = re.match(r"\s*\(declare-input \S+ \S+ \[([\d,\s]+)\]", line)
                    networks += line.lstrip().startswith("(declare-network")
                    if m:
                        seen.add(int(m.group(1)))
                    elif shape and networks <= 1:
                        entries += math.prod(int(d) for d in shape.group(1).split(","))
                    elif (seen or entries) and line.lstrip().startswith("(assert"):
                        break
            return len(seen) or entries
    return None


def fmt_dims(span):
    """The input dimension, or the range of a benchmark whose networks differ; thin spaces in thousands."""
    def num(n):
        return f"{n:,}".replace(",", r"\,")
    lo, hi = span
    return num(lo) if lo == hi else f"{num(lo)}--{num(hi)}"


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
                            "solved": c["unsat"] + c["sat"] - wrong,
                            "projected": any(r.get("projected") == "yes" for r in mine if r["tool"] == tool)}
    return out


def label(benchmark):
    """The benchmark's short name or its name without the year, underscores escaped."""
    return SHORT.get(benchmark, re.sub(r"_?20\d\d$", "", benchmark)).replace("_", r"\_")


def header(first, table, lead=2):
    """The header row: the bold tool names, CORA++ last; a star on a tool whose entries are all
    projected from the official results."""
    projected = {t for t, _ in TOOLS
                 if (ran := [table[b][t] for b in table if table[b][t]["run"]]) and all(e["projected"] for e in ran)}
    names = [rf"\textbf{{{h}}}" + (r"$^\ast$" if t in projected else "") for t, h in TOOLS if t not in SUB]
    machines = [SUB[t] for t, _ in TOOLS if t in SUB]
    if not machines:
        return rf"\textbf{{{first}}} & " + " & ".join(names) + r" \\"
    # one CORA++ heading over its machines' columns (they come last), the machines below it
    start = len(names) + lead + 1
    return (rf"\textbf{{{first}}} & " + " & ".join(names) + rf" & \multicolumn{{{len(machines)}}}{{c@{{}}}}"
            r"{\textbf{CORA++}} \\" + "\n" + rf"\cmidrule(l){{{start}-{start + len(machines) - 1}}}" + "\n"
            + " & " * (len(names) + lead) + " & ".join(rf"\textbf{{{m}}}" for m in machines) + r" \\")


def render(table, dims, source):
    """The table: a tabular in booktabs style (the paper's float holds it whole, on one page),
    verified/falsified/unknown per tool."""
    ncols = len(TOOLS) + 2
    # the header has the input dimension after the benchmark
    head = header("Benchmark", table).replace(r"\textbf{Benchmark} & ", r"\textbf{Benchmark} & \textbf{$n$} & ")
    # the tabular whole; the float, its caption and label are the paper's
    out = [
        f"% Generated by scripts/paper/vnn_table.py from {source}; do not edit.",
        "% Needs booktabs; the float, its caption and label are the paper's own.",
        # with a column per machine the table only fits the LNCS line at \tiny
        r"\tiny" if any(t in SUB for t, _ in TOOLS) else r"\scriptsize",
        r"\setlength{\tabcolsep}{1.25pt}" if any(t in SUB for t, _ in TOOLS) else r"\setlength{\tabcolsep}{2pt}",
        FIT, r"\begin{tabular}{@{}lr" + "c" * len(TOOLS) + "@{}}",
        r"\toprule", head, r"\midrule",
    ]
    used = set()
    # one row per benchmark under its size's heading; a benchmark that no tool has run says so across the tools
    order = [(name, b) for name, _ in SIZES for b in table if size_of(dims[b][1]) == name]
    for i, (group, b) in enumerate(order):
        entry = table[b]
        if i == 0 or order[i - 1][0] != group:
            out += ([r"\midrule"] if i else []) + [rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{group}}}}} \\"]
        # the benchmark's row: its cells, or that no tool has run it
        if not any(entry[t]["run"] for t, _ in TOOLS):
            across = rf"\multicolumn{{{len(TOOLS)}}}{{c@{{}}}}{{\textit{{not run yet}}}}"
            out.append(rf"{label(b)} & {fmt_dims(dims[b])} & {across} \\")
            continue
        shown = [entry[t] for t, _ in TOOLS if entry[t]["run"]]
        bests = (max(e["verified"] for e in shown), max(e["falsified"] for e in shown),
                 min(e["unknown"] + e["not run"] for e in shown))
        cells = []
        for tool, _ in TOOLS:
            e = entry[tool]
            if not e["run"]:
                cells.append("--")
                continue
            text = triple((e["verified"], e["falsified"], e["unknown"] + e["not run"]), bests, e["projected"])
            marks = [m for m, on in ((PARTIAL, e["not run"]), (WRONG, e["incorrect"])) if on]
            used.update(marks)
            cells.append(text + (rf"\textsuperscript{{{','.join(marks)}}}" if marks else ""))
        out.append(" & ".join([label(b), fmt_dims(dims[b]), *cells]) + r" \\")
    # the total over every benchmark; a benchmark a tool has not run counts as unsolved (VNN-COMP's rule)
    total, cells = sum_cells(table, list(table))
    out += [r"\midrule", " & ".join([rf"\multicolumn{{2}}{{@{{}}l}}{{All ({total} instances)}}", *cells]) + r" \\"]
    # the marks that occur, in a last row of the table
    notes = {PARTIAL: rf"\textsuperscript{{{PARTIAL}}}~{partial_note(table)}",
             WRONG: rf"\textsuperscript{{{WRONG}}}~incorrect results (contradicted by another tool) "
                    r"included, not counted as solved."}
    lines = [notes[m] for m in (PARTIAL, WRONG) if m in used]
    if lines:
        out.append(rf"\multicolumn{{{ncols}}}{{@{{}}p{{\linewidth}}@{{}}}}{{" + " ".join(lines) + r"} \\")
    out += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(out) + "\n"


def render_summary(table, dims, source):
    """The summary: per size of the input (and in total) per tool verified/falsified/unknown; a benchmark
    a tool has not run counts as unsolved, as in the total."""
    groups = [(name, [b for b in table if size_of(dims[b][1]) == name]) for name, _ in SIZES]
    # the tabular whole, a row per size and the total; the float is the paper's
    out = [
        f"% Generated by scripts/paper/vnn_table.py from {source}; do not edit.",
        "% Needs booktabs; the float, its caption and label are the paper's own.",
        r"\scriptsize" if any(t in SUB for t, _ in TOOLS) else r"\small",
        r"\setlength{\tabcolsep}{2pt}" if any(t in SUB for t, _ in TOOLS) else r"\setlength{\tabcolsep}{4pt}",
        FIT, r"\begin{tabular}{@{}l" + "c" * len(TOOLS) + "@{}}",
        r"\toprule", header("Input size", table, lead=1),
        r"\midrule",
    ]
    for i, (name, benches) in enumerate([*groups, ("All", list(table))]):
        if not benches:
            continue
        if i == len(groups):
            out.append(r"\midrule")
        # the group's row: per tool its sums
        _, cells = sum_cells(table, benches)
        out.append(" & ".join([name, *cells]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(out) + "\n"


def sum_cells(table, benches):
    """The instances of the benchmarks and per tool its summed cell; a benchmark the tool has not run
    counts as unsolved (VNN-COMP's rule)."""
    instances = sum(table[b]["instances"] for b in benches)
    sums = {}
    for t, _ in TOOLS:
        v = sum(table[b][t]["verified"] for b in benches if table[b][t]["run"])
        f = sum(table[b][t]["falsified"] for b in benches if table[b][t]["run"])
        ran = [table[b][t] for b in benches if table[b][t]["run"]]
        sums[t] = (v, f, instances - v - f, bool(ran) and all(e["projected"] for e in ran))
    bests = (max(x[0] for x in sums.values()), max(x[1] for x in sums.values()), min(x[2] for x in sums.values()))
    return instances, [triple(sums[t][:3], bests, sums[t][3]) for t, _ in TOOLS]


def size_of(n):
    """The summary row of a benchmark whose largest input has n entries."""
    return next(name for name, bound in SIZES if bound is None or n < bound)


def triple(vfu, bests, projected):
    """verified/falsified/unknown, each number bold where it is (tied) best of its row: the most verified,
    the most falsified, the fewest unknown; the slashes bold too when all three are; italics if projected."""
    bold = [x == b for x, b in zip(vfu, bests, strict=True)]
    if all(bold):
        text = rf"\textbf{{{vfu[0]}/{vfu[1]}/{vfu[2]}}}"
    else:
        text = "/".join(rf"\textbf{{{x}}}" if on else str(x) for x, on in zip(vfu, bold, strict=True))
    return rf"\textit{{{text}}}" if projected else text


def partial_note(table):
    """The note of the partial mark: which tool has not run how many instances of which benchmark."""
    parts = [f"{tool} {e['not run']} of {table[b]['instances']} on {label(b)}"
             for b in table for tool, _ in TOOLS
             for e in [table[b][tool]] if e["run"] and e["not run"]]
    return "not all instances run yet: " + "; ".join(parts) + "."


if __name__ == "__main__":
    sys.exit(main())

# ----------------------------------------  END OF CODE  ----------------------------------------- #
