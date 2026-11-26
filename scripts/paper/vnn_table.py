"""vnn_table - the paper's tables of the preliminary VNN-COMP results: solved instances per tool

One row per benchmark and, per tool, the verified, falsified and unknown instances (unsat, sat,
the rest), each number bold where it is the (tied) best of its row, then the sums over every
benchmark. Rows marked `projected` (official VNN-COMP shares scaled to the sample) are in italics,
their tools starred. An incorrect result (contradicted by another tool) is never solved; it is
marked. Tools without rows show `--`. The float and its caption are the paper's own.

Syntax:   python scripts/paper/vnn_table.py [--check] [--results CSV] [--output TEX]
Reads:    examples/vnncomp/results/dev_results.csv, or --results (one row per benchmark, instance and
          tool, optionally a `projected` column), and examples/vnncomp/results/input_dims.csv (each
          benchmark's input dimensions, the n of the paper)
Writes:   paper/generated/vnn-comp-table.tex, or --output
Checks:   --check renders the table and compares it with the committed file, writes nothing
Updates:  --input-dims DIR writes input_dims.csv from the vnnlib files of DIR/<benchmark>/1.0/
Exit code: 0 if the files are written or up to date, else 1 with what is wrong.

Authors:       Claude Opus 5.5
Written:       24-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import collections
import csv
import gzip
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
    dims = load_dims(REPO / DIMS)
    problems += [f"vnn_table: {DIMS} has no row of {b}"
                 for b in sorted({r["benchmark"] for r in rows}) if b not in dims]
    if problems:
        print("\n".join(problems))
        return 1
    table = tally(rows)
    source = Path(args.results).name if args.results else RESULTS
    texts = {Path(args.output) if args.output else Path(OUTPUT): render(table, dims, source)}
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
# the columns: the top tools of VNN-COMP, then CORA++ (the CSV's name, the header)
TOOLS = (*sorted((("alpha-beta-CROWN", r"$\alpha,\!\beta$-CROWN"), ("CORA", "CORA"), ("PyRAT", "PyRAT"),
                  ("vibecheck", "vibecheck")), key=lambda tool: tool[0].lower()),
         ("CORA++", "CORA++"))  # the others by name; CORA++ is always last
# the short names of the long benchmark names, so that the table fits the line width
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
    return rows, problems


def load_dims(path):
    """`{benchmark: (fewest, most)}` input dimensions of the committed CSV; empty if it is absent."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["benchmark"]: (int(r["min"]), int(r["max"])) for r in csv.DictReader(f)}


def write_input_dims(root):
    """Writes input_dims.csv: per benchmark the fewest and most inputs of a property of version 1.0."""
    out = []
    for inst in sorted(root.glob("*/1.0/instances.csv")):
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
    """The inputs X_i that a vnnlib property declares (plain or gzipped), or None without the file."""
    for p in (path, path.with_name(path.name + ".gz")):
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            seen = set()
            with opener(p, "rt", encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"\s*\(declare-const X_(\d+)", line)
                    if m:
                        seen.add(int(m.group(1)))
                    elif seen and line.lstrip().startswith("(assert"):
                        break
            return len(seen)
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


def header(first, table):
    """The header row: the bold tool names, CORA++ last; a star on a tool whose entries are all
    projected from the official results."""
    projected = {t for t, _ in TOOLS
                 if (ran := [table[b][t] for b in table if table[b][t]["run"]]) and all(e["projected"] for e in ran)}
    return rf"\textbf{{{first}}} & " + " & ".join(
        rf"\textbf{{{h}}}" + (r"$^\ast$" if t in projected else "") for t, h in TOOLS) + r" \\"


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
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lr" + "c" * len(TOOLS) + "@{}}",
        r"\toprule", head, r"\midrule",
    ]
    used = set()
    # one row per benchmark; a benchmark that no tool has run says so across the tools
    for b, entry in table.items():
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
    total = sum(table[b]["instances"] for b in table)
    sums = {}
    for t, _ in TOOLS:
        v = sum(table[b][t]["verified"] for b in table if table[b][t]["run"])
        f = sum(table[b][t]["falsified"] for b in table if table[b][t]["run"])
        ran = [table[b][t] for b in table if table[b][t]["run"]]
        sums[t] = (v, f, total - v - f, bool(ran) and all(e["projected"] for e in ran))
    bests = (max(x[0] for x in sums.values()), max(x[1] for x in sums.values()), min(x[2] for x in sums.values()))
    cells = [triple(sums[t][:3], bests, sums[t][3]) for t, _ in TOOLS]
    out += [r"\midrule", " & ".join([rf"\multicolumn{{2}}{{@{{}}l}}{{All ({total} instances)}}", *cells]) + r" \\"]
    # the marks that occur, in a last row of the table
    notes = {PARTIAL: rf"\textsuperscript{{{PARTIAL}}}~{partial_note(table)}",
             WRONG: rf"\textsuperscript{{{WRONG}}}~incorrect results (contradicted by another tool) "
                    r"included, not counted as solved."}
    lines = [notes[m] for m in (PARTIAL, WRONG) if m in used]
    if lines:
        out.append(rf"\multicolumn{{{ncols}}}{{@{{}}p{{\linewidth}}@{{}}}}{{" + " ".join(lines) + r"} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


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
