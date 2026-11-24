# Generated paper files

Files of the CORA++ paper that a script writes from the tree; never edited by hand. Copy them into
the paper repository under the path the paper names; rerun the generator after the tree changes.
`python3 scripts/paper/arch_table.py --check` and `python3 scripts/paper/status_figure.py --check`
fail in CI when their files are stale.

| File | Generator | Paper path |
|---|---|---|
| `status.tikz`, `status-caption.tex` | `scripts/paper/status_figure.py` | `figures/status/`: the status overview, Fig. `fig:status` |
| `status-classes-{sets,contdynamics,others,specification,misc}.tikz` | `scripts/paper/status_figure.py` | `figures/status/`: the per-class figures of the appendix |
| `appendix-status.tex` | `scripts/paper/status_figure.py` | `figures/status/`: the appendix section with those figures |
| `arch-comp-table.tex` | `scripts/paper/arch_table.py` | `sections/arch-comp-table.tex`: the ARCH-COMP26 results table of the appendix, `tab:arch-comp-long` |
