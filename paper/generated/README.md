# Generated paper files

The data of the CORA++ paper's figures and tables, written by a script from the tree; never edited
by hand. The captions, legends' surroundings and floats are the paper's own:
`python3 scripts/paper/sync_paper.py PAPER` copies the files to their paths in the paper repository
and keeps the caption the paper has there. Rerun the generators after the tree changes.
`python3 scripts/paper/arch_table.py --check`, `python3 scripts/paper/status_figure.py --check` and `python3 scripts/paper/vnn_table.py --check`
fail in CI when their files are stale.

| File | Generator | Paper path |
|---|---|---|
| `status.tikz` | `scripts/paper/status_figure.py` | `figures/status/`: the status overview, Fig. `fig:status` |
| `status-classes-{sets,contdynamics,others,specification,misc}.tikz` | `scripts/paper/status_figure.py` | `figures/status/`: the per-class figures of the appendix |
| `arch-comp-table.tex` | `scripts/paper/arch_table.py` | `sections/data/arch-comp-table.tex`: the ARCH-COMP26 results table of the appendix, `tab:arch-comp-long` |
| `arch-comp-summary.tex` | `scripts/paper/arch_table.py` | `sections/data/arch-comp-summary.tex`: the per-category summary of the main body, `tab:arch-comp-results` |
| `vnn-comp-table.tex` | `scripts/paper/vnn_table.py` | `sections/data/vnn-comp-table.tex`: the tabular of the preliminary VNN-COMP results per benchmark (the float and caption around it are in the paper), `tab:vnn-comp-results`; the paper's numbers come from a held-out sample tabulated with `--results` |
