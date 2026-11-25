# Preliminary VNN-COMP results on the dev sets

`dev_results.csv` holds the preliminary comparison on the development seeds, in long form: one row
per (benchmark, instance, tool). Adding a tool means adding its rows; the columns stay the same.

| Column | Meaning |
|---|---|
| `benchmark` | the VNN-COMP benchmark |
| `onnx`, `vnnlib` | the instance, relative to the benchmark's `1.0/` directory |
| `tool` | `CORA++` (this repository) or `alpha-beta-CROWN`; later also vibecheck, PyRAT and CORA (MATLAB) |
| `result` | `unsat`, `sat`, `unknown`, `timeout`, `error`, or `not run` |
| `time_s` | the tool's run time in seconds (blank when not run) |
| `timeout_s` | the instance's timeout from `instances.csv` |
| `expected` | the ground truth; blank, since the dev seeds have none |
| `incorrect` | for a decided result (`sat`/`unsat`): `yes` if another tool decided the opposite, else `no`; blank otherwise |
| `hardware` | the machine of the run |
| `commit` | the CORA.cpp commit of our run, or alpha-beta-CROWN's version |
| `preset` | the options of the run |
| `source` | the run's id: tool, sample and commit (the run logs stay with the developers) |

`input_dims.csv` holds each benchmark's input dimension (the n of the paper): the fewest and most inputs
`X_i` that a property of version 1.0 declares. `python3 scripts/paper/vnn_table.py --input-dims DIR` writes
it from a checkout of the benchmarks.

## The instances

- Dev sets v1 (`MANIFEST.json` sha256 `3501a505…6ae0`), seed set 1001, version 1.0 of every
  benchmark.
- The sample (`v1_s1001_seed213.json`): `numpy.random.default_rng(213).choice(n, k, replace=False)`
  per benchmark, sorted. The exception is cifar100_2024, where the sample is the first 40 instances.
- Instances per benchmark (k of n):

| Benchmark | k | n |
|---|---|---|
| cifar100_2024 | 40 | 187 |
| tinyimagenet_2024 | 20 | 157 |
| vit_2023 | 20 | 160 |
| metaroom_2023 | 12 | 52 |
| soundnessbench_2026 | 12 | 42 |
| relusplitter_2026 | 12 | 116 |
| nn4sys | 12 | 166 |
| challenging_certified_training_2026 | 8 | 56 |
| lsnc_relu | 8 | 80 |
| traffic_signs_recognition_2023 | 6 | 45 |
| ml4acopf_2024 | 6 | 69 |
| vggnet16_2022 | 4 | 18 |
| cgan2026 | 4 | 21 |
| tllverifybench_2023 | 4 | 32 |

- These are development numbers. No result here is compared against the official VNN-COMP results;
  that comparison waits for the final evaluation seed.

## The tools

- **Hardware:** both tools ran on the same machine: an NVIDIA A100-PCIE-40GB and a Xeon Platinum
  8380 (160 threads), one instance at a time with one CPU thread. The two tools never shared the
  GPU.
- **alpha-beta-CROWN:**
  - Release 0.8.0 at commit b58d83c, with its VNN-COMP'26 configurations (`exp_configs/vnncomp2x`),
    run by us on this machine. These are not the competition's published numbers. The competition
    ran on AWS g5.8xlarge (A10G).
  - safenlp and sat_relu need Gurobi and are not run.
- **CORA++:**
  - Each benchmark's row names the commit and options of its most recent run.
  - Most commits are on main. Two exceptions:
    - vit_2023 used a vit preset from a branch that is not on main (memory_budget 32 GiB, no ReLU
      slope steps).
    - cifar100_2024 used `{"refinement_method":"relu-bab"}` rather than a preset.
  - soundnessbench_2026 is from PR #320 (open).

## Not run yet

- **cifar100_2024:**
  - With relu-bab, only the 23 instances alpha-beta-CROWN proves were run. Its other 17 (7 sat,
    10 timeouts) are `not run` for CORA++.
  - alpha-beta-CROWN's cifar100 run used a copy of the instances made before dev sets v1 were frozen.
    All 80 of its onnx/vnnlib files are byte-identical to v1 (sha256 checked).
- **Small benchmarks:** acasxu_2023, cersyve, collins_rul_cnn_2022, cora_2024, dist_shift_2023,
  linearizenn_2024, malbeware, safenlp_2024 and sat_relu have no GPU run of either tool yet; they
  have one `not run` row per tool. CORA++ solves most of them on the CPU; they will be run at the
  next milestone.

## Scoring

`scripts/vnncomp/score.py` scores a run's results CSV by the VNN-COMP 2026 rules:

- **Points:** a correct holds or violated is +10, an incorrect one −150, everything else 0.
- **A violated (sat) counts** only if its counterexample passes the scoring's check
  (`cora.counterexample`).
- **A holds (unsat) is incorrect** if the expected verdicts say sat, or if a counterexample from
  another run is correct.

This CSV records `incorrect` only as disagreement between the tools. Every CORA++ `sat` is a
checked counterexample. So far no instance has conflicting verdicts.
