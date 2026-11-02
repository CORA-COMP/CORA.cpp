"""example_linear_reach_06_gpu - reachability on the GPU

Tensors on a CUDA device stay there: everything built from them runs on it. Prints a note and
stops if the machine has no CUDA device.

Syntax:   PYTHONPATH=build python examples/python/example_linear_reach_06_gpu.py
Outputs:  where the result lives, and the last center
See also: example_linear_reach_05_batch
"""
import torch

import cora

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

if not torch.cuda.is_available():
    print("no CUDA device: nothing to show")
    raise SystemExit(0)

# Parameters --------------------------------------------------------------------------------

R0 = cora.Zonotope(cora.Tensor([1.0, 0.0], device="gpu"), 0.1 * cora.eye(2, device="gpu"))

# System Dynamics ---------------------------------------------------------------------------

sys = cora.LinearSys(cora.Tensor([[-0.1, 1.0], [-1.0, -0.1]], device="gpu"))

# Reachability Analysis ---------------------------------------------------------------------

R = sys.reach(R0, time_step=0.1, t_final=1.0, taylor_terms=8)

# Evaluation --------------------------------------------------------------------------------

print("the result lives on", R.time_interval[-1].center.device)
print("center of the last enclosure:", R.time_interval[-1].center.cpu().tolist())

# example completed

# ----------------------------------------  END OF CODE  ----------------------------------------- #
