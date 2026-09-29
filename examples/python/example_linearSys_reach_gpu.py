"""The reachable set on the GPU: tensors on a CUDA device run there. Prints a note and exits
if the machine has none.

    PYTHONPATH=build python examples/python/example_linearSys_reach_gpu.py
"""
import torch

import coracpp

if not torch.cuda.is_available():
    print("no CUDA device: nothing to show")
    raise SystemExit(0)

f64 = dict(dtype=torch.float64, device="cuda")
A = torch.tensor([[-0.1, 1.0], [-1.0, -0.1]], **f64)
c = torch.tensor([1.0, 0.0], **f64)
G = 0.1 * torch.eye(2, **f64)

R = coracpp.reach(A, c, G, timeStep=0.1, tFinal=1.0)

print("the result lives on", R.timeInt_c.device)
print("center of the last enclosure:", R.timeInt_c[-1].cpu().tolist())
