// The reachable set of a damped oscillator. No backend is named: it is libtorch by default,
// or whatever CORACPP_BACKEND says (eigen, torch, torch:cuda).
//
//   make example                      builds every example into build/examples/cpp
//   CORACPP_BACKEND=eigen build/examples/cpp/example_linearSys_reach

#include "contDynamics/linearSys/linearSys.h"

#include <iostream>

using namespace cora::ct;

int main() {
    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}});
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));

    // Ten steps of 0.1; R.timeInt[k] covers [0.1 k, 0.1 (k+1)], R.timePoint[k] is at 0.1 k.
    const Reach R = LinearSys(A).reach(X0, /*timeStep=*/0.1, /*tFinal=*/1.0, /*taylorTerms=*/8);

    const Interval box = R.timeInt.back().interval();
    std::cout << "backend: " << backend().name() << "\n"
              << "steps: " << R.timeInt.size() << "\n"
              << "last enclosure lies in the box\n  lower " << box.inf << "\n  upper " << box.sup
              << "\n";
}
