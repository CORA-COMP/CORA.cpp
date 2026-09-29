// Specifications: check that a reachable set stays in a halfspace (safe set) or out of one
// (unsafe set), and find the step that first breaks it.
//
//   build/examples/cpp/example_linearSys_specification

#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"

#include <iostream>

using namespace cora::ct;

int main() {
    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}});
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const Reach R = LinearSys(A).reach(X0, 0.1, 6.0, 8);

    // A halfspace is {x | a'x <= b}. Safe: the set must stay inside; here x1 <= 1.2.
    const Specification stay_left = Specification::safe_set(Tensor({1.0, 0.0}), 1.2);
    std::cout << "stays in x1 <= 1.2: " << stay_left.check(R.time_int) << "\n";

    // Unsafe: the set must not touch it; here x1 <= -0.75, a wall on the left.
    const Specification wall = Specification::unsafe_set(Tensor({1.0, 0.0}), -0.75);
    std::cout << "avoids x1 <= -0.75: " << wall.check(R.time_int)
              << ", first touches it in step " << wall.first_violation(R.time_int) << "\n";

    // Several halfspaces make a polytope (safe sets only): -1 <= x1 <= 1.2 is two halfspaces.
    const Specification band = Specification::safe_set(
        {{Tensor({1.0, 0.0}), 1.2}, {Tensor({-1.0, 0.0}), 1.0}});
    std::cout << "stays in -1 <= x1 <= 1.2: " << band.check(R.time_int) << "\n";

    // A specification answers for any set, not only reachable ones.
    std::cout << "the initial set is safe: " << stay_left.check(X0) << "\n";
}
