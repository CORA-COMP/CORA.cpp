// example_linear_reach_03_specification - checking a reachable set against a specification
//
// A safe set must contain the reachable set, an unsafe set must not touch it. Both are
// halfspaces {x | a'x <= b}; several halfspaces make a polytope (safe sets only).
//
// Syntax:   build/examples/cpp/example_linear_reach_03_specification
// Outputs:  whether each specification holds, and the step that first violates it
// See also: example_linear_reach_01_5dim

#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora::ct;

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double tFinal = 6.0;
    const Zonotope R0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));

    // Reachability Settings -------------------------------------------------------------------

    const double timeStep = 0.1;
    const int taylorTerms = 8;

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys sys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}}));

    // Specifications --------------------------------------------------------------------------

    // Stay in x1 <= 1.2.
    const Specification specSafe = Specification::safeSet(Tensor({1.0, 0.0}), 1.2);

    // Never touch x1 <= -0.75, a wall on the left.
    const Specification specUnsafe = Specification::unsafeSet(Tensor({1.0, 0.0}), -0.75);

    // Stay in -1 <= x1 <= 1.2: a polytope of two halfspaces.
    const Specification specBand =
        Specification::safeSet({{Tensor({1.0, 0.0}), 1.2}, {Tensor({-1.0, 0.0}), 1.0}});

    // Reachability Analysis -------------------------------------------------------------------

    const Reach R = sys.reach(R0, timeStep, tFinal, taylorTerms);

    // Verification ----------------------------------------------------------------------------

    std::cout << "stays in x1 <= 1.2: " << specSafe.check(R.timeInt) << "\n"
              << "avoids x1 <= -0.75: " << specUnsafe.check(R.timeInt)
              << ", first touches it in step " << specUnsafe.firstViolation(R.timeInt) << "\n"
              << "stays in -1 <= x1 <= 1.2: " << specBand.check(R.timeInt) << "\n"
              << "the initial set is safe: " << specSafe.check(R0) << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
