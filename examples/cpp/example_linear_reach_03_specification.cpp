// example_linear_reach_03_specification - checking a reachable set against a specification
//
// A safe set must contain the reachable set, an unsafe set must not touch it. Both are
// sets, here polytopes {x | A x <= b}: one row per halfspace.
//
// Syntax:   build/examples/cpp/example_linear_reach_03_specification
// Outputs:  whether each specification holds, and the step that first violates it
// See also: example_linear_reach_01_5dim
//
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/specification/specification.h"

#include <iostream>


using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    // Parameters ------------------------------------------------------------------------------

    const double t_final = 6.0;
    const Zonotope R0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));

    // Reachability Settings -------------------------------------------------------------------

    const double time_step = 0.1;
    const int taylor_terms = 8;

    // System Dynamics -------------------------------------------------------------------------

    const LinearSys sys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}}));

    // Specifications --------------------------------------------------------------------------

    // Stay in x1 <= 1.2.
    const Specification spec_safe =
        Specification::safe_set(Polytope(Tensor({{1.0, 0.0}}), Tensor({{1.2}})));

    // Never touch x1 <= -0.75, a wall on the left.
    const Specification spec_unsafe =
        Specification::unsafe_set(Polytope(Tensor({{1.0, 0.0}}), Tensor({{-0.75}})));

    // Stay in -1 <= x1 <= 1.2: a polytope of two rows.
    const Specification spec_band = Specification::safe_set(
        Polytope(Tensor({{1.0, 0.0}, {-1.0, 0.0}}), Tensor({{1.2}, {1.0}})));

    // Reachability Analysis -------------------------------------------------------------------

    const ReachSet R =
        sys.reach({.initial_set = R0, .t_final = t_final},
                  alg::Standard{.time_step = time_step, .taylor_terms = taylor_terms});

    // Verification ----------------------------------------------------------------------------

    std::cout << "stays in x1 <= 1.2: " << spec_safe.check(R.time_interval) << "\n"
              << "avoids x1 <= -0.75: " << spec_unsafe.check(R.time_interval)
              << ", first touches it in step " << spec_unsafe.first_violation(R.time_interval)
              << "\n"
              << "stays in -1 <= x1 <= 1.2: " << spec_band.check(R.time_interval) << "\n"
              << "the initial set is safe: " << spec_safe.check(R0) << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
