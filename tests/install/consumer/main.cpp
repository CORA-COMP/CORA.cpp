// main - a program against the installed CORA.cpp: a reachable set, an error identifier, a pair
//
// Built by the ctest install/consumer with find_package(CORA) from the install tree only.
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cora.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    using namespace cora;
    // a reachable set of x' = A x from a box
    const LinearSys sys(Tensor({{-1.0, 2.0}, {-2.0, -1.0}}));
    const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    const ReachSet R =
        sys.reach({.initial_set = X0, .t_final = 1.0}, alg::Standard{.time_step = 0.1});
    std::cout << "consumer: " << R.time_point.size() << " time points, checks " << CORACPP_CHECKS
              << "\n";
    // an input error carries CORA's identifier
    try {
        sys.reach({.initial_set = X0, .t_final = -1.0}, alg::Standard{.time_step = 0.1});
        return 1;
    } catch (const Error &e) {
        if (e.id() != Error::wrong_value) return 1;
        std::cout << "consumer: " << e.identifier() << "\n";
    }
    // a pair of the registry
    const Set S = plus(X0, Interval(Tensor({0.0, 0.0}), Tensor({1.0, 1.0})));
    return R.time_point.size() == 11 && S.is<Zonotope>() ? 0 : 1;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
