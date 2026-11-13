// test_lean_linear_sys_output_set - the output set y = C x of a reachable set against CORALean
//
// The oracle's outputSet is C R + D U + k; with no input (the input dimension 0) and no offset it
// is C R, which CORA.cpp's output_set maps over every set of the reachable set.
//
// ORACLE: contDynamics/@linearSys/outputSet.m
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("linearSys.outputSet", [](test::oracle::Random &rnd) {
        for (const auto &[n, p, m] : {std::tuple{2, 1, 3}, {3, 2, 4}, {5, 5, 9}}) {
            const Tensor A = rnd.matrix(n, n), C = rnd.matrix(p, n);
            const Zonotope R = rnd.zonotope(n, m);
            const LinearSys sys(A, Tensor::zeros({n, 0}), C);
            ReachSet set;
            set.time_interval.push_back(R);
            test::oracle::check(sys.output_set(set).time_interval[0],
                                lean::LinearSys(A, std::nullopt, C).output_set(R));
        }
    });
    return test::finish("lean linear_sys output_set");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
