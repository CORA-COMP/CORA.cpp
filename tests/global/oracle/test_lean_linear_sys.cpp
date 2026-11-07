// test_lean_linear_sys - lean::LinearSys::reach: the argument checks, which need no oracle
//
// CORA: contDynamics/@linearSys/reach.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "cora/global/oracle/oracle.h"
#include "testing.h"

using namespace cora;
using test::throws;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &) {
        const lean::DTypeScope binary64("binary64");
        const Tensor A({{0.0, 1.0}, {-1.0, -0.2}});
        const Tensor c({{1.0}, {0.0}}), G({{0.2, 0.0}, {0.0, 0.2}});
        const lean::Zonotope X0(c, G);

        // a non-square matrix and a final time that is no multiple of the step are errors
        test::check(throws([] { lean::LinearSys(Tensor::zeros({2, 3})); }), "non-square A");
        test::check(throws([&] {
                        lean::LinearSys(A).reach(X0, 0.25,
                                                 alg::Standard{.time_step = 0.1,
                                                               .taylor_terms = 4,
                                                               .zonotope_order = 10});
                    }),
                    "tFinal");
        test::check(throws([&] {
                        lean::LinearSys(A).reach(X0, 1.0,
                                                 alg::Standard{.time_step = 0.0,
                                                               .taylor_terms = 4,
                                                               .zonotope_order = 10});
                    }),
                    "timeStep");
    });
    return test::finish("lean linearSys");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
