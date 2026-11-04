// test_lean_linear_sys - lean::LinearSys::reach: the argument checks, and with the oracle set the
// enclosure of CORA.cpp's own reachable sets
//
// CORA: contDynamics/@linearSys/reach.m  (enclosure against the oracle only, not bit-exact: not yet
// ORACLE)

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/oracle/linear_sys.h"
#include "cora/global/oracle/oracle.h"
#include "oracle_testing.h"
#include "testing.h"

using namespace cora;
using test::throws;

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

        if (!test::oracle::available()) return;

        // the lean sets enclose the intervals of CORA.cpp's reach at every time point
        const double time_step = 0.125, t_final = 1.0;
        const lean::ReachSet R = lean::LinearSys(A).reach(
            X0, t_final,
            alg::Standard{.time_step = time_step, .taylor_terms = 8, .zonotope_order = 10});
        test::check(R.time_point.size() == 9 && R.time_interval.size() == 8, "the number of sets");
        const auto ref =
            LinearSys(A).reach({.initial_set = Zonotope(c, G), .t_final = t_final},
                               alg::Standard{.time_step = time_step, .taylor_terms = 8});
        for (std::size_t k = 0; k < R.time_point.size(); ++k)
            test::oracle::check_enclosed("time point " + std::to_string(k),
                                         ref.time_point[k].interval(), R.time_point[k].interval());
    });
    return test::finish("lean linearSys");
}
