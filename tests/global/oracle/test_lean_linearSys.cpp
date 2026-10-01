// test_lean_linearSys - lean::LinearSys::reach: the argument checks, and with CORACPP_ORACLE set
// the enclosure of CORA.cpp's own reachable sets

#include "contDynamics/linearSys/linearSys.h"
#include "global/oracle/linearSys.h"
#include "global/oracle/oracle.h"
#include "testing.h"

#include <cstdlib>

using namespace cora;
using test::throws;

int main() {
    test::for_each_backend([](const std::string &) {
        lean::setDType("binary64");
        const Tensor A({{0.0, 1.0}, {-1.0, -0.2}});
        const Tensor c({{1.0}, {0.0}}), G({{0.2, 0.0}, {0.0, 0.2}});
        const lean::Zonotope X0(c, G);

        // a non-square matrix and a final time that is no multiple of the step are errors
        test::check(throws([] { lean::LinearSys(Tensor::zeros({2, 3})); }), "non-square A");
        test::check(throws([&] { lean::LinearSys(A).reach(X0, 0.1, 0.25, 4, 10); }), "tFinal");
        test::check(throws([&] { lean::LinearSys(A).reach(X0, 0.0, 1.0, 4, 10); }), "timeStep");

        if (!std::getenv("CORACPP_ORACLE")) return;

        // the lean sets enclose the intervals of CORA.cpp's reach at every time point
        const double timeStep = 0.125, tFinal = 1.0;
        const lean::ReachSet R = lean::LinearSys(A).reach(X0, timeStep, tFinal, 8, 10);
        test::check(R.timePoint.size() == 9 && R.timeInt.size() == 8, "the number of sets");
        const auto ref = LinearSys(A).reach(Zonotope(c, G), timeStep, tFinal, 8);
        for (std::size_t k = 0; k < R.timePoint.size(); ++k) {
            const auto [lo, hi] = R.timePoint[k].interval().gather();
            const Interval I = ref.timePoint[k].interval();
            const bool encloses = lo.data()[0] <= I.inf.data()[0] + 1e-9 &&
                                  hi.data()[0] >= I.sup.data()[0] - 1e-9;
            test::check(encloses, "time point " + std::to_string(k) + " encloses");
        }
    });
    return test::finish("lean linearSys");
}
