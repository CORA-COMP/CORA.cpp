// test_lean_interval_norm - Interval::norm against the CORALean oracle (its `norm`)
//
// The oracle answers the squared length rounded up (its `normSq`, the float definition has no
// square root). On the lean backend CORA.cpp's norm is the correctly rounded root of the same sum
// of squares, so it equals the root of the oracle's number bit for bit; in the sound mode it may
// exceed the root of the upper bound by the rounding of the root and the sum, kRelative (4 eps).
//
// ORACLE: contSet/@interval/norm_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <cmath>
#include <limits>

using namespace cora;

namespace {

constexpr double kRelative = 4 * std::numeric_limits<double>::epsilon();

}

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.norm", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            const double root = std::sqrt(lean::Interval(I).norm());
            const double mine = I.norm().data()[0];
            if (test::oracle::mode() == test::oracle::Mode::exact)
                test::oracle::check_exact("interval.norm", mine, root);
            else
                test::oracle::check_enclosed("interval.norm", mine, root * (1 + kRelative));
        }
    });
    return test::finish("lean interval norm");
}
