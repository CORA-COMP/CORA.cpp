// test_lean_interval_support_function_bound - Interval::support_function against the CORALean oracle,
// as a bound
//
// The oracle's value is `Interval.boundDot`, an upper bound on the support function that sums in
// its own order, so CORA.cpp's value (the same sum in another order) differs by a few ulp. CORA.cpp's
// value does not exceed the oracle's upper bound (sound mode) and agrees with it to a relative
// 1e-12 (both modes). No bit-exact test exists, so there is no ORACLE tag.

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.supportFunc", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            const Tensor d = rnd.direction(n);
            const double mine = I.support_function(d).item();
            const double theirs = lean::Interval(I).support_function(d);
            test::check(std::abs(mine - theirs) <= 1e-12 * std::max(1.0, std::abs(theirs)),
                        "interval.supportFunc: agrees with the oracle's bound");
            if (test::oracle::mode() == test::oracle::Mode::sound)
                test::oracle::check_enclosed("interval.supportFunc", mine, theirs);
        }
    });
    return test::finish("lean interval support_function");
}
