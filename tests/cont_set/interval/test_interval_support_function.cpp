// test_interval_support_function - interval support_function: the best corner of the box
//
// CORA: contSet/@interval/supportFunc_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The support of a box along `d` is the best corner, `Σ max(d_i lo_i, d_i hi_i)`.
void support_function_matches_the_corners(const std::string &b) {
    cora::Rng rng(1);
    for (const int n : {1, 2, 4}) {
        const Interval I = Interval::generate_random(n, rng);
        const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
        for (int k = 0; k < 5; ++k) {
            std::vector<double> d(n);
            rng.normal(d.data(), d.size(), 1.0);
            double want = 0.0;
            for (int i = 0; i < n; ++i) want += std::max(d[i] * lo[i], d[i] * hi[i]);
            check(close(I.support_function(column(d)).data()[0], want, 1e-12),
                  b + ": support_function in " + std::to_string(n) + "d");
        }
    }
}

/// A direction of the wrong dimension is refused; with the checks off the value is the same.
void checks_and_their_switch() {
    const Interval Z = Interval(test::column({0, -1}), test::column({1, 1}));
    check(test::raises([&] { Z.support_function(test::column({1, 0, 0})); },
                       Error::dimension_mismatch),
          "support_function: a direction of the wrong dimension");
    const double on = test::support(Z, {1, 0});
    ChecksScope off(false);
    check(test::close(test::support(Z, {1, 0}), on, 0), "support_function: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { support_function_matches_the_corners(b); });
    checks_and_their_switch();
    return test::finish("interval support_function");
}
