// test_interval_cosh - interval cosh: the hyperbolic cosine of a box
//
// CORA: contSet/@interval/cosh.m
// PROPERTY: contSet/@interval/cosh.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// One box per case of CORA: left of 0, right of 0, around 0 (either side larger), a point at 0.
void cosh_on_known_boxes(const std::string &b) {
    const Interval I(column({-2.0, 1.0, -1.0, -3.0, 0.0, -1.0, 0.0}),
                     column({-1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 1.0}));
    const Interval R = I.cosh();
    const std::vector<double> inf{std::cosh(1.0), std::cosh(1.0), 1.0, 1.0, 1.0, 1.0, 1.0};
    const std::vector<double> sup{
        std::cosh(2.0), std::cosh(2.0), std::cosh(2.0), std::cosh(3.0), 1.0,
        std::cosh(1.0), std::cosh(1.0)};
    check(close(R.inf(), inf) && close(R.sup(), sup), b + ": cosh of known boxes");
    const Interval S = Interval::stack({I, I}).cosh();
    check(S.inf().shape() == std::vector<int64_t>({2, 7, 1}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({7, 1}), R.sup()), b + ": a batch member");
}

/// The image of every sampled point lies in the result, and the result is the range.
void cosh_is_the_range(const std::string &b) {
    cora::Rng rng(34);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> lo(1), hi(1);
        rng.uniform(lo.data(), 1, -4, 4);
        rng.uniform(hi.data(), 1, 0, 4);
        hi[0] += lo[0];
        const Interval I(column(lo), column(hi)), R = I.cosh();
        double low = 1e300, high = 0;
        for (int k = 0; k <= 400; ++k) {
            const double c = std::cosh(lo[0] + (hi[0] - lo[0]) * k / 400);
            low = std::min(low, c);
            high = std::max(high, c);
        }
        const double inf = R.inf().data()[0], sup = R.sup().data()[0];
        check(inf <= low + 1e-12 && sup >= high - 1e-12, b + ": the samples are in the result");
        check(low - inf < 1e-3 && sup - high < 1e-12, b + ": the result is the range");
    }
}

/// cosh has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Tensor on = I.cosh().sup();
    ChecksScope off(false);
    check(close(I.cosh().sup(), on, 0), "cosh: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        cosh_on_known_boxes(b);
        cosh_is_the_range(b);
    });
    checks_and_their_switch();
    return test::finish("interval cosh");
}
