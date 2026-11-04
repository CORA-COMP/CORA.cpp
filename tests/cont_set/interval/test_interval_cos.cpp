// test_interval_cos - interval cos: the cosine of a box by CORA's case analysis
//
// CORA: contSet/@interval/cos.m
// PROPERTY: contSet/@interval/cos.m

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

constexpr double kPi = 3.14159265358979323846;

/// Hand-computed boxes: monotone pieces, a maximum, a minimum, both, and a wide box.
void cos_on_known_boxes(const std::string &b) {
    const Interval I(column({0.0, -1.0, 3.0, 1.0, 1.0, -10.0, 2.0}),
                     column({1.0, 1.0, 4.0, 2.0, 7.0, 20.0, 2.0}));
    const Interval R = I.cos();
    const std::vector<double> inf{std::cos(1.0), std::cos(1.0), -1.0,         std::cos(2.0),
                                  -1.0,          -1.0,          std::cos(2.0)};
    const std::vector<double> sup{1.0, 1.0, std::cos(4.0), std::cos(1.0), 1.0, 1.0, std::cos(2.0)};
    check(close(R.inf(), inf, 1e-12) && close(R.sup(), sup, 1e-12), b + ": cos of known boxes");
    const Interval S = Interval::stack({I, I}).cos();
    check(S.inf().shape() == std::vector<int64_t>({2, 7, 1}), b + ": a batch has a batch shape");
    check(close(S.inf().slice(0, 1, 2).reshape({7, 1}), R.inf()), b + ": a batch member");
}

/// Dense samples of the box give the exact range, which the result matches (up to the sampling)
/// and contains (up to rounding).
void cos_is_the_range(const std::string &b) {
    cora::Rng rng(32);
    const int kSamples = 4000;
    for (int trial = 0; trial < 30; ++trial) {
        std::vector<double> lo(1), hi(1);
        rng.uniform(lo.data(), 1, -12, 12);
        rng.uniform(hi.data(), 1, 0, 8);
        hi[0] += lo[0];
        const Interval R = Interval(column(lo), column(hi)).cos();
        double low = 2, high = -2;
        for (int k = 0; k <= kSamples; ++k) {
            const double c = std::cos(lo[0] + (hi[0] - lo[0]) * k / kSamples);
            low = std::min(low, c);
            high = std::max(high, c);
        }
        const double inf = R.inf().data()[0], sup = R.sup().data()[0];
        check(inf <= low + 1e-12 && sup >= high - 1e-12, b + ": the samples are in the result");
        check(low - inf < 1e-3 && sup - high < 1e-3, b + ": the result is the range");
        check(inf >= -1 - 1e-15 && sup <= 1 + 1e-15, b + ": within [-1, 1]");
    }
}

/// cos of a point is its cosine, at multiples of pi too.
void cos_of_points(const std::string &b) {
    const std::vector<double> x{0.0, kPi, 2 * kPi, -kPi, 5.0};
    const Interval R = Interval(column(x), column(x)).cos();
    std::vector<double> c;
    c.reserve(x.size());
    for (const double v : x) c.push_back(std::cos(v));
    check(close(R.inf(), c, 1e-12) && close(R.sup(), c, 1e-12), b + ": cos of points");
}

/// cos has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Tensor on = I.cos().sup();
    ChecksScope off(false);
    check(close(I.cos().sup(), on, 0), "cos: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        cos_on_known_boxes(b);
        cos_is_the_range(b);
        cos_of_points(b);
    });
    checks_and_their_switch();
    return test::finish("interval cos");
}
