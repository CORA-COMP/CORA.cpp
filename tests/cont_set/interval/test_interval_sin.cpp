// test_interval_sin - interval sin: the sine of a box
//
// CORA: contSet/@interval/sin.m
// PROPERTY: contSet/@interval/sin.m
// CORA authors: Matthias Althoff, Dmitry Grebenyuk, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <numbers>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kPi = std::numbers::pi;

void sin_on_known_boxes(const std::string &b) {
    // monotone pieces, a maximum, a minimum, both, and a full period
    const Interval I(column({0.0, -kPi / 2, 0.0, kPi, -kPi / 2, 0.0}),
                     column({kPi / 2, kPi / 2, kPi, 2 * kPi, 3 * kPi / 2, 10.0}));
    const Interval R = I.sin();
    check(close(R.inf(), std::vector<double>{0, -1, 0, -1, -1, -1}) &&
              close(R.sup(), std::vector<double>{1, 1, 1, 0, 1, 1}),
          b + ": sin of known boxes");
    const Interval S = Interval::stack({I, I}).sin();
    check(S.inf().shape() == std::vector<int64_t>({2, 6, 1}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({6, 1}), R.sup()), b + ": a batch member");
}

/// The sine of every sampled point lies in the result, which is the range of the sine.
void sin_is_the_range(const std::string &b) {
    cora::Rng rng(35);
    for (int trial = 0; trial < 30; ++trial) {
        std::vector<double> lo(1), width(1);
        rng.uniform(lo.data(), 1, -8, 8);
        rng.uniform(width.data(), 1, 0, 7);
        const Interval I(column(lo), column({lo[0] + width[0]})), R = I.sin();
        double low = 1, high = -1;
        for (int k = 0; k <= 2000; ++k) {
            const double s = std::sin(lo[0] + width[0] * k / 2000);
            low = std::min(low, s);
            high = std::max(high, s);
        }
        const double inf = R.inf().data()[0], sup = R.sup().data()[0];
        check(inf <= low + 1e-12 && sup >= high - 1e-12, b + ": the samples are in the result");
        check(low - inf < 1e-4 && sup - high < 1e-4, b + ": the result is the range");
    }
}

/// sin has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 4.0}));
    const Tensor on = I.sin().sup();
    ChecksScope off(false);
    check(close(I.sin().sup(), on, 0), "sin: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        sin_on_known_boxes(b);
        sin_is_the_range(b);
    });
    checks_and_their_switch();
    return test::finish("interval sin");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
