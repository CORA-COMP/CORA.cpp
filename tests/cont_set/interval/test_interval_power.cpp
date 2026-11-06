// test_interval_power - interval power: the elementwise power of a box
//
// CORA: contSet/@interval/power.m
// PROPERTY: contSet/@interval/power.m
// CORA authors: Dmitry Grebenyuk
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Integer exponents: the hull of the end powers, 0 below an even power over a sign change.
void power_with_integer_exponents(const std::string &b) {
    const Interval I(column({-2.0, 1.0, -3.0, -2.0}), column({3.0, 2.0, -1.0, 0.0}));
    check(close(I.power(2).inf(), std::vector<double>{0, 1, 1, 0}) &&
              close(I.power(2).sup(), std::vector<double>{9, 4, 9, 4}),
          b + ": the square");
    check(close(I.power(3).inf(), std::vector<double>{-8, 1, -27, -8}) &&
              close(I.power(3).sup(), std::vector<double>{27, 8, -1, 0}),
          b + ": the cube");
    check(close(I.power(0).inf(), std::vector<double>{1, 1, 1, 1}) &&
              close(I.power(0).sup(), std::vector<double>{1, 1, 1, 1}),
          b + ": the power 0 is 1");
    check(I.power(1).inf().data() == I.inf().data(), b + ": the power 1 is the box");
    const Interval S = Interval::stack({I, I}).power(2);
    check(S.inf().shape() == std::vector<int64_t>({2, 4, 1}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({4, 1}), I.power(2).sup()), b + ": a batch member");
}

/// The powers are the products 1 * x * ... * x in this order, and pow beyond kPowerProduct.
void power_multiplies_in_order(const std::string &b) {
    cora::Rng rng(8);
    std::vector<double> x(50);
    rng.uniform(x.data(), 50, 1, 2);
    bool same = true;
    for (const double v : x) {
        const Interval R = Interval(column({v}), column({v})).power(3);
        same = same && R.inf().data()[0] == (v * v) * v;
    }
    check(same, b + ": x^3 is (x x) x");
    const Interval big = Interval(column({1.01}), column({1.02})).power(100);
    check(close(big.inf(), std::vector<double>{std::pow(1.01, 100)}, 1e-12) &&
              close(big.sup(), std::vector<double>{std::pow(1.02, 100)}, 1e-12),
          b + ": a long power");
}

/// A negative exponent is the power of the reciprocal.
void power_with_negative_exponents(const std::string &b) {
    const Interval I(column({2.0, -4.0}), column({4.0, -2.0}));
    check(close(I.power(-1).inf(), std::vector<double>{0.25, -0.5}) &&
              close(I.power(-1).sup(), std::vector<double>{0.5, -0.25}),
          b + ": the power -1");
    check(close(I.power(-2).inf(), std::vector<double>{1.0 / 16, 1.0 / 16}) &&
              close(I.power(-2).sup(), std::vector<double>{0.25, 0.25}),
          b + ": the power -2");
    const Interval Z(column({-1.0}), column({2.0}));
    check(I.power(0).inf().data()[0] == 1 && Z.power(-1).inf().data()[0] == -kInf &&
              Z.power(-1).sup().data()[0] == kInf,
          b + ": a box around 0 has the whole line as reciprocal");
    check(Z.power(-2).inf().data()[0] == 0 && Z.power(-2).sup().data()[0] == kInf,
          b + ": and a half line as even power");
}

/// Other exponents are [inf^e, sup^e] for a nonnegative box, and NaN (refused) below 0.
void power_with_other_exponents(const std::string &b) {
    const Interval I(column({4.0, 0.0}), column({9.0, 1.0}));
    check(close(I.power(0.5).inf(), std::vector<double>{2, 0}) &&
              close(I.power(0.5).sup(), std::vector<double>{3, 1}),
          b + ": the square root");
    check(close(I.power(-0.5).inf(), std::vector<double>{1.0 / 3, 1}) &&
              close(I.power(-0.5).sup().slice(0, 0, 1), std::vector<double>{0.5}) &&
              I.power(-0.5).sup().data()[1] == kInf,
          b + ": a negative fractional exponent is the power of the reciprocal");
}

/// An array of exponents; a box that is one number with an array of exponents.
void power_with_arrays(const std::string &b) {
    const Interval I(column({2.0, -3.0, 4.0}), column({3.0, 2.0, 5.0}));
    const Tensor e = column({2.0, 3.0, 0.5});
    const Interval R = I.power(e);
    check(close(R.inf(), std::vector<double>{4, -27, 2}) &&
              close(R.sup(), std::vector<double>{9, 8, std::sqrt(5.0)}),
          b + ": mixed integer and fractional exponents");
    const Interval one(column({2.0}), column({3.0}));
    const Interval T = one.power(column({1.0, 2.0, -1.0}));
    check(close(T.inf(), std::vector<double>{2, 4, 1.0 / 3}) &&
              close(T.sup(), std::vector<double>{3, 9, 0.5}),
          b + ": one box, an array of exponents");
}

/// Boxes as exponents: numbers to a box, and a box to a box.
void power_with_box_exponents(const std::string &b) {
    const Interval E(column({1.0}), column({3.0}));
    const Interval R = Interval::power(column({2.0}), E);
    check(close(R.inf(), std::vector<double>{2}) && close(R.sup(), std::vector<double>{8}),
          b + ": a number to a box");
    const Interval T =
        Interval(column({2.0}), column({3.0})).power(Interval(column({1.0}), column({2.0})));
    check(close(T.inf(), std::vector<double>{2}) && close(T.sup(), std::vector<double>{9}),
          b + ": a box to a box");
}

/// Every power of a sampled point is in the result of the exponents -3..4.
void power_encloses_the_image(const std::string &b) {
    cora::Rng rng(41);
    for (int e = -3; e <= 4; ++e)
        for (int trial = 0; trial < 10; ++trial) {
            std::vector<double> lo(2), hi(2);
            rng.uniform(lo.data(), 2, 0.5, 2);
            rng.uniform(hi.data(), 2, 0.5, 2);
            for (int i = 0; i < 2; ++i) {
                if (lo[i] > hi[i]) std::swap(lo[i], hi[i]);
                if (trial % 2) lo[i] = -lo[i];
            }
            if (e < 0 && (lo[0] < 0 || lo[1] < 0)) continue;
            const Interval I(column(lo), column(hi)), R = I.power(e);
            const Tensor x = I.rand_point(20, rng);
            bool inside = true;
            for (int k = 0; k < 20; ++k)
                inside = inside && R.contains(x.slice(-1, k, k + 1).pow(double(e)));
            check(inside, b + ": the e-th power of a sample is in the result");
        }
}

/// A negative base with a fractional exponent, a base without a value, and exponents of the wrong
/// shape are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({1.0, 2.0}), column({2.0, 3.0}));
    check(test::raises([&] { Interval(column({-1.0}), column({4.0})).power(0.5); },
                       Error::out_of_domain),
          "power: a negative base with a fractional exponent");
    check(test::raises([&] { Interval(column({0.0}), column({0.0})).power(-1); },
                       Error::out_of_domain),
          "power: the reciprocal of [0, 0]");
    check(test::raises(
              [&] { Interval::power(column({-2.0}), Interval(column({1.0}), column({2.0}))); },
              Error::out_of_domain),
          "power: a negative number to a box");
    check(test::raises([&] { I.power(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "power: three exponents for two coordinates");
    check(test::raises([&] { I.power(Interval(column({1, 2, 3}), column({1, 2, 3}))); },
                       Error::dimension_mismatch),
          "power: three exponent boxes for two coordinates");
    const Tensor on = I.power(3).sup();
    ChecksScope off(false);
    // the same with the checks off
    check(close(I.power(3).sup(), on, 0), "power: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        power_with_integer_exponents(b);
        power_multiplies_in_order(b);
        power_with_negative_exponents(b);
        power_with_other_exponents(b);
        power_with_arrays(b);
        power_with_box_exponents(b);
        power_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval power");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
