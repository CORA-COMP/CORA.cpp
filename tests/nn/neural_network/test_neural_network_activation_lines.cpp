// test_neural_network_activation_lines - the lines of softplus, root and inverse square root
//
// A line y in slope x + offset +- radius must enclose the function over [l, u] and be tight: the
// radius equals half the range of f(x) - slope x (sampled on a fine grid), for Singh's slope and
// for the regression slope. A point range has no radius, and the domains are checked.
//
// CORA: nn/layers/nonlinear/nnSoftPlusLayer.m
// CORA: nn/layers/nonlinear/nnRootLayer.m
// CORA: nn/layers/nonlinear/nnInvSqrtRootLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <functional>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The function of an activation on a number.
std::function<double(double)> aux_f(Activation a) {
    if (a == Activation::softplus) return [](double x) { return std::log(1.0 + std::exp(x)); };
    if (a == Activation::root) return [](double x) { return std::sqrt(x); };
    return [](double x) { return std::pow(x, -0.5); };
}

/// The range of f(x) - m x over a fine grid of [l, u]: its half width.
double aux_half_range(const std::function<double(double)> &f, double m, double l, double u) {
    double lo = 1e300, hi = -1e300;
    // 20001 points
    for (int i = 0; i <= 20000; ++i) {
        const double x = l + (u - l) * double(i) / 20000.0;
        lo = std::min(lo, f(x) - m * x);
        hi = std::max(hi, f(x) - m * x);
    }
    return (hi - lo) / 2;
}

void lines(const std::string &b) {
    // each activation with the ranges of four neurons (the last a point)
    struct Case {
        Activation a;
        std::vector<double> l, u;
    };
    for (const Case &k : std::vector<Case>{
             {Activation::softplus, {-3.0, 0.0, -0.5, 2.0}, {1.0, 6.0, 0.5, 2.0}},
             {Activation::root, {0.0, 0.25, 1.0, 4.0}, {1.0, 9.0, 1.5, 4.0}},
             {Activation::inv_sqrt_root, {0.1, 0.5, 2.0, 3.0}, {1.0, 9.0, 2.5, 3.0}}}) {
        const auto f = aux_f(k.a);
        for (const bool regression : {false, true}) {
            const Tensor l = test::column(k.l), u = test::column(k.u);
            const PrivLine line = priv_monotone_line(k.a, l, u, regression);
            const std::vector<double> m = line.slope.data(), o = line.offset.data(),
                                      r = line.radius.data();
            // the line against the function on a grid, and against the sampled range
            const std::string what = b + ": activation " + std::to_string(int(k.a)) +
                                     (regression ? " regression" : " singh");
            for (std::size_t i = 0; i < m.size(); ++i) {
                // the line encloses the function, and the radius is the range's half
                double worst = 0.0;
                for (int j = 0; j <= 200; ++j) {
                    const double x = k.l[i] + (k.u[i] - k.l[i]) * double(j) / 200.0;
                    worst = std::max(worst, std::abs(f(x) - (m[i] * x + o[i])) - r[i]);
                }
                check(worst <= 1e-12, what + ": the line encloses the function");
                const double half = aux_half_range(f, m[i], k.l[i], k.u[i]);
                check(r[i] >= half - 1e-12 && r[i] <= half + 1e-5,
                      what + ": the radius is the range's half");
                if (k.l[i] == k.u[i]) check(r[i] <= 1e-12, what + ": a point has no radius");
            }
        }
    }
    // the domains: root takes x >= 0, inv_sqrt_root x > 0
    check(test::raises(
              [] {
                  priv_monotone_line(Activation::root, test::column({-1.0}), test::column({1.0}),
                                     true);
              },
              Error::out_of_domain),
          b + ": root below 0");
    // inv_sqrt_root at 0, another activation, and an overflow
    check(test::raises(
              [] {
                  priv_monotone_line(Activation::inv_sqrt_root, test::column({0.0}),
                                     test::column({1.0}), false);
              },
              Error::out_of_domain),
          b + ": inv_sqrt_root at 0");
    check(test::raises(
              [] {
                  priv_monotone_line(Activation::relu, test::column({0.0}), test::column({1.0}),
                                     true);
              },
              Error::wrong_value),
          b + ": another activation");
    // exp overflows: softplus of a huge range has no line
    check(test::raises(
              [] {
                  priv_monotone_line(Activation::softplus, test::column({0.0}),
                                     test::column({1000.0}), false);
              },
              Error::special_error),
          b + ": a bound beyond the float range");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(priv_monotone_line(Activation::root, test::column({1.0}), test::column({2.0}), true)
                      .radius.numel() == 1,
              b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { lines(b); });
    return test::finish("neuralNetwork activation lines");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
