// test_neural_network_evaluate_taylm - the network on Taylor models
//
// A Taylor model per input goes through the layers (the map of each layer, then per neuron the
// polynomial of its bounds with the error as the remainder). The tests check exact cases by hand
// (an identity network, a stable ReLU) and that the outputs of sampled points lie in the bounds
// (orders 1 to 3, both methods). The fits: at order 1 a neuron takes the zonotope path's line as it
// is (priv_line), whose band is the true range of relu(x) - s x; the ReLU's parabola of order 2
// (singh) has the band of the true range of p - relu, the vertex value in closed form, on boxes of
// every scale (long double, up to the roundings of the terms). The ORACLE test is
// test_lean_nn_evaluate_taylm (ReLU orders 1 and 2, sigmoid and tanh at order 1).
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateTaylm.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/evaluateTaylm.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       23-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_activation_poly.h"
#include "nn/neural_network/private/priv_layers.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// A two-layer network 2 -> 3 -> 2 with the activation a after the first layer.
NeuralNetwork aux_network(Activation a) {
    return NeuralNetwork(
        {{Tensor({{1.0, -2.0}, {0.5, 1.5}, {-1.0, 1.0}}), Tensor({0.1, -0.2, 0.3})},
         {Tensor({{1.0, 1.0, -1.0}, {0.5, -1.0, 2.0}}), Tensor({0.0, 0.2})}},
        {a, Activation::identity});
}

/// The models of the box [lo, hi]^2 and the outputs of the network at the points of its grid
/// lie in the bounds of the models of the outputs.
void taylm_contains(const std::string &b, Activation a, const NnOptions &options, double lo,
                    double hi) {
    const NeuralNetwork nn = aux_network(a);
    const std::vector<Taylm> y = nn.evaluate_taylm(
        Taylm::from_interval(Interval(Tensor({lo, lo}), Tensor({hi, hi}))), options);
    check(y.size() == 2, b + ": one model per output");
    bool inside = true;
    // the grid of the box and 20 random points
    Rng rng(4);
    std::vector<double> random(40);
    rng.uniform(random.data(), random.size(), lo, hi);
    for (int i = 0; i < 9 + 20; ++i)
        for (int j = 0; j <= (i < 9 ? 8 : 0); ++j) {
            const double x = i < 9 ? lo + (hi - lo) * i / 8.0 : random[2 * std::size_t(i - 9)],
                         z = i < 9 ? lo + (hi - lo) * j / 8.0 : random[2 * std::size_t(i - 9) + 1];
            const Tensor out = nn.evaluate(Tensor({{x, z}}));
            const std::vector<double> v = out.data();
            for (std::size_t k = 0; k < 2; ++k) {
                const std::array<double, 2> r = y[k].bounds();
                inside = inside && r[0] <= v[k] + 1e-12 && v[k] - 1e-12 <= r[1];
            }
        }
    check(inside, b + ": the sampled outputs lie in the bounds");
}

void taylm(const std::string &b) {
    // an identity network is exact: the bounds of the image of the box
    const NeuralNetwork lin({{Tensor({{1.0, 2.0}}), Tensor({0.5})}}, {Activation::identity});
    const std::vector<Taylm> y =
        lin.evaluate_taylm(Taylm::from_interval(Interval(Tensor({0.0, -1.0}), Tensor({1.0, 1.0}))));
    check(test::close(y[0].bounds()[0], -1.5, 1e-12) && test::close(y[0].bounds()[1], 3.5, 1e-12),
          b + ": the identity network is the image of the box");
    // a ReLU that is active over the whole box keeps the line
    const NeuralNetwork relu({{Tensor({{1.0}}), Tensor({3.0})}}, {Activation::relu});
    const std::vector<Taylm> r = relu.evaluate_taylm({Taylm::from_range(0.0, 1.0, 1)});
    check(test::close(r[0].bounds()[0], 3.0, 1e-12) && test::close(r[0].bounds()[1], 4.0, 1e-12),
          b + ": a stable ReLU is exact");
    // the sampled outputs lie in the models, per activation, method and order
    NnOptions singh, regression, order3;
    regression.poly_method = PolyMethod::regression;
    order3.orders = {{3}};
    order3.poly_method = PolyMethod::regression;
    taylm_contains(b + " relu", Activation::relu, singh, -1.0, 1.0);
    taylm_contains(b + " sigmoid regression", Activation::sigmoid, regression, -1.0, 1.0);
    taylm_contains(b + " tanh order 3", Activation::tanh, order3, -0.5, 1.0);
    NnOptions per_neuron;
    per_neuron.orders = {{1, 2, 3}};
    per_neuron.poly_method = PolyMethod::regression;
    taylm_contains(b + " tanh per neuron", Activation::tanh, per_neuron, -1.0, 0.5);
    // a list of another size takes its largest order
    NnOptions other;
    other.orders = {{1, 2}};
    taylm_contains(b + " relu largest order", Activation::relu, other, -1.0, 1.0);
    // the checks
    check(test::raises([&] { lin.evaluate_taylm({Taylm::constant(0.0, 1)}); },
                       Error::dimension_mismatch),
          b + ": a wrong number of models");
    check(test::raises(
              [&] { lin.evaluate_taylm({Taylm::constant(0.0, 1), Taylm::constant(0.0, 2)}); },
              Error::dimension_mismatch),
          b + ": models over different variables");
    const NeuralNetwork graph({NeuralNetwork::Layer::linear(Tensor({{1.0}}), Tensor({0.0})),
                               [] {
                                   NeuralNetwork::Layer L =
                                       NeuralNetwork::Layer::linear(Tensor({{1.0}}), Tensor({0.0}));
                                   L.source = -1;
                                   L.residuals = {{0, nullptr}};
                                   return L;
                               }()},
                              {Activation::identity, Activation::identity});
    check(test::raises([&] { graph.evaluate_taylm({Taylm::constant(0.0, 1)}); },
                       Error::not_supported),
          b + ": a graph");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(lin.evaluate_taylm(
                     Taylm::from_interval(Interval(Tensor({0.0, 0.0}), Tensor({1.0, 1.0}))))
                      .size() == 1,
              b + ": a valid call without checks");
    }
}


/// Boxes [l, u] across 0 of the scales 1e-3 to 1e3.
std::vector<std::pair<double, double>> aux_crossing_boxes(int count) {
    std::mt19937_64 rng(17);
    std::uniform_real_distribution<double> unit(1e-3, 1.0);
    std::vector<std::pair<double, double>> out;
    for (int i = 0; i < count; ++i) {
        const double scale = std::pow(10.0, double(i % 7) - 3.0);
        out.emplace_back(-unit(rng) * scale, unit(rng) * scale);
    }
    return out;
}

/// The slack of the roundings of the terms of a fit over [l, u]: 16 ulps of the largest end.
long double aux_slack(double l, double u) {
    return 16.0L * std::numeric_limits<double>::epsilon() * std::max(std::abs(l), std::abs(u));
}

/// At order 1 a neuron is the line of priv_line as it is: offset + s x, the radius the remainder.
/// Its band is the true range of relu(x) - s x over [l, u] (at l, 0 and u).
void order_one(const std::string &b) {
    int structure = 0, misses = 0;
    for (const auto &[l0, u0] : aux_crossing_boxes(70)) {
        // the model c + h t over t in [-1, 1], its bounds about [l0, u0]
        const double c = (l0 + u0) * 0.5, h = (u0 - l0) * 0.5;
        const Taylm x = Taylm::constant(c, 1, 2).plus(
            Taylm::from_interval(Interval(Tensor({-h}), Tensor({h})), 2)[0]);
        const auto [l, u] = x.bounds();
        const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::relu});
        const Taylm y = nn.evaluate_taylm({x})[0];
        const PrivLine line = priv_line(Activation::relu, Tensor({l}), Tensor({u}), NnOptions{});
        const double s = line.slope.item(), o = line.offset.item(), r = line.radius.item();
        // the terms: the offset plus s c, the slope times h, the radius as the remainder
        const Taylm expected =
            Taylm::constant(o, 1, 2).plus(Taylm::from_range(-r, r, 1, 2)).plus(x.power(1).times(s));
        structure +=
            !(y.coefficients() == expected.coefficients() && y.remainder() == expected.remainder());
        // the band holds relu(x) - s x at the ends and at 0
        for (const long double v : {(long double)l, 0.0L, (long double)u}) {
            const long double g = std::max(v, 0.0L) - (long double)s * v;
            misses += g < (long double)o - r - aux_slack(l, u) ||
                      (long double)o + r + aux_slack(l, u) < g;
        }
    }
    check(structure == 0, b + ": order 1 is the zonotope path's line as it is");
    check(misses == 0, b + ": the line's band holds relu over the box");
}

/// The least and largest of a x^2 + b1 x + c over [x0, x1] in long double, the vertex exactly.
std::pair<long double, long double> aux_parabola(long double a, long double b1, long double c,
                                                 long double x0, long double x1) {
    const auto q = [&](long double x) { return (a * x + b1) * x + c; };
    long double lo = std::min(q(x0), q(x1));
    const long double hi = std::max(q(x0), q(x1)), vertex = -b1 / (2.0L * a);
    if (x0 <= vertex && vertex <= x1) lo = std::min(lo, q(vertex));
    return {lo, hi};
}

/// The ReLU's parabola of order 2 (singh): -d <= p - relu <= d on [l, u], d the larger distance of
/// the band's middle to its ends (tight: one end at d), the vertex value in closed form; a stable
/// box is the line; a box whose (u - l)^2 overflows falls back to the line of order 1.
void order_two(const std::string &b) {
    int misses = 0, loose = 0;
    for (const auto &[l, u] : aux_crossing_boxes(140)) {
        const PrivApprox p = priv_approx_poly(Activation::relu, l, u, 2, PolyMethod::singh);
        const long double a = p.coeffs[0], b1 = p.coeffs[1], c = p.coeffs[2], d = p.d;
        const auto [lo1, hi1] = aux_parabola(a, b1, c, l, 0.0L);
        const auto [lo2, hi2] = aux_parabola(a, b1 - 1.0L, c, 0.0L, u);
        const long double lo = std::min(lo1, lo2), hi = std::max(hi1, hi2), slack = aux_slack(l, u);
        misses += lo < -d - slack || d + slack < hi;
        loose += std::max(-lo, hi) < d - slack;
    }
    check(misses == 0, b + ": the parabola's band holds relu over the box");
    check(loose == 0, b + ": the band is the true range (the vertex value, not above it)");
    // a stable box: the line, no error
    const PrivApprox above = priv_approx_poly(Activation::relu, 1.0, 2.0, 2, PolyMethod::singh);
    const PrivApprox below = priv_approx_poly(Activation::relu, -2.0, -1.0, 2, PolyMethod::singh);
    check(above.coeffs == std::vector<double>{1.0, 0.0} && above.d == 0.0 &&
              below.coeffs == std::vector<double>{0.0, 0.0} && below.d == 0.0,
          b + ": a stable box is the line");
    // (u - l)^2 beyond the float range: the line of order 1
    const PrivApprox wide = priv_approx_poly(Activation::relu, -1e200, 1e200, 2, PolyMethod::singh);
    const PrivLine line =
        priv_line(Activation::relu, Tensor({-1e200}), Tensor({1e200}), NnOptions{});
    check(wide.coeffs == std::vector<double>{line.slope.item(), line.offset.item()} &&
              wide.d == line.radius.item(),
          b + ": an overflowing box falls back to the line");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        taylm(b);
        order_one(b);
        order_two(b);
    });
    return test::finish("neuralNetwork evaluate on Taylor models");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
