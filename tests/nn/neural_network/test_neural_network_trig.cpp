// test_neural_network_trig - sin and cos as activations (ml4acopf's power flows)
//
// On points, their lines and boxes on random ranges (narrow: the tangent, wide: slope 0), the
// sensitivity, and in verify (the zonotope, alpha-CROWN, the gradient heuristics).
//
// CORA.cpp only: the sin and cos activations (no CORA layer)
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"
#include "testing.h"

#include <cmath>
#include <numbers>
#include <random>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// y = sin(x_1 - x_2) for two inputs: ml4acopf's angle differences.
NeuralNetwork aux_network(Activation a) {
    return NeuralNetwork({{Tensor({{1.0, -1.0}}), Tensor({0.0})}, {Tensor({{1.0}}), Tensor({0.0})}},
                         {a, Activation::identity});
}

/// The box [lo, hi]^2 and the unsafe set y <= t.
NnProperty aux_below(double lo, double hi, double t) {
    NnProperty p;
    p.lo = Tensor::from_data(std::vector<double>{lo, lo}, {1, 2});
    p.hi = Tensor::from_data(std::vector<double>{hi, hi}, {1, 2});
    p.A = Tensor::from_data(std::vector<double>{1.0}, {1, 1, 1});
    p.b = Tensor::from_data(std::vector<double>{t}, {1, 1});
    p.poly_case = {0};
    return p;
}

void points(const std::string &b) {
    check(test::close(aux_network(Activation::sin).evaluate(Tensor({{1.0, 0.5}})),
                      std::vector<double>{std::sin(0.5)}),
          b + ": sin at a point");
    check(test::close(aux_network(Activation::cos).evaluate(Tensor({{1.0, 0.5}})),
                      std::vector<double>{std::cos(0.5)}),
          b + ": cos at a point");
}

/// The lines enclose sin and cos at sampled points of random boxes (widths 0.01 to 20), and the
/// boxes hold the functions' values; a narrow range takes the tangent, a wide one slope 0.
void lines_hold(const std::string &b) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> U(-10.0, 10.0), V(-2.0, 1.3);
    int violations = 0;
    for (const bool cosine : {false, true}) {
        const auto f = [&](double x) { return cosine ? std::cos(x) : std::sin(x); };
        for (int k = 0; k < 80; ++k) {
            const double l = U(rng), u = l + std::pow(10.0, V(rng));
            const PrivLine line = priv_trig_line(Tensor({l}), Tensor({u}), cosine);
            const auto [lo, hi] = priv_trig_box(Tensor({l}), Tensor({u}), cosine);
            const double m = line.slope.reshape({1}).item(), o = line.offset.reshape({1}).item();
            const double r = line.radius.reshape({1}).item();
            for (int s = 0; s <= 50; ++s) {
                const double x = l + (u - l) * s / 50.0, y = f(x);
                violations += y < m * x + o - r || y > m * x + o + r;
                violations += y < lo.reshape({1}).item() || y > hi.reshape({1}).item();
            }
        }
    }
    check(violations == 0, b + ": the lines and boxes of sin and cos hold 8160 points");
    // a narrow range: the tangent (slope cos c); [0, 2 pi]: slope 0 over [-1, 1]
    const PrivLine narrow = priv_trig_line(Tensor({0.4}), Tensor({0.6}), false);
    check(std::abs(narrow.slope.reshape({1}).item() - std::cos(0.5)) < 1e-12 &&
              narrow.radius.reshape({1}).item() < 0.0051,
          b + ": sin over [0.4, 0.6] takes the tangent at 0.5");
    const PrivLine wide = priv_trig_line(Tensor({0.0}), Tensor({2.0 * std::numbers::pi}), true);
    check(wide.slope.reshape({1}).item() == 0.0 &&
              std::abs(wide.radius.reshape({1}).item() - 1.0) < 1e-12,
          b + ": cos over [0, 2 pi] is slope 0 over [-1, 1]");
}

/// verify: sin(x_1 - x_2) on [0, 0.5]^2 is at least sin(-0.5) = -0.479 (y <= -0.6 unsat), and
/// y <= -0.4 is sat; cos(x_1 - x_2) on [0, 3]^2 is at least cos 3 = -0.99 (y <= -1.2 unsat).
void verify_decides(const std::string &b) {
    std::vector<NnVerifyOptions> options{NnVerifyOptions{}, vnncomp_options("trig")};
    // the gradient heuristics with and without the held slopes
    for (const bool exact : {true, false}) {
        NnVerifyOptions o = cora_vnncomp_options("test");
        o.bounding_method = "conzonotope";
        o.input_split_heuristic = "zono-norm-gradient";
        o.input_generator_heuristic = "zono-norm-gradient";
        o.num_init_gens = 1;
        o.exact_backprop = exact;
        options.push_back(o);
    }
    for (NnVerifyOptions o : options) {
        if (b == "lean") o.dtype = DType::float64;
        const std::string name = b + " " + o.bounding_method;
        const NeuralNetwork s = aux_network(Activation::sin), c = aux_network(Activation::cos);
        check(s.verify(aux_below(0.0, 0.5, -0.6), 20, o).status == "unsat",
              name + ": sin: y <= -0.6 is unsat");
        check(s.verify(aux_below(0.0, 0.5, -0.4), 20, o).status == "sat",
              name + ": sin: y <= -0.4 is sat");
        check(c.verify(aux_below(0.0, 3.0, -1.2), 20, o).status == "unsat",
              name + ": cos: y <= -1.2 is unsat");
        // tight: cos(x_1 - x_2) >= cos 3 = -0.98999, so y <= -0.995 needs splits
        check(c.verify(aux_below(0.0, 3.0, -0.995), 60, o).status != "sat",
              name + ": cos: y <= -0.995 is not sat");
    }
    // the sensitivity: d cos(x_1 - x_2) / dx = -sin(x_1 - x_2) (1, -1)
    const Tensor Jc = aux_network(Activation::cos).sensitivity(Tensor({1.0, 0.5}));
    check(test::close(Jc.reshape({2}), std::vector<double>{-std::sin(0.5), std::sin(0.5)}, 1e-12),
          b + ": the sensitivity through cos");
    // the polynomial methods refuse sin (verify's lines only)
    bool refused = false;
    try {
        const PolyZonotope X(Tensor({0.5, 0.5}), Tensor({{0.5}, {0.5}}), Tensor::zeros({0, 2}),
                             {{1}}, {1});
        (void)aux_network(Activation::sin).evaluate(X, {.poly_method = PolyMethod::taylor});
    } catch (const Error &) {
        refused = true;
    }
    check(refused, b + ": a polynomial zonotope through sin is refused");
    // the sensitivity: d sin(x_1 - x_2) / dx = cos(x_1 - x_2) (1, -1)
    const Tensor J = aux_network(Activation::sin).sensitivity(Tensor({1.0, 0.5}));
    check(test::close(J.reshape({2}), std::vector<double>{std::cos(0.5), -std::cos(0.5)}, 1e-12),
          b + ": the sensitivity through sin");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points(b);
        lines_hold(b);
        verify_decides(b);
    });
    return test::finish("neuralNetwork trig");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
