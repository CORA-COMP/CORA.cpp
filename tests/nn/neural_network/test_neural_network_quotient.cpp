// test_neural_network_quotient - x^3, 1 / x and the quotient x^3 / sum x^3 (nn4sys' pensieve)
//
// On points, their lines on random boxes, the out-of-range failure of 1 / x at l <= 0, the bounds
// of 1 / x from the cubes' box, and in verify (the conzonotope, alpha-CROWN, the gradient
// heuristics).
//
// CORA.cpp only: the cube and inverse activations (no CORA layer)
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"
#include "nn/neural_network/private/priv_zono_batch.h"
#include "testing.h"

#include <cmath>
#include <limits>
#include <random>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// y = (p_1 - p_2) / (p_1 + p_2), p = relu(W x + b)^3: pensieve's x^3 / sum x^3 as the front end
/// lowers it (a gather to the cubes, their sum to 1 / x, a product layer of both, its read-out).
NeuralNetwork quotient_network() {
    using Layer = NeuralNetwork::Layer;
    const Layer relu{Tensor({{1.0, -0.5}, {0.3, 0.8}}), Tensor({0.2, 0.1})};
    const Layer cube = Layer::gather({0, 1}, 2, Tensor({0.0, 0.0}));
    const Layer sum{Tensor({{1.0, 1.0}}), Tensor({0.0})};
    // [p_1, p_2, 1 / s, 1 / s]: the cubes by a gather, the inverse by a residual
    Layer product = Layer::gather({0, 1, -1, -1}, 2, Tensor({0.0, 0.0, 0.0, 0.0}));
    product.source = 1;
    product.residuals = {{2, std::make_shared<Layer>(
                                 Layer::gather({-1, -1, 0, 0}, 1, Tensor({0.0, 0.0, 0.0, 0.0})))}};
    const Layer out{Tensor({{1.0, -1.0, 0.0, 0.0}}), Tensor({0.0})};
    return NeuralNetwork({relu, cube, sum, product, out},
                         {Activation::relu, Activation::cube, Activation::inverse,
                          Activation::product, Activation::identity});
}

/// The box [lo, hi] of two inputs and the unsafe set y <= t.
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
    const NeuralNetwork c({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::cube});
    const NeuralNetwork i({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::inverse});
    check(test::close(c.evaluate(Tensor({{2.0}, {-0.5}})), std::vector<double>{8.0, -0.125}),
          b + ": x^3 at points");
    check(test::close(i.evaluate(Tensor({{4.0}, {0.5}})), std::vector<double>{0.25, 2.0}),
          b + ": 1 / x at points");
    // the quotient at x = (0.5, 1): z = (0.2, 0.95)
    const double p1 = 0.008, p2 = 1.05 * 1.05 * 1.05;
    check(test::close(quotient_network().evaluate(Tensor({{0.5, 1.0}})),
                      std::vector<double>{(p1 - p2) / (p1 + p2)}, 1e-12),
          b + ": the quotient at a point");
}

/// The lines enclose x^3 (boxes across 0 and on either side) and 1 / x (l > 0) at sampled points.
void lines_hold(const std::string &b) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(-3.0, 3.0), V(0.0, 1.0);
    int violations = 0;
    const auto inside = [&](const PrivLine &line, double x, double y) {
        const double m = line.slope.reshape({1}).item(), o = line.offset.reshape({1}).item();
        const double r = line.radius.reshape({1}).item();
        return y >= m * x + o - r && y <= m * x + o + r;
    };
    for (int k = 0; k < 60; ++k) {
        const double l = U(rng), u = l + 4.0 * V(rng) * V(rng);
        const PrivLine cube = priv_cube_line(Tensor({l}), Tensor({u}));
        const double lp = 1e-3 + 3.0 * V(rng), up = lp * (1.0 + 20.0 * V(rng));
        const PrivLine inv = priv_inverse_line(Tensor({lp}), Tensor({up}));
        for (int s = 0; s <= 40; ++s) {
            const double x = l + (u - l) * s / 40.0, xp = lp + (up - lp) * s / 40.0;
            violations += !inside(cube, x, x * x * x) + !inside(inv, xp, 1.0 / xp);
        }
    }
    check(violations == 0, b + ": the lines of x^3 and 1 / x hold 4920 points of 120 boxes");
    // a point range: the line is the value (up to the rounding)
    const PrivLine at = priv_cube_line(Tensor({2.0}), Tensor({2.0}));
    check(std::abs(at.slope.reshape({1}).item() - 12.0) < 1e-12 &&
              at.radius.reshape({1}).item() < 1e-12,
          b + ": x^3's line of a point is its tangent");
}

/// 1 / x's band reaches below g(x) = 1 / x - m x at its least point x* = 1 / sqrt(-m) (long
/// double), where g at a rounded sqrt(l u) need not: the lower end is 2 sqrt(-m) (CORALean).
void inverse_lower_end(const std::string &b) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> V(0.0, 1.0);
    int violations = 0;
    for (int k = 0; k < 200; ++k) {
        const double l = std::pow(10.0, -3.0 + 6.0 * V(rng)), u = l * (1.0 + 1e3 * V(rng) * V(rng));
        const PrivLine line = priv_inverse_line(Tensor({l}), Tensor({u}));
        const long double m = line.slope.reshape({1}).item();
        const long double lo = (long double)line.offset.reshape({1}).item() -
                               (long double)line.radius.reshape({1}).item();
        const long double x = 1.0L / std::sqrt(-m);
        violations += lo > 1.0L / x - m * x;
    }
    check(violations == 0, b + ": 1 / x's band is below its least value at 200 boxes");
}

/// 1 / x has no line where l <= 0 (or NaN): verify's out-of-range failure.
void inverse_out_of_range(const std::string &b) {
    for (const double l : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
        bool failed = false;
        try {
            (void)priv_inverse_line(Tensor({l, 1.0}), Tensor({2.0, 2.0}));
        } catch (const Error &error) {
            failed = priv_is_out_of_range(error);
        }
        check(failed, b + ": 1 / x over [" + std::to_string(l) + ", 2] is out of range");
    }
    // x^3 whose cube overflows
    bool failed = false;
    try {
        (void)priv_cube_line(Tensor({1.0}), Tensor({1e120}));
    } catch (const Error &error) {
        failed = priv_is_out_of_range(error);
    }
    check(failed, b + ": x^3 of 1e120 is out of range");
}

/// The bounds of 1 / x's input from the cubes' box: on [-2, -1]^2 a ReLU of the first neuron
/// crosses 0 and the second is 0, so the sum of cubes may be 0 (out of range); on [0.5, 1]^2 its
/// lower bound is at least the sum of the cubes of the ReLUs' lower bounds.
void inverse_reads_the_cubes(const std::string &b) {
    const double inf = std::numeric_limits<double>::infinity();
    const NeuralNetwork nn = quotient_network();
    const ZonoNet net(nn, 2.0, inf, "singh", "length");
    const ZonoInput in = net.input_zonotope(Tensor({{0.75, 0.75}}), Tensor({{0.25, 0.25}}));
    const ZonoPropagation out = net.forward(in.c, in.dr, in.G, Tensor(), nullptr, true);
    // the records of relu, cube, inverse, product: the inverse's input bound
    const double l = out.layers[2].l.reshape({1}).item();
    check(l >= 0.2 * 0.2 * 0.2 + 0.65 * 0.65 * 0.65 - 1e-12,
          b + ": 1 / x's input is at least the cubes' sum of the ReLUs' lower bounds");
    bool failed = false;
    const ZonoInput neg = net.input_zonotope(Tensor({{-1.5, -1.5}}), Tensor({{0.5, 0.5}}));
    try {
        (void)net.forward(neg.c, neg.dr, neg.G, Tensor(), nullptr, true);
    } catch (const Error &error) {
        failed = priv_is_out_of_range(error);
    }
    check(failed, b + ": a sum of cubes that may be 0 is out of range");
}

/// verify on [0.5, 1]^2: y >= -1 always (y <= -1.5 unsat), y <= -0.9 is sat near (0.5, 1); on
/// [-2, -1]^2 the sum may be 0 and the verdict stays open. The conzonotope, alpha-CROWN with the
/// backward bound, and the gradient heuristics with and without exact_backprop.
void verify_decides(const std::string &b) {
    const NeuralNetwork nn = quotient_network();
    // (the zonotope bounding takes chains only: the graph's is the conzonotope one)
    NnVerifyOptions plain;
    plain.bounding_method = "conzonotope";
    std::vector<NnVerifyOptions> options{plain, vnncomp_options("quotient")};
    // the gradient heuristics (the interval gradient) with and without the held slopes
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
        check(nn.verify(aux_below(0.5, 1.0, -1.5), 20, o).status == "unsat",
              name + ": y <= -1.5 is unsat on [0.5, 1]^2");
        const NnVerifyResult sat = nn.verify(aux_below(0.5, 1.0, -0.9), 20, o);
        check(sat.status == "sat" && sat.x.size() == 2, name + ": y <= -0.9 is sat");
        check(nn.verify(aux_below(-2.0, -1.0, -1.5), 1, o).status == "unknown",
              name + ": a sum that may be 0 leaves the verdict open");
    }
    // the sensitivity through x^3 and 1 / x (a chain: 1 / x^3)
    const NeuralNetwork chain({{Tensor({{1.0}}), Tensor({0.5})}, {Tensor({{1.0}}), Tensor({0.0})}},
                              {Activation::cube, Activation::inverse});
    const Tensor J = chain.sensitivity(Tensor({0.5}));
    check(std::abs(J.reshape({1}).item() + 3.0) < 1e-12,
          b + ": d(1 / (x + 0.5)^3)/dx at 0.5 is -3");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points(b);
        lines_hold(b);
        inverse_lower_end(b);
        inverse_out_of_range(b);
        inverse_reads_the_cubes(b);
        verify_decides(b);
    });
    return test::finish("neuralNetwork quotient");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
