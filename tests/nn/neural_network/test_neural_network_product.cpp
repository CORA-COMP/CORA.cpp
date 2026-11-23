// test_neural_network_product - the product of two values: points, verify's enclosure, the passes
//
// A product layer maps its output [a; b] to [a o b; 0]. On points; verify's zonotope of it holds
// sampled values (with and without an interval center) and is tighter than the squares
// ((a + b)^2 - (a - b)^2) / 4 when one factor varies much less than the other; verify decides
// through it with the backward bound, the gradient heuristics and alpha. Also the interval center
// through a negative slope (x^2 left of 0, 1 / max(x, 1)), whose radius must stay >= 0.
//
// CORA.cpp only: verify's product of two values (no CORA layer)
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
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

constexpr double kInf = std::numeric_limits<double>::infinity();

/// x (n) -> W x + b, the product of its halves, then the first half summed: one output.
NeuralNetwork aux_product_network(const Tensor &W, const Tensor &b) {
    const int64_t N = W.shape()[0] / 2;
    std::vector<double> first(std::size_t(2 * N), 0.0);
    for (int64_t i = 0; i < N; ++i) first[std::size_t(i)] = 1.0;
    return NeuralNetwork({{W, b}, {Tensor::from_data(first, {1, 2 * N}), Tensor({0.0})}},
                         {Activation::product, Activation::identity});
}

/// The same products as squares: [(a + b) / 2; (a - b) / 2], x^2, then the difference summed.
NeuralNetwork aux_square_network(const Tensor &W, const Tensor &b) {
    const int64_t N = W.shape()[0] / 2, n = W.shape()[1];
    const std::vector<double> w = W.data(), c = b.data();
    std::vector<double> half(std::size_t(2 * N * n)), hb(std::size_t(2 * N)),
        diff(std::size_t(2 * N), 1.0);
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const double a = w[std::size_t(i * n + j)], v = w[std::size_t((i + N) * n + j)];
            half[std::size_t(i * n + j)] = (a + v) / 2;
            half[std::size_t((i + N) * n + j)] = (a - v) / 2;
        }
        hb[std::size_t(i)] = (c[std::size_t(i)] + c[std::size_t(i + N)]) / 2;
        hb[std::size_t(i + N)] = (c[std::size_t(i)] - c[std::size_t(i + N)]) / 2;
        diff[std::size_t(i + N)] = -1.0;
    }
    return NeuralNetwork({{Tensor::from_data(half, {2 * N, n}), Tensor::from_data(hb, {2 * N})},
                          {Tensor::from_data(diff, {1, 2 * N}), Tensor({0.0})}},
                         {Activation::square, Activation::identity});
}

/// verify's output bounds of the box (x, r) (1, n): every input a generator (q0 = n), or q0.
std::pair<double, double> aux_bounds(const NeuralNetwork &nn, const Tensor &x, const Tensor &r,
                                     double q0 = kInf) {
    const ZonoNet net(nn, q0, kInf, "singh", "length");
    const ZonoInput in = net.input_zonotope(x, r);
    const ZonoPropagation out = net.forward(in.c, in.dr, in.G, Tensor(), nullptr, false);
    double rad = out.G.abs().sum(1).reshape({1}).item();
    if (out.dr.defined()) rad += out.dr.reshape({1}).item();
    const double c = out.c.reshape({1}).item();
    return {c - rad, c + rad};
}

void points(const std::string &b) {
    const NeuralNetwork nn(
        {{Tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, -1.0}}), Tensor({0.0, 0.0, 0.0, 0.0})}},
        {Activation::product});
    // [x0 (x0 + x1), x1 (x0 - x1), 0, 0]
    check(test::close(nn.evaluate(Tensor({{2.0, 3.0}})), std::vector<double>{10.0, -3.0, 0.0, 0.0}),
          b + ": the products of the halves at a point");
    // an odd width has no halves
    bool refused = false;
    try {
        (void)NeuralNetwork({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::product});
    } catch (const Error &) {
        refused = true;
    }
    check(refused, b + ": a product layer of an odd width is refused");
    // the set operations outside verify and the sensitivity refuse it
    const NeuralNetwork two = aux_product_network(Tensor({{1.0}, {2.0}}), Tensor({0.0, 1.0}));
    const PolyZonotope X(Tensor({0.5}), Tensor({{0.5}}), Tensor::zeros({0, 1}), {{1}}, {1});
    check(test::raises([&] { (void)two.sensitivity(Tensor({1.0})); }, Error::not_supported) &&
              test::raises([&] { (void)two.evaluate(Zonotope(Tensor({0.5}), Tensor({{0.5}}))); },
                           Error::not_supported) &&
              test::raises(
                  [&] {
                      (void)two.evaluate(X, {.poly_method = PolyMethod::taylor, .orders = {{2}}});
                  },
                  Error::not_supported),
          b + ": the sensitivity and the sets outside verify refuse a product");
}

/// Random products of random maps of a box: verify's bounds hold 200 sampled points, with every
/// input a generator and with one (the rest in the interval center).
void enclosure_holds(const std::string &b) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    int violations = 0;
    for (int trial = 0; trial < 6; ++trial) {
        std::vector<double> w(12), c(4), x0(3), r0(3);
        // a random map, offsets and box
        for (double &v : w) v = U(rng) * 2.0;
        for (double &v : c) v = U(rng) * 3.0;
        for (int i = 0; i < 3; ++i) {
            x0[std::size_t(i)] = U(rng);
            r0[std::size_t(i)] = 0.1 + 0.5 * (U(rng) + 1.0);
        }
        const NeuralNetwork nn =
            aux_product_network(Tensor::from_data(w, {4, 3}), Tensor::from_data(c, {4}));
        const Tensor x = Tensor::from_data(x0, {1, 3}), r = Tensor::from_data(r0, {1, 3});
        // every input a generator, then one
        for (const double q0 : {kInf, 1.0}) {
            const auto [lo, hi] = aux_bounds(nn, x, r, q0);
            for (int s = 0; s < 200; ++s) {
                std::vector<double> p(3);
                for (int i = 0; i < 3; ++i)
                    p[std::size_t(i)] = x0[std::size_t(i)] + r0[std::size_t(i)] * U(rng);
                const double y = nn.evaluate(Tensor::from_data(p, {1, 3})).reshape({1}).item();
                violations += y < lo || y > hi;
            }
        }
    }
    check(violations == 0, b + ": the product's enclosure holds 2400 sampled points");
}

/// x0 in [9, 11], x1 in [-0.01, 0.01]: x0 x1 lies in [-0.11, 0.11]. The product's bound is that
/// (lambda = 0.1); the squares' is [-0.355, 0.355].
void tighter_than_squares(const std::string &b) {
    const Tensor W({{1.0, 0.0}, {0.0, 1.0}}), c({0.0, 0.0});
    const Tensor x({{10.0, 0.0}}), r({{1.0, 0.01}});
    const auto [plo, phi] = aux_bounds(aux_product_network(W, c), x, r);
    const auto [slo, shi] = aux_bounds(aux_square_network(W, c), x, r);
    check(plo <= -0.11 && plo > -0.1101 && phi >= 0.11 && phi < 0.1101,
          b + ": the product's bound is the range [-0.11, 0.11]");
    check(slo < -0.35 && shi > 0.35, b + ": the squares' bound is [-0.355, 0.355]");
    // the same factor twice: x0^2 over [9, 11], the square's line 20 x0 - 100 + [0, 1] (lambda = 1)
    const Tensor twice({{1.0, 0.0}, {1.0, 0.0}});
    const auto [qlo, qhi] = aux_bounds(aux_product_network(twice, c), x, r);
    check(qlo <= 80.0 && qlo > 79.99 && qhi >= 121.0 && qhi < 121.01,
          b + ": x0 x0 is the square's bound [80, 121]");
}

/// A box (1, n) and the unsafe set y_0 >= t (or <= t with below).
NnProperty aux_property(const std::vector<double> &lo, const std::vector<double> &hi, double t,
                        bool below) {
    NnProperty p;
    const auto n = int64_t(lo.size());
    p.lo = Tensor::from_data(lo, {1, n});
    p.hi = Tensor::from_data(hi, {1, n});
    p.A = Tensor::from_data(std::vector<double>{below ? 1.0 : -1.0}, {1, 1, 1});
    p.b = Tensor::from_data(std::vector<double>{below ? t : -t}, {1, 1});
    p.poly_case = {0};
    return p;
}

/// verify on x0 x1 over [9, 11] x [-0.01, 0.01] (at most 0.11): y >= 0.2 is unsat, y >= 0.1 sat,
/// with the VNN-COMP preset (alpha, the backward bound), the gradient heuristics (exact and held
/// slopes) and one input generator (the interval gradient through the product).
void verify_decides(const std::string &b) {
    const NeuralNetwork nn =
        aux_product_network(Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({0.0, 0.0}));
    const std::vector<double> lo{9.0, -0.01}, hi{11.0, 0.01};
    // CORA's naive zonotope bounding goes through evaluate, which refuses a product
    check(test::raises([&] { (void)nn.verify(aux_property(lo, hi, 0.2, false), 5); },
                       Error::not_supported),
          b + ": the naive bounding refuses a product");
    // the preset, and the gradient heuristics with one input generator
    std::vector<std::pair<std::string, NnVerifyOptions>> presets{
        {"vnncomp", vnncomp_options("product")}};
    for (const bool exact : {true, false}) {
        NnVerifyOptions o = vnncomp_options("product");
        o.bounding_method = "conzonotope";
        o.input_split_heuristic = "zono-norm-gradient";
        o.input_generator_heuristic = "zono-norm-gradient";
        o.num_init_gens = 1;
        o.exact_backprop = exact;
        presets.emplace_back(exact ? "gradients" : "held gradients", o);
    }
    // x0 (x0 + x1) over [0, 1]^2 lies in [0, 2], its first bound is [-0.5625, 2]: splits decide
    const NeuralNetwork split =
        aux_product_network(Tensor({{1.0, 0.0}, {1.0, 1.0}}), Tensor({0.0, 0.0}));
    const std::vector<double> lo2{0.0, 0.0}, hi2{1.0, 1.0};
    for (auto [name, o] : presets) {
        if (b == "lean") o.dtype = DType::float64;
        check(nn.verify(aux_property(lo, hi, 0.2, false), 20, o).status == "unsat",
              b + " " + name + ": x0 x1 >= 0.2 is unsat");
        check(nn.verify(aux_property(lo, hi, 0.1, false), 20, o).status == "sat",
              b + " " + name + ": x0 x1 >= 0.1 is sat");
        check(split.verify(aux_property(lo2, hi2, -0.3, true), 20, o).status == "unsat",
              b + " " + name + ": x0 (x0 + x1) <= -0.3 is unsat after splits");
        check(split.verify(aux_property(lo2, hi2, 1.9, false), 20, o).status == "sat",
              b + " " + name + ": x0 (x0 + x1) >= 1.9 is sat");
    }
}

/// The helpers: the halves of an odd width are refused, the derivative of every activation as a
/// slope against central differences, and a product has no line of its bounds alone.
void helpers(const std::string &b) {
    check(test::raises([] { (void)priv_halves(Tensor({{1.0, 2.0, 3.0}})); },
                       Error::dimension_mismatch),
          b + ": the halves of an odd width are refused");
    const Tensor x({{0.3, -0.7, 1.6, 2.4}});
    int wrong = 0;
    for (const Activation a : {Activation::identity, Activation::relu, Activation::sigmoid,
                               Activation::tanh, Activation::square, Activation::sign,
                               Activation::exp, Activation::reciprocal, Activation::product}) {
        // d/dx_j of sum_i g_i f(x)_i with g = 1 for every j: the slope's transpose times ones
        const Tensor g = priv_slope_times_t(a, priv_derivative(a, x), x.ones_like());
        const std::vector<double> got = g.data();
        for (int64_t j = 0; j < 4; ++j) {
            // the central difference in x_j
            std::vector<double> up = x.data(), down = x.data();
            up[std::size_t(j)] += 1e-6;
            down[std::size_t(j)] -= 1e-6;
            const double fd = (priv_apply(a, Tensor::from_data(up, {1, 4})).sum() -
                               priv_apply(a, Tensor::from_data(down, {1, 4})).sum())
                                  .item() /
                              2e-6;
            wrong += std::abs(fd - got[std::size_t(j)]) > 1e-5;
        }
    }
    check(wrong == 0, b + ": the derivatives of the nine activations as slopes");
    check(test::raises(
              [] {
                  (void)priv_verify_line(Activation::product, Tensor({{0.0, 0.0}}),
                                         Tensor({{1.0, 1.0}}), "singh");
              },
              Error::not_supported),
          b + ": a product has no line of its bounds");
}

/// The interval center through a negative slope: (x0 + x1)^2 and 1 / max(x0 + x1, 1) with x1
/// boxed (q0 = 1); the bounds hold sampled points (a negative slope times the radius was not one).
void negative_slopes(const std::string &b) {
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    int violations = 0;
    for (const auto &[a, center] : std::vector<std::pair<Activation, double>>{
             {Activation::square, -3.0}, {Activation::reciprocal, 2.0}}) {
        const NeuralNetwork nn(
            {{Tensor({{1.0, 1.0}}), Tensor({0.0})}, {Tensor({{2.0}}), Tensor({0.5})}},
            {a, Activation::identity});
        const Tensor x({{center, 0.0}}), r({{0.3, 0.4}});
        const auto [lo, hi] = aux_bounds(nn, x, r, 1.0);
        for (int s = 0; s < 400; ++s) {
            const double p0 = center + 0.3 * U(rng), p1 = 0.4 * U(rng);
            const double y = nn.evaluate(Tensor({{p0, p1}})).reshape({1}).item();
            violations += y < lo || y > hi;
        }
    }
    check(violations == 0, b + ": the interval center through negative slopes holds 800 points");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points(b);
        enclosure_holds(b);
        tighter_than_squares(b);
        verify_decides(b);
        negative_slopes(b);
        helpers(b);
    });
    return test::finish("neuralNetwork product");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
