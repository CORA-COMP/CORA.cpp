// test_neural_network_softmax - exp, 1 / max(x, 1), a softmax of them and gathers with factors
//
// On points, on sets (the lines hold, a softmax of a box) and in verify; the heuristics, the
// sensitivity and order-2 polynomials through x^2, sign, exp and 1 / max(x, 1); the interval
// center through their negative slopes.
//
// CORA.cpp only: exp and the reciprocal 1 / max(x, 1), a softmax's parts (no CORA layer)
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   23-November-2026
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

/// softmax(x) of n logits as the ONNX front end lowers it: 1 / sum_j exp(x_j - x_i).
NeuralNetwork softmax_network(int64_t n) {
    std::vector<double> diff(std::size_t(n * n * n), 0.0), sum(std::size_t(n * n * n), 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            // row (i, j) of the differences: x_j - x_i
            diff[std::size_t((i * n + j) * n + j)] += 1.0;
            diff[std::size_t((i * n + j) * n + i)] -= 1.0;
            sum[std::size_t(i * n * n + i * n + j)] = 1.0;
        }
    std::vector<double> eye(std::size_t(n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) eye[std::size_t(i * n + i)] = 1.0;
    const std::vector<double> zero(std::size_t(n * n), 0.0);
    return NeuralNetwork(
        {{Tensor::from_data(diff, {n * n, n}), Tensor::from_data(zero, {n * n})},
         {Tensor::from_data(sum, {n, n * n}), Tensor::from_data(std::vector<double>(n, 0.0), {n})},
         {Tensor::from_data(eye, {n, n}), Tensor::from_data(std::vector<double>(n, 0.0), {n})}},
        {Activation::exp, Activation::reciprocal, Activation::identity});
}

/// The softmax of a point.
std::vector<double> aux_softmax(const std::vector<double> &x) {
    double total = 0;
    for (const double v : x) total += std::exp(v);
    std::vector<double> y;
    y.reserve(x.size());
    for (const double v : x) y.push_back(std::exp(v) / total);
    return y;
}

void points(const std::string &b) {
    const NeuralNetwork e({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::exp});
    check(test::close(e.evaluate(Tensor({{0.0}, {1.0}})), std::vector<double>{1.0, std::exp(1.0)}),
          b + ": exp at points");
    const NeuralNetwork r({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::reciprocal});
    check(test::close(r.evaluate(Tensor({{4.0}, {0.5}})), std::vector<double>{0.25, 1.0}),
          b + ": 1 / max(x, 1) at points");
    const std::vector<double> want = aux_softmax({0.3, -1.2, 2.0});
    check(test::close(softmax_network(3).evaluate(Tensor({{0.3, -1.2, 2.0}})), want, 1e-12),
          b + ": the lowered softmax at a point");
}

/// The lines on ranges: every sample of the function inside the enclosure of the range.
void lines_hold(const std::string &b) {
    int violations = 0;
    for (const Activation a : {Activation::exp, Activation::reciprocal}) {
        const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {a});
        for (const auto &[l, u] : std::vector<std::pair<double, double>>{
                 {-3.0, 2.0}, {0.5, 3.0}, {1.0, 40.0}, {-1e-3, 1e-3}, {-50.0, -49.0}, {2.0, 2.0}}) {
            const Interval I =
                nn.evaluate(Zonotope(Tensor({(l + u) / 2}), Tensor({{(u - l) / 2}}))).interval();
            for (int j = 0; j <= 200; ++j) {
                const double x = l + (u - l) * j / 200.0;
                const double y = a == Activation::exp ? std::exp(x) : 1.0 / std::max(x, 1.0);
                violations += y < I.inf().data()[0] || y > I.sup().data()[0];
            }
        }
    }
    check(violations == 0, b + ": the lines of exp and 1 / max(x, 1) hold on six ranges");
}

/// The softmax of a box: the enclosure holds sampled softmax values, inside (0, 1).
void softmax_of_a_box(const std::string &b) {
    const NeuralNetwork nn = softmax_network(3);
    const std::vector<double> c{0.5, -0.5, 1.0}, r{0.2, 0.3, 0.1};
    const Interval I =
        nn.evaluate(Zonotope(Tensor({c[0], c[1], c[2]}),
                             Tensor({{r[0], 0.0, 0.0}, {0.0, r[1], 0.0}, {0.0, 0.0, r[2]}})))
            .interval();
    // a 5 x 5 x 5 grid of the box
    int violations = 0;
    for (int s = 0; s < 125; ++s) {
        const int d0 = s % 5 - 2, d1 = (s / 5) % 5 - 2, d2 = s / 25 - 2;
        const std::vector<double> x{c[0] + r[0] * d0 / 2.0, c[1] + r[1] * d1 / 2.0,
                                    c[2] + r[2] * d2 / 2.0};
        const std::vector<double> y = aux_softmax(x);
        for (std::size_t i = 0; i < 3; ++i)
            violations += y[i] < I.inf().data()[i] || y[i] > I.sup().data()[i];
    }
    check(violations == 0, b + ": the softmax of a box holds 125 sampled points");
    check(I.inf().data()[1] > 0.0 && I.sup().data()[2] < 1.0,
          b + ": the bounds stay inside (0, 1)");
}

/// The gather with factors: points, the transpose and |W| against the dense matrix.
void scaled_gather(const std::string &b) {
    const NeuralNetwork::Layer g = NeuralNetwork::Layer::gather(
        {1, 0, 1, -1}, 2, Tensor({0.0, 0.0, 0.0, 1.0}), Tensor({0.5, -2.0, -0.5, 3.0}));
    const Tensor x({{1.0, 4.0}}), y({{1.0, 2.0, 3.0, 4.0}});
    check(test::close(g.rows(x), std::vector<double>{2.0, -2.0, -2.0, 0.0}),
          b + ": a gather times its factors");
    // the transpose: W^T y with W = [[0, .5], [-2, 0], [0, -.5], [0, 0]]
    check(test::close(g.rows_t(y), std::vector<double>{-4.0, -1.0}), b + ": its transpose");
    check(test::close(g.abs_rows(x), std::vector<double>{2.0, 2.0, 2.0, 0.0}) &&
              test::close(g.abs_rows_t(y), std::vector<double>{4.0, 2.0}),
          b + ": |W| and |W|^T");
    check(test::close(g.dense(), std::vector<double>{0.0, 0.5, -2.0, 0.0, 0.0, -0.5, 0.0, 0.0}),
          b + ": its matrix");
}

/// A box (1, n) and the unsafe set y_k <= t.
NnProperty aux_below(const std::vector<double> &lo, const std::vector<double> &hi, int64_t k,
                     int64_t m, double t) {
    NnProperty p;
    const auto n = int64_t(lo.size());
    p.lo = Tensor::from_data(lo, {1, n});
    p.hi = Tensor::from_data(hi, {1, n});
    std::vector<double> row(std::size_t(m), 0.0);
    row[std::size_t(k)] = 1.0;
    p.A = Tensor::from_data(row, {1, 1, m});
    p.b = Tensor::from_data(std::vector<double>{t}, {1, 1});
    p.poly_case = {0};
    return p;
}

/// verify: class 0 of two logits with x0 - x1 in [0.5, 1.5] has p0 = sigmoid(x0 - x1) in [0.62,
/// 0.82], so p0 <= 0.5 is unsat and p0 <= 0.7 is sat; exp beyond the float range is unknown.
void verify_decides(const std::string &b) {
    const NeuralNetwork nn = softmax_network(2);
    for (const NnVerifyOptions &o : {NnVerifyOptions{}, vnncomp_options("softmax")}) {
        const std::string name = b + " " + o.bounding_method;
        check(nn.verify(aux_below({1.0, 0.0}, {1.5, 0.5}, 0, 2, 0.5), 20, o).status == "unsat",
              name + ": p0 <= 0.5 is unsat");
        check(nn.verify(aux_below({1.0, 0.0}, {1.5, 0.5}, 0, 2, 0.7), 20, o).status == "sat",
              name + ": p0 <= 0.7 is sat");
        const NeuralNetwork e({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::exp});
        check(e.verify(aux_below({0.0}, {2000.0}, 0, 1, -1.0), 2, o).status == "unknown",
              name + ": exp beyond the float range is unknown");
    }
}

/// The four activations of Phase 4 (x^2, sign, exp, 1 / max(x, 1)) after a 2 x 2 map.
NeuralNetwork aux_phase4_network(Activation a) {
    return NeuralNetwork({{Tensor({{1.0, -0.5}, {0.3, 0.8}}), Tensor({0.2, -0.1})},
                          {Tensor({{1.0, 1.0}}), Tensor({0.0})}},
                         {a, Activation::identity});
}

/// The input-generator and split heuristics' interval gradient, the F-radius gradients with and
/// without exact_backprop, and the sensitivity, on the four activations: verify still decides.
void heuristics_through_the_activations(const std::string &b) {
    for (const Activation a :
         {Activation::square, Activation::sign, Activation::exp, Activation::reciprocal}) {
        const NeuralNetwork nn = aux_phase4_network(a);
        // the bound far below every output: unsat with a gradient heuristic and one generator
        Tensor y = nn.evaluate(Tensor({{1.0, 1.0}, {1.4, 0.6}, {0.6, 1.4}, {1.0, 1.4}}));
        const double least = y.reshape({4}).min(0).item();
        for (const bool exact : {true, false}) {
            NnVerifyOptions o = cora_vnncomp_options("test");
            o.bounding_method = "conzonotope";
            o.input_split_heuristic = "zono-norm-gradient";
            o.input_generator_heuristic = "zono-norm-gradient";
            o.num_init_gens = 1;
            o.exact_backprop = exact;
            o.mini_batch_size = 16;
            if (b == "lean") o.dtype = DType::float64;
            const std::string status =
                nn.verify(aux_below({0.6, 0.6}, {1.4, 1.4}, 0, 1, least - 100.0), 10, o).status;
            check(status == "unsat", b + ": a gradient heuristic decides through activation " +
                                         std::to_string(int(a)) + (exact ? "" : " (held slopes)"));
        }
        // the sensitivity: the network's Jacobian (1, 2) at a point off the kinks
        const Tensor J = nn.sensitivity(Tensor({1.0, 1.2}));
        check(J.numel() == 2, b + ": the sensitivity through activation " + std::to_string(int(a)));
    }
}

/// A polynomial zonotope through the four activations by Taylor and Bernstein of order 2: the
/// enclosure holds sampled values, or the method refuses the activation.
void polynomials_of_the_activations(const std::string &b) {
    int violations = 0, evaluated = 0, refused = 0;
    for (const Activation a :
         {Activation::square, Activation::sign, Activation::exp, Activation::reciprocal}) {
        const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {a});
        for (const PolyMethod m : {PolyMethod::taylor, PolyMethod::bernstein}) {
            const double l = a == Activation::reciprocal ? 1.5 : 0.5, u = l + 1.0;
            const PolyZonotope X(Tensor({(l + u) / 2}), Tensor({{(u - l) / 2}}),
                                 Tensor::zeros({0, 1}), {{1}}, {1});
            try {
                const Interval I =
                    nn.evaluate(X, {.poly_method = m, .orders = {{2}}}).zonotope().interval();
                ++evaluated;
                for (int j = 0; j <= 50; ++j) {
                    const double x = l + (u - l) * j / 50.0;
                    const double v = nn.evaluate(Tensor({{x}})).reshape({1}).item();
                    violations += v < I.inf().data()[0] - 1e-9 || v > I.sup().data()[0] + 1e-9;
                }
            } catch (const Error &) {
                ++refused; // a method without a polynomial of this activation
            }
        }
    }
    check(evaluated > 0 && evaluated + refused == 8 && violations == 0,
          b + ": order-2 polynomials of the four activations hold sampled values");
}

/// The interval center through a negative slope: (x0 + x1)^2 and 1 / max(x0 + x1, 1) with x1
/// boxed (q0 = 1); the bounds hold sampled points (a negative slope times the radius was not one).
void negative_slopes(const std::string &b) {
    const double inf = std::numeric_limits<double>::infinity();
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    int violations = 0;
    for (const auto &[a, center] : std::vector<std::pair<Activation, double>>{
             {Activation::square, -3.0}, {Activation::reciprocal, 2.0}}) {
        const NeuralNetwork nn(
            {{Tensor({{1.0, 1.0}}), Tensor({0.0})}, {Tensor({{2.0}}), Tensor({0.5})}},
            {a, Activation::identity});
        // x1 has no generator (q0 = 1): its radius is the interval center's
        const ZonoNet net(nn, 1.0, inf, "singh", "length");
        const ZonoInput in = net.input_zonotope(Tensor({{center, 0.0}}), Tensor({{0.3, 0.4}}));
        const ZonoPropagation out = net.forward(in.c, in.dr, in.G, Tensor(), nullptr, false);
        double rad = out.G.abs().sum(1).reshape({1}).item();
        if (out.dr.defined()) rad += out.dr.reshape({1}).item();
        const double c = out.c.reshape({1}).item();
        // sampled points of the box inside the bounds
        for (int s = 0; s < 400; ++s) {
            const double p0 = center + 0.3 * U(rng), p1 = 0.4 * U(rng);
            const double y = nn.evaluate(Tensor({{p0, p1}})).reshape({1}).item();
            violations += y < c - rad || y > c + rad;
        }
    }
    check(violations == 0, b + ": the interval center through negative slopes holds 800 points");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points(b);
        lines_hold(b);
        softmax_of_a_box(b);
        scaled_gather(b);
        verify_decides(b);
        heuristics_through_the_activations(b);
        polynomials_of_the_activations(b);
        negative_slopes(b);
    });
    return test::finish("neuralNetwork softmax");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
