// test_neural_network_order_k - polynomials of order k on polynomial zonotopes, and refine
//
// A neuron of order k is enclosed by a polynomial of that order and an error, evaluated by the
// powers of its set. The tests check an exact case by hand (a ReLU of order 2 by Singh's
// polynomial), that order 1 through the neuron-wise path is the vectorised line bit for bit, that
// the sets contain the outputs of sampled points (every poly_method, orders 1 to 4, sets with
// even exponents and independent generators, which CORA's calcSquared misses), that a limit on the
// generators keeps the enclosure, and that refine raises the orders its heuristic names. CORALean
// has the float polynomial evaluation (Layer.activationPoly) but the oracle serves no op for it
// (docs/oracle-requests.md), so these are property tests.
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotopeNeuron.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/nnActivationLayer.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotopeNeuron.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/nnActivationLayer.m
// CORA authors: Tobias Ladner, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

double aux_f(Activation a, double x) {
    return a == Activation::tanh      ? std::tanh(x)
           : a == Activation::sigmoid ? 1.0 / (1.0 + std::exp(-x))
                                      : std::max(x, 0.0);
}

/// The enclosure [lo, hi] of the set of a one-dimensional polynomial zonotope.
std::pair<double, double> aux_hull(const PolyZonotope &P) {
    const Interval I = P.zonotope().interval();
    return {I.inf().data()[0], I.sup().data()[0]};
}

/// The neuron a(x) over x = c + g alpha (plus the independent b with the radius gi) as a network.
PolyZonotope aux_neuron(Activation a, const PolyZonotope &X, int order, PolyMethod method,
                        bool in_gi = true) {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {a});
    return nn.evaluate(
        X, {.add_approx_error_to_gi = in_gi, .poly_method = method, .orders = {{order}}});
}

/// x = c + g alpha over alpha in [-1, 1].
PolyZonotope aux_interval_set(double l, double u) {
    return PolyZonotope(Tensor({(l + u) / 2}), Tensor({{(u - l) / 2}}), Tensor::zeros({1, 0}),
                        {{1}}, {1});
}

/// The ReLU of order 2 by Singh's polynomial over [-1, 3] is 3/16 (x + 1)^2 + 7/96 +- 25/96.
void a_relu_of_order_2_is_the_hand_computed_polynomial(const std::string &b) {
    const PolyZonotope Y =
        aux_neuron(Activation::relu, aux_interval_set(-1.0, 3.0), 2, PolyMethod::singh);
    // x = 1 + 2 alpha: the centre, alpha, alpha^2 and the error as an independent generator
    check(test::close(Y.center(), {79.0 / 96.0}), b + ": the centre");
    check(Y.exponents() == Exponents({{1}, {2}}), b + ": the exponents are alpha and alpha^2");
    check(test::close(Y.generators(), Tensor({{1.5, 0.75}})), b + ": the generators");
    check(test::close(Y.independent_generators(), Tensor({{25.0 / 96.0}})), b + ": the error");
}

/// Order 1 through the neuron-wise path (a limit on the generators that is never reached) is the
/// vectorised line of the other path, bit for bit, for the errors as generators and as factors.
void order_1_is_the_line_bit_for_bit(const std::string &b) {
    // a network of two layers on a zonotope as polynomial zonotope
    Rng rng(4);
    const NeuralNetwork nn(
        {{Tensor({{1.0, -2.0}, {0.5, 1.5}, {-1.0, 1.0}}), Tensor({0.1, -0.2, 0.3})},
         {Tensor({{1.0, 1.0, -1.0}}), Tensor({0.0})}},
        {Activation::tanh, Activation::sigmoid});
    const PolyZonotope X(Zonotope::generate_random(rng, 2, std::nullopt, 3));
    // both error forms and both lines
    for (const bool in_gi : {true, false}) {
        for (const PolyMethod method : {PolyMethod::singh, PolyMethod::regression}) {
            const PolyZonotope line =
                nn.evaluate(X, {.add_approx_error_to_gi = in_gi, .poly_method = method});
            const PolyZonotope poly = nn.evaluate(X, {.add_approx_error_to_gi = in_gi,
                                                      .poly_method = method,
                                                      .num_generators = 1000000});
            check(
                test::close(line.center(), poly.center(), 0) &&
                    test::close(line.generators(), poly.generators(), 0) &&
                    test::close(line.independent_generators(), poly.independent_generators(), 0) &&
                    line.exponents() == poly.exponents() && line.ids() == poly.ids(),
                b + ": order 1 equals the line (" + (in_gi ? "generators" : "factors") + ")");
        }
    }
}

/// The outputs of the points of a network enclosed by the zonotope of the output set.
int aux_outside(const NeuralNetwork &nn, const PolyZonotope &X, const PolyZonotope &Y, Rng &rng) {
    const Tensor x = X.rand_point(400, rng);
    const std::vector<double> y = nn.evaluate(x).data();
    const Interval I = Y.zonotope().interval();
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    const std::size_t m = lo.size();
    int outside = 0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < 400; ++j)
            outside += y[i * 400 + j] < lo[i] - 1e-9 || y[i * 400 + j] > hi[i] + 1e-9;
    return outside;
}

/// Every poly_method and order, on narrow, wide and one-sided ranges: the curve stays inside.
void every_method_encloses_the_activation(const std::string &b) {
    // the cases: S-shapes by five methods, the ReLU by five, and Singh's order 2
    struct Case {
        Activation a;
        PolyMethod m;
        int order;
    };
    std::vector<Case> cases;
    for (const Activation a : {Activation::sigmoid, Activation::tanh})
        for (const PolyMethod m :
             {PolyMethod::regression, PolyMethod::ridge_regression, PolyMethod::bernstein,
              PolyMethod::taylor, PolyMethod::throw_catch})
            for (int order = 1; order <= 4; ++order) cases.push_back({a, m, order});
    for (int order = 1; order <= 4; ++order)
        for (const PolyMethod m : {PolyMethod::regression, PolyMethod::ridge_regression,
                                   PolyMethod::bernstein, PolyMethod::taylor, PolyMethod::bounds})
            cases.push_back({Activation::relu, m, order});
    cases.push_back({Activation::relu, PolyMethod::singh, 2});
    // every case on every range
    int bad = 0;
    for (const Case &c : cases) {
        for (const auto &[l, u] : std::vector<std::pair<double, double>>{
                 {-1.0, 3.0}, {-0.3, 0.4}, {0.5, 2.5}, {-6.0, -1.0}, {-8.0, 9.0}}) {
            const auto [lo, hi] = aux_hull(aux_neuron(c.a, aux_interval_set(l, u), c.order, c.m));
            for (int k = 0; k <= 400; ++k) {
                const double y = aux_f(c.a, l + (u - l) * k / 400.0);
                bad += y < lo - 1e-9 || y > hi + 1e-9;
            }
        }
    }
    check(bad == 0, b + ": " + std::to_string(bad) + " points of an activation left its enclosure");
}

/// A set with even exponents and an independent generator, x = alpha^2 + b, is squared: CORA's
/// set reaches 3 where x^2 reaches 4. The outputs of every orders stay inside ours.
void powers_of_sets_with_even_exponents_and_independent_generators_hold(const std::string &b) {
    Rng rng(6);
    // x = 0.2 + alpha^2 + 0.8 b + 0.5 alpha beta, with alpha in two factors
    const PolyZonotope X(Tensor({{0.2}}), Tensor({{1.0, 0.5, -0.4}}), Tensor({{0.8}}),
                         {{2, 0}, {1, 1}, {0, 3}}, {1, 2});
    int outside = 0;
    for (const Activation a : {Activation::sigmoid, Activation::tanh, Activation::relu})
        for (const bool in_gi : {true, false})
            for (int order = 2; order <= 4; ++order) {
                const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {a});
                const PolyZonotope Y = nn.evaluate(X, {.add_approx_error_to_gi = in_gi,
                                                       .poly_method = PolyMethod::regression,
                                                       .orders = {{order}}});
                outside += aux_outside(nn, X, Y, rng);
            }
    check(outside == 0, b + ": " + std::to_string(outside) + " outputs left the set");
}

/// A network of three layers with orders per layer and per neuron, and a limit on the generators.
void networks_of_order_k_enclose_the_outputs(const std::string &b) {
    Rng rng(8);
    const auto matrix = [&](int64_t r, int64_t c) {
        std::vector<double> data(std::size_t(r * c));
        rng.normal(data.data(), data.size(), 0.8);
        return Tensor::from_data(data, {r, c});
    };
    const NeuralNetwork nn(
        {{matrix(5, 3), matrix(5, 1)}, {matrix(4, 5), matrix(4, 1)}, {matrix(2, 4), matrix(2, 1)}},
        {Activation::tanh, Activation::sigmoid, Activation::identity});
    // dependent generators with a shared factor, and independent ones
    const PolyZonotope X(
        Tensor({{0.1}, {-0.3}, {0.2}}), Tensor({{0.4, 0.1, 0.0}, {0.0, 0.3, 0.2}, {0.2, 0.0, 0.3}}),
        Tensor({{0.05}, {0.0}, {0.1}}), {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {1, 2, 3});
    int outside = 0;
    for (const auto &orders : std::vector<std::vector<std::vector<int>>>{
             {{2}, {2}, {}}, {{3}, {1}, {}}, {{2, 1, 3, 2, 1}, {2, 3, 1, 2}, {}}, {{1}, {3}, {}}}) {
        for (const bool in_gi : {true, false}) {
            const PolyZonotope Y = nn.evaluate(X, {.add_approx_error_to_gi = in_gi,
                                                   .poly_method = PolyMethod::regression,
                                                   .orders = orders});
            outside += aux_outside(nn, X, Y, rng);
        }
    }
    check(outside == 0, b + ": " + std::to_string(outside) + " outputs left the set");
    // a limit on the generators keeps the enclosure, and fewer generators
    const NnOptions two{.poly_method = PolyMethod::regression, .orders = {{2}, {2}, {}}};
    NnOptions limited = two;
    limited.num_generators = 6;
    const PolyZonotope full = nn.evaluate(X, two), cut = nn.evaluate(X, limited);
    check(aux_outside(nn, X, cut, rng) == 0, b + ": the reduced set encloses the outputs");
    check(cut.num_dependent() + cut.independent_generators().shape()[1] <
              full.num_dependent() + full.independent_generators().shape()[1],
          b + ": the limit reduces the generators");
    // a generator with all exponents even in the reduction: alpha^2 + alpha^4 reaches 2
    const PolyZonotope Z(Tensor({{-1.0}}), Tensor({{1.0, 1.0}}), Tensor::zeros({1, 0}), {{2}, {4}},
                         {1});
    const NeuralNetwork one({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::tanh});
    const PolyZonotope R = one.evaluate(Z, {.num_generators = 1});
    check(aux_outside(one, Z, R, rng) == 0, b + ": the reduction of even exponents keeps 2");
}

/// The input rows are the input set, and the output depends on its factors (the pairs).
void the_pairs_keep_the_input_rows(const std::string &b) {
    // one tanh layer on a set of two dependent factors
    const NeuralNetwork nn({{Tensor({{1.0, 0.5}}), Tensor({0.1})}}, {Activation::tanh});
    const PolyZonotope X(Tensor({{0.2}, {-0.1}}), Tensor({{0.5, 0.1}, {0.0, 0.4}}),
                         Tensor({{0.1}, {0.05}}), {{1, 0}, {0, 1}}, {1, 2});
    for (const bool in_gi : {true, false}) {
        const PolyZonotope XY = nn.evaluate_with_input(X, {.add_approx_error_to_gi = in_gi,
                                                           .poly_method = PolyMethod::regression,
                                                           .orders = {{3}}});
        // the input rows against the input set
        const Interval hull = XY.zonotope().interval(), in = X.zonotope().interval();
        check(test::close(hull.inf().slice(0, 0, 2), in.inf()) &&
                  test::close(hull.sup().slice(0, 0, 2), in.sup()),
              b + ": the input rows are the input set");
        check(XY.dim() == 3, b + ": the pairs have dimension in + out");
    }
}

void wrong_arguments_are_described(const std::string &b) {
    // an order, a limit and a method the set cannot take
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::sigmoid});
    const PolyZonotope X = aux_interval_set(-1.0, 1.0);
    check(test::raises([&] { nn.evaluate(X, {.orders = {{0}}}); }, Error::wrong_value),
          b + ": an order below 1");
    check(test::raises([&] { nn.evaluate(X, {.num_generators = -1}); }, Error::wrong_value),
          b + ": a negative limit");
    check(test::raises([&] { nn.evaluate(X, {.orders = {{2}}}); }, Error::nn_layer_not_supported),
          b + ": Singh's polynomial of a sigmoid has no order 2");
    // a zonotope takes a line
    const Zonotope Z(Tensor({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { nn.evaluate(Z, {.poly_method = PolyMethod::taylor}); },
                       Error::not_supported),
          b + ": a zonotope takes a line");
}

/// The same evaluation with the checks off.
void checks_and_their_switch() {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::tanh});
    const PolyZonotope X = aux_interval_set(-1.0, 2.0);
    const NnOptions options{.poly_method = PolyMethod::regression, .orders = {{3}}};
    const PolyZonotope on = nn.evaluate(X, options);
    ChecksScope off(false);
    const PolyZonotope again = nn.evaluate(X, options);
    check(test::close(on.generators(), again.generators(), 0),
          "evaluate: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_relu_of_order_2_is_the_hand_computed_polynomial(b);
        order_1_is_the_line_bit_for_bit(b);
        every_method_encloses_the_activation(b);
        powers_of_sets_with_even_exponents_and_independent_generators_hold(b);
        networks_of_order_k_enclose_the_outputs(b);
        the_pairs_keep_the_input_rows(b);
        wrong_arguments_are_described(b);
    });
    checks_and_their_switch();
    return test::finish("neuralNetwork order k");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
