// test_neural_network_evaluate_con_zonotope - the network on constrained zonotopes
//
// A constrained zonotope goes through as a star set, each unstable ReLU becoming a factor with the
// triangle's constraints. The tests check exact cases by hand (a stable ReLU, the triangle of one
// neuron, the bound CORA gets wrong) and that the outputs of sampled points lie in the set (both
// ways to bound a neuron). CORALean serves no op for it (docs/oracle-requests.md), so these are
// property tests.
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateConZonotope.m
// CORA: nn/layers/nonlinear/nnReLULayer.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/evaluateConZonotope.m
// PROPERTY: nn/layers/nonlinear/nnReLULayer.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
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


/// A two-layer network 2 -> 3 -> 2 with the activation a after the first layer.
NeuralNetwork aux_network(Activation a) {
    return NeuralNetwork(
        {{Tensor({{1.0, -2.0}, {0.5, 1.5}, {-1.0, 1.0}}), Tensor({0.1, -0.2, 0.3})},
         {Tensor({{1.0, 1.0, -1.0}, {0.5, -1.0, 2.0}}), Tensor({0.0, 0.2})}},
        {a, Activation::identity});
}

/// Whether the points of the set c + G^T x with x1 + x2 = 0.5 lie in the output set Y.
bool aux_samples_inside(const NeuralNetwork &nn, const ConZonotope &Y, const Tensor &c,
                        const Tensor &G) {
    const std::vector<double> cv = c.data(), gv = G.data();
    // the grid of x1 and 20 random pairs (x1, x3) from a fixed seed
    Rng rng(4);
    std::vector<double> random(40);
    rng.uniform(random.data(), random.size(), 0.0, 1.0);
    for (int i = 0; i <= 10 + 20; ++i)
        for (int j = 0; j <= (i <= 10 ? 4 : 0); ++j) {
            // the factors (x1, 0.5 - x1, x3) in the unit box
            const double x1 = i <= 10 ? -0.5 + 1.5 * i / 10.0 : -0.5 + 1.5 * random[2 * (i - 11)],
                         x3 = i <= 10 ? -1.0 + 0.5 * j : -1.0 + 2.0 * random[2 * (i - 11) + 1];
            const double x[3] = {x1, 0.5 - x1, x3};
            std::vector<double> p(cv);
            for (std::size_t k = 0; k < 3; ++k)
                for (std::size_t a = 0; a < 2; ++a) p[a] += x[k] * gv[k * 2 + a];
            const Tensor out = nn.evaluate(Tensor::from_data(p, {1, 2})).squeeze(0);
            if (!Y.contains(out, "exact", 1e-9)) return false;
        }
    return true;
}

void con_zonotope(const std::string &b) {
    const Tensor c({0.2, -0.1});
    const Tensor G({{1.0, 0.5}, {-0.5, 1.0}, {0.3, 0.2}});
    // the constraint x1 + x2 = 0.5 on the factors
    const ConZonotope X(c, G, Tensor({{1.0, 1.0, 0.0}}), Tensor({0.5}));
    const NeuralNetwork nn = aux_network(Activation::relu);
    for (const bool approx : {true, false}) {
        NnOptions options;
        options.bound_approx = approx;
        const ConZonotope Y = nn.evaluate_con_zonotope(X, options);
        check(Y.dim() == 2 && aux_samples_inside(nn, Y, c, G),
              b + ": the outputs of sampled points lie in the set, bound_approx " +
                  std::to_string(approx));
    }
    // a set without constraints: a box through a ReLU that is stable (active and zero)
    const NeuralNetwork stable({{Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({5.0, -5.0})}},
                               {Activation::relu});
    const ConZonotope U(Tensor({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const ConZonotope S = stable.evaluate_con_zonotope(U);
    check(S.nr_generators() == 2 && test::close(S.c(), {5.0, 0.0}, 1e-12),
          b + ": a stable ReLU adds no factor");
    const Interval box = S.support_function_range(Tensor({1.0, 0.0}));
    check(test::close(box.inf(), Tensor({4.0}), 1e-9) &&
              test::close(box.sup(), Tensor({6.0}), 1e-9),
          b + ": the active neuron keeps its range");
    // an unstable ReLU in one dimension: y between max(x, 0) and the chord over [-1, 1]
    const NeuralNetwork one({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::relu});
    const ConZonotope O = one.evaluate_con_zonotope(ConZonotope(Tensor({0.0}), Tensor({{1.0}})));
    const Interval range = O.support_function_range(Tensor({1.0}));
    check(O.nr_generators() >= 2 && test::close(range.inf(), Tensor({0.0}), 1e-9) &&
              test::close(range.sup(), Tensor({1.0}), 1e-9),
          b + ": an unstable ReLU is its triangle");
    // the bounds take the middle of the factors' ranges: -0.1 - x on [-1, 1] reaches 0.9
    const NeuralNetwork shifted({{Tensor({{1.0}}), Tensor({-0.1})}}, {Activation::relu});
    const ConZonotope W =
        shifted.evaluate_con_zonotope(ConZonotope(Tensor({0.0}), Tensor({{-1.0}})));
    check(W.contains(Tensor({0.4}), "exact", 1e-9) && W.contains(Tensor({0.9}), "exact", 1e-9),
          b + ": a neuron that is positive somewhere is not zeroed");
    // no generators: the constraints are rows 0 = b
    const NeuralNetwork id({{Tensor({{2.0}}), Tensor({1.0})}}, {Activation::relu});
    const ConZonotope P(Tensor({1.0}), Tensor::zeros({0, 1}), Tensor::zeros({0, 0}),
                        Tensor::zeros({0}));
    check(test::close(id.evaluate_con_zonotope(P).c(), {3.0}, 1e-12),
          b + ": a set without generators");
    // rows 0 = 0 without generators keep the point
    const ConZonotope Q(Tensor({1.0}), Tensor::zeros({0, 1}), Tensor::zeros({1, 0}),
                        Tensor::zeros({1}));
    check(test::close(id.evaluate_con_zonotope(Q).c(), {3.0}, 1e-12), b + ": rows 0 = 0");
    // an infeasible set stays empty, and what a ReLU cannot take is refused
    check(nn.evaluate_con_zonotope(ConZonotope::empty(2)).representsa("emptySet"),
          b + ": the empty set");
    check(test::raises([&] { aux_network(Activation::sigmoid).evaluate_con_zonotope(X); },
                       Error::not_supported),
          b + ": a sigmoid has no constrained zonotope");
    check(
        test::raises([&] { nn.evaluate_con_zonotope(ConZonotope(Tensor({0.0}), Tensor({{1.0}}))); },
                     Error::dimension_mismatch),
        b + ": a wrong dimension");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(nn.evaluate_con_zonotope(X).dim() == 2, b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { con_zonotope(b); });
    return test::finish("neuralNetwork evaluate on constrained zonotopes");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
