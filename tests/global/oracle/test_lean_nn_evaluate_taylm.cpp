// test_lean_nn_evaluate_taylm - NeuralNetwork::evaluate_taylm against the CORALean oracle
//
// ORACLE: nn/@neuralNetwork/evaluate.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/evaluateTaylm.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
//
// The constructors of a model (taylm.constant, taylm.fromRange) on random numbers and ranges, a
// zero among them (no term). Then random networks on Taylor models (neuralNetwork.evaluateTaylm):
// ReLU at order 1, at order 2 (singh: the parabola with the closed-form error) and at mixed orders
// per neuron, with neurons below, above and across 0; sigmoid and tanh at order 1 (singh, and
// regression with the bracket's error, the sigmoid also with the extremum's), at the oracle's
// default k as the zonotope path pins it. The inputs are the models of boxes and zonotopes, and
// products of them that carry a remainder. Every term (its monomial in the map's order, its
// coefficient) and the remainder is compared; the oracle's error box is zero at nearest. The
// oracle's wire form has no -0: CORA.cpp's zeros are compared as +0 (a zero slope times a negative
// coefficient is -0 there).
//
// Authors:       Claude Opus 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "cora/global/oracle/neural_network.h"
#include "cora/global/oracle/taylm.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

#include <string>
#include <utility>
#include <vector>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORA.cpp's model against the oracle's: the terms in order, the remainder (zeros as +0), no
/// error box.
void aux_check(const std::string &what, const cora::Taylm &mine, const lean::Taylm &theirs) {
    const auto &terms = theirs.terms();
    test::check(mine.coefficients().size() == terms.size(), what + ": the number of terms");
    std::size_t i = 0;
    for (const auto &[m, c] : mine.coefficients()) {
        if (i >= terms.size()) break;
        test::check(m == terms[i].first, what + ": the monomials in the map's order");
        oracle::check(what + " coefficient " + std::to_string(i), c + 0.0, terms[i].second);
        ++i;
    }
    const auto r = mine.remainder();
    oracle::check(what + " remainder inf", r[0] + 0.0, theirs.remainder()[0]);
    oracle::check(what + " remainder sup", r[1] + 0.0, theirs.remainder()[1]);
    // the roundings of the sound operations cost nothing at nearest
    test::check(theirs.error()[0] == 0.0 && theirs.error()[1] == 0.0, what + ": no error box");
}

/// A random network of the layer sizes with the activation after each layer, the biases scaled
/// (a large bias puts neurons above or below 0 over the whole input).
NeuralNetwork aux_random_network(oracle::Random &rnd, const std::vector<int64_t> &sizes,
                                 const std::vector<Activation> &activations, double bias) {
    std::vector<NeuralNetwork::Layer> layers;
    for (std::size_t k = 0; k + 1 < sizes.size(); ++k)
        layers.push_back({rnd.matrix(sizes[k + 1], sizes[k]), rnd.point(sizes[k + 1]) * bias});
    return {layers, activations};
}

/// The layers of the network as the oracle's, each activation with its orders.
std::vector<lean::NnLayer> aux_lean_layers(const NeuralNetwork &nn,
                                           const std::vector<std::vector<int>> &orders) {
    std::vector<lean::NnLayer> layers;
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        layers.push_back(lean::NnLayer::linear(nn.layers()[k].W, nn.layers()[k].b));
        const Activation a = nn.activations()[k];
        if (a == Activation::identity) continue;
        lean::NnLayer layer = a == Activation::relu      ? lean::NnLayer::relu()
                              : a == Activation::sigmoid ? lean::NnLayer::sigmoid()
                                                         : lean::NnLayer::tanh();
        if (k < orders.size()) layer.order = orders[k];
        layers.push_back(layer);
    }
    return layers;
}

/// The network on the models in CORA.cpp and in the oracle, output by output.
void aux_check_network(const std::string &what, const NeuralNetwork &nn,
                       const std::vector<cora::Taylm> &x, const NnOptions &options,
                       const std::string &poly_method, const std::string &regression_error = "") {
    const std::vector<cora::Taylm> mine = nn.evaluate_taylm(x, options);
    std::vector<lean::Taylm> in;
    in.reserve(x.size());
    for (const cora::Taylm &t : x) in.emplace_back(t);
    const std::vector<lean::Taylm> theirs = lean::nn_evaluate_taylm(
        aux_lean_layers(nn, options.orders), in, poly_method, regression_error);
    test::check(mine.size() == theirs.size(), what + ": one model per output");
    for (std::size_t i = 0; i < mine.size() && i < theirs.size(); ++i)
        aux_check(what + " output " + std::to_string(i), mine[i], theirs[i]);
}

/// The inputs: the models of a box, of a zonotope, and of a box with a product in place of the
/// first coordinate (a remainder from the truncation at the order).
std::vector<std::vector<cora::Taylm>> aux_inputs(oracle::Random &rnd, int64_t n, int max_order) {
    std::vector<std::vector<cora::Taylm>> out;
    out.push_back(cora::Taylm::from_interval(rnd.box(n), max_order));
    out.push_back(cora::Taylm::from_zonotope(rnd.zonotope(n, n + 1), max_order));
    std::vector<cora::Taylm> x = cora::Taylm::from_interval(rnd.box(n), max_order);
    x[0] = x[0].plus(x[n - 1].times(x[0]).power(max_order).times(0.25));
    out.push_back(x);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    // the constant: one term, none for a zero
    oracle::run("taylm.constant", [](oracle::Random &rnd) {
        for (const auto &[vars, order] :
             std::vector<std::pair<int64_t, int>>{{1, 2}, {2, 3}, {3, 5}})
            for (const double c : {rnd.scalar(), 0.0, rnd.scalar() * 1e-3})
                aux_check("constant", cora::Taylm::constant(c, vars, order),
                          lean::taylm_constant(c, vars, order));
    });

    // the range: its middle the constant, its half width the remainder
    oracle::run("taylm.fromRange", [](oracle::Random &rnd) {
        for (const auto &[vars, order] :
             std::vector<std::pair<int64_t, int>>{{1, 2}, {2, 4}, {3, 3}})
            for (int i = 0; i < 4; ++i) {
                const double c = i == 0 ? 0.0 : rnd.scalar(), r = rnd.uniform(0.0, 2.0);
                aux_check("fromRange", cora::Taylm::from_range(c - r, c + r, vars, order),
                          lean::taylm_from_range(c - r, c + r, vars, order));
            }
    });

    // ReLU networks at orders 1 and 2 and per neuron, biases small and large
    oracle::run("neuralNetwork.evaluateTaylm", [](oracle::Random &rnd) {
        const std::vector<std::vector<std::vector<int>>> order_sets{
            {}, {{2}, {2}}, {{2, 1, 2}, {1, 2, 2}}, {{1, 2}, {2}}};
        for (const double bias : {0.3, 3.0})
            for (const std::vector<std::vector<int>> &orders : order_sets) {
                const NeuralNetwork nn = aux_random_network(
                    rnd, {2, 3, 3, 2}, {Activation::relu, Activation::relu, Activation::identity},
                    bias);
                NnOptions options;
                options.orders = orders;
                for (const int max_order : {2, 3})
                    for (const std::vector<cora::Taylm> &x : aux_inputs(rnd, 2, max_order))
                        aux_check_network("relu order " + std::to_string(orders.size()), nn, x,
                                          options, "singh");
            }
        // a wider layer and three variables
        const NeuralNetwork wide =
            aux_random_network(rnd, {3, 6, 1}, {Activation::relu, Activation::identity}, 1.0);
        NnOptions two;
        two.orders = {{2}};
        for (const std::vector<cora::Taylm> &x : aux_inputs(rnd, 3, 4))
            aux_check_network("relu wide order 2", wide, x, two, "singh");
    });

    // sigmoid and tanh at order 1: singh, and regression with the bracket's or extremum's error
    oracle::run("neuralNetwork.evaluateTaylm", [](oracle::Random &rnd) {
        for (const Activation a : {Activation::sigmoid, Activation::tanh}) {
            const std::string name = a == Activation::sigmoid ? "sigmoid" : "tanh";
            const NeuralNetwork nn =
                aux_random_network(rnd, {2, 3, 2}, {a, Activation::identity}, 1.0);
            NnOptions singh, bracket, extremum;
            bracket.poly_method = extremum.poly_method = PolyMethod::regression;
            bracket.regression_error = RegressionError::bracket;
            extremum.regression_error = RegressionError::extremum;
            for (const std::vector<cora::Taylm> &x : aux_inputs(rnd, 2, 3)) {
                aux_check_network(name + " singh", nn, x, singh, "singh");
                aux_check_network(name + " regression bracket", nn, x, bracket, "regression",
                                  "bracket");
                // the oracle's extremum is the sigmoid's only
                if (a == Activation::sigmoid)
                    aux_check_network(name + " regression extremum", nn, x, extremum, "regression",
                                      "extremum");
            }
        }
        // two layers, ReLU then tanh, at order 1
        const NeuralNetwork mixed = aux_random_network(
            rnd, {2, 3, 3, 1}, {Activation::relu, Activation::tanh, Activation::identity}, 0.5);
        for (const std::vector<cora::Taylm> &x : aux_inputs(rnd, 2, 3))
            aux_check_network("relu tanh", mixed, x, NnOptions{}, "singh");
    });
    return test::finish("neuralNetwork evaluateTaylm against CORALean");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
