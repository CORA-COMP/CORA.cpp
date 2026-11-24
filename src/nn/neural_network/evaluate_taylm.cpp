// evaluate_taylm - the network on Taylor models, as CORA's evaluateTaylm of each layer
//
// A layer maps the models x_j of its input by W x + b; an activation is applied neuron by neuron:
// over the bounds [l, u] of the neuron's model the polynomial p of poly_method and the order of the
// neuron (options.orders) with the error radius d, and the output is p(x) + [-d, d], the powers of
// the model by Taylm::power and the error added to its remainder. At order 1 of singh and
// regression p is the line of evaluate (priv_line, sound, CORALean's), else priv_approx_poly's.
// An identity layer leaves the models. Only a chain of layers is evaluated; an activation without
// a line or polynomial (product, group sort; floor and the like above order 1) is not supported.
//
// Syntax:   std::vector<Taylm> y = nn.evaluate_taylm(x);   nn.evaluate_taylm(x, options);
// Inputs:   x - one model per input of the network, all over the same variables
//           options - the poly_method (singh or regression) and the orders per layer
// Outputs:  y - one model per output, enclosing the network's outputs over the variables' box
// See also: neural_network.h, evaluate.cpp, private/priv_activation_poly.h
// References:
//    [1] N. Kochdumper, C. Schilling, M. Althoff, S. Bak. "Open- and closed-loop neural network
//        verification using polynomial zonotopes", NFM 2023 (arXiv:2207.02715).
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateTaylm.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       23-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_activation_poly.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The layers --------------------------------------------------------------------------------------

/// The order of each neuron of layer k (CORA's layer.order): none is 1, one value is every
/// neuron's, a list of the layer's size is one per neuron, any other count takes the largest.
std::vector<int> aux_orders(const NnOptions &options, std::size_t k, std::size_t neurons) {
    const std::vector<int> given =
        k < options.orders.size() ? options.orders[k] : std::vector<int>{};
    if (given.empty()) return std::vector<int>(neurons, 1);
    if (given.size() == neurons) return given;
    return std::vector<int>(neurons, *std::max_element(given.begin(), given.end()));
}

/// W x + b on the models, each output the sum of the weighted inputs from the first.
std::vector<Taylm> aux_affine(const NeuralNetwork::Layer &L, const std::vector<Taylm> &x) {
    const std::vector<double> W = L.dense().data(), b = L.b.data();
    const std::size_t in = x.size(), out = b.size();
    std::vector<Taylm> y;
    y.reserve(out);
    for (std::size_t i = 0; i < out; ++i) {
        Taylm acc = x[0].times(W[i * in]);
        for (std::size_t j = 1; j < in; ++j) acc = acc.plus(x[j].times(W[i * in + j]));
        y.push_back(acc.plus(b[i]));
    }
    return y;
}

// The neurons -------------------------------------------------------------------------------------

/// The fit of a neuron over [l, u]: at order 1 of singh or regression the zonotope path's line
/// as it is (slope, offset, radius; CORALean's Relaxation.toPoly), else CORA's polynomial.
PrivApprox aux_fit(Activation activation, double l, double u, int order, const NnOptions &options) {
    const bool lines =
        options.poly_method == PolyMethod::singh || options.poly_method == PolyMethod::regression;
    if (!lines || order > 1) return priv_approx_poly(activation, l, u, order, options.poly_method);
    return priv_line_fit(activation, l, u, options);
}

/// One neuron: the polynomial of its bounds with the error as a remainder, p(x) + [-d, d].
Taylm aux_neuron(Activation activation, const Taylm &x, int order, const NnOptions &options) {
    const std::array<double, 2> bounds = x.bounds();
    const PrivApprox a = aux_fit(activation, bounds[0], bounds[1], order, options);
    // the constant term and the error, then the powers of the model from the lowest
    const int64_t vars = x.num_vars();
    Taylm r = Taylm::constant(a.coeffs.back(), vars, x.max_order())
                  .plus(Taylm::from_range(-a.d, a.d, vars, x.max_order()));
    for (std::size_t i = 1; i < a.coeffs.size(); ++i)
        r = r.plus(x.power(int(i)).times(a.coeffs[a.coeffs.size() - 1 - i]));
    return r;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Taylm> NeuralNetwork::evaluate_taylm(const std::vector<Taylm> &x,
                                                 const NnOptions &options) const {
    detail::Entry entry;
    CORA_CHECK(int64_t(x.size()) == input_dim(), Error::dimension_mismatch, [&] {
        return "evaluate_taylm: " + std::to_string(x.size()) + " models, the network takes " +
               std::to_string(input_dim());
    });
    CORA_CHECK(is_chain(), Error::not_supported, [] {
        return std::string("evaluate_taylm: it supports a chain of layers, not a graph");
    });
    for (const Taylm &t : x)
        CORA_CHECK(t.num_vars() == x[0].num_vars(), Error::dimension_mismatch, [] {
            return std::string("evaluate_taylm: the models must be over the same variables");
        });
    // the layers in turn: the map, then the activation of each neuron
    std::vector<Taylm> h = x;
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        h = aux_affine(layers_[k], h);
        if (activations_[k] == Activation::identity) continue;
        const std::vector<int> orders = aux_orders(options, k, h.size());
        for (std::size_t j = 0; j < h.size(); ++j)
            h[j] = aux_neuron(activations_[k], h[j], orders[j], options);
    }
    return h;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
