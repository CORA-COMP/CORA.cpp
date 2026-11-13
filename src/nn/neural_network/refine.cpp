// refine - raises the orders of the polynomials of the activations where the heuristic is large
//
// CORA's neuralNetwork/refine: after an evaluation each neuron has a heuristic, by default the
// radius of its approximation error. The type "all" raises every refinable neuron by one (to at
// most max_order); "layer" raises the layer whose heuristic has the largest 2-norm, among the
// layers below max_order; "neuron" raises the neurons whose heuristic exceeds gamma times the
// largest one, in the layer of that neuron. The heuristics "sensitivity", "both" and "random"
// are not ported (a sensitivity of every layer, and a random draw that CORA.cpp cannot repeat).
//
// Syntax:   orders = nn.refine(orders, nn.approx_errors(X));
//           orders = nn.refine(orders, errors, {.max_order = 4, .type = RefineType::neuron});
// Inputs:   orders - per layer: none (order 1), one value, or one per neuron
//           heuristic - per layer, one value per neuron (approx_errors)
//           options - max_order (5), type (layer), heuristic (approx_error), gamma (0.1)
// Outputs:  orders - one entry per neuron of every layer
// See also: cora/nn/neural_network/neural_network.h, evaluate.cpp
//
// CORA: nn/@neuralNetwork/refine.m
// CORA: nn/@neuralNetwork/getRefinableLayers.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The orders of a layer of n neurons: none is 1, one value is every neuron's, n values as they
/// are, any other count takes the largest.
std::vector<int> aux_expand(const std::vector<int> &given, int64_t n) {
    if (given.empty()) return std::vector<int>(std::size_t(n), 1);
    if (int64_t(given.size()) == n) return given;
    return std::vector<int>(std::size_t(n), *std::max_element(given.begin(), given.end()));
}

// The candidates ----------------------------------------------------------------------------------

/// One candidate of the neuron-wise refinement: the heuristic, the layer, the neuron.
struct Candidate {
    double heuristic;
    std::size_t layer, neuron;
};

} // namespace


// ===========================================  MAIN  =========================================== //

// Refine ------------------------------------------------------------------------------------------

std::vector<std::vector<int>>
NeuralNetwork::refine(const std::vector<std::vector<int>> &orders,
                      const std::vector<std::vector<double>> &heuristic,
                      const RefineOptions &options) const {
    detail::Entry entry;
    CORA_CHECK(options.max_order >= 1, Error::wrong_value, [&] {
        return "refine: max_order must be at least 1, got " + std::to_string(options.max_order);
    });
    // the orders of every neuron, and the heuristic of the refinable layers
    std::vector<std::vector<int>> out;
    for (std::size_t k = 0; k < layers_.size(); ++k)
        out.push_back(aux_expand(k < orders.size() ? orders[k] : std::vector<int>{},
                                 layers_[k].W.shape()[0]));
    const std::vector<int64_t> refinable = refinable_layers();
    std::vector<std::vector<double>> heu(layers_.size());
    for (std::size_t i = 0; i < refinable.size(); ++i) {
        const std::size_t k = std::size_t(refinable[i]);
        heu[k] = k < heuristic.size() ? heuristic[k] : std::vector<double>{};
        heu[k].resize(out[k].size(), 0.0);
        // layer_bias: earlier layers count more
        if (options.heuristic == RefineHeuristic::layer_bias)
            for (double &v : heu[k]) v /= double(i + 1);
    }
    const RefineType type =
        options.heuristic == RefineHeuristic::all ? RefineType::all : options.type;
    // every refinable neuron by one
    if (type == RefineType::all) {
        for (const int64_t k : refinable)
            for (int &order : out[std::size_t(k)]) order = std::min(order + 1, options.max_order);
        return out;
    }
    // the layer of the largest norm among those below max_order: all its neurons
    if (type == RefineType::layer) {
        std::vector<std::pair<double, std::size_t>> rows;
        for (const int64_t k : refinable) {
            const std::vector<int> &o = out[std::size_t(k)];
            double norm = 0;
            for (const double v : heu[std::size_t(k)]) norm += v * v;
            norm = std::sqrt(norm);
            if (*std::max_element(o.begin(), o.end()) < options.max_order && norm > 0)
                rows.emplace_back(norm, std::size_t(k));
        }
        std::stable_sort(rows.begin(), rows.end(),
                         [](const auto &a, const auto &b) { return a.first > b.first; });
        if (!rows.empty())
            for (int &order : out[rows.front().second]) ++order;
        return out;
    }
    // the neurons above gamma times the largest heuristic, in the layer of the largest
    std::vector<Candidate> rows;
    for (const int64_t k : refinable)
        for (std::size_t j = 0; j < out[std::size_t(k)].size(); ++j)
            if (out[std::size_t(k)][j] < options.max_order && heu[std::size_t(k)][j] > 0)
                rows.push_back({heu[std::size_t(k)][j], std::size_t(k), j});
    std::stable_sort(rows.begin(), rows.end(), [](const Candidate &a, const Candidate &b) {
        return a.heuristic > b.heuristic;
    });
    if (rows.empty()) return out;
    const double largest = rows.front().heuristic;
    const std::size_t layer = rows.front().layer;
    for (const Candidate &c : rows)
        if (c.heuristic > options.gamma * largest && c.layer == layer) ++out[c.layer][c.neuron];
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
