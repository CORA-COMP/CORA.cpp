// network_forms - the order of a network's inputs and its normal form
//
// input_neuron_order lists the inputs (the pixels of an image, row after row) in the order in which
// they are given a generator: as they come, along a snake (the first row, then the rest turned
// by a quarter, so that neighbouring pixels stay neighbours), or by the mean absolute Jacobian of
// the network, the least sensitive first. normal_form merges the linear layers in front of every
// activation, so that the network alternates a linear layer and an activation.
//
// Syntax:   network_forms() - the members of NeuralNetwork, e.g.
//           std::vector<int64_t> order = nn.input_neuron_order("snake", x, {1, 28, 28});
//           NeuralNetwork normal = nn.normal_form();
// Inputs:   method - "in-order", "snake" or "sensitivity"
//           x - a point (in)
//           input_size - (C, H, W)
// Outputs:  order - the inputs, 0-based (a pixel is y * W + x)
//           normal - the network
// See also: cora/nn/neural_network/neural_network.h, sensitivity.cpp
// References: none (CORA's helpers of a network, no paper's algorithm)
//
// CORA: nn/@neuralNetwork/getInputNeuronOrder.m
// CORA: nn/@neuralNetwork/getNormalForm.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The pixels of an h x w image along the snake: the first row, then the rest turned (MATLAB's
/// A = fliplr(A(2:end, :))') until nothing is left; a pixel is y * w + x.
std::vector<int64_t> aux_snake(int64_t h, int64_t w) {
    std::vector<std::vector<int64_t>> A(static_cast<std::size_t>(h),
                                        std::vector<int64_t>(static_cast<std::size_t>(w)));
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x) A[size_t(y)][size_t(x)] = y * w + x;
    std::vector<int64_t> order;
    while (!A.empty() && !A.front().empty()) {
        order.insert(order.end(), A.front().begin(), A.front().end());
        // the rest, columns reversed, transposed: row j is the rest's column (width - 1 - j)
        const std::size_t rows = A.size() - 1, cols = A.front().size();
        std::vector<std::vector<int64_t>> turned(cols, std::vector<int64_t>(rows));
        for (std::size_t j = 0; j < cols; ++j)
            for (std::size_t i = 0; i < rows; ++i) turned[j][i] = A[i + 1][cols - 1 - j];
        A = rows == 0 ? std::vector<std::vector<int64_t>>{} : turned;
    }
    return order;
}

/// The matrix of a layer (the unit vectors through a convolution or a gather).
Tensor aux_matrix(const NeuralNetwork::Layer &layer) {
    return layer.plan ? layer.dense() : layer.W;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The order of the inputs -------------------------------------------------------------------------

std::vector<int64_t>
NeuralNetwork::input_neuron_order(const std::string &method, const Tensor &x,
                                  const std::vector<int64_t> &input_size) const {
    detail::Entry entry;
    // the method and the sizes
    CORA_CHECK(method == "in-order" || method == "snake" || method == "sensitivity",
               Error::wrong_value, [&] {
                   return "input_neuron_order: the method is 'in-order', 'snake' or "
                          "'sensitivity', got '" +
                          method + "'";
               });
    CORA_CHECK(input_size.empty() || (input_size.size() == 3 &&
                                      input_size[0] * input_size[1] * input_size[2] == input_dim()),
               Error::dimension_mismatch, [&] {
                   return "input_neuron_order: the input size (C, H, W) " + shape_text(input_size) +
                          " must have the network's " + std::to_string(input_dim()) + " inputs";
               });
    CORA_CHECK(method != "snake" || !input_size.empty(), Error::wrong_value,
               [] { return std::string("input_neuron_order: a snake needs the input size"); });
    // the pixels in order, or the snake's
    const int64_t pixels = input_size.empty() ? input_dim() : input_size[1] * input_size[2];
    std::vector<int64_t> order(static_cast<std::size_t>(pixels));
    if (method == "snake") return aux_snake(input_size[1], input_size[2]);
    std::iota(order.begin(), order.end(), int64_t(0));
    if (method == "in-order") return order;
    // the mean absolute Jacobian per input, over the channels of a pixel, the least first
    Tensor S = sensitivity(x).abs().mean(0);
    if (!input_size.empty()) S = S.reshape(input_size).mean(0).reshape({pixels});
    const std::vector<double> s = S.data();
    // ascending, ties by the index (MATLAB's sort is stable)
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return s[size_t(a)] < s[size_t(b)] || (s[size_t(a)] == s[size_t(b)] && a < b);
    });
    return order;
}

// The normal form ---------------------------------------------------------------------------------

NeuralNetwork NeuralNetwork::normal_form() const {
    detail::Entry entry;
    CORA_CHECK(is_chain(), Error::not_supported, [] {
        return std::string("normal_form: the layers must be a chain (no residuals or skips)");
    });
    std::vector<Layer> layers;
    std::vector<Activation> activations;
    // the linear layers merged so far (none: the next layer starts a new one)
    std::optional<Layer> pending;
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        Layer merged = layers_[k];
        // a chain again, whatever the layer's place was
        merged.source = kPrevious;
        if (pending) {
            // W_k (W h + b) + b_k
            const Tensor next = aux_matrix(layers_[k]);
            const Tensor W = next.matmul(aux_matrix(*pending));
            const Tensor b = next.matmul(pending->b.expand_dims(-1)).squeeze(-1) + layers_[k].b;
            merged = Layer{W, b};
        }
        if (activations_[k] == Activation::identity) {
            pending = merged;
            continue;
        }
        layers.push_back(merged);
        activations.push_back(activations_[k]);
        pending.reset();
    }
    // linear layers after the last activation
    if (pending) {
        layers.push_back(*pending);
        activations.push_back(Activation::identity);
    }
    return NeuralNetwork(layers, activations);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
