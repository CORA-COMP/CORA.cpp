// linear_layers - CORA's linear layers as the layers of a NeuralNetwork
//
// Syntax:   linear_layers() - the factories of Layer, e.g.
//           Layer L = NeuralNetwork::Layer::batch_norm(scale, offset, mean, var, {C, H, W});
// Inputs:   scale, offset, mean, var - one entry or one per channel
//           input - (channels, height, width) of the input in C order
// Outputs:  L - the affine layer (a gather with factors, or a convolution)
// See also: cora/nn/neural_network/neural_network.h, conv2d.cpp
// References: none (CORA's layers of a network, no paper's algorithm)
//
// CORA: nn/layers/nnLayer.m
// CORA: nn/layers/linear/nnLinearLayer.m
// CORA: nn/layers/linear/nnIdentityLayer.m
// CORA: nn/layers/linear/nnElementwiseAffineLayer.m
// CORA: nn/layers/linear/nnBatchNormLayer.m
// CORA: nn/layers/linear/nnAvgPool2DLayer.m
// CORA: nn/layers/linear/nnLipConstrLinearLayer.m
// CORA: nn/@neuralNetwork/getNumNeurons.m
// CORA: nn/@neuralNetwork/getNumParams.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether p has one entry or one per group.
bool aux_fits(const Tensor &p, int64_t groups) {
    return p.defined() && (p.numel() == 1 || p.numel() == groups);
}

/// The values p (one entry or one per group of `per` consecutive values) as one entry per value.
Tensor aux_expand(const Tensor &p, int64_t groups, int64_t per) {
    const Tensor column = p.reshape({p.numel(), 1}) + p.zeros_like({groups, per});
    return column.reshape({groups * per});
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The layers --------------------------------------------------------------------------------------

NeuralNetwork::Layer NeuralNetwork::Layer::linear(const Tensor &W, const Tensor &b) {
    detail::Entry entry;
    return Layer{W, b};
}

CORA_FROM_NUMBERS NeuralNetwork::Layer NeuralNetwork::Layer::identity(int64_t n) {
    detail::Entry entry;
    // the gather of every input to its own output, without a bias
    std::vector<int64_t> index(static_cast<std::size_t>(n));
    std::iota(index.begin(), index.end(), int64_t(0));
    return gather(index, n, Tensor::zeros({n}));
}

NeuralNetwork::Layer NeuralNetwork::Layer::elementwise_affine(const Tensor &scale,
                                                              const Tensor &offset, int64_t n) {
    detail::Entry entry;
    // CORA: scale and offset are scalars or vectors of the size of the input
    CORA_CHECK(aux_fits(scale, n) && aux_fits(offset, n), Error::wrong_input_in_constructor, [&] {
        return "Layer::elementwise_affine: scale and offset need one entry or " +
               std::to_string(n) + " entries, got " + shape_text(scale.shape()) + " and " +
               shape_text(offset.shape());
    });
    // a gather of every input to its own output with the factors
    std::vector<int64_t> index(static_cast<std::size_t>(n));
    std::iota(index.begin(), index.end(), int64_t(0));
    return gather(index, n, aux_expand(offset, n, 1), aux_expand(scale, n, 1));
}

NeuralNetwork::Layer NeuralNetwork::Layer::batch_norm(const Tensor &scale, const Tensor &offset,
                                                      const Tensor &mean, const Tensor &var,
                                                      const std::vector<int64_t> &input,
                                                      double epsilon) {
    detail::Entry entry;
    CORA_CHECK(input.size() == 3 && epsilon >= 0, Error::wrong_input_in_constructor, [&] {
        return "Layer::batch_norm: the input is (channels, height, width), got " +
               shape_text(input);
    });
    const int64_t C = input[0], pixels = input[1] * input[2];
    CORA_CHECK(aux_fits(scale, C) && aux_fits(offset, C) && aux_fits(mean, C) && aux_fits(var, C),
               Error::wrong_input_in_constructor, [&] {
                   return "Layer::batch_norm: scale, offset, mean and variance need one entry or " +
                          std::to_string(C) + " entries (one per channel)";
               });
    // the folded map per channel: factor = scale / sqrt(var + epsilon), bias = offset - mean factor
    const Tensor spread = (aux_expand(var, C, 1) + epsilon).sqrt();
    const Tensor factor = aux_expand(scale, C, 1).divide(spread);
    const Tensor shift = aux_expand(mean, C, 1).multiply(factor);
    const Tensor bias = aux_expand(offset, C, 1).subtract(shift);
    // each channel's values over its pixels, as a gather with the factors
    return elementwise_affine(aux_expand(factor, C, pixels), aux_expand(bias, C, pixels),
                              C * pixels);
}

CORA_FROM_NUMBERS NeuralNetwork::Layer NeuralNetwork::Layer::avg_pool2d(
    const std::array<int64_t, 2> &pool, const std::vector<int64_t> &input,
    const std::array<int64_t, 2> &stride, const std::array<int64_t, 4> &pads) {
    detail::Entry entry;
    CORA_CHECK(input.size() == 3 && pool[0] >= 1 && pool[1] >= 1, Error::wrong_input_in_constructor,
               [&] {
                   return "Layer::avg_pool2d: the pool size is positive and the input (channels, "
                          "height, width), got " +
                          shape_text(input);
               });
    Conv2d spec;
    spec.stride = stride[0] > 0 ? stride : pool;
    spec.pads = pads;
    spec.groups = input[0];
    // a depthwise kernel of ones over the window, scaled by its size; the bias of its output size
    const Tensor kernel = Tensor::ones({input[0], 1, pool[0], pool[1]}) / double(pool[0] * pool[1]);
    const Layer probe = conv2d(kernel, Tensor::zeros({1}), input, spec, false);
    return conv2d(kernel, Tensor::zeros({probe.outputs()}), input, spec);
}

NeuralNetwork::Layer NeuralNetwork::Layer::lip_constr_linear(const Tensor &W, const Tensor &b,
                                                             double lambda) {
    detail::Entry entry;
    CORA_CHECK(lambda > 0, Error::wrong_input_in_constructor,
               [] { return std::string("Layer::lip_constr_linear: lambda must be positive"); });
    // N = diag(1 / max(1, |W| 1 / lambda)) scales the rows of W and b
    const Tensor norm = (W.abs().sum(1) / lambda).maximum(1.0);
    return Layer{W.divide(norm.reshape({norm.numel(), 1})), b.divide(norm)};
}

// The counts --------------------------------------------------------------------------------------

std::vector<int64_t> NeuralNetwork::num_neurons() const {
    detail::Entry entry;
    // the inputs of every layer, then the outputs of the last
    std::vector<int64_t> n;
    n.reserve(layers_.size() + 1);
    for (const Layer &layer : layers_) n.push_back(layer.inputs());
    n.push_back(layers_.back().outputs());
    return n;
}

int64_t NeuralNetwork::num_params() const {
    detail::Entry entry;
    CORA_CHECK(is_dense(), Error::not_supported, [] {
        return std::string("NeuralNetwork::num_params: only dense layers keep their parameters");
    });
    int64_t count = 0;
    for (const Layer &layer : layers_) count += layer.W.numel() + layer.b.numel();
    return count;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
