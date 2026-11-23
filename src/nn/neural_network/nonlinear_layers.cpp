// nonlinear_layers - CORA's exp, softmax and max-pool layers as layers of a NeuralNetwork
//
// A layer of this port is an affine map and an activation. Exp is the identity map with the exp
// activation. Softmax and max pooling have no neuron by neuron activation: they are lowered, as
// the ONNX front end does, to affine layers with the activations exp, reciprocal and relu, which
// the set propagation encloses neuron by neuron (a softmax is 1 / sum_j exp(x_j - x_i), a maximum
// is max(a, b) = b + relu(a - b)). The layers of a block are numbered from the one after the layer
// `previous`, so that the residuals of the max pool name their sources.
//
// Syntax:   nonlinear_layers() - the factories, e.g.
//           NeuralNetwork nn(NeuralNetwork::softmax_layer(10));
//           NeuralNetwork nn(NeuralNetwork::max_pool2d_layer({2, 2}, {C, H, W}));
// Inputs:   n - the number of values
//           pool - the window; input - (channels, height, width) in C order
// Outputs:  B - the layers and their activations
// See also: cora/nn/neural_network/neural_network.h, conv2d.cpp
// References: none (CORA's layers of a network, no paper's algorithm)
//
// CORA: nn/layers/nonlinear/nnExpLayer.m
// CORA: nn/layers/nonlinear/nnSoftmaxLayer.m
// CORA: nn/layers/nonlinear/nnMaxPool2DLayer.m
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

using Layer = NeuralNetwork::Layer;

/// The zeros of n values.
CORA_FROM_NUMBERS Tensor aux_zeros(int64_t n) { return Tensor::zeros({n}); }

/// The tensor of the values with the shape.
CORA_FROM_NUMBERS Tensor aux_tensor(const std::vector<double> &values,
                                    const std::vector<int64_t> &shape) {
    return Tensor::from_data(values, shape);
}

/// A grouped convolution of one max-pool round: row j of `rows` (m, items) per channel, over the
/// window (first round) or one item per channel and pixel (the others).
CORA_FROM_NUMBERS Layer aux_round_conv(const std::vector<std::vector<double>> &rows,
                                       int64_t channels, int64_t items,
                                       const std::vector<int64_t> &from,
                                       const NeuralNetwork::Conv2d &spec,
                                       const std::array<int64_t, 2> &pool, bool first,
                                       int64_t pixels) {
    const int64_t m = int64_t(rows.size());
    // channel ch's kernels are the rows (the kernel tiled over the channels)
    std::vector<double> data;
    for (int64_t ch = 0; ch < channels; ++ch)
        for (const std::vector<double> &row : rows) data.insert(data.end(), row.begin(), row.end());
    const std::vector<int64_t> shape = first
                                           ? std::vector<int64_t>{channels * m, 1, pool[0], pool[1]}
                                           : std::vector<int64_t>{channels * m, items, 1, 1};
    return Layer::conv2d(aux_tensor(data, shape), Tensor::zeros({channels * m * pixels}), from,
                         spec);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Exp and softmax ---------------------------------------------------------------------------------

CORA_FROM_NUMBERS NeuralNetwork::Block NeuralNetwork::exp_layer(int64_t n, int64_t previous) {
    detail::Entry entry;
    Layer layer = Layer::identity(n);
    layer.source = previous;
    return {{layer}, {Activation::exp}};
}

CORA_FROM_NUMBERS NeuralNetwork::Block NeuralNetwork::softmax_layer(int64_t n, int64_t previous) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_input_in_constructor,
               [] { return std::string("Layer softmax_layer: the input needs a value"); });
    // exp(x_t - x_q) for every pair (t, q) at output t * n + q: the gather of x_t, and x_q times -1
    std::vector<int64_t> it, iq;
    for (int64_t t = 0; t < n; ++t)
        for (int64_t q = 0; q < n; ++q) {
            it.push_back(t);
            iq.push_back(q);
        }
    const Tensor zero = Tensor::zeros({n * n});
    Layer differences = Layer::gather(it, n, zero);
    differences.source = previous;
    differences.residuals.push_back(
        {previous,
         std::make_shared<Layer>(Layer::gather(iq, n, zero, Tensor::ones({n * n}).negative()))});
    // the sums over t by a 1 x 1 convolution (t the channels), then 1 / max(s, 1) (s >= 1: t = q)
    const Layer sums = Layer::conv2d(Tensor::ones({1, n, 1, 1}), Tensor::zeros({n}), {n, n, 1}, {});
    return {{differences, sums}, {Activation::exp, Activation::reciprocal}};
}

// Max pooling -------------------------------------------------------------------------------------

CORA_FROM_NUMBERS NeuralNetwork::Block
NeuralNetwork::max_pool2d_layer(const std::array<int64_t, 2> &pool,
                                const std::vector<int64_t> &input,
                                const std::array<int64_t, 2> &stride_, int64_t previous) {
    detail::Entry entry;
    CORA_CHECK(input.size() == 3 && pool[0] >= 1 && pool[1] >= 1 && input[1] >= pool[0] &&
                   input[2] >= pool[1],
               Error::wrong_input_in_constructor, [&] {
                   return "Layer max_pool2d_layer: the pool fits into the input (channels, "
                          "height, width), got " +
                          shape_text(input);
               });
    const std::array<int64_t, 2> stride = stride_[0] > 0 ? stride_ : pool;
    const int64_t channels = input[0];
    const int64_t ho = (input[1] - pool[0]) / stride[0] + 1,
                  wo = (input[2] - pool[1]) / stride[1] + 1;
    const int64_t pixels = ho * wo;
    Block block;
    // the index of the next layer, the layer it reads, and the items of each window
    int64_t src = previous, items = pool[0] * pool[1];
    bool first = true;
    while (items > 1 || first) {
        const int64_t pairs = items / 2, left = items % 2;
        NeuralNetwork::Conv2d spec;
        spec.groups = channels;
        if (first) spec.stride = stride;
        // the input of the round: the image, then one item per channel and pixel
        const std::vector<int64_t> from =
            first ? input : std::vector<int64_t>{channels * items, ho, wo};
        // a 1 x 1 window: the selection only
        if (pairs == 0) {
            Layer pick = aux_round_conv({{1.0}}, channels, items, from, spec, pool, first, pixels);
            pick.source = src;
            block.layers.push_back(pick);
            block.activations.push_back(Activation::identity);
            return block;
        }
        // the differences a - b of each pair, and the second items (and an odd last one)
        std::vector<std::vector<double>> diff(size_t(pairs),
                                              std::vector<double>(size_t(items), 0.0));
        std::vector<std::vector<double>> pick(size_t(pairs + left),
                                              std::vector<double>(size_t(items), 0.0));
        for (int64_t j = 0; j < pairs; ++j) {
            diff[size_t(j)][size_t(2 * j)] = 1.0;
            diff[size_t(j)][size_t(2 * j + 1)] = -1.0;
            pick[size_t(j)][size_t(2 * j + 1)] = 1.0;
        }
        // an odd last item goes on as it is
        if (left) pick[size_t(pairs)][size_t(items - 1)] = 1.0;
        // the ReLU layer of the differences
        Layer relu = aux_round_conv(diff, channels, items, from, spec, pool, first, pixels);
        relu.source = src;
        block.layers.push_back(relu);
        block.activations.push_back(Activation::relu);
        const int64_t relu_index = previous + int64_t(block.layers.size());
        // the ReLU of pair j at item j of each channel, 0 at an odd last item
        std::vector<int64_t> index;
        for (int64_t ch = 0; ch < channels; ++ch)
            for (int64_t j = 0; j < pairs + left; ++j)
                for (int64_t p = 0; p < pixels; ++p)
                    index.push_back(j < pairs ? (ch * pairs + j) * pixels + p : -1);
        const int64_t size = channels * (pairs + left) * pixels;
        Layer join = Layer::gather(index, channels * pairs * pixels, aux_zeros(size));
        join.source = relu_index;
        // b + relu(a - b): the second items are the residual, read from where the round began
        join.residuals.push_back(
            {src, std::make_shared<Layer>(
                      aux_round_conv(pick, channels, items, from, spec, pool, first, pixels))});
        block.layers.push_back(join);
        block.activations.push_back(Activation::identity);
        src = previous + int64_t(block.layers.size());
        items = pairs + left;
        first = false;
    }
    return block;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
