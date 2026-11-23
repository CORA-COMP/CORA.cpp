// backprop - the gradient of a network at its input, at points and over boxes
//
// The reverse pass of the chain rule through the layers: an activation scales the gradient by its
// derivative at its input, an affine layer maps it by W^T. At points the input of each layer is
// the value of the forward pass; over a box the gradient's bounds [gl, gu] go back with the
// interval arithmetic's boxes as the layers' inputs (Gowal et al. 2019: an affine layer maps the
// midpoint of the gradient by W^T and its radius by |W|^T). The weights are not updated (the
// network is immutable here). propagate_bounds carries the bounds of one activation layer's input
// through the layers to the next activation layer, where they meet the bounds stored there.
//
// Syntax:   Tensor g = nn.backprop(x, grad_out);
//           auto [gl, gu] = nn.backprop_interval_batch(l, u, gl_out, gu_out);
//           bounds = nn.propagate_bounds(bounds, i);
// Inputs:   x, l, u - (B, in) points and box; grad_out, gl_out, gu_out - (B, out)
// Outputs:  g, gl, gu - (B, in)
// See also: cora/nn/neural_network/neural_network.h, sensitivity.cpp
// References:
//    [1] S. Gowal, K. Dvijotham, R. Stanforth, R. Bunel, C. Qin, J. Uesato, R. Arandjelovic,
//        T. Mann, P. Kohli. "On the effectiveness of interval bound propagation for training
//        verifiably robust models", arXiv:1810.12715, 2018.
//
// CORA: nn/@neuralNetwork/backprop.m
// CORA: nn/@neuralNetwork/backpropIntervalBatch.m
// CORA: nn/@neuralNetwork/propagateBounds.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_layers.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The image of the box [l, u] under the activation (an interval function of each neuron).
std::pair<Tensor, Tensor> aux_activation_box(Activation a, const Tensor &l, const Tensor &u) {
    switch (a) {
    case Activation::identity:
        return {l, u};
    case Activation::reciprocal:
        // 1 / max(x, 1) falls
        return {priv_apply(a, u), priv_apply(a, l)};
    case Activation::square: {
        // the least square is 0 where the box crosses it
        const Tensor lo = l.square().minimum(u.square()), hi = l.square().maximum(u.square());
        const Tensor crosses =
            l.less_equal(l.zeros_like()).multiply(u.greater_equal(u.zeros_like()));
        return {Tensor::where(crosses, lo.zeros_like(), lo), hi};
    }
    case Activation::product:
        CORA_FAIL(Error::not_supported, "NeuralNetwork: a product layer has no box image here");
    default:
        // relu, sigmoid, tanh, exp, sign and floor rise
        return {priv_apply(a, l), priv_apply(a, u)};
    }
}

/// The boxes of the pre-activations of every layer (rows (B, out)) by interval arithmetic from the
/// box [l, u] (B, in): the midpoint through the layer, the radius through |W|.
std::vector<std::pair<Tensor, Tensor>> aux_boxes(const NeuralNetwork &nn, const Tensor &l,
                                                 const Tensor &u) {
    std::vector<std::pair<Tensor, Tensor>> pre;
    Tensor lo = l, hi = u;
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        const NeuralNetwork::Layer &layer = nn.layers()[k];
        const Tensor mid = (lo + hi) * 0.5, rad = (hi - lo) * 0.5;
        const Tensor c = layer.rows(mid) + layer.b, r = layer.abs_rows(rad);
        pre.emplace_back(c - r, c + r);
        std::tie(lo, hi) = aux_activation_box(nn.activations()[k], c - r, c + r);
    }
    return pre;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The gradient at points --------------------------------------------------------------------------

Tensor NeuralNetwork::backprop(const Tensor &x, const Tensor &grad_out) const {
    detail::Entry entry;
    CORA_CHECK(x.ndim() == 2 && x.shape()[1] == input_dim() && grad_out.ndim() == 2 &&
                   grad_out.shape()[0] == x.shape()[0] && grad_out.shape()[1] == output_dim(),
               Error::dimension_mismatch, [&] {
                   return "backprop: the points are rows (B, " + std::to_string(input_dim()) +
                          ") and the gradients (B, " + std::to_string(output_dim()) + "), got " +
                          shape_text(x.shape()) + " and " + shape_text(grad_out.shape());
               });
    CORA_CHECK(is_chain(), Error::not_supported, [] {
        return std::string("backprop: the layers must be a chain (no residuals or skips)");
    });
    // the forward pass: the input of each activation
    std::vector<Tensor> pre;
    Tensor h = x;
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        pre.push_back(layers_[k].rows(h) + layers_[k].b);
        h = priv_apply(activations_[k], pre.back());
    }
    // the reverse pass: the derivative of the activation, then W^T
    Tensor g = grad_out;
    for (std::size_t k = layers_.size(); k-- > 0;) {
        g = priv_slope_times_t(activations_[k], priv_derivative(activations_[k], pre[k]), g);
        g = layers_[k].rows_t(g);
    }
    return g;
}

// The gradient over a box -------------------------------------------------------------------------

std::pair<Tensor, Tensor> NeuralNetwork::backprop_interval_batch(const Tensor &l, const Tensor &u,
                                                                 const Tensor &gl,
                                                                 const Tensor &gu) const {
    detail::Entry entry;
    // the box and the gradients' bounds are rows of the sizes of the network
    CORA_CHECK(l.shape() == u.shape() && l.ndim() == 2 && l.shape()[1] == input_dim() &&
                   gl.shape() == gu.shape() && gl.ndim() == 2 && gl.shape()[0] == l.shape()[0] &&
                   gl.shape()[1] == output_dim(),
               Error::dimension_mismatch, [&] {
                   return "backprop_interval_batch: the box is rows (B, " +
                          std::to_string(input_dim()) + ") and the gradient bounds (B, " +
                          std::to_string(output_dim()) + "), got " + shape_text(l.shape()) +
                          " and " + shape_text(gl.shape());
               });
    CORA_CHECK(is_chain(), Error::not_supported, [] {
        return std::string(
            "backprop_interval_batch: the layers must be a chain (no residuals or skips)");
    });
    const auto pre = aux_boxes(*this, l, u);
    Tensor lo = gl, hi = gu;
    for (std::size_t k = layers_.size(); k-- > 0;) {
        // the activation: the derivative at the lower (upper) end scales the lower (upper) bound
        lo =
            priv_slope_times_t(activations_[k], priv_derivative(activations_[k], pre[k].first), lo);
        hi = priv_slope_times_t(activations_[k], priv_derivative(activations_[k], pre[k].second),
                                hi);
        // the affine map of the layer: W^T the middle of the bounds, |W|^T the radius
        const Tensor mid = layers_[k].rows_t((hi + lo) * 0.5);
        const Tensor rad = layers_[k].abs_rows_t((hi - lo) * 0.5);
        lo = mid - rad;
        hi = mid + rad;
    }
    return {lo, hi};
}

// The bounds of a layer's input -------------------------------------------------------------------

std::vector<NnBounds> NeuralNetwork::propagate_bounds(std::vector<NnBounds> bounds,
                                                      int64_t i) const {
    detail::Entry entry;
    const int64_t K = int64_t(layers_.size());
    CORA_CHECK(i >= 0 && i < K && int64_t(bounds.size()) == K, Error::wrong_value, [&] {
        return "propagate_bounds: the layer " + std::to_string(i) + " and one bound per layer (" +
               std::to_string(K) + "), got " + std::to_string(bounds.size());
    });
    CORA_CHECK(is_chain(), Error::not_supported, [] {
        return std::string("propagate_bounds: the layers must be a chain (no residuals or skips)");
    });
    // the bounds count when they are set and without a NaN (CORA's isempty and isnan)
    const auto has = [](const NnBounds &b) {
        return b.l.defined() && b.u.defined() && !b.l.isnan().any(0).item() &&
               !b.u.isnan().any(0).item();
    };
    const std::size_t first = std::size_t(i);
    if (activations_[first] == Activation::identity || !has(bounds[first])) return bounds;
    // the bounds after layer i's activation, through the layers up to the next activation
    auto [lo, hi] = aux_activation_box(activations_[first], bounds[first].l.expand_dims(0),
                                       bounds[first].u.expand_dims(0));
    for (std::size_t j = first + 1; j < layers_.size(); ++j) {
        const Tensor mid = (lo + hi) * 0.5, rad = (hi - lo) * 0.5;
        const Tensor c = layers_[j].rows(mid) + layers_[j].b, r = layers_[j].abs_rows(rad);
        lo = c - r;
        hi = c + r;
        if (activations_[j] == Activation::identity) continue;
        // an activation layer: its bounds are these, or the intersection with the stored ones
        if (has(bounds[j])) {
            lo = lo.maximum(bounds[j].l.expand_dims(0));
            hi = hi.minimum(bounds[j].u.expand_dims(0));
        }
        bounds[j] = {lo.squeeze(0), hi.squeeze(0)};
        return bounds;
    }
    return bounds;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
