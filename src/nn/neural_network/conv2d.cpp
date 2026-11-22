// conv2d - the linear part of a network layer: a matrix, a 2-D convolution or a gather
//
// A layer maps rows h (..., in) to W h + b (..., out). W is a dense matrix, a convolution (ONNX's
// Conv of a (channels, height, width) input in C order, with strides, pads, dilations and groups)
// or a gather (each output an input or 0: ONNX's Transpose, Pad, Slice; Reshape and Flatten are the
// identity on rows in C order). A convolution is never made a matrix: its rows are the patches of
// the input times the kernel (im2col), and its transpose gathers, for every input pixel and
// kernel offset, the output pixel it feeds (at most one), so W^T y is the same kind of product.
// The generator axis of a batch of zonotopes is just more rows, a batched convolution.
//
// Syntax:   Layer L = NeuralNetwork::Layer::conv2d(kernel, bias, {C, H, W}, spec);
//           Tensor y = L.rows(x) + L.b;   Tensor g = L.rows_t(y);   Tensor r = L.abs_rows(dr);
// Inputs:   kernel - (out, in / groups, kh, kw)
//           bias - (out channels times output pixels)
//           input - (channels, height, width) of the input
//           spec - strides, dilations, pads, groups
// Outputs:  L - the layer; rows (..., in) -> (..., out), rows_t (..., out) -> (..., in)
// See also: cora/nn/neural_network/neural_network.h, priv_zono_batch.h
// References:
//    [1] K. Chellapilla, S. Puri, P. Simard. "High performance convolutional neural networks for
//        document processing", IWFHR 2006.
//
// CORA: nn/layers/linear/nnConv2DLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

// The plan ----------------------------------------------------------------------------------------

/// What a convolution or a gather needs per product: the index lists and the kernel as matrices.
struct NeuralNetwork::LinearPlan {
    bool conv = false, transposed = false; ///< transposed: ConvTranspose, the convolution's W^T
    int64_t in = 0, out = 0;
    // a convolution: channels, pixels and kernel offsets; the patch and transposed index lists
    int64_t cin = 0, cout = 0, pixels_in = 0, pixels_out = 0, offsets = 0;
    std::vector<int64_t> patch, patch_t;
    Tensor K, abs_k, KT, abs_kt; ///< (cin offsets, cout) and (cout offsets, cin)
    // a gather: the input of each output, and the fan outputs of each input (out: none)
    std::vector<int64_t> index, inverse;
    int64_t fan = 1;
};

namespace {

// Patches and gathers ----------------------------------------------------------------------------

/// The rows (N, k) of the leading axes of x (..., k), and back.
Tensor aux_flat(const Tensor &x, int64_t k) { return x.reshape({x.numel() / k, k}); }
Tensor aux_unflat(const Tensor &y, const Tensor &like, int64_t k) {
    std::vector<int64_t> shape = like.shape();
    shape.back() = k;
    return y.reshape(shape);
}

/// The columns of x (N, c, p) at the indices of a list into p + 1 (p: zero), as (N, c, size).
Tensor aux_gather(const Tensor &x, const std::vector<int64_t> &idx) {
    const Tensor padded = Tensor::concat({x, x.zeros_like({x.shape()[0], x.shape()[1], 1})}, 2);
    return padded.take(idx, 2);
}

/// The convolution's product: rows X (N, cin pixels_in) with kernel M (cin offsets, cout), or the
/// transposed one with the roles of in and out swapped.
Tensor aux_conv(const Tensor &X, int64_t channels, int64_t pixels, const std::vector<int64_t> &idx,
                int64_t positions, int64_t offsets, const Tensor &M) {
    const int64_t N = X.shape()[0], cols = M.shape()[1];
    const Tensor patches = aux_gather(X.reshape({N, channels, pixels}), idx)
                               .reshape({N, channels, positions, offsets})
                               .permute_dims({0, 2, 1, 3})
                               .reshape({N * positions, channels * offsets});
    return patches.matmul(M)
        .reshape({N, positions, cols})
        .permute_dims({0, 2, 1})
        .reshape({N, cols * positions});
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The layers --------------------------------------------------------------------------------------

NeuralNetwork::Layer NeuralNetwork::Layer::conv2d(const Tensor &kernel, const Tensor &bias,
                                                  const std::vector<int64_t> &input,
                                                  const Conv2d &spec, bool check_bias) {
    detail::Entry entry;
    const std::vector<int64_t> k = kernel.shape();
    CORA_CHECK(k.size() == 4 && input.size() == 3 && spec.groups >= 1 &&
                   input[0] == k[1] * spec.groups && k[0] % spec.groups == 0,
               Error::wrong_input_in_constructor, [&] {
                   return "Layer::conv2d: the kernel (out, in / groups, kh, kw) " + shape_text(k) +
                          " does not fit the input (channels, height, width) " + shape_text(input) +
                          " with " + std::to_string(spec.groups) + " groups";
               });
    // the sizes of the input and the output
    auto plan = std::make_shared<LinearPlan>();
    const int64_t cin = input[0], H = input[1], Wd = input[2], cout = k[0], kh = k[2], kw = k[3];
    const auto [sh, sw] = spec.stride;
    const auto [dh, dw] = spec.dilation;
    const int64_t Ho = (H + spec.pads[0] + spec.pads[2] - dh * (kh - 1) - 1) / sh + 1;
    const int64_t Wo = (Wd + spec.pads[1] + spec.pads[3] - dw * (kw - 1) - 1) / sw + 1;
    CORA_CHECK(Ho >= 1 && Wo >= 1, Error::wrong_input_in_constructor,
               [] { return std::string("Layer::conv2d: the output has no pixel"); });
    // the plan's counts
    plan->conv = true;
    plan->cin = cin;
    plan->cout = cout;
    plan->pixels_in = H * Wd;
    plan->pixels_out = Ho * Wo;
    plan->offsets = kh * kw;
    plan->in = cin * H * Wd;
    plan->out = cout * Ho * Wo;
    const int64_t P = plan->pixels_out, KK = plan->offsets, HW = plan->pixels_in;
    // the input pixel of each output pixel and offset, and the output pixel of each input one
    plan->patch.assign(std::size_t(P * KK), HW);
    plan->patch_t.assign(std::size_t(HW * KK), P);
    for (int64_t oh = 0; oh < Ho; ++oh)
        for (int64_t ow = 0; ow < Wo; ++ow)
            for (int64_t a = 0; a < kh; ++a)
                for (int64_t c = 0; c < kw; ++c) {
                    const int64_t ih = oh * sh - spec.pads[0] + a * dh;
                    const int64_t iw = ow * sw - spec.pads[1] + c * dw;
                    if (ih < 0 || ih >= H || iw < 0 || iw >= Wd) continue;
                    const int64_t o = oh * Wo + ow, i = ih * Wd + iw, kk = a * kw + c;
                    plan->patch[std::size_t(o * KK + kk)] = i;
                    plan->patch_t[std::size_t(i * KK + kk)] = o;
                }
    // the kernel as (cin offsets, cout), zero between groups, and transposed as (cout offsets, cin)
    const std::vector<double> kv = kernel.data();
    const int64_t cg = cin / spec.groups, og = cout / spec.groups;
    std::vector<double> M(std::size_t(cin * KK * cout), 0.0), MT(std::size_t(cout * KK * cin), 0.0);
    for (int64_t o = 0; o < cout; ++o)
        for (int64_t j = 0; j < cg; ++j)
            for (int64_t kk = 0; kk < KK; ++kk) {
                const int64_t ci = (o / og) * cg + j;
                const double v = kv[std::size_t((o * cg + j) * KK + kk)];
                M[std::size_t((ci * KK + kk) * cout + o)] = v;
                MT[std::size_t((o * KK + kk) * cin + ci)] = v;
            }
    // the matrices on the kernel's backend, and their absolute values
    plan->K = Tensor::like(kernel, M, {cin * KK, cout});
    plan->KT = Tensor::like(kernel, MT, {cout * KK, cin});
    plan->abs_k = plan->K.abs();
    plan->abs_kt = plan->KT.abs();
    CORA_CHECK(!check_bias || bias.shape() == std::vector<int64_t>({plan->out}),
               Error::wrong_input_in_constructor, [&] {
                   return "Layer::conv2d: the bias must be a vector (" + std::to_string(plan->out) +
                          "), one entry per output, got " + shape_text(bias.shape());
               });
    Layer layer{Tensor(), bias};
    layer.plan = plan;
    return layer;
}

NeuralNetwork::Layer NeuralNetwork::Layer::gather(const std::vector<int64_t> &index, int64_t inputs,
                                                  const Tensor &bias) {
    detail::Entry entry;
    auto plan = std::make_shared<LinearPlan>();
    plan->in = inputs;
    plan->out = int64_t(index.size());
    // the outputs of each input (an index of none is a zero)
    std::vector<std::vector<int64_t>> outputs(static_cast<std::size_t>(inputs));
    for (std::size_t i = 0; i < index.size(); ++i) {
        const int64_t j = index[i];
        CORA_CHECK(j < inputs, Error::wrong_input_in_constructor, [&] {
            return "Layer::gather: the index " + std::to_string(j) + " is out of range";
        });
        plan->index.push_back(j < 0 ? inputs : j);
        if (j >= 0) outputs[std::size_t(j)].push_back(int64_t(i));
    }
    // the transpose sums the fan outputs of each input (padded with none)
    for (const std::vector<int64_t> &o : outputs)
        plan->fan = std::max(plan->fan, int64_t(o.size()));
    plan->inverse.assign(std::size_t(inputs * plan->fan), plan->out);
    for (std::size_t j = 0; j < outputs.size(); ++j)
        for (std::size_t t = 0; t < outputs[j].size(); ++t)
            plan->inverse[j * std::size_t(plan->fan) + t] = outputs[j][t];
    CORA_CHECK(bias.shape() == std::vector<int64_t>({plan->out}), Error::wrong_input_in_constructor,
               [] { return std::string("Layer::gather: the bias needs one entry per output"); });
    Layer layer{Tensor(), bias};
    layer.plan = plan;
    return layer;
}

NeuralNetwork::Layer NeuralNetwork::Layer::conv_transpose2d(const Tensor &kernel,
                                                            const Tensor &bias,
                                                            const std::vector<int64_t> &output,
                                                            const Conv2d &spec) {
    detail::Entry entry;
    // the convolution from the output (out channels, H, W) back to the input is its transpose
    const Layer conv = conv2d(kernel, kernel.zeros_like({1}), output, spec, false);
    auto plan = std::make_shared<LinearPlan>(*conv.plan);
    plan->transposed = true;
    std::swap(plan->in, plan->out);
    CORA_CHECK(bias.shape() == std::vector<int64_t>({plan->out}), Error::wrong_input_in_constructor,
               [&] {
                   return "Layer::conv_transpose2d: the bias must be a vector (" +
                          std::to_string(plan->out) + "), one entry per output";
               });
    Layer layer{Tensor(), bias};
    layer.plan = plan;
    return layer;
}

// The products ------------------------------------------------------------------------------------

int64_t NeuralNetwork::Layer::inputs() const { return plan ? plan->in : W.shape()[1]; }
int64_t NeuralNetwork::Layer::outputs() const { return plan ? plan->out : W.shape()[0]; }

Tensor NeuralNetwork::Layer::aux_product(const Tensor &x, bool absolute, bool transposed_) const {
    detail::Entry entry;
    // a ConvTranspose is the transposed product of its convolution
    const bool transposed = plan && plan->transposed ? !transposed_ : transposed_;
    const int64_t from = transposed_ ? outputs() : inputs(),
                  to = transposed_ ? inputs() : outputs();
    // the rows as a matrix (N, from)
    const Tensor X = aux_flat(x, from);
    Tensor Y;
    if (!plan) {
        // a matrix: rows times W^T, or times W
        const Tensor M = absolute ? W.abs() : W;
        Y = transposed ? X.matmul(M) : X.matmul(M.matrix_transpose());
    } else if (plan->conv) {
        // a convolution: the patches times the kernel, or the transposed patches
        const LinearPlan &p = *plan;
        Y = transposed ? aux_conv(X, p.cout, p.pixels_out, p.patch_t, p.pixels_in, p.offsets,
                                  absolute ? p.abs_kt : p.KT)
                       : aux_conv(X, p.cin, p.pixels_in, p.patch, p.pixels_out, p.offsets,
                                  absolute ? p.abs_k : p.K);
    } else {
        // a gather and its transpose (the sum over each input's outputs) are gathers (|W| = W)
        const int64_t N = X.shape()[0];
        const Tensor X3 = X.reshape({N, 1, from});
        Y = transposed ? aux_gather(X3, plan->inverse).reshape({N, to, plan->fan}).sum(-1)
                       : aux_gather(X3, plan->index).reshape({N, to});
    }
    return aux_unflat(Y, x, to);
}

NeuralNetwork::Layer NeuralNetwork::Layer::to(DType dtype, const std::string &device) const {
    detail::Entry entry;
    const auto move = [&](const Tensor &t) {
        if (!t.defined()) return t;
        return (device.empty() ? t : t.to(device)).to(dtype);
    };
    Layer layer{move(W), move(b)};
    // the kernel matrices move with the layer; the index lists stay on the host
    if (plan) {
        auto p = std::make_shared<LinearPlan>(*plan);
        p->K = move(p->K);
        p->abs_k = move(p->abs_k);
        p->KT = move(p->KT);
        p->abs_kt = move(p->abs_kt);
        layer.plan = p;
    }
    return layer;
}

Tensor NeuralNetwork::Layer::dense() const {
    detail::Entry entry;
    if (!plan) return W;
    // the image of the unit vectors, column by column
    const Tensor eye = b.eye_like(inputs());
    return rows(eye).matrix_transpose();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
