// conv2d - the linear part of a network layer: a matrix, a 2-D convolution or a gather
//
// A layer maps rows h (..., in) to W h + b (..., out). W is a dense matrix, a convolution (ONNX's
// Conv of a (channels, height, width) input in C order, with strides, pads, dilations and groups)
// or a gather (each output an input or 0, times an optional factor: ONNX's Transpose, Pad, Slice,
// the front end's products and softmax; Reshape and Flatten are the identity on rows in C order).
// A convolution is never made a matrix: its rows are the patches of the input times the kernel
// (im2col), and its transpose multiplies by the kernel's transpose and folds the patches back onto
// the input pixels (col2im), so W^T y costs what W x does.
// The generator axis of a batch of zonotopes is just more rows, a batched convolution.
// A ConvTranspose (CORA's nnConvTranspose2DLayer) is the transposed product of the convolution it
// inverts (kernel (in, out / groups, kh, kw), the output's shape given). CORA's nnReshapeLayer
// (its output the input entries at an index list) is a gather: its gradient scatters back (the
// transpose, a repeated input summing its outputs).
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
// CORA: nn/layers/linear/nnConvTranspose2DLayer.m
// CORA: nn/layers/other/nnReshapeLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "global/tensor/derived.h"

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
    // (col2im: the fold positions of each input pixel, into P offsets, fold of them)
    std::vector<int64_t> patch, col2im;
    int64_t fold = 1;
    Tensor K, abs_k, KT, abs_kt; ///< (cin offsets, cout) and its transpose (cout, cin offsets)
    // a gather: the input of each output, the fan outputs of each input (out: none)
    std::vector<int64_t> index, inverse;
    int64_t fan = 1;
    Tensor scale, abs_scale; ///< the factor of each output (1, out) and |factor|; undefined: 1
    /// A convolution's kernel (cout, cin / groups, kh, kw), its absolute values, its input and
    /// output sizes and its spec, for the tensors' own convolution (`native`: use it).
    Tensor kernel, abs_kernel;
    std::array<int64_t, 2> size{0, 0}, size_out{0, 0};
    Tensor::ConvSpec spec;
    bool native = false;
    /// The largest sum of |kernel| over the entries of one output (`mass_out`) and of one input
    /// channel (`mass_in`): the scale of the rounding of the native products of radii.
    double mass_out = 0, mass_in = 0;
    /// Windows side by side (conv_windows, gather_windows): the lists index all of them, the
    /// rows' first axis counts them, and `in`, `out` are one window's (0: none).
    int64_t windows = 0;
};

namespace {

// Patches and gathers ----------------------------------------------------------------------------

/// The rows (N, k) of the leading axes of x (..., k), and back.
Tensor aux_flat(const Tensor &x, int64_t k) {
    // the rows are the leading dimensions (numel / k fails for a layer without neurons, k = 0)
    int64_t rows = 1;
    for (int64_t i = 0; i + 1 < x.ndim(); ++i) rows *= x.shape()[i];
    return x.reshape({rows, k});
}
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

/// The convolution's product: rows X (N, cin pixels_in), their patches (N, cin offsets, P) by
/// the patch list (offset-major), times KT (cout, cin offsets) per row: (N, cout P), no transposes.
Tensor aux_conv(const Tensor &X, const NeuralNetwork::LinearPlan &p, const Tensor &KT) {
    const int64_t N = X.shape()[0], P = p.pixels_out;
    const Tensor patches =
        aux_gather(X.reshape({N, p.cin, p.pixels_in}), p.patch).reshape({N, p.cin * p.offsets, P});
    return KT.matmul(patches).reshape({N, p.cout * P});
}

/// The transposed product: K (cin offsets, cout) times the rows Y (N, cout, P) are the patches'
/// values, folded back onto the input pixels (col2im), (N, cin pixels_in).
Tensor aux_conv_t(const Tensor &Y, const NeuralNetwork::LinearPlan &p, const Tensor &K) {
    const int64_t N = Y.shape()[0], P = p.pixels_out, KK = p.offsets;
    const Tensor cols = K.matmul(Y.reshape({N, p.cout, P})).reshape({N, p.cin, KK * P});
    return aux_gather(cols, p.col2im)
        .reshape({N, p.cin, p.pixels_in, p.fold})
        .sum(-1)
        .reshape({N, p.cin * p.pixels_in});
}

/// The convolution or its transpose by the tensors' own operation (Tensor::conv2d). The products
/// of radii (`absolute`) gain the rounding's slack: (4 n + 256) ulps of the largest radius times
/// the kernel's mass, n the terms of a sum (a sum in any order, a library's fast transforms).
Tensor aux_native(const Tensor &X, const NeuralNetwork::LinearPlan &p, bool absolute,
                  bool transposed) {
    const int64_t N = X.shape()[0];
    const Tensor &k = absolute ? p.abs_kernel : p.kernel;
    const Tensor Y = transposed ? X.reshape({N, p.cout, p.size_out[0], p.size_out[1]})
                                      .conv_transpose2d(k, p.size, p.spec)
                                      .reshape({N, p.cin * p.pixels_in})
                                : X.reshape({N, p.cin, p.size[0], p.size[1]})
                                      .conv2d(k, p.spec)
                                      .reshape({N, p.cout * p.pixels_out});
    if (!absolute) return Y;
    const double terms = double(std::max(p.cin, p.cout) / p.spec.groups * p.offsets);
    const double slack =
        (4.0 * terms + 256.0) * machine_eps(X.dtype()) * (transposed ? p.mass_in : p.mass_out);
    return Y + X.abs().max(-1, true) * slack;
}

/// The fold lists of a convolution's transpose from the patch entries of each input pixel.
void aux_fold(NeuralNetwork::LinearPlan &p, const std::vector<std::vector<int64_t>> &entries) {
    for (const std::vector<int64_t> &e : entries) p.fold = std::max(p.fold, int64_t(e.size()));
    p.col2im.assign(std::size_t(p.pixels_in * p.fold), p.pixels_out * p.offsets);
    for (std::size_t i = 0; i < entries.size(); ++i)
        for (std::size_t t = 0; t < entries[i].size(); ++t)
            p.col2im[i * std::size_t(p.fold) + t] = entries[i][t];
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
    const int64_t KK = plan->offsets;
    // the patch and fold lists of the tensors' convolution (derived.cpp), its kernel and spec
    plan->spec = {spec.stride, spec.dilation, spec.pads, spec.groups};
    derived::ConvLists lists = derived::conv_lists(H, Wd, kh, kw, plan->spec);
    plan->patch = std::move(lists.patch);
    plan->col2im = std::move(lists.col2im);
    plan->fold = lists.fold;
    plan->kernel = kernel;
    plan->abs_kernel = kernel.abs();
    plan->size = {H, Wd};
    plan->size_out = {Ho, Wo};
    const int64_t per = (cin / spec.groups) * kh * kw;
    plan->mass_out = plan->abs_kernel.reshape({cout, per}).sum(1).max().item();
    plan->mass_in =
        plan->abs_kernel.sum(0).reshape({cin / spec.groups, kh * kw}).sum(1).max().item();
    // the kernel as (cin offsets, cout), zero between groups, and transposed as (cout offsets, cin)
    const std::vector<double> kv = kernel.data();
    const int64_t cg = cin / spec.groups, og = cout / spec.groups;
    std::vector<double> M(std::size_t(cin * KK * cout), 0.0);
    for (int64_t o = 0; o < cout; ++o)
        for (int64_t j = 0; j < cg; ++j)
            for (int64_t kk = 0; kk < KK; ++kk) {
                const int64_t ci = (o / og) * cg + j;
                const double v = kv[std::size_t((o * cg + j) * KK + kk)];
                M[std::size_t((ci * KK + kk) * cout + o)] = v;
            }
    // the matrices on the kernel's backend, and their absolute values
    plan->K = Tensor::like(kernel, M, {cin * KK, cout});
    plan->KT = plan->K.matrix_transpose();
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
                                                  const Tensor &bias, const Tensor &scale) {
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
    // the factors as a row, broadcast over the rows of a product
    if (scale.defined()) {
        CORA_CHECK(
            scale.shape() == std::vector<int64_t>({plan->out}), Error::wrong_input_in_constructor,
            [] { return std::string("Layer::gather: the scale needs one entry per output"); });
        plan->scale = scale.reshape({1, plan->out});
        plan->abs_scale = plan->scale.abs();
    }
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

// The windows -------------------------------------------------------------------------------------

int64_t NeuralNetwork::Layer::conv_channels(bool out) const {
    detail::Entry entry;
    const bool is = plan && plan->conv && !plan->transposed && plan->windows == 0;
    return !is ? 0 : out ? plan->cout : plan->cin;
}

int64_t NeuralNetwork::Layer::conv_pixels(bool out) const {
    detail::Entry entry;
    const bool is = plan && plan->conv && !plan->transposed && plan->windows == 0;
    return !is ? 0 : out ? plan->pixels_out : plan->pixels_in;
}

std::vector<int64_t>
NeuralNetwork::Layer::conv_reads(const std::vector<int64_t> &out_pixels) const {
    detail::Entry entry;
    CORA_CHECK(conv_pixels(true) > 0, Error::wrong_value,
               [] { return std::string("Layer::conv_reads: the layer is no convolution"); });
    // the patch entries of the pixels, without the padding's
    std::vector<int64_t> reads;
    const int64_t P = plan->pixels_out;
    for (const int64_t o : out_pixels)
        for (int64_t kk = 0; o >= 0 && kk < plan->offsets; ++kk)
            if (const int64_t i = plan->patch[std::size_t(kk * P + o)]; i < plan->pixels_in)
                reads.push_back(i);
    // sorted, each once
    std::sort(reads.begin(), reads.end());
    reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
    return reads;
}

NeuralNetwork::Layer NeuralNetwork::Layer::conv_windows(const Layer &conv,
                                                        const std::vector<int64_t> &out_pixels,
                                                        int64_t m_out,
                                                        const std::vector<int64_t> &in_pixels,
                                                        int64_t m_in) {
    detail::Entry entry;
    // a convolution, and as many windows of output pixels as of input pixels
    CORA_CHECK(conv.conv_pixels(true) > 0 && m_out > 0 && m_in > 0 &&
                   int64_t(out_pixels.size()) % m_out == 0 &&
                   int64_t(in_pixels.size()) / m_in == int64_t(out_pixels.size()) / m_out &&
                   int64_t(in_pixels.size()) % m_in == 0,
               Error::wrong_input_in_constructor, [] {
                   return std::string("Layer::conv_windows: a convolution and as many windows of "
                                      "output pixels as of input pixels are needed");
               });
    // the plan of all windows side by side: one window's sizes, the lists of all
    const LinearPlan &c = *conv.plan;
    const int64_t B = int64_t(out_pixels.size()) / m_out, KK = c.offsets;
    auto plan = std::make_shared<LinearPlan>();
    plan->conv = true;
    plan->windows = B;
    plan->cin = c.cin;
    plan->cout = c.cout;
    plan->offsets = KK;
    plan->pixels_in = B * m_in;
    plan->pixels_out = B * m_out;
    plan->in = c.cin * m_in;
    plan->out = c.cout * m_out;
    // each window's patch entries: the slot of the input pixel in its window (none: zero)
    const int64_t P = plan->pixels_out, HW = plan->pixels_in;
    plan->patch.assign(std::size_t(P * KK), HW);
    std::vector<std::vector<int64_t>> entries(static_cast<std::size_t>(HW));
    for (int64_t w = 0; w < B; ++w) {
        const auto first = in_pixels.begin() + std::ptrdiff_t(w * m_in);
        for (int64_t s = 0; s < m_out; ++s) {
            const int64_t o = out_pixels[std::size_t(w * m_out + s)];
            for (int64_t kk = 0; o >= 0 && kk < KK; ++kk) {
                const int64_t i = c.patch[std::size_t(kk * c.pixels_out + o)];
                if (i >= c.pixels_in) continue;
                // the window's input pixels are sorted (after them: -1)
                const auto end = std::find(first, first + std::ptrdiff_t(m_in), int64_t(-1));
                const auto at = std::lower_bound(first, end, i);
                if (at == end || *at != i) continue;
                const int64_t slot = w * m_in + int64_t(at - first), e = kk * P + w * m_out + s;
                plan->patch[std::size_t(e)] = slot;
                entries[std::size_t(slot)].push_back(e);
            }
        }
    }
    // the transpose folds the patches back onto each window's input pixels
    aux_fold(*plan, entries);
    // the convolution's own kernel matrices
    plan->K = c.K;
    plan->KT = c.KT;
    plan->abs_k = c.abs_k;
    plan->abs_kt = c.abs_kt;
    Layer layer{Tensor(), c.K.zeros_like({plan->out})};
    layer.plan = plan;
    return layer;
}

NeuralNetwork::Layer NeuralNetwork::Layer::gather_windows(const std::vector<int64_t> &index,
                                                          int64_t m, int64_t inputs,
                                                          const Tensor &like) {
    detail::Entry entry;
    CORA_CHECK(
        m > 0 && inputs > 0 && int64_t(index.size()) % m == 0, Error::wrong_input_in_constructor,
        [] { return std::string("Layer::gather_windows: the index needs m entries a window"); });
    // the windows side by side: one gather of all their inputs
    const int64_t B = int64_t(index.size()) / m;
    std::vector<int64_t> all(index.size(), -1);
    for (std::size_t e = 0; e < index.size(); ++e) {
        CORA_CHECK(index[e] < inputs, Error::wrong_input_in_constructor,
                   [] { return std::string("Layer::gather_windows: an index is out of range"); });
        if (index[e] >= 0) all[e] = (int64_t(e) / m) * inputs + index[e];
    }
    const Layer whole = gather(all, B * inputs, like.zeros_like({B * m}));
    auto plan = std::make_shared<LinearPlan>(*whole.plan);
    plan->windows = B;
    plan->in = inputs;
    plan->out = m;
    Layer layer{Tensor(), like.zeros_like({m})};
    layer.plan = plan;
    return layer;
}

// The products ------------------------------------------------------------------------------------

int64_t NeuralNetwork::Layer::inputs() const { return plan ? plan->in : W.shape()[1]; }
int64_t NeuralNetwork::Layer::outputs() const { return plan ? plan->out : W.shape()[0]; }

int64_t NeuralNetwork::Layer::work() const {
    detail::Entry entry;
    // the patches of either direction, or the fan of a gather's transpose
    if (!plan) return std::max(inputs(), outputs());
    if (!plan->conv) return std::max(plan->out, plan->in * plan->fan);
    // the tensors' own convolution: no patches, a library's workspace about 4 layers (measured)
    if (plan->native) return 4 * std::max(plan->in, plan->out);
    return std::max(plan->pixels_out * plan->offsets, plan->pixels_in * plan->fold) * plan->cin;
}

NeuralNetwork::Layer NeuralNetwork::Layer::native_convolution(bool on) const {
    detail::Entry entry;
    // a convolution (or its transpose) of whole layers: the same plan with the switch
    Layer layer = *this;
    if (plan && plan->conv && plan->windows == 0 && plan->kernel.defined()) {
        auto p = std::make_shared<LinearPlan>(*plan);
        p->native = on;
        layer.plan = p;
    }
    // a residual's own map as well
    for (Residual &r : layer.residuals)
        if (r.map) r.map = std::make_shared<Layer>(r.map->native_convolution(on));
    return layer;
}

Tensor NeuralNetwork::Layer::aux_product(const Tensor &x, bool absolute, bool transposed_) const {
    detail::Entry entry;
    // a ConvTranspose is the transposed product of its convolution
    const bool transposed = plan && plan->transposed ? !transposed_ : transposed_;
    const int64_t from = transposed_ ? outputs() : inputs(),
                  to = transposed_ ? inputs() : outputs();
    if (plan && plan->windows > 0) return aux_windows(x, absolute, transposed);
    // the rows as a matrix (N, from)
    const Tensor X = aux_flat(x, from);
    Tensor Y;
    if (!plan) {
        // a matrix: rows times W^T, or times W
        const Tensor M = absolute ? W.abs() : W;
        Y = transposed ? X.matmul(M) : X.matmul(M.matrix_transpose());
    } else if (plan->conv && plan->native) {
        // a convolution by the tensors' own operation
        Y = aux_native(X, *plan, absolute, transposed);
    } else if (plan->conv) {
        // a convolution: the patches times the kernel, or the transposed patches
        const LinearPlan &p = *plan;
        Y = transposed ? aux_conv_t(X, p, absolute ? p.abs_k : p.K)
                       : aux_conv(X, p, absolute ? p.abs_kt : p.KT);
    } else {
        // a gather and its transpose (sums over each input's outputs), times the factors
        const int64_t N = X.shape()[0];
        const Tensor &f = absolute ? plan->abs_scale : plan->scale;
        if (transposed) {
            const Tensor X3 = (f.defined() ? X.multiply(f) : X).reshape({N, 1, from});
            Y = aux_gather(X3, plan->inverse).reshape({N, to, plan->fan}).sum(-1);
        } else {
            Y = aux_gather(X.reshape({N, 1, from}), plan->index).reshape({N, to});
            if (f.defined()) Y = Y.multiply(f);
        }
    }
    return aux_unflat(Y, x, to);
}

// The products of the windows ---------------------------------------------------------------------

Tensor NeuralNetwork::Layer::aux_windows(const Tensor &x, bool absolute, bool transposed) const {
    detail::Entry entry;
    // rows (B, R, from) (or (B, from)) of B windows: each row's windows side by side (R, B from)
    const LinearPlan &p = *plan;
    const int64_t B = p.windows, from = transposed ? p.out : p.in, to = transposed ? p.in : p.out;
    const std::vector<int64_t> shape = x.shape();
    CORA_CHECK((shape.size() == 2 || shape.size() == 3) && shape[0] == B && shape.back() == from,
               Error::dimension_mismatch, [&] {
                   return "Layer: the rows of " + std::to_string(B) + " windows need the shape (" +
                          std::to_string(B) + ", ..., " + std::to_string(from) + "), got " +
                          shape_text(shape);
               });
    const int64_t R = shape.size() == 3 ? shape[1] : 1;
    Tensor Y;
    if (p.conv) {
        // channel-major: (B, R, c, m) -> (R, c, B m), the convolution, and back
        const int64_t cf = transposed ? p.cout : p.cin, ct = transposed ? p.cin : p.cout;
        const Tensor X = x.reshape({B, R, cf, from / cf})
                             .permute_dims({1, 2, 0, 3})
                             .reshape({R, cf * B * (from / cf)});
        Y = transposed ? aux_conv_t(X, p, absolute ? p.abs_k : p.K)
                       : aux_conv(X, p, absolute ? p.abs_kt : p.KT);
        Y = Y.reshape({R, ct, B, to / ct}).permute_dims({2, 0, 1, 3});
    } else {
        // a gather: (B, R, from) -> (R, B from), the gather of all windows, and back
        const Tensor X = x.reshape({B, R, from}).permute_dims({1, 0, 2}).reshape({R, 1, B * from});
        Y = transposed ? aux_gather(X, p.inverse).reshape({R, B * to, p.fan}).sum(-1)
                       : aux_gather(X, p.index);
        Y = Y.reshape({R, B, to}).permute_dims({1, 0, 2});
    }
    std::vector<int64_t> out = shape;
    out.back() = to;
    return Y.reshape(out);
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
        p->kernel = move(p->kernel);
        p->abs_kernel = move(p->abs_kernel);
        p->scale = move(p->scale);
        p->abs_scale = move(p->abs_scale);
        layer.plan = p;
    }
    // the graph: the same sources, the residuals' maps moved as well
    layer.source = source;
    for (const Residual &r : residuals)
        layer.residuals.push_back(
            {r.source, r.map ? std::make_shared<Layer>(r.map->to(dtype, device)) : nullptr});
    return layer;
}

CORA_FROM_NUMBERS Tensor NeuralNetwork::Layer::dense() const {
    detail::Entry entry;
    if (!plan) return W;
    // the image of the unit vectors, column by column (a residual's map has no bias)
    const Tensor eye = b.defined() ? b.eye_like(inputs()) : Tensor::eye(inputs());
    return rows(eye).matrix_transpose();
}

bool NeuralNetwork::Layer::nonnegative() const {
    detail::Entry entry;
    // the weights themselves: a convolution's kernel, a gather's factors (undefined: all 1)
    const auto least = [](const Tensor &t) {
        return t.numel() == 0 ? 0.0 : t.reshape({t.numel()}).min(0).item();
    };
    if (!plan) return least(W) >= 0;
    if (plan->conv) return least(plan->K) >= 0;
    return !plan->scale.defined() || least(plan->scale) >= 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
