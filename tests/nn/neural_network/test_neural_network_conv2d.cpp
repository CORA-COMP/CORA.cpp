// test_neural_network_conv2d - convolution and gather layers on rows, sets and in verify
//
// A convolution with groups, strides, dilations and asymmetric pads maps rows as the direct sum of
// its definition; its transpose and absolute value are those of its matrix (Layer::dense), on
// every backend. A gather moves and zeroes entries. A network of convolutions evaluates at points
// as the same network of dense layers, its batched zonotopes in verify are those of the dense one
// up to rounding, and verify gives the same verdicts.
//
// CORA: nn/layers/linear/nnConv2DLayer.m
// CORA: nn/layers/linear/nnConvTranspose2DLayer.m
// CORA: nn/layers/other/nnReshapeLayer.m
// CORA: nn/layers/nnLayer.m
// PROPERTY: nn/layers/linear/nnConvTranspose2DLayer.m
// PROPERTY: nn/layers/other/nnReshapeLayer.m
// PROPERTY: nn/layers/nnLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_zono_batch.h"
#include "testing.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

using namespace cora;
using test::check;

namespace {

/// Random values of a shape.
Tensor aux_random(Rng &rng, const std::vector<int64_t> &shape, double scale) {
    int64_t n = 1;
    for (const int64_t s : shape) n *= s;
    std::vector<double> v(static_cast<std::size_t>(n));
    rng.normal(v.data(), v.size(), scale);
    return Tensor::from_data(v, shape);
}

/// The convolution by its definition, input (C, H, W) in C order, per row of x.
std::vector<double> aux_direct(const std::vector<double> &x, int64_t N,
                               const std::vector<double> &k, int64_t cin, int64_t H, int64_t W,
                               int64_t cout, int64_t kh, int64_t kw, const NeuralNetwork::Conv2d &s,
                               int64_t &Ho, int64_t &Wo) {
    Ho = (H + s.pads[0] + s.pads[2] - s.dilation[0] * (kh - 1) - 1) / s.stride[0] + 1;
    Wo = (W + s.pads[1] + s.pads[3] - s.dilation[1] * (kw - 1) - 1) / s.stride[1] + 1;
    const int64_t cg = cin / s.groups, og = cout / s.groups;
    std::vector<double> y(static_cast<std::size_t>(N * cout * Ho * Wo), 0.0);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t o = 0; o < cout; ++o)
            for (int64_t oh = 0; oh < Ho; ++oh)
                for (int64_t ow = 0; ow < Wo; ++ow) {
                    // the sum over the group's channels and the kernel's taps inside the input
                    double sum = 0;
                    for (int64_t j = 0; j < cg; ++j)
                        for (int64_t a = 0; a < kh; ++a)
                            for (int64_t c = 0; c < kw; ++c) {
                                const int64_t ih = oh * s.stride[0] - s.pads[0] + a * s.dilation[0];
                                const int64_t iw = ow * s.stride[1] - s.pads[1] + c * s.dilation[1];
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                const int64_t ci = (o / og) * cg + j;
                                sum += k[std::size_t(((o * cg + j) * kh + a) * kw + c)] *
                                       x[std::size_t(((r * cin + ci) * H + ih) * W + iw)];
                            }
                    y[std::size_t(((r * cout + o) * Ho + oh) * Wo + ow)] = sum;
                }
    return y;
}

/// The largest absolute difference of two tensors.
double aux_gap(const Tensor &a, const Tensor &b) { return (a - b).abs().max().item(); }

/// A layer's native convolution (Tensor::conv2d and conv_transpose2d) is the patches' products
/// up to the rounding, for a convolution and a ConvTranspose; its products of radii are at least
/// the exact ones (the slack) and as close.
void native(const std::string &b) {
    Rng rng(11);
    NeuralNetwork::Conv2d spec;
    spec.stride = {2, 1};
    spec.dilation = {1, 2};
    spec.pads = {1, 0, 2, 1};
    spec.groups = 2;
    const int64_t cin = 4, H = 6, W = 5, cout = 6, kh = 3, kw = 2;
    const Tensor kernel = aux_random(rng, {cout, cin / 2, kh, kw}, 1.0);
    const int64_t Ho = (H + 3 - (kh - 1) - 1) / 2 + 1, Wo = (W + 1 - 2 * (kw - 1) - 1) + 1;
    const NeuralNetwork::Layer conv =
        NeuralNetwork::Layer::conv2d(kernel, Tensor::zeros({cout * Ho * Wo}), {cin, H, W}, spec);
    const NeuralNetwork::Layer back = NeuralNetwork::Layer::conv_transpose2d(
        kernel, Tensor::zeros({cin * H * W}), {cin, H, W}, spec);
    bool same = true, above = true, kept = true;
    for (const NeuralNetwork::Layer *layer : {&conv, &back}) {
        const NeuralNetwork::Layer fast = layer->native_convolution();
        // rows of a batch (2, 3, n) through the layer and its transpose
        const Tensor x = aux_random(rng, {2, 3, layer->inputs()}, 1.0);
        const Tensor y = aux_random(rng, {2, 3, layer->outputs()}, 1.0);
        same = same && aux_gap(fast.rows(x), layer->rows(x)) < 1e-12 &&
               aux_gap(fast.rows_t(y), layer->rows_t(y)) < 1e-12;
        // the radii: not below the patches' products, and within the slack of them
        const Tensor r = x.abs(), q = y.abs();
        const Tensor up = fast.abs_rows(r) - layer->abs_rows(r);
        const Tensor up_t = fast.abs_rows_t(q) - layer->abs_rows_t(q);
        above = above && up.min().item() >= 0 && up_t.min().item() >= 0 &&
                up.max().item() < 1e-10 && up_t.max().item() < 1e-10;
        kept = kept && fast.inputs() == layer->inputs() && fast.outputs() == layer->outputs() &&
               aux_gap(fast.native_convolution(false).rows(x), layer->rows(x)) == 0;
    }
    check(same, b + ": a native convolution and its transpose are the patches' products");
    check(above, b + ": a native product of radii is at least the patches' and as close");
    check(kept, b + ": the native switch keeps the sizes and turns off again");
    // a matrix layer has no convolution: itself
    const NeuralNetwork::Layer dense{aux_random(rng, {3, 4}, 1.0), Tensor::zeros({3})};
    const Tensor v = aux_random(rng, {2, 4}, 1.0);
    check(aux_gap(dense.native_convolution().rows(v), dense.rows(v)) == 0,
          b + ": a matrix layer's native switch is the layer");
}

/// A convolution's rows are its definition; its transpose, absolute value and a gather are those
/// of the matrix.
void layers(const std::string &b) {
    Rng rng(5);
    NeuralNetwork::Conv2d spec;
    spec.stride = {2, 1};
    spec.dilation = {1, 2};
    spec.pads = {1, 0, 2, 1};
    spec.groups = 2;
    const int64_t cin = 4, H = 6, W = 5, cout = 6, kh = 3, kw = 2;
    const Tensor kernel = aux_random(rng, {cout, cin / 2, kh, kw}, 1.0);
    const Tensor x = aux_random(rng, {3, cin * H * W}, 1.0);
    int64_t Ho = 0, Wo = 0;
    const std::vector<double> want =
        aux_direct(x.data(), 3, kernel.data(), cin, H, W, cout, kh, kw, spec, Ho, Wo);
    const int64_t out = cout * Ho * Wo;
    const NeuralNetwork::Layer L =
        NeuralNetwork::Layer::conv2d(kernel, Tensor::zeros({out}), {cin, H, W}, spec);
    check(L.inputs() == cin * H * W && L.outputs() == out, b + ": the convolution's sizes");
    check(aux_gap(L.rows(x), Tensor::from_data(want, {3, out})) < 1e-12,
          b + ": the convolution's rows are its definition");
    // the transpose, the absolute values and a batch of generator rows (B, q, n)
    const Tensor M = L.dense(), y = aux_random(rng, {2, 4, out}, 1.0);
    check(aux_gap(L.rows_t(y), y.matmul(M)) < 1e-12, b + ": W^T of the convolution");
    check(aux_gap(L.abs_rows(x), x.matmul(M.abs().matrix_transpose())) < 1e-12 &&
              aux_gap(L.abs_rows_t(y), y.matmul(M.abs())) < 1e-12,
          b + ": |W| of the convolution, both ways");
    // a gather: a permutation with a zero and a dropped input, plus the bias
    const NeuralNetwork::Layer G =
        NeuralNetwork::Layer::gather({2, -1, 0, 3}, 5, Tensor({1.0, 2.0, 3.0, 4.0}));
    const Tensor g = G.rows(Tensor({{10.0, 11.0, 12.0, 13.0, 14.0}})) + G.b;
    check(aux_gap(g, Tensor({{13.0, 2.0, 13.0, 17.0}})) == 0, b + ": the gather's rows");
    check(aux_gap(G.rows_t(Tensor({{1.0, 2.0, 3.0, 4.0}})), Tensor({{3.0, 0.0, 1.0, 4.0, 0.0}})) ==
              0,
          b + ": the gather's transpose");
    // a repeated input (a nearest upsample): the transpose sums its outputs
    const NeuralNetwork::Layer U =
        NeuralNetwork::Layer::gather({1, 1, 0}, 2, Tensor({0.0, 0.0, 0.0}));
    check(aux_gap(U.rows_t(Tensor({{1.0, 2.0, 5.0}})), Tensor({{5.0, 3.0}})) == 0,
          b + ": a repeated input's transpose sums its outputs");
    // the signs of the weights: a convolution's kernel, a gather's factors, a matrix
    const NeuralNetwork::Layer S =
        NeuralNetwork::Layer::gather({0, 1}, 2, Tensor({0.0, 0.0}), Tensor({1.0, -2.0}));
    const NeuralNetwork::Layer D =
        NeuralNetwork::Layer::linear(Tensor({{1.0, 0.0}}), Tensor({0.0}));
    const NeuralNetwork::Layer N =
        NeuralNetwork::Layer::linear(Tensor({{1.0, -1e-30}}), Tensor({0.0}));
    check(G.nonnegative() && U.nonnegative() && !S.nonnegative() && D.nonnegative() &&
              !N.nonnegative() && L.nonnegative() == (M.min().item() >= 0),
          b + ": whether every weight is at least 0, of each kind of layer");
    // a wrong index is refused while the checks are on
    if (checks_enabled())
        check(test::raises([&] { NeuralNetwork::Layer::gather({2}, 2, Tensor({0.0})); },
                           Error::wrong_input_in_constructor),
              b + ": a gather refuses an index out of range");
}

/// A ConvTranspose by its definition: each input pixel scatters its kernel into the output.
std::vector<double> aux_scatter(const std::vector<double> &x, int64_t N,
                                const std::vector<double> &k, int64_t cin, int64_t Hi, int64_t Wi,
                                int64_t cout, int64_t kh, int64_t kw, int64_t H, int64_t W,
                                const NeuralNetwork::Conv2d &s) {
    const int64_t cg = cin / s.groups, og = cout / s.groups;
    std::vector<double> y(static_cast<std::size_t>(N * cout * H * W), 0.0);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t i = 0; i < cin; ++i)
            for (int64_t ih = 0; ih < Hi; ++ih)
                for (int64_t iw = 0; iw < Wi; ++iw)
                    for (int64_t o = 0; o < og; ++o)
                        for (int64_t a = 0; a < kh; ++a)
                            for (int64_t c = 0; c < kw; ++c) {
                                // the output pixel and channel this tap lands on
                                const int64_t oh = ih * s.stride[0] - s.pads[0] + a * s.dilation[0];
                                const int64_t ow = iw * s.stride[1] - s.pads[1] + c * s.dilation[1];
                                if (oh < 0 || oh >= H || ow < 0 || ow >= W) continue;
                                const int64_t oc = (i / cg) * og + o;
                                y[std::size_t(((r * cout + oc) * H + oh) * W + ow)] +=
                                    x[std::size_t(((r * cin + i) * Hi + ih) * Wi + iw)] *
                                    k[std::size_t(((i * og + o) * kh + a) * kw + c)];
                            }
    return y;
}

/// A ConvTranspose is the scatter of its definition, its transpose the convolution's rows, and a
/// reshape (a gather) moves entries and its gradient moves them back.
void transposed_and_reshape(const std::string &b) {
    Rng rng(11);
    NeuralNetwork::Conv2d spec;
    spec.stride = {2, 1};
    spec.dilation = {1, 2};
    spec.pads = {1, 0, 0, 1};
    spec.groups = 2;
    const int64_t cin = 4, cout = 6, kh = 3, kw = 2, H = 7, W = 6;
    const Tensor kernel = aux_random(rng, {cin, cout / 2, kh, kw}, 1.0);
    const NeuralNetwork::Layer L = NeuralNetwork::Layer::conv_transpose2d(
        kernel, Tensor::zeros({cout * H * W}), {cout, H, W}, spec);
    // the input is the output of the convolution this one inverts
    const int64_t Hi =
        (H + spec.pads[0] + spec.pads[2] - spec.dilation[0] * (kh - 1) - 1) / spec.stride[0] + 1;
    const int64_t Wi =
        (W + spec.pads[1] + spec.pads[3] - spec.dilation[1] * (kw - 1) - 1) / spec.stride[1] + 1;
    check(L.inputs() == cin * Hi * Wi && L.outputs() == cout * H * W,
          b + ": the ConvTranspose's sizes");
    const Tensor x = aux_random(rng, {3, L.inputs()}, 1.0);
    check(aux_gap(L.rows(x), Tensor::from_data(aux_scatter(x.data(), 3, kernel.data(), cin, Hi, Wi,
                                                           cout, kh, kw, H, W, spec),
                                               {3, L.outputs()})) < 1e-12,
          b + ": the ConvTranspose's rows are its definition");
    const Tensor M = L.dense(), y = aux_random(rng, {2, 4, L.outputs()}, 1.0);
    check(aux_gap(L.rows_t(y), y.matmul(M)) < 1e-12 &&
              aux_gap(L.abs_rows(x), x.matmul(M.abs().matrix_transpose())) < 1e-12,
          b + ": the ConvTranspose's transpose and absolute value");
    // a reshape of a 2 x 3 matrix to its transpose: a gather; its gradient is the inverse
    const NeuralNetwork::Layer R =
        NeuralNetwork::Layer::gather({0, 3, 1, 4, 2, 5}, 6, Tensor::zeros({6}));
    const Tensor v({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0}});
    check(aux_gap(R.rows(v), Tensor({{0.0, 3.0, 1.0, 4.0, 2.0, 5.0}})) == 0 &&
              aux_gap(R.rows_t(R.rows(v)), v) == 0,
          b + ": a reshape moves the entries, its gradient moves them back");
}

/// A small convolutional network and the same network of dense layers.
std::pair<NeuralNetwork, NeuralNetwork> aux_networks(Rng &rng) {
    NeuralNetwork::Conv2d spec;
    spec.pads = {1, 1, 0, 0};
    const NeuralNetwork::Layer c1 = NeuralNetwork::Layer::conv2d(
        aux_random(rng, {3, 1, 2, 2}, 0.7), aux_random(rng, {3 * 4 * 4}, 0.2), {1, 4, 4}, spec);
    NeuralNetwork::Conv2d s2;
    s2.stride = {2, 2};
    const NeuralNetwork::Layer c2 = NeuralNetwork::Layer::conv2d(
        aux_random(rng, {2, 3, 2, 2}, 0.5), aux_random(rng, {2 * 2 * 2}, 0.2), {3, 4, 4}, s2);
    const NeuralNetwork::Layer d3{aux_random(rng, {2, 8}, 0.5), aux_random(rng, {2}, 0.1)};
    const std::vector<Activation> acts = {Activation::relu, Activation::relu, Activation::identity};
    const NeuralNetwork conv({c1, c2, d3}, acts);
    const NeuralNetwork dense(
        {NeuralNetwork::Layer{c1.dense(), c1.b}, NeuralNetwork::Layer{c2.dense(), c2.b}, d3}, acts);
    return {conv, dense};
}

/// The network of convolutions is the dense one at points, in the batched zonotopes and in verify.
void networks(const std::string &b) {
    Rng rng(8);
    const auto [conv, dense] = aux_networks(rng);
    const Tensor x = aux_random(rng, {5, 16}, 1.0);
    check(aux_gap(conv.evaluate(x), dense.evaluate(x)) < 1e-12, b + ": evaluate at points");
    for (const double slots : {2.0, std::numeric_limits<double>::infinity()}) {
        const ZonoNet zc(conv, 16, slots, "bounds", "length"),
            zd(dense, 16, slots, "bounds", "length");
        const Tensor r = x.abs() * 0.05 + 0.01;
        const ZonoInput in = zc.input_zonotope(x, r);
        const ZonoPropagation oc = zc.forward(in.c, in.dr, in.G, Tensor(), nullptr, true);
        const ZonoPropagation od = zd.forward(in.c, in.dr, in.G, Tensor(), nullptr, true);
        bool same = aux_gap(oc.c, od.c) < 1e-12 && aux_gap(oc.G, od.G) < 1e-12;
        if (oc.dr.defined()) same = same && aux_gap(oc.dr, od.dr) < 1e-12;
        // CROWN's bound through the transposed convolutions
        const Tensor rows = Tensor({{1.0, -1.0}});
        same = same && aux_gap(priv_backward_lower(oc, zc, Tensor(), x, r, rows),
                               priv_backward_lower(od, zd, Tensor(), x, r, rows)) < 1e-12;
        check(same, b + ": the batched zonotopes and CROWN's bound of the convolutions");
    }
    // the same verdicts on a box with an unsafe set far below and one around an output
    const Tensor y0 = dense.evaluate(x.slice(0, 0, 1));
    for (const double t : {-100.0, y0.data()[0] + 0.5}) {
        NnProperty p;
        p.lo = x.slice(0, 0, 1) - 0.01;
        p.hi = x.slice(0, 0, 1) + 0.01;
        p.poly_case = {0};
        p.A = Tensor::from_data({1.0, 0.0}, {1, 1, 2});
        p.b = Tensor::from_data({t}, {1, 1});
        const NnVerifyOptions o = vnncomp_options("x");
        // a memory budget of a few sets: fewer slots and input generators, the same verdict
        NnVerifyOptions tight = o;
        tight.memory_budget = 20000;
        const NnVerifyResult small = conv.verify(p, 20, tight);
        // (a sat verdict may come from the attack, before the bounding is prepared)
        check(small.status == dense.verify(p, 20, o).status &&
                  (small.status == "sat" || small.stats.count("slot_budget") == 1),
              b + ": the slot budget keeps the verdict");
        check(conv.verify(p, 20, o).status == dense.verify(p, 20, o).status,
              b + ": verify gives the dense network's verdict");
        // a larger batch budget: more sets per batch, the same cuts of one set
        if (small.status == "sat") continue;
        NnVerifyOptions batched = tight;
        batched.batch_memory_budget = int64_t{1} << 30;
        const NnVerifyResult wide = conv.verify(p, 20, batched);
        const auto cut = [](const NnVerifyResult &r, const std::string &key) {
            return r.stats.count(key) ? r.stats.at(key) : -1.0;
        };
        check(wide.status == small.status &&
                  cut(wide, "slot_budget") == cut(small, "slot_budget") &&
                  cut(wide, "init_gen_budget") == cut(small, "init_gen_budget"),
              b + ": the batch budget keeps the verdict and the cuts of one set");
        check(small.stats.at("batch_size") == 1 &&
                  wide.stats.at("batch_size") == double(o.mini_batch_size),
              b + ": the batch budget sets the boxes of a batch");
        // the verified sets are the sum of their three kinds
        check(wide.stats.at("verified_sets") == wide.stats.at("check_verified") +
                                                    wide.stats.at("empty_children") +
                                                    wide.stats.at("point_children") &&
                  wide.stats.at("queue_peak") >= 1,
              b + ": the verified sets by kind and the queue's peak");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        layers(b);
        transposed_and_reshape(b);
        native(b);
        networks(b);
    });
    // the same with the checks off
    {
        ChecksScope off(false);
        layers("checks off");
    }
    return test::finish("neuralNetwork convolutions");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
