// test_lean_nn_relaxation - the lines and bounds of NeuralNetwork::verify against CORALean
//
// Bit for bit against the oracle, every rounding cost of CORALean's definitions zero at nearest:
// the lines of sign, the square, exp and 1 / max(x, 1) over boxes (nnLayer.relaxation), the ReLU
// line at optimized slopes alpha, the product line on a zonotope (nnLayer.productLine), the box of
// a softmax's sums (nnLayer.softmaxSumBounds), the dual certificate (conZonotope.dualBound) and a
// convolution's map (the layer conv of nnLayer.evaluateZonotope).
//
// CORA.cpp only: tests verify's lines (CORA has no zonotope line of these activations)
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"
#include "nn/neural_network/private/priv_zono_batch.h"
#include "oracle_testing.h"

#include <cmath>
#include <limits>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Boxes [l, u] as columns (n, 1): random ones of every scale, plus points and boxes at zero.
std::pair<Tensor, Tensor> aux_boxes(oracle::Random &rnd, int64_t n) {
    std::vector<double> l, u;
    for (int64_t i = 0; i < n; ++i) {
        const double scale = std::pow(10.0, double(i % 7) - 3.0);
        const double a = rnd.scalar() * scale, b = rnd.scalar() * scale;
        l.push_back(std::min(a, b));
        u.push_back(std::max(a, b));
    }
    // a point, a box ending at zero, one starting at zero
    l.insert(l.end(), {0.75, -2.0, 0.0});
    u.insert(u.end(), {0.75, 0.0, 3.0});
    const auto m = int64_t(l.size());
    return {Tensor::from_data(l, {m, 1}), Tensor::from_data(u, {m, 1})};
}

/// Boxes for exp: u at most 40 (no overflow), widths from 0 to 30, and two subnormal widths.
std::pair<Tensor, Tensor> aux_exp_boxes(oracle::Random &rnd, int64_t n) {
    std::vector<double> l, u;
    for (int64_t i = 0; i < n; ++i) {
        const double a = rnd.uniform(-30.0, 30.0), w = std::pow(10.0, rnd.uniform(-9.0, 1.4));
        l.push_back(a);
        u.push_back(std::min(a + w, 40.0));
    }
    l.insert(l.end(), {0.5, 1e-310, -700.0});
    u.insert(u.end(), {0.5, 2e-310, -690.0});
    const auto m = int64_t(l.size());
    return {Tensor::from_data(l, {m, 1}), Tensor::from_data(u, {m, 1})};
}

/// The line against the oracle's slope, offset and error.
void aux_check(const std::string &kind, const PrivLine &line, const lean::NnLine &lean) {
    // + 0.0: the wire prints -0 as 0 (a ReLU off below has the zero line -0)
    oracle::check(kind + " slope", line.slope + 0.0, lean.slope);
    oracle::check(kind + " offset", line.offset + 0.0, lean.offset);
    oracle::check(kind + " radius", line.radius + 0.0, lean.error);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available() || !oracle::serves("nnLayer.relaxation")) return 0;
    oracle::run("nnLayer.relaxation", [&](oracle::Random &rnd) {
        for (const int64_t n : {1, 6, 40}) {
            const auto [l, u] = aux_boxes(rnd, n);
            aux_check("sign", priv_sign_line(l, u), lean::relaxation("sign", l, u));
            aux_check("square", priv_square_line(l, u), lean::relaxation("square", l, u));
            aux_check("reciprocal", priv_reciprocal_line(l, u),
                      lean::relaxation("reciprocal", l, u));
            const auto [el, eu] = aux_exp_boxes(rnd, n);
            aux_check("exp", priv_exp_line(el, eu), lean::relaxation("exp", el, eu));
            // the ReLU at slopes alpha in [0, 1] (CORALean's line where a neuron does not cross)
            const Tensor a = (rnd.vector(l.shape()[0]) * 0.5 + 0.5).clip(0.0, 1.0);
            const PrivLine base = priv_verify_line(Activation::relu, l, u, "singh");
            aux_check("alpha", priv_alpha_line(base, l, u, a), lean::relaxation("relu", l, u, a));
        }
    });

    // the sign and square layers on a small zonotope (NeuralNetwork::evaluate's lines)
    oracle::run("nnLayer.evaluateZonotope", [&](oracle::Random &rnd) {
        const Zonotope Z(rnd.point(3), rnd.matrix(3, 4).matrix_transpose());
        for (const auto &[layer, activation] : std::vector<std::pair<lean::NnLayer, Activation>>{
                 {lean::NnLayer::sign(), Activation::sign},
                 {lean::NnLayer::square(), Activation::square}}) {
            const NeuralNetwork nn({{Tensor::eye(3), Tensor::zeros({3})}}, {activation});
            const lean::Zonotope Y = lean::evaluate_layer(layer, lean::Zonotope(Z), "singh");
            const Zonotope mine = nn.evaluate(Z);
            oracle::check(layer.kind + " center", mine.center().reshape({3, 1}), Y.center());
        }
    });

    // the product line on zonotopes with and without an interval center
    oracle::run("nnLayer.productLine", [&](oracle::Random &rnd) {
        for (const auto &[n, q] :
             std::vector<std::pair<int64_t, int64_t>>{{1, 1}, {3, 4}, {6, 9}}) {
            const Tensor c = rnd.vector(2 * n), G = rnd.matrix(2 * n, q);
            const Tensor dr = rnd.vector(2 * n).abs() * 0.1;
            // CORA.cpp's batch: c (1, 2 n), G (1, q, 2 n), dr (1, 2 n)
            const Tensor cb = c.reshape({1, 2 * n}), Gb = G.matrix_transpose().expand_dims(0);
            for (const bool center : {false, true}) {
                const PrivLine line =
                    priv_product_line(cb, Gb, center ? dr.reshape({1, 2 * n}) : Tensor());
                const lean::NnLine lean = lean::product_line(c, G, center ? dr : Tensor());
                oracle::check("product slope", line.slope.reshape({2 * n, 1}), lean.slope);
                oracle::check("product offset", line.offset.slice(-1, 0, n).reshape({n, 1}),
                              lean.offset);
                oracle::check("product radius", line.radius.slice(-1, 0, n).reshape({n, 1}),
                              lean.error);
            }
        }
    });

    // the box of a softmax's sums: the exps' box through the sum layer's map
    oracle::run("nnLayer.softmaxSumBounds", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : std::vector<std::pair<int64_t, int64_t>>{{2, 1}, {5, 3}}) {
            const Tensor W = rnd.matrix(m, n), b = rnd.vector(m) * 0.1;
            const NeuralNetwork nn({{rnd.matrix(n, n), Tensor::zeros({n})}, {W, b.squeeze(-1)}},
                                   {Activation::exp, Activation::reciprocal});
            const ZonoNet net(nn, kInf, kInf, "singh", "length");
            const Tensor l = rnd.vector(n) - 1.0, u = l + rnd.vector(n).abs();
            const auto [lo, hi] =
                priv_box_image(net, 1, l.exp().reshape({1, n}), u.exp().reshape({1, n}));
            const auto [inf, sup] = lean::softmax_sum_bounds(W, b, l, u);
            oracle::check("softmax inf", lo.reshape({m, 1}), inf);
            oracle::check("softmax sup", hi.reshape({m, 1}), sup);
        }
    });

    // the dual certificate at multipliers of both signs (the negative ones clamp to zero)
    oracle::run("conZonotope.dualBound", [&](oracle::Random &rnd) {
        for (const auto &[p, q] :
             std::vector<std::pair<int64_t, int64_t>>{{1, 2}, {4, 3}, {7, 6}}) {
            const Tensor C = rnd.matrix(p, q), d = rnd.vector(p), w = rnd.vector(q);
            const Tensor lambda = rnd.vector(p);
            const Tensor h = priv_dual_certificate(w.reshape({1, 1, q}), C.expand_dims(0),
                                                   d.reshape({1, p}), lambda.reshape({1, 1, p}));
            oracle::check("dual certificate", h.item(), lean::dual_bound(C, d, w, lambda));
        }
    });

    // a convolution's map of the center and of the generator rows (the generator axis a batch)
    oracle::run("nnLayer.evaluateZonotope", [&](oracle::Random &rnd) {
        NeuralNetwork::Conv2d strided, grouped;
        strided.stride = {2, 1};
        strided.pads = {1, 0, 0, 1};
        grouped.dilation = {2, 1};
        grouped.groups = 2;
        for (const auto &[spec, out] : std::vector<std::pair<NeuralNetwork::Conv2d, int64_t>>{
                 {NeuralNetwork::Conv2d{}, 3}, {strided, 2}, {grouped, 4}}) {
            const std::vector<int64_t> input{2, 4, 3};
            const int64_t n = 24, h = 5;
            const Tensor kernel =
                rnd.matrix(out, (2 / spec.groups) * 4).reshape({out, 2 / spec.groups, 2, 2});
            // the output (out, Ho, Wo) of the 2 x 2 kernel
            const int64_t Ho =
                (4 + spec.pads[0] + spec.pads[2] - spec.dilation[0] - 1) / spec.stride[0] + 1;
            const int64_t Wo =
                (3 + spec.pads[1] + spec.pads[3] - spec.dilation[1] - 1) / spec.stride[1] + 1;
            const int64_t m = out * Ho * Wo;
            const Tensor bias = rnd.vector(m).squeeze(-1) * 0.1;
            const NeuralNetwork::Layer L = NeuralNetwork::Layer::conv2d(kernel, bias, input, spec);
            const Tensor c = rnd.vector(n), G = rnd.matrix(n, h);
            const Tensor mine_c = L.rows(c.reshape({1, n})).reshape({m, 1}) + bias.reshape({m, 1});
            const Tensor mine_G = L.rows(G.matrix_transpose()).matrix_transpose();
            const lean::Zonotope Y = lean::evaluate_layer(
                lean::NnLayer::conv(kernel, bias, input, spec), lean::Zonotope(c, G), "singh");
            oracle::check("conv center", mine_c, Y.center());
            oracle::check("conv generators", mine_G, Y.generators().gather().slice(1, 0, h));
        }
    });
    return test::finish("lean nn relaxation");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
