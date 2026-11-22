// test_neural_network_residual - graphs of layers (residuals, earlier sources), points to verify
//
// A layer may read an earlier layer's output or the input, and add residuals (the identity or a
// map). The points are the definition; without activations the graph is affine and its batched
// zonotope, sensitivity and CROWN bound are exact; with ReLUs the zonotope contains the outputs
// of sampled points, the exact F-radius gradient (the reverse pass) is the finite differences',
// and verify's verdicts agree with a grid. The constructor refuses a source that is not earlier,
// sizes that do not fit and an output that no layer reads.
//
// CORA.cpp only: graph networks (CORA's verify takes a chain of layers).
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_zono_batch.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

using namespace cora;
using test::check;

namespace {

using Layer = NeuralNetwork::Layer;

/// Random values of a shape.
Tensor aux_random(Rng &rng, const std::vector<int64_t> &shape, double scale) {
    int64_t n = 1;
    for (const int64_t s : shape) n *= s;
    std::vector<double> v(static_cast<std::size_t>(n));
    rng.normal(v.data(), v.size(), scale);
    return Tensor::from_data(v, shape);
}

/// The largest absolute difference of two tensors.
double aux_gap(const Tensor &a, const Tensor &b) { return (a - b).abs().max().item(); }

/// The weights of the graph: input (3) -> h0 (4) -> h1 = act(W1 h0 + h0) -> y = W2 h1 + S x (2).
struct Weights {
    Tensor W0, b0, W1, b1, W2, b2, S, s;
};

Weights aux_weights(Rng &rng) {
    return {aux_random(rng, {4, 3}, 0.8), aux_random(rng, {4}, 0.3),
            aux_random(rng, {4, 4}, 0.6), aux_random(rng, {4}, 0.3),
            aux_random(rng, {2, 4}, 0.7), aux_random(rng, {2}, 0.2),
            aux_random(rng, {2, 3}, 0.5), aux_random(rng, {2}, 0.2)};
}

/// The graph: layer 1 adds h0 (the identity), layer 2 reads h1 and adds S x + s (the input).
NeuralNetwork aux_graph(const Weights &w, Activation a) {
    Layer l0{w.W0, w.b0}, l1{w.W1, w.b1}, l2{w.W2, w.b2};
    l1.residuals = {{0, nullptr}};
    l2.residuals = {{-1, std::make_shared<Layer>(Layer{w.S, w.s})}};
    return NeuralNetwork({l0, l1, l2}, {a, a, Activation::identity});
}

/// The graph's definition at the rows x (N, 3).
Tensor aux_direct(const Weights &w, Activation a, const Tensor &x) {
    const auto act = [&](const Tensor &z) { return a == Activation::relu ? z.maximum(0.0) : z; };
    const Tensor h0 = act(x.matmul(w.W0.matrix_transpose()) + w.b0);
    const Tensor h1 = act(h0.matmul(w.W1.matrix_transpose()) + w.b1 + h0);
    return h1.matmul(w.W2.matrix_transpose()) + w.b2 + x.matmul(w.S.matrix_transpose()) + w.s;
}

/// A graph whose second layer reads the input instead of the layer before, and whose last adds
/// the first layer's output through a convolution-free gather.
NeuralNetwork aux_branches(const Weights &w) {
    Layer l0{w.W0, w.b0}, l1{w.W0 * 0.5, w.b0 * -1.0}, l2{w.W2, w.b2};
    l1.source = -1;
    l2.residuals = {{0, std::make_shared<Layer>(Layer{w.W2 * -0.3, w.b2})}};
    return NeuralNetwork({l0, l1, l2}, {Activation::relu, Activation::relu, Activation::identity});
}

/// Points, the affine graph's exact sets, the ReLU graph's enclosure and gradients.
void graphs(const std::string &b) {
    Rng rng(21);
    const Weights w = aux_weights(rng);
    const Tensor x = aux_random(rng, {6, 3}, 1.0);
    for (const Activation a : {Activation::identity, Activation::relu}) {
        const NeuralNetwork net = aux_graph(w, a);
        check(!net.is_chain() && net.source(2) == 1, b + ": a graph, layer 2 reads layer 1");
        check(aux_gap(net.evaluate(x), aux_direct(w, a, x)) < 1e-12, b + ": evaluate at points");
        const ZonoNet zn(net, 3, std::numeric_limits<double>::infinity(), "bounds", "length");
        check(aux_gap(zn.evaluate(x), aux_direct(w, a, x)) < 1e-12, b + ": the batch at points");
    }
    // without activations: the zonotope is the affine image, CROWN's bound the exact minimum
    const NeuralNetwork affine = aux_graph(w, Activation::identity);
    const ZonoNet za(affine, 3, 4, "bounds", "length");
    const Tensor r = x.abs() * 0.1 + 0.05;
    const ZonoInput in = za.input_zonotope(x, r);
    const ZonoPropagation out = za.forward(in.c, in.dr, in.G, Tensor(), nullptr, true);
    // the Jacobian J (2, 3): the images of the unit vectors less the image of zero
    const Tensor eye = x.eye_like(3), zero = x.zeros_like({1, 3});
    const Tensor J =
        (aux_direct(w, Activation::identity, eye) - aux_direct(w, Activation::identity, zero))
            .matrix_transpose();
    const Tensor G = r.expand_dims(-1).multiply(J.matrix_transpose().expand_dims(0));
    check(aux_gap(out.c, aux_direct(w, Activation::identity, x)) < 1e-12 &&
              aux_gap(out.G, G) < 1e-12,
          b + ": the affine graph's zonotope is exact");
    const Tensor rows = Tensor({{1.0, -1.0}});
    const Tensor exact =
        out.c.matmul(rows.matrix_transpose()) - r.matmul(rows.matmul(J).abs().matrix_transpose());
    check(aux_gap(priv_backward_lower(out, za, Tensor(), x, r, rows), exact) < 1e-12,
          b + ": CROWN's bound of the affine graph is the minimum");
    check(aux_gap(za.sensitivity(x).jacobian, J.expand_dims(0).broadcast_to({6, 2, 3})) < 1e-12,
          b + ": the sensitivity is the Jacobian");
    // with ReLUs, interval centers and slots: the zonotope contains sampled outputs
    for (const auto &[net, name] : {std::pair{aux_graph(w, Activation::relu), "residual"},
                                    std::pair{aux_branches(w), "branches"}}) {
        for (const double gens : {3.0, 2.0}) {
            const ZonoNet zr(net, gens, 2, "bounds", "length");
            const ZonoInput ir = zr.input_zonotope(x, r);
            const ZonoPropagation o = zr.forward(ir.c, ir.dr, ir.G, Tensor(), nullptr, true);
            Tensor rad = o.G.abs().sum(1);
            if (o.dr.defined()) rad = rad + o.dr;
            bool inside = true;
            for (int i = 0; i < 50; ++i) {
                const Tensor p = x + r.multiply(aux_random(rng, {6, 3}, 1.0).clip(-1.0, 1.0));
                inside = inside && ((net.evaluate(p) - o.c).abs() - rad).max().item() <= 1e-12;
            }
            check(inside, b + ": the " + name + " zonotope contains the outputs");
            // the F-radius gradient of the reverse pass, by central differences
            ZonoPropagation rec = o;
            const Tensor g = priv_frad_gradients(rec, ir, zr, Tensor(), true);
            const auto frad = [&](const Tensor &rr) {
                const ZonoInput i2 = zr.input_zonotope(x, rr, ir.dims);
                const ZonoPropagation p2 = zr.forward(i2.c, i2.dr, i2.G, Tensor(), nullptr, false);
                return p2.G.square().sum(-1).sum(-1).sqrt();
            };
            double worst = 0;
            const int64_t q0 = ir.dims.shape()[1];
            for (int64_t j = 0; j < q0; ++j) {
                const Tensor e = ir.dims.slice(1, j, j + 1).equal(r.arange_like(3).reshape({1, 3}));
                const double h = 1e-6;
                const Tensor fd = (frad(r + e * h) - frad(r - e * h)) / (2 * h);
                worst = std::max(worst, aux_gap(fd, g.slice(1, j, j + 1).squeeze(1)));
            }
            check(worst < 1e-5, b + ": the " + name + " F-radius gradient is the differences'");
        }
    }
}

/// verify on a graph agrees with a grid of its outputs; the constructor's refusals.
void verdicts(const std::string &b) {
    Rng rng(5);
    Weights w = aux_weights(rng);
    w.W0 = aux_random(rng, {4, 2}, 0.8);
    w.S = aux_random(rng, {2, 2}, 0.5);
    const NeuralNetwork net = aux_graph(w, Activation::relu);
    // the minimum of y1 - y2 over [-1, 1]^2 on a grid
    std::vector<double> grid;
    for (int i = 0; i <= 100; ++i)
        for (int j = 0; j <= 100; ++j) grid.insert(grid.end(), {i / 50.0 - 1, j / 50.0 - 1});
    const Tensor y = net.evaluate(Tensor::from_data(grid, {101 * 101, 2}));
    const double low = (y.slice(1, 0, 1) - y.slice(1, 1, 2)).min().item();
    for (const double t : {low - 0.3, low + 0.05}) {
        NnProperty p;
        p.lo = Tensor::from_data({-1.0, -1.0}, {1, 2});
        p.hi = Tensor::from_data({1.0, 1.0}, {1, 2});
        p.poly_case = {0};
        p.A = Tensor::from_data({1.0, -1.0}, {1, 1, 2});
        p.b = Tensor::from_data({t}, {1, 1});
        const std::string expected = t < low ? "unsat" : "sat";
        check(net.verify(p, 20, vnncomp_options("x")).status == expected,
              b + ": verify on the graph is " + expected);
    }
    // a source that is not earlier, a residual of the wrong size, an output no layer reads
    Layer l0{w.W0, w.b0}, l1{w.W1, w.b1}, l2{w.W2, w.b2};
    Layer bad = l1;
    bad.source = 1;
    check(test::throws([&] { NeuralNetwork({l0, bad, l2}); }), b + ": a layer reads itself");
    bad = l2;
    bad.residuals = {{0, nullptr}};
    check(test::throws([&] { NeuralNetwork({l0, l1, bad}); }),
          b + ": an identity residual of 4 values into 2 outputs");
    Layer skip{w.W2, w.b2};
    skip.source = -1;
    skip.W = aux_random(rng, {2, 2}, 1.0);
    check(test::throws([&] { NeuralNetwork({l0, l1, skip}); }),
          b + ": layer 1's output is not read");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        graphs(b);
        verdicts(b);
    });
    return test::finish("neuralNetwork graphs");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
