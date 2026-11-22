// test_lean_nn_verify - the sets of CORA's verify (the bounding "conzonotope") against CORALean
//
// ORACLE: nn/@neuralNetwork/evaluateZonotopeBatch_.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/evaluateZonotopeBatch.m
//
// The batched zonotope of verify with a slot for every error (a point center) is, member by member,
// the oracle's neuralNetwork.evaluate at poly_method "singh" (ReLU, sigmoid, tanh), the bias boxes
// of its linear layers dropped. Then a whole verify
// of an ACAS Xu-sized network (5-24-24-5) on lean: unsat, and the sets of its first batch (the
// output zonotope and the factor system of Prop. 2) through the oracle. Not served (docs/oracle-
// requests.md): the slots with an interval center, the clamps, the "bounds" line of sigmoid and
// tanh, the zonotack candidates, the constraints of Prop. 2 and verify itself.
//
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/verify.h"
#include "nn/neural_network/private/priv_zono_batch.h"
#include "oracle_testing.h"

#include <limits>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The network of the layer sizes, the activation after every hidden layer.
NeuralNetwork aux_net(oracle::Random &rnd, const std::vector<int64_t> &sizes, Activation hidden,
                      double scale) {
    std::vector<NeuralNetwork::Layer> layers;
    std::vector<Activation> acts;
    for (std::size_t k = 0; k + 1 < sizes.size(); ++k) {
        layers.push_back(
            {rnd.matrix(sizes[k + 1], sizes[k]) * scale, rnd.point(sizes[k + 1]) * 0.1});
        acts.push_back(k + 2 < sizes.size() ? hidden : Activation::identity);
    }
    return {layers, acts};
}

/// The oracle's output of member i of the batch (c (B, n), G (B, q, n)), without the zero
/// generators of its linear layers (the bias boxes).
lean::Zonotope aux_oracle_member(const NeuralNetwork &nn, const Tensor &c, const Tensor &G,
                                 int64_t i) {
    std::vector<lean::NnLayer> layers;
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        layers.push_back(lean::NnLayer::linear(nn.layers()[k].W, nn.layers()[k].b));
        if (nn.activations()[k] == Activation::relu) layers.push_back(lean::NnLayer::relu());
        if (nn.activations()[k] == Activation::sigmoid) layers.push_back(lean::NnLayer::sigmoid());
        if (nn.activations()[k] == Activation::tanh) layers.push_back(lean::NnLayer::tanh());
    }
    const Zonotope Z(c.slice(0, i, i + 1).squeeze(0), G.slice(0, i, i + 1).squeeze(0));
    const lean::Zonotope Y = lean::evaluate_network(layers, lean::Zonotope(Z), "singh");
    // the generators of the input, then those of each activation; a linear layer's are zero
    const Tensor rows = Y.generators().gather();
    std::vector<int64_t> keep;
    int64_t h = Z.nr_generators(), n = 0;
    keep.reserve(std::size_t(h));
    for (int64_t j = 0; j < h; ++j) keep.push_back(j);
    for (const lean::NnLayer &layer : layers) {
        if (layer.kind == "linear") {
            n = layer.W.shape()[0];
        } else {
            for (int64_t j = 0; j < n; ++j) keep.push_back(h + j);
        }
        h += n;
    }
    return lean::Zonotope(Y.center().gather(), rows.take(keep, -1));
}

/// Every member of the batched output against the oracle's.
void aux_check_batch(const std::string &what, const NeuralNetwork &nn, const Tensor &x,
                     const Tensor &r) {
    const ZonoNet net(nn, kInf, kInf, "singh", "length");
    const ZonoInput in = net.input_zonotope(x, r);
    const ZonoPropagation out = net.forward(in.c, in.dr, in.G, Tensor(), nullptr, false);
    for (int64_t i = 0; i < x.shape()[0]; ++i)
        oracle::check(
            what + " member " + std::to_string(i),
            Zonotope(out.c.slice(0, i, i + 1).squeeze(0), out.G.slice(0, i, i + 1).squeeze(0)),
            aux_oracle_member(nn, in.c, in.G, i));
}

/// CORALean's order of Fourier-Motzkin against the oracle, member by member (finite rows only).
void aux_check_fm(const std::string &what, const Tensor &A, const Tensor &b, int64_t passes) {
    const int64_t M = A.shape()[0], q = A.shape()[2];
    const Tensor one = A.ones_like({M, q});
    const auto [bl, bu] = priv_fourier_motzkin(A, b, -one, one, passes, FmOrder::coralean, 0.0);
    for (int64_t m = 0; m < M; ++m) {
        const Tensor Am = A.slice(0, m, m + 1).squeeze(0), bm = b.slice(0, m, m + 1).squeeze(0);
        const std::vector<int64_t> finite = bm.isfinite().nonzero();
        const lean::Interval I = lean::fourier_motzkin_bounds(
            Am.take(finite, 0), bm.take(finite, 0).expand_dims(-1), passes);
        oracle::check(what + " inf " + std::to_string(m), bl.slice(0, m, m + 1).reshape({q, 1}),
                      I.inf());
        oracle::check(what + " sup " + std::to_string(m), bu.slice(0, m, m + 1).reshape({q, 1}),
                      I.sup());
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available() || !oracle::serves("neuralNetwork.evaluate")) return 0;

    // the batched zonotope with every slot, member by member
    oracle::run("neuralNetwork.evaluate", [&](oracle::Random &rnd) {
        for (const Activation a : {Activation::relu, Activation::sigmoid, Activation::tanh}) {
            const NeuralNetwork nn = aux_net(rnd, {3, 6, 5, 2}, a, 0.5);
            const Tensor x = Tensor::stack({rnd.point(3), rnd.point(3) * 0.5}, 0);
            const Tensor r =
                Tensor::stack({rnd.point(3).abs() * 0.25, rnd.point(3).abs() * 0.125}, 0);
            aux_check_batch("batch", nn, x, r);
        }
    });

    // a whole verify on lean, and the sets of its first batch
    oracle::run("conZonotope.fourierMotzkinBounds", [&](oracle::Random &rnd) {
        if (oracle::pass() != "lean") return;
        const NeuralNetwork nn = aux_net(rnd, {5, 24, 24, 5}, Activation::relu, 0.4);
        const Tensor lo = nn.layers()[0].W.full_like({1, 5}, -0.1), hi = lo.negative();
        // unsafe below the least sampled y_0 (a corner box of points), less a margin
        Tensor pts = (rnd.matrix(400, 5) * 0.1).clip(-0.1, 0.1);
        const double least = nn.evaluate(pts).slice(1, 0, 1).min().item();
        NnProperty p;
        p.lo = lo;
        p.hi = hi;
        std::vector<double> row(5, 0.0);
        row[0] = 1.0;
        p.A = Tensor::like(lo, row, {1, 1, 5});
        p.b = Tensor::like(lo, {least - 0.05}, {1, 1});
        p.poly_case = {0};
        NnVerifyOptions o = vnncomp_options("test");
        o.falsification_order = "bound-first";
        o.falsification_method = "center";
        o.mini_batch_size = 4;
        o.dtype = DType::float64;
        o.fourier_motzkin_order = "coralean";
        const NnVerifyResult result = nn.verify(p, 300, o);
        test::check(result.status == "unsat" && result.stats.at("boxes") > 1,
                    "verify on lean: unsat after splits");
        // the first batch: its output zonotope and the factor system of the unsafe outputs
        aux_check_batch("verify's first set", nn, (lo + hi) * 0.5, (hi - lo) * 0.5);
        const ZonoNet net(nn, kInf, kInf, o.poly_method, o.approx_error_order);
        const ZonoInput in = net.input_zonotope((lo + hi) * 0.5, (hi - lo) * 0.5);
        const ZonoPropagation out = net.forward(in.c, in.dr, in.G, Tensor(), nullptr, false);
        const Tensor Ap = p.A.slice(0, 0, 1);
        const Tensor C = Ap.matmul(out.G.matrix_transpose());
        const Tensor d = p.b - Ap.matmul(out.c.expand_dims(-1)).squeeze(-1) + o.verified_margin;
        aux_check_fm("verify's first system", C, d, o.polytope_bound_approx_max_iter);
    });
    return test::finish("verify's sets against the oracle");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
