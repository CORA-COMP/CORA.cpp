// test_lean_nn_layers - the linear layers of the NN port against the CORALean oracle
//
// ORACLE: nn/@neuralNetwork/neuralNetwork.m
// ORACLE: nn/layers/linear/nnLinearLayer.m
// ORACLE: nn/layers/linear/nnIdentityLayer.m
// ORACLE: nn/layers/linear/nnElementwiseAffineLayer.m
// ORACLE: nn/layers/linear/nnBatchNormLayer.m
// ORACLE: nn/layers/linear/nnAvgPool2DLayer.m
// ORACLE: nn/layers/linear/nnLipConstrLinearLayer.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/nnActivationLayer.m
//
// The oracle's neuralNetwork.evaluate at poly_method "singh" on a zonotope: the batch of one
// zonotope through a network of each linear layer of CORA.cpp's port (Layer::linear, identity,
// elementwise_affine with one factor and with one per value, batch_norm per channel, avg_pool2d
// with and without a stride and pads, lip_constr_linear at lambda 1 and below the rows' norms),
// each followed by a ReLU, has the oracle's center and generators (its layer kinds linear,
// elementwiseAffine, batchNorm, avgPool and lipLinear), without the zero generators that the
// oracle's linear layers add. The networks are built by the constructor; a dense network of
// sigmoid and tanh activation layers runs the same way.
//
// Authors:       Claude Opus 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

#include <string>
#include <vector>

using namespace cora;

namespace oracle = test::oracle;
using Layer = NeuralNetwork::Layer;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The oracle's set without the zero generators that its affine layers add: the m input
/// generators stay, then each activation layer's (`sizes`: the outputs of the oracle's layers).
lean::Zonotope aux_without_bias(const lean::Zonotope &Y, int64_t m,
                                const std::vector<lean::NnLayer> &layers,
                                const std::vector<int64_t> &sizes) {
    const Tensor G = Y.generators().gather();
    std::vector<int64_t> keep;
    keep.reserve(static_cast<std::size_t>(G.shape()[1]));
    for (int64_t j = 0; j < m; ++j) keep.push_back(j);
    int64_t h = m;
    for (std::size_t k = 0; k < layers.size(); ++k) {
        const std::string &kind = layers[k].kind;
        const bool activation = kind == "relu" || kind == "sigmoid" || kind == "tanh";
        if (activation)
            for (int64_t i = 0; i < sizes[k]; ++i) keep.push_back(h + i);
        h += sizes[k];
    }
    return lean::Zonotope(Y.center().gather(), G.take(keep, -1));
}

/// The batch of one zonotope through the network, as a zonotope.
Zonotope aux_batch(const NeuralNetwork &nn, const Zonotope &Z) {
    ZonotopeBatchOptions options;
    options.poly_method = "singh";
    const ZonotopeBatch Y = nn.evaluate_zonotope_batch(Z.center().expand_dims(0),
                                                       Z.generators().expand_dims(0), options);
    return Zonotope(Y.c.squeeze(0), Y.G.squeeze(0));
}

/// The layer followed by a ReLU, in CORA.cpp and in the oracle, on Z (m generators).
void aux_check(const std::string &what, const Layer &layer, const lean::NnLayer &lean_layer,
               const Zonotope &Z, int64_t m) {
    const NeuralNetwork nn({layer}, {Activation::relu});
    const std::vector<lean::NnLayer> ls{lean_layer, lean::NnLayer::relu()};
    const int64_t n = layer.outputs();
    oracle::check(
        what, aux_batch(nn, Z),
        aux_without_bias(lean::evaluate_network(ls, lean::Zonotope(Z), "singh"), m, ls, {n, n}));
}

/// A zonotope of n dimensions and m generators, the center scaled by 2.
Zonotope aux_set(oracle::Random &rnd, int64_t n, int64_t m) {
    return Zonotope(rnd.point(n) * 2.0, rnd.matrix(n, m).matrix_transpose());
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("neuralNetwork.evaluate", [&](oracle::Random &rnd) {
        // the dense layer and the identity (the oracle's linear layer of the unit matrix)
        const Zonotope Z = aux_set(rnd, 4, 3);
        const Tensor W = rnd.matrix(5, 4), b = rnd.point(5);
        aux_check("linear", Layer::linear(W, b), lean::NnLayer::linear(W, b), Z, 3);
        aux_check("identity", Layer::identity(4),
                  lean::NnLayer::linear(Tensor::eye(4), Tensor::zeros({4})), Z, 3);
        // the elementwise affine map, one factor per value and one for all
        const Tensor s = rnd.point(4), o = rnd.point(4);
        aux_check("elementwise affine", Layer::elementwise_affine(s, o, 4),
                  lean::NnLayer::elementwise_affine(s, o), Z, 3);
        const Tensor s1 = rnd.point(1), o1 = rnd.point(1);
        aux_check("elementwise affine, one factor", Layer::elementwise_affine(s1, o1, 4),
                  lean::NnLayer::elementwise_affine(s1, o1), Z, 3);
        // batch norm of 2 channels of 2 x 2 pixels, the variances positive
        const Zonotope B = aux_set(rnd, 8, 3);
        const Tensor gamma = rnd.point(2), beta = rnd.point(2), mu = rnd.point(2),
                     var = rnd.point(2).abs() + 0.25;
        for (const double eps : {1e-5, 0.0})
            aux_check("batch norm", Layer::batch_norm(gamma, beta, mu, var, {2, 2, 2}, eps),
                      lean::NnLayer::batch_norm(gamma, beta, mu, var, {2, 2, 2}, eps), B, 3);
        // the average pools: 2 x 2 of 2 x 4 x 4, and 3 x 2 at stride 1 with pads
        const Zonotope P = aux_set(rnd, 32, 3);
        aux_check("avg pool", Layer::avg_pool2d({2, 2}, {2, 4, 4}),
                  lean::NnLayer::avg_pool({2, 2}, {2, 4, 4}, {0, 0}, {0, 0, 0, 0}), P, 3);
        aux_check("avg pool, stride and pads",
                  Layer::avg_pool2d({3, 2}, {2, 4, 4}, {1, 1}, {1, 0, 1, 1}),
                  lean::NnLayer::avg_pool({3, 2}, {2, 4, 4}, {1, 1}, {1, 0, 1, 1}), P, 3);
        // the Lipschitz-constrained layer: lambda 1 and one below the rows' norms
        for (const double lambda : {1.0, 0.7, 3.0})
            aux_check("lip constr linear", Layer::lip_constr_linear(W, b, lambda),
                      lean::NnLayer::lip_linear(W, b, lambda), Z, 3);
        // a dense network of sigmoid and tanh activation layers
        const Tensor W1 = rnd.matrix(5, 4), b1 = rnd.point(5), W2 = rnd.matrix(3, 5),
                     b2 = rnd.point(3);
        const NeuralNetwork dense({{W1, b1}, {W2, b2}}, {Activation::sigmoid, Activation::tanh});
        const std::vector<lean::NnLayer> dl{lean::NnLayer::linear(W1, b1), lean::NnLayer::sigmoid(),
                                            lean::NnLayer::linear(W2, b2), lean::NnLayer::tanh()};
        oracle::check("sigmoid tanh", aux_batch(dense, Z),
                      aux_without_bias(lean::evaluate_network(dl, lean::Zonotope(Z), "singh"), 3,
                                       dl, {5, 5, 3, 3}));
    });
    return test::finish("lean nn layers");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
