// test_lean_nn_batch - evaluate_zonotope_batch against the CORALean oracle
//
// ORACLE: nn/@neuralNetwork/evaluateZonotopeBatch.m
// ORACLE: nn/layers/linear/nnConv2DLayer.m
// ORACLE: nn/layers/nonlinear/nnSquareLayer.m
// ORACLE: nn/layers/linear/nnConvTranspose2DLayer.m
// ORACLE: nn/layers/other/nnReshapeLayer.m
// ORACLE: nn/layers/nnLayer.m
//
// The oracle's neuralNetwork.evaluate at poly_method "singh" on a zonotope, with the layer kinds
// linear, conv, relu, sign and square: the batch of one zonotope through the network (an error
// generator per neuron of every activation layer, in the order of the neurons) has the oracle's
// center and generators, without the zero generators that the oracle's linear layers add. A
// ConvTranspose of stride 1 is the conv kind with the flipped kernel and the pads k - 1 - p, a
// reshape the linear kind of its permutation matrix.
//
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

using namespace cora;

namespace oracle = test::oracle;
using Layer = NeuralNetwork::Layer;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The oracle's set without the zero generators that its linear and conv layers add: the m
/// input generators stay, then each activation layer's (`sizes`: the outputs of the oracle's
/// layers, of an activation the size before).
lean::Zonotope aux_without_bias(const lean::Zonotope &Y, int64_t m,
                                const std::vector<lean::NnLayer> &layers,
                                const std::vector<int64_t> &sizes) {
    const Tensor G = Y.generators().gather();
    // the input's columns, then the activations' (every layer adds one per output)
    std::vector<int64_t> keep;
    keep.reserve(static_cast<std::size_t>(m));
    for (int64_t j = 0; j < m; ++j) keep.push_back(j);
    int64_t h = m;
    for (std::size_t k = 0; k < layers.size(); ++k) {
        const bool affine = layers[k].kind == "linear" || layers[k].kind == "conv";
        if (!affine)
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

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available() || !oracle::serves("neuralNetwork.evaluate")) return 0;
    oracle::run("neuralNetwork.evaluate", [&](oracle::Random &rnd) {
        // a dense network: relu, then the identity
        const Zonotope Z(rnd.point(3) * 2.0, rnd.matrix(3, 4).matrix_transpose());
        const Tensor W1 = rnd.matrix(5, 3), b1 = rnd.point(5), W2 = rnd.matrix(2, 5),
                     b2 = rnd.point(2);
        const NeuralNetwork dense({{W1, b1}, {W2, b2}}, {Activation::relu, Activation::identity});
        const std::vector<lean::NnLayer> dl{lean::NnLayer::linear(W1, b1), lean::NnLayer::relu(),
                                            lean::NnLayer::linear(W2, b2)};
        oracle::check("dense relu", aux_batch(dense, Z),
                      aux_without_bias(lean::evaluate_network(dl, lean::Zonotope(Z), "singh"), 4,
                                       dl, {5, 5, 2}));
        // the squares and signs of a layer each
        const Tensor W3 = rnd.matrix(3, 3), b3 = rnd.point(3);
        for (const Activation a : {Activation::square, Activation::sign}) {
            const NeuralNetwork nn({{W3, b3}}, {a});
            const std::vector<lean::NnLayer> ls{lean::NnLayer::linear(W3, b3),
                                                a == Activation::square ? lean::NnLayer::square()
                                                                        : lean::NnLayer::sign()};
            oracle::check(a == Activation::square ? "square" : "sign", aux_batch(nn, Z),
                          aux_without_bias(lean::evaluate_network(ls, lean::Zonotope(Z), "singh"),
                                           4, ls, {3, 3}));
        }
        // a convolution of a 1 x 4 x 4 input by two 2 x 2 kernels, relu, then a linear layer
        const Zonotope X(rnd.point(16) * 2.0, rnd.matrix(16, 3).matrix_transpose());
        const Tensor K = rnd.matrix(2, 4), bc = rnd.point(18), W4 = rnd.matrix(3, 18),
                     b4 = rnd.point(3);
        const NeuralNetwork cnn(
            {Layer::conv2d(K.reshape({2, 1, 2, 2}), bc, {1, 4, 4}, {}), {W4, b4}},
            {Activation::relu, Activation::identity});
        const std::vector<lean::NnLayer> cl{
            lean::NnLayer::conv(K.reshape({2, 1, 2, 2}), bc, {1, 4, 4}, {}), lean::NnLayer::relu(),
            lean::NnLayer::linear(W4, b4)};
        oracle::check("conv relu", aux_batch(cnn, X),
                      aux_without_bias(lean::evaluate_network(cl, lean::Zonotope(X), "singh"), 3,
                                       cl, {18, 18, 3}));
        // a ConvTranspose (stride 1) is a padded convolution; small integers: the sums are exact
        const auto whole = [](const Tensor &t) { return (t * 4.0).round(); };
        const Zonotope T(whole(rnd.point(18)), whole(rnd.matrix(18, 3).matrix_transpose()));
        const Tensor Kt = whole(rnd.matrix(2, 3 * 4)), bt = whole(rnd.point(48)),
                     W5 = rnd.matrix(2, 48), b5 = rnd.point(2);
        // the kernel flipped in both axes, its channel axes swapped
        std::vector<double> flipped(3 * 2 * 4);
        for (int64_t o = 0; o < 3; ++o)
            for (int64_t i = 0; i < 2; ++i)
                for (int64_t a = 0; a < 2; ++a)
                    for (int64_t c = 0; c < 2; ++c)
                        flipped[std::size_t(((o * 2 + i) * 2 + a) * 2 + c)] =
                            Kt.data()[std::size_t(((i * 3 + o) * 2 + (1 - a)) * 2 + (1 - c))];
        const NeuralNetwork ct(
            {Layer::conv_transpose2d(Kt.reshape({2, 3, 2, 2}), bt, {3, 4, 4}, {}), {W5, b5}},
            {Activation::relu, Activation::identity});
        NeuralNetwork::Conv2d pad1;
        pad1.pads = {1, 1, 1, 1};
        const std::vector<lean::NnLayer> tl{
            lean::NnLayer::conv(Tensor::from_data(flipped, {3, 2, 2, 2}), bt, {2, 3, 3}, pad1),
            lean::NnLayer::relu(), lean::NnLayer::linear(W5, b5)};
        oracle::check("convTranspose relu", aux_batch(ct, T),
                      aux_without_bias(lean::evaluate_network(tl, lean::Zonotope(T), "singh"), 3,
                                       tl, {48, 48, 2}));
        // a reshape (a gather) is the linear layer of its permutation matrix
        const Zonotope S(rnd.point(6) * 2.0, rnd.matrix(6, 3).matrix_transpose());
        const Layer shuffle = Layer::gather({0, 3, 1, 4, 2, 5}, 6, Tensor::zeros({6}));
        const NeuralNetwork rs({shuffle}, {Activation::relu});
        const std::vector<lean::NnLayer> rl{
            lean::NnLayer::linear(shuffle.dense(), Tensor::zeros({6})), lean::NnLayer::relu()};
        oracle::check("reshape relu", aux_batch(rs, S),
                      aux_without_bias(lean::evaluate_network(rl, lean::Zonotope(S), "singh"), 3,
                                       rl, {6, 6}));
    });
    return test::finish("lean nn batch");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
