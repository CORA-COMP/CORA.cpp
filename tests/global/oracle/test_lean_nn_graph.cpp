// test_lean_nn_graph - graph networks (residuals, MaxPool, a dot product) against CORALean
//
// ORACLE: nn/@neuralNetwork/evaluateZonotopeBatch_.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/evaluateZonotopeBatch.m
// ORACLE: nn/layers/nonlinear/nnMaxPool2DLayer.m
// ORACLE: nn/layers/other/nnCompositeLayer.m
//
// ZonoNet::forward of verify on graph networks is, bit for bit, the oracle's
// neuralNetwork.evaluateGraph: a residual block (a ReLU layer that adds the input), MaxPool as
// b + relu(a - b) (relu of the differences, then the second entries as a residual of an earlier
// output), the two in a row, a dot product (a product layer, then the sum) and square and sign
// layers; the port's max_pool2d_layer (its convolutions and gathers as matrices) after a ReLU
// layer, and composite_layer of a ReLU and an identity branch, added and concatenated. Each with
// every error in a slot, every error in the interval center (dr, the oracle's
// error box) and the first layer's alone in it; with the record and without (a layer's error
// generators then wait for the next map). Not served: splits, alpha slopes, the slots chosen by
// length, the ranges of sources (reciprocal, 1 / x, x^3), sigmoid and tanh.
//
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "nn/neural_network/private/priv_zono_batch.h"
#include "oracle_testing.h"

#include <limits>
#include <memory>
#include <stdexcept>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

using Layer = NeuralNetwork::Layer;

/// The oracle's name of an activation of a graph layer.
std::string aux_activation(Activation a) {
    switch (a) {
    case Activation::identity:
        return "identity";
    case Activation::relu:
        return "relu";
    case Activation::sign:
        return "sign";
    case Activation::square:
        return "square";
    case Activation::product:
        return "product";
    default:
        throw std::invalid_argument("aux_activation: evaluateGraph has no layer of this kind");
    }
}

/// The network as the oracle's graph layers; the layers marked in none keep their errors in the
/// error box.
std::vector<lean::NnGraphLayer> aux_graph(const NeuralNetwork &nn, const std::vector<bool> &none) {
    std::vector<lean::NnGraphLayer> graph;
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        const Layer &layer = nn.layers()[k];
        lean::NnGraphLayer g{layer.dense(), layer.b, nn.source(k)};
        // each residual through its map's matrix, the identity without one
        for (const NeuralNetwork::Residual &r : layer.residuals)
            g.residuals.push_back({r.source, r.map ? r.map->dense() : Tensor()});
        g.activation = aux_activation(nn.activations()[k]);
        g.slots = none[k] ? "none" : "all";
        graph.push_back(g);
    }
    return graph;
}

/// ZonoNet::forward of the zonotope (one member) against the oracle: c, G, and dr (zero where
/// undefined) against the error box's upper end.
void aux_check(const std::string &what, const NeuralNetwork &nn, const Zonotope &X,
               const std::vector<bool> &none, bool record) {
    ZonoNet net(nn, kInf, kInf, "singh", "sequential");
    for (std::size_t k = 0; k < none.size(); ++k)
        if (none[k]) net.slots[k] = 0;
    const ZonoPropagation out =
        net.forward(X.center().expand_dims(0), Tensor(), X.generators().expand_dims(0), Tensor(),
                    nullptr, record);
    const lean::Zonotope Y = lean::evaluate_graph(aux_graph(nn, none), lean::Zonotope(X));
    // CORA's columns (m, 1) and (m, q); the radius of the interval center as a column
    const int64_t m = out.c.shape()[1];
    const std::string name = what + (record ? " (recorded)" : "");
    oracle::check(name + " c", out.c.reshape({m, 1}), Y.center());
    oracle::check(name + " G", out.G.squeeze(0).matrix_transpose(), Y.generators());
    const Tensor dr = out.dr.defined() ? out.dr.reshape({m, 1}) : out.c.zeros_like({m, 1});
    oracle::check(name + " dr", dr, Y.error().sup());
}

/// The network with every error in a slot, every one in dr, and the first layer's alone in dr;
/// with the record and without.
void aux_check_all(const std::string &what, const NeuralNetwork &nn, const Zonotope &X) {
    const std::size_t L = nn.layers().size();
    std::vector<bool> first(L, false);
    first[0] = true;
    for (const bool record : {false, true}) {
        aux_check(what + ", slots all", nn, X, std::vector<bool>(L, false), record);
        aux_check(what + ", slots none", nn, X, std::vector<bool>(L, true), record);
        aux_check(what + ", the first layer's in dr", nn, X, first, record);
    }
}

/// A layer of the weights and the bias, reading `source`, with residuals.
Layer aux_layer(const Tensor &W, const Tensor &b, int64_t source = NeuralNetwork::kPrevious,
                std::vector<NeuralNetwork::Residual> residuals = {}) {
    Layer layer{W, b};
    layer.source = source;
    layer.residuals = std::move(residuals);
    return layer;
}

/// A residual of the output `source` through the matrix R.
NeuralNetwork::Residual aux_through(int64_t source, const Tensor &R) {
    Layer map;
    map.W = R;
    return {source, std::make_shared<const Layer>(map)};
}

/// The matrix of the rows (r, c).
Tensor aux_mat(const std::vector<double> &data, int64_t rows, int64_t cols) {
    return Tensor::from_data(data, {rows, cols});
}

// The networks ------------------------------------------------------------------------------------

/// relu(W1 x + b1), relu(W2 y + b2 + x), then W3 z + b3 (2-3-2-1).
NeuralNetwork aux_residual_block(oracle::Random &rnd, double scale) {
    return {{aux_layer(rnd.matrix(3, 2) * scale, rnd.point(3) * 0.25),
             aux_layer(rnd.matrix(2, 3) * scale, rnd.point(2) * 0.25, NeuralNetwork::kPrevious,
                       {{-1, nullptr}}),
             aux_layer(rnd.matrix(1, 2), rnd.point(1))},
            {Activation::relu, Activation::relu, Activation::identity}};
}

/// relu(W x + b) (4 values), then MaxPool of the pairs: relu(a - b), plus b as a residual.
NeuralNetwork aux_max_pool(oracle::Random &rnd, double scale) {
    const Tensor D = aux_mat({1, -1, 0, 0, 0, 0, 1, -1}, 2, 4);
    const Tensor B = aux_mat({0, 1, 0, 0, 0, 0, 0, 1}, 2, 4), zero = rnd.point(2) * 0.0;
    return {{aux_layer(rnd.matrix(4, 3) * scale, rnd.point(4) * 0.25), aux_layer(D, zero),
             aux_layer(aux_mat({1, 0, 0, 1}, 2, 2), zero, NeuralNetwork::kPrevious,
                       {aux_through(0, B)})},
            {Activation::relu, Activation::relu, Activation::identity}};
}

/// The residual block's two outputs, then their max: relu(y0 - y1) plus y1 of layer 1.
NeuralNetwork aux_block_then_pool(oracle::Random &rnd, double scale) {
    const NeuralNetwork block = aux_residual_block(rnd, scale);
    std::vector<Layer> layers{block.layers()[0], block.layers()[1]};
    const Tensor zero = rnd.point(1) * 0.0;
    layers.push_back(aux_layer(aux_mat({1, -1}, 1, 2), zero));
    layers.push_back(aux_layer(aux_mat({1}, 1, 1), zero, NeuralNetwork::kPrevious,
                               {aux_through(1, aux_mat({0, 1}, 1, 2))}));
    return {layers, {Activation::relu, Activation::relu, Activation::relu, Activation::identity}};
}

/// a . b of two linear images of the input: a product of the halves, then their sum.
NeuralNetwork aux_dot(oracle::Random &rnd, int64_t n, double scale) {
    std::vector<double> sum(std::size_t(2 * n), 0.0);
    for (int64_t i = 0; i < n; ++i) sum[std::size_t(i)] = 1.0;
    return {{aux_layer(rnd.matrix(2 * n, 2) * scale, rnd.point(2 * n) * 0.25),
             aux_layer(aux_mat(sum, 1, 2 * n), rnd.point(1) * 0.0)},
            {Activation::product, Activation::identity}};
}

/// x^2, then sign, then a sum with the input through R.
NeuralNetwork aux_square_sign(oracle::Random &rnd, double scale) {
    return {{aux_layer(rnd.matrix(2, 2) * scale, rnd.point(2) * 0.125),
             aux_layer(rnd.matrix(2, 2), rnd.point(2) * 0.125),
             aux_layer(aux_mat({1, 1}, 1, 2), rnd.point(1) * 0.0, NeuralNetwork::kPrevious,
                       {aux_through(-1, rnd.matrix(1, 2) * 0.25)})},
            {Activation::square, Activation::sign, Activation::identity}};
}

/// relu(W x + b) (8 values, one channel of 2 x 4), then the port's MaxPool of 2 x 2 windows.
NeuralNetwork aux_port_max_pool(oracle::Random &rnd, double scale) {
    const NeuralNetwork::Block pool = NeuralNetwork::max_pool2d_layer({2, 2}, {1, 2, 4}, {0, 0}, 0);
    std::vector<Layer> layers{aux_layer(rnd.matrix(8, 3) * scale, rnd.point(8) * 0.25)};
    std::vector<Activation> activations{Activation::relu};
    layers.insert(layers.end(), pool.layers.begin(), pool.layers.end());
    activations.insert(activations.end(), pool.activations.begin(), pool.activations.end());
    return {layers, activations};
}

/// The port's composite of a ReLU branch and an identity branch on the input, added or
/// concatenated, then a dense layer.
NeuralNetwork aux_composite(oracle::Random &rnd, Aggregation aggregation, double scale) {
    const NeuralNetwork::Block relu{{Layer::linear(rnd.matrix(3, 2) * scale, rnd.point(3) * 0.25)},
                                    {Activation::relu}};
    const NeuralNetwork::Block line{{Layer::linear(rnd.matrix(3, 2) * scale, rnd.point(3) * 0.25)},
                                    {Activation::identity}};
    const NeuralNetwork::Block both = NeuralNetwork::composite_layer({relu, line}, aggregation, 2);
    std::vector<Layer> layers = both.layers;
    std::vector<Activation> activations = both.activations;
    const int64_t m = layers.back().outputs();
    layers.push_back(aux_layer(rnd.matrix(2, m), rnd.point(2)));
    activations.push_back(Activation::identity);
    return {layers, activations};
}

/// A zonotope of n dimensions and m generators of radius about `scale`.
Zonotope aux_input(oracle::Random &rnd, int64_t n, int64_t m, double scale) {
    return {rnd.point(n) * 0.5, rnd.matrix(m, n) * scale};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available() || !oracle::serves("neuralNetwork.evaluateGraph")) return 0;

    oracle::run("neuralNetwork.evaluateGraph", [&](oracle::Random &rnd) {
        // narrow and wide inputs: some neurons fixed, most crossing zero
        for (const double scale : {0.125, 0.5}) {
            const std::string at = " at " + std::to_string(scale);
            aux_check_all("residual block" + at, aux_residual_block(rnd, 1.0),
                          aux_input(rnd, 2, 3, scale));
            aux_check_all("max pool" + at, aux_max_pool(rnd, 1.0), aux_input(rnd, 3, 4, scale));
            aux_check_all("residual block, then max pool" + at, aux_block_then_pool(rnd, 1.0),
                          aux_input(rnd, 2, 2, scale));
            aux_check_all("dot product" + at, aux_dot(rnd, 2, 1.0), aux_input(rnd, 2, 2, scale));
            aux_check_all("dot product of 3" + at, aux_dot(rnd, 3, 1.0),
                          aux_input(rnd, 2, 3, scale));
            aux_check_all("square, sign" + at, aux_square_sign(rnd, 1.0),
                          aux_input(rnd, 2, 2, scale));
            // the port's MaxPool and composite layers
            aux_check_all("port max pool" + at, aux_port_max_pool(rnd, 1.0),
                          aux_input(rnd, 3, 3, scale));
            aux_check_all("composite add" + at, aux_composite(rnd, Aggregation::add, 1.0),
                          aux_input(rnd, 2, 3, scale));
            aux_check_all("composite concat" + at, aux_composite(rnd, Aggregation::concat, 1.0),
                          aux_input(rnd, 2, 3, scale));
        }
    });
    return test::finish("graph networks against the oracle");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
