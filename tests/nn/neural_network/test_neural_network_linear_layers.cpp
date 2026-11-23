// test_neural_network_linear_layers - CORA's linear layers as layers of a NeuralNetwork
//
// Linear, identity, element-wise affine, batch norm (inference), average pooling and the Lipschitz
// constrained layer map rows as their definitions say; the counts of neurons and parameters.
//
// CORA: nn/layers/linear/nnLinearLayer.m
// CORA: nn/layers/linear/nnIdentityLayer.m
// CORA: nn/layers/linear/nnElementwiseAffineLayer.m
// CORA: nn/layers/linear/nnBatchNormLayer.m
// CORA: nn/layers/linear/nnAvgPool2DLayer.m
// CORA: nn/layers/linear/nnLipConstrLinearLayer.m
// CORA: nn/@neuralNetwork/getNumNeurons.m
// CORA: nn/@neuralNetwork/getNumParams.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using Layer = NeuralNetwork::Layer;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The layer's map of the rows x, with its bias.
Tensor aux_apply(const Layer &layer, const Tensor &x) { return layer.rows(x) + layer.b; }

void elementwise(const std::string &b) {
    const Layer id = Layer::identity(3);
    const Tensor x({{1.0, -2.0, 3.0}});
    check(test::close(aux_apply(id, x), std::vector<double>{1.0, -2.0, 3.0}), b + ": identity");
    const Layer ea = Layer::elementwise_affine(Tensor({2.0, -1.0, 0.5}), Tensor({1.0}), 3);
    check(test::close(aux_apply(ea, x), std::vector<double>{3.0, 3.0, 2.5}),
          b + ": scale .* x + offset, the offset one entry");
    const Layer sc = Layer::elementwise_affine(Tensor({3.0}), Tensor({0.0, 1.0, 2.0}), 3);
    check(test::close(aux_apply(sc, x), std::vector<double>{3.0, -5.0, 11.0}),
          b + ": the scale one entry");
    check(test::raises([] { Layer::elementwise_affine(Tensor({1.0, 2.0}), Tensor({0.0}), 3); },
                       Error::wrong_input_in_constructor),
          b + ": scale of the wrong length");
}

void batch_norm(const std::string &b) {
    // two channels of two pixels: (x - mean) / sqrt(var + eps) * scale + offset per channel
    const double eps = 1e-5;
    // the folded map equals the three steps of CORA's inference mode
    const Layer bn = Layer::batch_norm(Tensor({2.0, 1.0}), Tensor({0.5, -1.0}), Tensor({1.0, 0.0}),
                                       Tensor({3.0, 1.0}), {2, 1, 2}, eps);
    const Tensor x({{1.0, 4.0, 2.0, -2.0}});
    const double f0 = 2.0 / std::sqrt(3.0 + eps), f1 = 1.0 / std::sqrt(1.0 + eps);
    const std::vector<double> want{(1.0 - 1.0) * f0 + 0.5, (4.0 - 1.0) * f0 + 0.5, 2.0 * f1 - 1.0,
                                   -2.0 * f1 - 1.0};
    check(test::close(aux_apply(bn, x), want, 1e-12), b + ": batch norm in inference mode");
    // statistics of 0 and 1 leave the input as it is
    const Layer one = Layer::batch_norm(Tensor({1.0}), Tensor({0.0}), Tensor({0.0}), Tensor({1.0}),
                                        {1, 1, 2}, 0.0);
    check(test::close(aux_apply(one, Tensor({{5.0, -7.0}})), std::vector<double>{5.0, -7.0}),
          b + ": the identity statistics, one entry each");
    check(test::raises(
              [] {
                  Layer::batch_norm(Tensor({1.0}), Tensor({0.0}), Tensor({0.0}), Tensor({1.0}),
                                    {2, 2});
              },
              Error::wrong_input_in_constructor),
          b + ": an input that is no image");
}

void pooling(const std::string &b) {
    // one channel, 2 x 4 image, 2 x 2 windows: the means of the left and the right square
    const Layer avg = Layer::avg_pool2d({2, 2}, {1, 2, 4});
    const Tensor x({{1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0}});
    check(test::close(aux_apply(avg, x), std::vector<double>{2.5, 6.5}), b + ": the window means");
    // two channels with a stride of one
    const Layer two = Layer::avg_pool2d({1, 2}, {2, 1, 3}, {1, 1});
    check(two.outputs() == 4, b + ": the output size");
    check(test::close(aux_apply(two, Tensor({{1.0, 3.0, 5.0, 0.0, 2.0, 4.0}})),
                      std::vector<double>{2.0, 4.0, 1.0, 3.0}),
          b + ": channels stay apart");
    check(test::raises([] { Layer::avg_pool2d({0, 2}, {1, 2, 4}); },
                       Error::wrong_input_in_constructor),
          b + ": an empty window");
}

void lipschitz_and_counts(const std::string &b) {
    const Tensor W({{3.0, -1.0}, {0.25, 0.25}}), bias({4.0, 1.0});
    const Layer lip = Layer::lip_constr_linear(W, bias, 2.0);
    // row 0 has the absolute sum 4 > 2: scaled by 1/2; row 1 stays
    check(test::close(aux_apply(lip, Tensor({{1.0, 1.0}})), std::vector<double>{3.0, 1.5}),
          b + ": rows of absolute sum above lambda are scaled down");
    check(test::raises([&] { Layer::lip_constr_linear(W, bias, 0.0); },
                       Error::wrong_input_in_constructor),
          b + ": lambda must be positive");
    const NeuralNetwork nn(
        {Layer::linear(W, bias), Layer::linear(Tensor({{1.0, 2.0}}), Tensor({0.0}))});
    check(nn.num_neurons() == std::vector<int64_t>({2, 2, 1}), b + ": numbers of neurons");
    check(nn.num_params() == 4 + 2 + 2 + 1, b + ": number of parameters");
    const NeuralNetwork sparse({Layer::identity(2)});
    check(test::raises([&] { sparse.num_params(); }, Error::not_supported),
          b + ": parameters of a gather layer");
    {
        ChecksScope off(false);
        check(
            Layer::batch_norm(Tensor({1.0}), Tensor({0.0}), Tensor({0.0}), Tensor({1.0}), {1, 1, 1})
                    .outputs() == 1,
            b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        elementwise(b);
        batch_norm(b);
        pooling(b);
        lipschitz_and_counts(b);
    });
    return test::finish("neuralNetwork linear layers");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
