// neural_network - the class of feed-forward networks: construction and its checks
//
// See also: neuralNetwork.h, evaluate.cpp

#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) : layers_(std::move(layers)) {
    detail::Entry entry;
    CORA_CHECK(!layers_.empty(), Error::wrong_input_in_constructor,
               [] { return std::string("NeuralNetwork: give at least one layer"); });
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        const std::vector<int64_t> w = layers_[k].W.shape(), b = layers_[k].b.shape();
        CORA_CHECK(w.size() == 2, Error::wrong_input_in_constructor, [&] {
            return "NeuralNetwork: the weights of layer " + std::to_string(k) +
                   " must be a matrix (out, in), got the shape " + shape_text(w);
        });
        CORA_CHECK(b == std::vector<int64_t>({w[0], 1}), Error::wrong_input_in_constructor, [&] {
            return "NeuralNetwork: the bias of layer " + std::to_string(k) + " must be a column (" +
                   std::to_string(w[0]) + ", 1), one entry per output, got " + shape_text(b);
        });
        CORA_CHECK(k == 0 || w[1] == layers_[k - 1].W.shape()[0], Error::wrong_input_in_constructor,
                   [&] {
                       return "NeuralNetwork: layer " + std::to_string(k) + " takes " +
                              std::to_string(w[1]) + " inputs, the layer before gives " +
                              std::to_string(layers_[k - 1].W.shape()[0]);
                   });
    }
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
