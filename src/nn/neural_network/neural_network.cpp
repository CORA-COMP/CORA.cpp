// neural_network - the class of feed-forward networks: construction and its checks
//
// See also: neuralNetwork.h, evaluate.cpp
//
// CORA.cpp only: the class's construction and checks (the header names the CORA file).
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// the old rule: a ReLU after every layer but the last
NeuralNetwork::NeuralNetwork(std::vector<Layer> layers)
    : NeuralNetwork(layers, [&] {
          std::vector<Activation> relus(layers.size(), Activation::relu);
          if (!relus.empty()) relus.back() = Activation::identity;
          return relus;
      }()) {}

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers, std::vector<Activation> activations)
    : layers_(std::move(layers)), activations_(std::move(activations)) {
    detail::Entry entry;
    // at least one layer, one activation each, and matching sizes
    CORA_CHECK(!layers_.empty(), Error::wrong_input_in_constructor,
               [] { return std::string("NeuralNetwork: give at least one layer"); });
    CORA_CHECK(activations_.size() == layers_.size(), Error::wrong_input_in_constructor, [&] {
        return "NeuralNetwork: give one activation per layer, got " +
               std::to_string(activations_.size()) + " for " + std::to_string(layers_.size()) +
               " layers";
    });
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
                       // a layer needs as many inputs as the one before gives
                       return "NeuralNetwork: layer " + std::to_string(k) + " takes " +
                              std::to_string(w[1]) + " inputs, the layer before gives " +
                              std::to_string(layers_[k - 1].W.shape()[0]);
                   });
    }
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
