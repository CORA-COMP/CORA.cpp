// neural_network - the class of feed-forward networks: construction and its checks
//
// See also: neuralNetwork.h, evaluate.cpp
//
// CORA.cpp only: the class's construction and checks (the header names the CORA file).
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   20-November-2026
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
        // the weights a matrix (out, in), the bias a vector (out)
        const Tensor &bias = layers_[k].b;
        CORA_CHECK(bias.defined(), Error::wrong_input_in_constructor,
                   [&] { return "NeuralNetwork: layer " + std::to_string(k) + " needs a bias"; });
        // a convolution or a gather checked its own sizes
        if (!layers_[k].plan) {
            CORA_CHECK(layers_[k].W.defined(), Error::wrong_input_in_constructor, [&] {
                return "NeuralNetwork: layer " + std::to_string(k) + " needs weights";
            });
            const std::vector<int64_t> w = layers_[k].W.shape(), b = bias.shape();
            CORA_CHECK(w.size() == 2, Error::wrong_input_in_constructor, [&] {
                return "NeuralNetwork: the weights of layer " + std::to_string(k) +
                       " must be a matrix (out, in), got the shape " + shape_text(w);
            });
            CORA_CHECK(b == std::vector<int64_t>({w[0]}), Error::wrong_input_in_constructor, [&] {
                return "NeuralNetwork: the bias of layer " + std::to_string(k) +
                       " must be a vector (" + std::to_string(w[0]) +
                       "), one entry per output, got " + shape_text(b);
            });
        }
        const int64_t inputs = layers_[k].inputs();
        CORA_CHECK(k == 0 || inputs == layers_[k - 1].outputs(), Error::wrong_input_in_constructor,
                   [&] {
                       // a layer needs as many inputs as the one before gives
                       return "NeuralNetwork: layer " + std::to_string(k) + " takes " +
                              std::to_string(inputs) + " inputs, the layer before gives " +
                              std::to_string(layers_[k - 1].outputs());
                   });
    }
}

bool NeuralNetwork::is_dense() const {
    for (const Layer &layer : layers_)
        if (layer.plan) return false;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
