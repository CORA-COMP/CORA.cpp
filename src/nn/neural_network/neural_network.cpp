// neural_network - the class of feed-forward networks: construction and its checks, the graph
//
// See also: neuralNetwork.h, evaluate.cpp
//
// CORA.cpp only: the class's construction and checks (the header names the CORA file).
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   22-November-2026
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <memory>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// Construction ------------------------------------------------------------------------------------

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
    // layer k's source resolved and checked, its residuals checked, their biases moved into its b
    const auto check_inputs = [&](std::size_t k) {
        Layer &layer = layers_[k];
        const std::string name = "NeuralNetwork: layer " + std::to_string(k);
        // a source before the layer, -1 the network's input
        const auto width = [&](int64_t j) {
            CORA_CHECK(j >= -1 && j < int64_t(k), Error::wrong_input_in_constructor, [&] {
                return name + " reads " + std::to_string(j) +
                       ", not an earlier layer or the input (-1)";
            });
            return j < 0 ? layers_.front().inputs() : layers_[std::size_t(j)].outputs();
        };
        if (layer.source == kPrevious) layer.source = int64_t(k) - 1;
        CORA_CHECK(k > 0 || layer.source == -1, Error::wrong_input_in_constructor,
                   [&] { return name + " must read the network's input"; });
        const int64_t given = k == 0 ? layer.inputs() : width(layer.source);
        CORA_CHECK(layer.inputs() == given, Error::wrong_input_in_constructor, [&] {
            return name + " takes " + std::to_string(layer.inputs()) +
                   " inputs, its source gives " + std::to_string(given);
        });
        // the residuals' maps fit their sources and the layer; their biases go into the layer's b
        for (Residual &r : layer.residuals) {
            CORA_CHECK(!r.map || r.map->W.defined() || r.map->plan,
                       Error::wrong_input_in_constructor,
                       [&] { return name + ": a residual's map needs weights or a plan"; });
            const int64_t in = width(r.source), out = r.map ? r.map->outputs() : in;
            CORA_CHECK((!r.map || r.map->inputs() == in) && out == layer.outputs(),
                       Error::wrong_input_in_constructor, [&] {
                           return name + ": a residual of layer " + std::to_string(r.source) +
                                  " maps " + std::to_string(in) + " values to " +
                                  std::to_string(out) + ", the layer has " +
                                  std::to_string(layer.outputs());
                       });
            // (a map without a bias, so that the layers make the same network again)
            if (r.map && r.map->b.defined()) {
                layer.b = layer.b + r.map->b;
                auto map = std::make_shared<Layer>(*r.map);
                map->b = Tensor();
                r.map = map;
            }
        }
    };
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
        // its source and residuals
        check_inputs(k);
    }
    // every output but the last is read
    std::vector<bool> read(layers_.size(), false);
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        if (source(k) >= 0) read[std::size_t(source(k))] = true;
        for (const Residual &r : layers_[k].residuals)
            if (r.source >= 0) read[std::size_t(r.source)] = true;
    }
    for (std::size_t k = 0; k + 1 < layers_.size(); ++k)
        CORA_CHECK(read[k], Error::wrong_input_in_constructor, [&] {
            return "NeuralNetwork: no later layer reads the output of layer " + std::to_string(k);
        });
}


// The graph ---------------------------------------------------------------------------------------

bool NeuralNetwork::is_dense() const {
    for (const Layer &layer : layers_)
        if (layer.plan) return false;
    return true;
}

bool NeuralNetwork::is_chain() const {
    for (std::size_t k = 0; k < layers_.size(); ++k)
        if (layers_[k].source != int64_t(k) - 1 || !layers_[k].residuals.empty()) return false;
    return true;
}

int64_t NeuralNetwork::source(std::size_t k) const { return layers_[k].source; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
