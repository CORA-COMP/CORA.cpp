// neuralNetwork - the class of feed-forward networks: construction and its checks
//
// See also: neuralNetwork.h, evaluate.cpp

#include "nn/neuralNetwork/neuralNetwork.h"

#include <stdexcept>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) : layers_(std::move(layers)) {
    if (layers_.empty()) throw std::invalid_argument("cora: a network has at least one layer");
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        const std::vector<int64_t> w = layers_[k].W.shape(), b = layers_[k].b.shape();
        const std::string layer = std::to_string(k);
        if (w.size() != 2)
            throw std::invalid_argument("cora: the weights of a layer are a matrix (out, in); "
                                        "layer " + layer + " has another shape");
        if (b != std::vector<int64_t>({w[0], 1}))
            throw std::invalid_argument("cora: the bias of a layer is a column with one entry per "
                                        "output; layer " + layer + " has another shape");
        if (k > 0 && w[1] != layers_[k - 1].W.shape()[0])
            throw std::invalid_argument("cora: a layer takes as many inputs as the layer before "
                                        "gives outputs; layer " + layer + " does not");
    }
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
