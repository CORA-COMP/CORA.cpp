// random_network - random and re-initialized networks, and a network in another dtype or device
//
// CORA's generateRandom: a dense network of a number of layers, each a random affine map with
// entries in [-1, 1) followed by an activation (default sigmoid); a size not given is drawn as
// CORA draws it (the inputs and outputs from 1 to 5, the layers from 1 to 5, the hidden neurons
// from 2 to 20). initWeights draws the weights of the dense layers again: 'glorot' uniform in
// [-1 / sqrt(n_in), 1 / sqrt(n_in)], 'shi' normal with the standard deviation sqrt(2 pi) / n_in,
// the biases 0. castWeights is `to`: every weight in another dtype or on another device; the
// copy of a network (CORA's copyNeuralNetwork) is the copy constructor, as a network is a value.
// A convolution, a gather and the other layers of a plan keep their weights in initWeights (CORA
// draws a convolution's kernel and sets a batch norm's scale to 1 and offset to 0).
//
// Syntax:   random_network() - the members of NeuralNetwork, e.g.
//           NeuralNetwork nn = NeuralNetwork::random({.inputs = 3, .layers = 2});
//           NeuralNetwork fresh = nn.init_weights("glorot", seed);
//           NeuralNetwork single = nn.to(DType::float32);
// Inputs:   options - inputs, outputs, layers, hidden (0: drawn), activation, seed
//           method - "glorot" or "shi"
//           which - the layers to draw again (empty: all)
// Outputs:  nn - the network
// See also: cora/nn/neural_network/neural_network.h, cora/global/rng.h
// References: none (CORA's utilities of a network, no paper's algorithm)
//
// CORA: nn/@neuralNetwork/generateRandom.m
// CORA: nn/@neuralNetwork/initWeights.m
// CORA: nn/@neuralNetwork/castWeights.m
// CORA: nn/@neuralNetwork/copyNeuralNetwork.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kPi = 3.14159265358979323846;

/// An integer drawn uniformly from lo to hi (CORA's randi).
int64_t aux_randi(Rng &rng, int64_t lo, int64_t hi) {
    double u = 0.0;
    rng.uniform(&u, 1, 0.0, 1.0);
    return std::min(hi, lo + int64_t(std::floor(u * double(hi - lo + 1))));
}

/// A tensor of the shape with entries uniform in [lo, hi), or normal with the scale hi.
CORA_FROM_NUMBERS Tensor aux_draw(Rng &rng, const std::vector<int64_t> &shape, bool normal,
                                  double lo, double hi) {
    int64_t n = 1;
    for (const int64_t s : shape) n *= s;
    std::vector<double> v(static_cast<std::size_t>(n));
    if (normal) {
        // standard normals times the scale afterwards, as CORA's normrnd (randn .* sigma)
        rng.normal(v.data(), v.size(), 1.0);
        for (double &x : v) x *= hi;
    } else
        rng.uniform(v.data(), v.size(), lo, hi);
    return Tensor::from_data(v, shape);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Random ------------------------------------------------------------------------------------------

NeuralNetwork NeuralNetwork::random(const RandomOptions &options) {
    detail::Entry entry;
    CORA_CHECK(
        options.inputs >= 0 && options.outputs >= 0 && options.layers >= 0 && options.hidden >= 0,
        Error::wrong_value,
        [] { return std::string("NeuralNetwork::random: the sizes are positive, or 0 to draw"); });
    // the sizes not given, drawn in CORA's order
    Rng rng(options.seed);
    const int64_t inputs = options.inputs ? options.inputs : aux_randi(rng, 1, 5);
    const int64_t outputs = options.outputs ? options.outputs : aux_randi(rng, 1, 5);
    const int64_t depth = options.layers ? options.layers : aux_randi(rng, 1, 5);
    const int64_t hidden = options.hidden ? options.hidden : aux_randi(rng, 2, 20);
    // the neurons per layer, and a random affine map and the activation for each
    std::vector<int64_t> neurons(std::size_t(depth) + 1, hidden);
    neurons.front() = inputs;
    neurons.back() = outputs;
    std::vector<Layer> layers;
    for (std::size_t i = 0; i + 1 < neurons.size(); ++i) {
        const Tensor W = aux_draw(rng, {neurons[i + 1], neurons[i]}, false, -1.0, 1.0);
        layers.push_back({W, aux_draw(rng, {neurons[i + 1]}, false, -1.0, 1.0)});
    }
    const std::vector<Activation> activations(layers.size(), options.activation);
    return NeuralNetwork(std::move(layers), activations);
}

// InitWeights -------------------------------------------------------------------------------------

NeuralNetwork NeuralNetwork::init_weights(const std::string &method, uint64_t seed,
                                          const std::vector<int64_t> &which) const {
    detail::Entry entry;
    CORA_CHECK(method == "glorot" || method == "shi", Error::wrong_value, [&] {
        return "init_weights: the method is 'glorot' or 'shi', got '" + method + "'";
    });
    // the layers to draw again
    std::vector<bool> redraw(layers_.size(), which.empty());
    for (const int64_t k : which) {
        CORA_CHECK(k >= 0 && k < num_layers(), Error::wrong_value, [&] {
            return "init_weights: the layer " + std::to_string(k) + " of " +
                   std::to_string(num_layers()) + " layers";
        });
        redraw[std::size_t(k)] = true;
    }
    // a dense layer's weights by the method, its biases 0
    Rng rng(seed);
    std::vector<Layer> layers = layers_;
    for (std::size_t k = 0; k < layers.size(); ++k) {
        Layer &layer = layers[k];
        if (!redraw[k] || layer.plan || !layer.W.defined()) continue;
        const double n_in = double(layer.inputs());
        layer.W = method == "glorot"
                      ? aux_draw(rng, layer.W.shape(), false, -1.0 / std::sqrt(n_in),
                                 1.0 / std::sqrt(n_in))
                      : aux_draw(rng, layer.W.shape(), true, 0.0, std::sqrt(2.0 * kPi) / n_in);
        layer.b = layer.b.zeros_like();
    }
    return NeuralNetwork(std::move(layers), activations_);
}

// CastWeights -------------------------------------------------------------------------------------

NeuralNetwork NeuralNetwork::to(DType dtype, const std::string &device) const {
    detail::Entry entry;
    // every layer's weights, the activations stay
    std::vector<Layer> layers;
    layers.reserve(layers_.size());
    for (const Layer &layer : layers_) layers.push_back(layer.to(dtype, device));
    return NeuralNetwork(std::move(layers), activations_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
