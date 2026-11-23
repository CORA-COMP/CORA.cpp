// explain - CORA's explain: a minimal abductive explanation of a classification
//
// The features of the input that can be freed (given the noise radius epsilon) while the class
// stays: the network gets a layer of the logit differences to the label and goes to its normal
// form; the inputs are tried in an order (the least sensitive first); a feature is freed for good
// if the network on the box of the freed features (as a zonotope) has every logit difference at
// most 0 (a regression network: an output box of radius at most the threshold). "abstract+refine"
// first tries reduced networks (compute_reduced_network at the rates of refine_steps, then the
// original network) and refines them while no counterexample of the fast gradient sign method is
// found. A feature is a pixel (freed in all channels at once) if the input size is given. Two
// differences to CORA, both sound: the attack moves only the freed features (CORA moves all of
// them, so its counterexample need not lie in the set) and uses the one-hot label (CORA subtracts
// the label's index from every output), and the random refinement rises toward the original
// network (CORA's can stay at the same rate forever). No verbose output and no plots.
//
// Syntax:   explain() - the member of NeuralNetwork, e.g.
//           Explanation e = nn.explain(x, label, 0.05, options);
//           Tensor z = nn.compute_fgsm_attack(x, t, 0.05);
// Inputs:   x - the point (in)
//           label - the class (0-based)
//           epsilon - the noise radius
//           options - method, order, refinement, threshold, timeout
// Outputs:  e - the freed features, the order tried, the seconds per feature
// See also: cora/nn/neural_network/neural_network.h, reduce_network.cpp, network_forms.cpp
// References:
//    [1] S. Bassan et al. "Explaining, Fast and Slow: Abstraction and Refinement of Provable
//        Explanations", ICML 2025.
//    [2] I. J. Goodfellow, J. Shlens, C. Szegedy. "Explaining and harnessing adversarial
//        examples", ICLR 2015.
//
// CORA: nn/@neuralNetwork/explain.m
// CORA: nn/@neuralNetwork/computeFGSMAttack.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_verify.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The features ------------------------------------------------------------------------------------

/// The input features of the pixels (every channel of a pixel), or the features themselves.
std::vector<int64_t> aux_features(const std::vector<int64_t> &pixels,
                                  const std::vector<int64_t> &input_size) {
    if (input_size.empty()) return pixels;
    const int64_t hw = input_size[1] * input_size[2];
    std::vector<int64_t> features;
    for (int64_t c = 0; c < input_size[0]; ++c)
        for (const int64_t p : pixels) features.push_back(c * hw + p);
    return features;
}

/// The box of x with the features in free_features moved by +- epsilon, as a zonotope.
CORA_FROM_NUMBERS Zonotope aux_input_set(const Tensor &x, double epsilon,
                                         const std::vector<int64_t> &free_features) {
    const int64_t n = x.numel(), h = int64_t(free_features.size());
    std::vector<double> G(std::size_t(h * n), 0.0);
    for (int64_t k = 0; k < h; ++k) G[std::size_t(k * n + free_features[std::size_t(k)])] = epsilon;
    return Zonotope(x, Tensor::from_data(G, {h, n}));
}

// The specification -------------------------------------------------------------------------------

/// Whether the output set meets the specification: a classifier's logit differences are at most 0,
/// a regression network's output box has the radius at most delta.
bool aux_check_specs(const Zonotope &Y, double delta) {
    const Interval box = Y.interval();
    const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
    if (hi.size() == 1) return (hi[0] - lo[0]) / 2.0 <= delta;
    return std::all_of(hi.begin(), hi.end(), [](double v) { return v <= 0.0; });
}

/// Whether the points of the network show a violation (a classifier's logit difference above 0;
/// a regression network has none to find).
bool aux_violated(const Tensor &y) {
    const std::vector<double> v = y.data();
    return v.size() > 1 &&
           std::any_of(v.begin(), v.end(), [](double value) { return value > 0.0; });
}

// The abstract network ----------------------------------------------------------------------------

/// The abstract network of abstract+refine: the network, its reduction rate, and the output set.
struct Abstract {
    NeuralNetwork net;
    double rate;
    Zonotope Y;
};

/// Whether every layer of the network has a neuron (a reduction can merge a whole layer away).
bool aux_has_neurons(const NeuralNetwork &net) {
    return std::all_of(net.layers().begin(), net.layers().end(),
                       [](const NeuralNetwork::Layer &l) { return l.outputs() > 0; });
}

/// The network at the reduction rate: the reduced network, or the network itself at rate 1.
Abstract aux_abstract(const NeuralNetwork &nn, const Zonotope &X, double rate,
                      const ExplainOptions &o) {
    NnOptions nno;
    nno.poly_method = PolyMethod::regression;
    if (rate >= 1.0) return {nn, 1.0, nn.evaluate(X, nno)};
    ReduceOptions ro;
    ro.dynamic_buckets = o.bucket_type == "dynamic";
    ro.reduction_rate = {rate};
    ro.seed = o.seed;
    ReducedNetwork R = nn.compute_reduced_network(X, ro);
    return {R.net, rate, R.output};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The fast gradient sign method -------------------------------------------------------------------

Tensor NeuralNetwork::compute_fgsm_attack(const Tensor &x, const Tensor &t, double epsilon,
                                          const Tensor &mask) const {
    detail::Entry entry;
    CORA_CHECK(x.ndim() == 2 && t.ndim() == 2 && x.shape()[1] == input_dim() &&
                   t.shape()[0] == x.shape()[0] && t.shape()[1] == output_dim(),
               Error::dimension_mismatch, [&] {
                   return "compute_fgsm_attack: the points are rows (B, " +
                          std::to_string(input_dim()) + ") and the targets (B, " +
                          std::to_string(output_dim()) + "), got " + shape_text(x.shape()) +
                          " and " + shape_text(t.shape());
               });
    CORA_CHECK(!mask.defined() || mask.numel() == input_dim(), Error::dimension_mismatch,
               [] { return std::string("compute_fgsm_attack: the mask has one entry per input"); });
    // the gradient of the loss, whose derivative at the output is softmax(y) - t
    const Tensor grad = backprop(x, evaluate(x).softmax(1) - t);
    // a step of the full radius along the gradient's sign (against it with a negative alpha)
    Tensor alpha = x.full_like(x.shape(), -epsilon);
    if (mask.defined()) alpha = alpha.multiply(mask.reshape({1, input_dim()}));
    return priv_sign_step(x, grad, alpha);
}

// The explanation ---------------------------------------------------------------------------------

CORA_FROM_NUMBERS Explanation NeuralNetwork::explain(const Tensor &x, int64_t label, double epsilon,
                                                     const ExplainOptions &options) const {
    detail::Entry entry;
    const ExplainOptions &o = options;
    const int64_t m = output_dim();
    // the arguments
    CORA_CHECK(x.ndim() == 1 && x.numel() == input_dim(), Error::dimension_mismatch, [&] {
        return "explain: x has the network's " + std::to_string(input_dim()) + " inputs, got " +
               shape_text(x.shape());
    });
    CORA_CHECK(label >= 0 && (m == 1 || label < m), Error::wrong_value,
               [&] { return "explain: the label must be a class below " + std::to_string(m); });
    // the radii, the methods and the sizes
    CORA_CHECK(epsilon >= 0.0 && o.output_threshold >= 0.0 && o.timeout >= 0.0, Error::wrong_value,
               [] { return std::string("explain: epsilon, the threshold and timeout are >= 0"); });
    CORA_CHECK(o.method == "standard" || o.method == "abstract+refine", Error::wrong_value, [&] {
        return "explain: the method is 'standard' or 'abstract+refine', got '" + o.method + "'";
    });
    CORA_CHECK(
        o.refine_method == "all" || o.refine_method == "rand" || o.refine_method == "sensitivity",
        Error::wrong_value, [&] {
            return "explain: the refine method is 'all' or 'rand', got '" + o.refine_method + "'";
        });
    CORA_CHECK(o.refine_method != "sensitivity" || o.method == "standard", Error::not_supported,
               [] { return std::string("explain: refinement by sensitivity is not supported"); });
    // the buckets and the ladder of reduction rates
    CORA_CHECK(o.bucket_type == "static" || o.bucket_type == "dynamic", Error::wrong_value, [&] {
        return "explain: the bucket type is 'static' or 'dynamic', got '" + o.bucket_type + "'";
    });
    // the reduction rates: CORA's 0.1:0.1:1 unless given
    std::vector<double> steps = o.refine_steps;
    for (int k = 1; steps.empty() && k <= 10; ++k) steps.push_back(double(k) / 10.0);
    CORA_CHECK(std::is_sorted(steps.begin(), steps.end()) && steps.front() > 0.0 &&
                   steps.back() <= 1.0,
               Error::wrong_value,
               [] { return std::string("explain: refine_steps ascend within (0, 1]"); });
    CORA_CHECK(o.input_size.empty() ||
                   (o.input_size.size() == 3 &&
                    o.input_size[0] * o.input_size[1] * o.input_size[2] == input_dim()),
               Error::dimension_mismatch, [&] {
                   return "explain: the input size (C, H, W) " + shape_text(o.input_size) +
                          " must have the network's " + std::to_string(input_dim()) + " inputs";
               });
    // the order's method (input_neuron_order has no checks of its own inside this call)
    CORA_CHECK(!o.feat_order.empty() ||
                   ((o.feat_order_method == "in-order" || o.feat_order_method == "snake" ||
                     o.feat_order_method == "sensitivity") &&
                    (o.feat_order_method != "snake" || !o.input_size.empty())),
               Error::wrong_value, [&] {
                   return "explain: the order is 'in-order', 'snake' (with the input size) or "
                          "'sensitivity', got '" +
                          o.feat_order_method + "'";
               });
    // the logit differences to the label as the output of a classifier, in normal form
    NeuralNetwork nn = *this;
    if (m > 1) {
        std::vector<double> W(std::size_t(m * m), 0.0);
        for (int64_t i = 0; i < m; ++i) W[std::size_t(i * m + i)] = 1.0;
        for (int64_t i = 0; i < m; ++i) W[std::size_t(i * m + label)] -= 1.0;
        std::vector<Layer> layers = layers_;
        std::vector<Activation> activations = activations_;
        layers.push_back(Layer::linear(Tensor::from_data(W, {m, m}), Tensor::zeros({m})));
        activations.push_back(Activation::identity);
        nn = NeuralNetwork(layers, activations);
    }
    nn = nn.normal_form();
    // sets go through dense layers only: a convolution becomes its matrix
    if (!nn.is_dense()) {
        std::vector<Layer> dense;
        for (const Layer &layer : nn.layers()) dense.push_back(Layer{layer.dense(), layer.b});
        nn = NeuralNetwork(dense, nn.activations());
    }
    // the reduction takes ReLU, sigmoid and tanh (CORA's nested checks are off in this call)
    CORA_CHECK(
        o.method == "standard" || std::all_of(nn.activations().begin(), nn.activations().end(),
                                              [](Activation a) {
                                                  return a == Activation::relu ||
                                                         a == Activation::sigmoid ||
                                                         a == Activation::tanh ||
                                                         a == Activation::identity;
                                              }),
        Error::not_supported, [] {
            return std::string("explain: abstract+refine needs ReLU, sigmoid or tanh activations");
        });
    // the order of the features
    Explanation out;
    out.feat_order = o.feat_order.empty()
                         ? nn.input_neuron_order(o.feat_order_method, x, o.input_size)
                         : o.feat_order;
    const int64_t pixels = o.input_size.empty() ? input_dim() : o.input_size[1] * o.input_size[2];
    CORA_CHECK(std::all_of(out.feat_order.begin(), out.feat_order.end(),
                           [&](int64_t f) { return f >= 0 && f < pixels; }),
               Error::wrong_value, [&] {
                   return "explain: the feature order lists values below " + std::to_string(pixels);
               });
    CORA_CHECK(!out.feat_order.empty(), Error::wrong_value,
               [] { return std::string("explain: the feature order is empty"); });
    out.times.assign(out.feat_order.size(), std::numeric_limits<double>::quiet_NaN());
    NnOptions nno;
    nno.poly_method = PolyMethod::regression;
    // can all features be freed?
    const Zonotope all = aux_input_set(x, epsilon, aux_features(out.feat_order, o.input_size));
    if (aux_check_specs(nn.evaluate(all, nno), o.output_threshold)) {
        out.freed = out.feat_order;
        out.times.assign(out.feat_order.size(), 0.0);
        return out;
    }
    // free the features one by one
    std::optional<Abstract> abstract;
    Rng rng(o.seed);
    double total = 0.0;
    for (std::size_t i = 0; i < out.feat_order.size(); ++i) {
        const double started = now_seconds();
        std::vector<int64_t> trial = out.freed;
        trial.push_back(out.feat_order[i]);
        const std::vector<int64_t> features = aux_features(trial, o.input_size);
        const Zonotope X = aux_input_set(x, epsilon, features);
        bool verified = false;
        if (o.method == "standard") {
            verified = aux_check_specs(nn.evaluate(X, nno), o.output_threshold);
        } else {
            // an abstract network at the current rate (the first rate at the start)
            abstract = aux_abstract(nn, X, abstract ? abstract->rate : steps.front(), o);
            // the mask of the freed features, and the one-hot class to attack
            std::vector<double> mask(std::size_t(input_dim()), 0.0), hot(std::size_t(m), 0.0);
            for (const int64_t f : features) mask[std::size_t(f)] = 1.0;
            hot[std::size_t(m > 1 ? label : 0)] = 1.0;
            while (true) {
                verified = aux_check_specs(abstract->Y, o.output_threshold);
                if (verified) break;
                // a counterexample of the abstract net that holds on the original one
                if (aux_has_neurons(abstract->net)) {
                    const Tensor z = abstract->net.compute_fgsm_attack(
                        x.reshape({1, input_dim()}), Tensor::from_data(hot, {1, m}), epsilon,
                        Tensor::from_data(mask, {input_dim()}));
                    if (aux_violated(nn.evaluate(z))) break;
                }
                // refine, or give up on the feature at the original network
                const double old = abstract->rate;
                if (old >= 1.0) break;
                double next = 1.0;
                if (o.refine_method == "all") {
                    if (old >= steps.back()) break;
                    next = *std::find_if(steps.begin(), steps.end(),
                                         [&](double r) { return r > old; });
                } else {
                    double u = 0.0;
                    rng.uniform(&u, 1, 0.0, 1.0);
                    next = 1.0 - (1.0 - old) * u;
                }
                abstract = aux_abstract(nn, X, next, o);
            }
        }
        out.times[i] = now_seconds() - started;
        if (verified) out.freed = trial;
        // stop if the next feature would exceed the timeout
        total += out.times[i];
        if (double(i + 2) * total / double(i + 1) > o.timeout) break;
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
