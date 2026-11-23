// reduce_network - CORA's computeReducedNetwork: merge the neurons whose outputs settle
//
// The network in normal form is a chain of layers (W, b, activation). Going through the
// activations in order, the output range of every neuron (interval arithmetic from the set's box)
// is compared with the buckets of the activation (its limits at -inf and inf for the static type,
// the neurons' centers for the dynamic one): the neurons whose range lies within the tolerance of
// the first bucket that holds it, in a bucket with at least two of them, are merged away. Their
// rows leave this layer, their columns the next, and their output ranges, mapped by the next
// matrix, are added to that layer as an interval d (the layer's approximation error). The set
// goes through the reduced layer with its error interval, so the output contains the outputs of
// the original network. With a target rate per layer the tolerance is searched by bisection; with
// input compression a ReLU of the identity reduces the (positive) input first. The result keeps
// the intervals beside the network, as this port's layers carry no interval offset. A set takes
// regression lines; no verbose output or plots (CORA prints and plots them).
//
// Syntax:   reduce_network() - the reduction, e.g.
//           ReducedNetwork R = nn.compute_reduced_network(X, options);
// Inputs:   X - the input set
//           options - bucket type, tolerance, rate, compression
// Outputs:  R - network, error intervals, merged neurons, rates, and the output set
// See also: cora/nn/neural_network/neural_network.h, network_forms.cpp, region_polys.cpp
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/computeReducedNetwork.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_layers.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Layer = NeuralNetwork::Layer;

constexpr int kMaxSearches = 2000;
/// CORA's default bucket width, and where the bisection of the tolerance ends (withinTol).
constexpr double kBucketWidth = 1e-2, kBisectionEnd = 1.0 / 1e10;

// The chain ---------------------------------------------------------------------------------------

/// The layers of the normal form with their error intervals (center, radius; undefined: none).
struct Chain {
    std::vector<Tensor> W, b, dc, dr, merged;
    std::vector<Activation> act;
    std::size_t size() const { return W.size(); }
};

/// A box (lo, hi) of rows (n).
using Box = std::pair<Tensor, Tensor>;

Tensor aux_times(const Tensor &W, const Tensor &x) {
    return W.matmul(x.expand_dims(-1)).squeeze(-1);
}

/// The box of a zonotope: c -+ sum |g|.
Box aux_box(const Zonotope &Z) {
    const Tensor reach = Z.generators().abs().sum(-2);
    return {Z.center() - reach, Z.center() + reach};
}

/// Interval arithmetic through layer j (IBP: the midpoint by W, the radius by |W|), plus d.
Box aux_affine_box(const Chain &c, std::size_t j, const Box &x) {
    const Tensor mid = (x.second + x.first) * 0.5, rad = (x.second - x.first) * 0.5;
    const Tensor mu = aux_times(c.W[j], mid) + c.b[j], r = aux_times(c.W[j].abs(), rad);
    Box out{mu - r, mu + r};
    if (c.dc[j].defined())
        out = {out.first + (c.dc[j] - c.dr[j]), out.second + (c.dc[j] + c.dr[j])};
    return out;
}

/// The image of a box under a rising activation.
Box aux_act_box(Activation a, const Box &x) {
    if (a == Activation::identity) return x;
    return {priv_apply(a, x.first), priv_apply(a, x.second)};
}

/// The zonotope after layer j's affine map and its error interval (a box generator per neuron).
Zonotope aux_affine_set(const Chain &c, std::size_t j, const Zonotope &Z) {
    Tensor center = aux_times(c.W[j], Z.center()) + c.b[j];
    Tensor G = Z.generators().matmul(c.W[j].matrix_transpose());
    if (c.dc[j].defined()) {
        center = center + c.dc[j];
        G = Tensor::concat({G, c.dr[j].expand_dims(-1).diag()}, -2);
    }
    return Zonotope(center, G);
}

/// The zonotope after an activation, by CORA's regression lines (the line of the network class).
Zonotope aux_act_set(Activation a, const Zonotope &Z) {
    if (a == Activation::identity) return Z;
    NnOptions options;
    options.poly_method = PolyMethod::regression;
    const int64_t n = Z.dim();
    const NeuralNetwork act({Layer{Z.center().eye_like(n), Z.center().zeros_like({n})}}, {a});
    return act.evaluate(Z, options);
}

/// CORA's withinTol: absolute or relative distance at most tol.
bool aux_within(double a, double b, double tol) {
    const double d = std::abs(a - b);
    return d <= tol || d / std::min(std::abs(a), std::abs(b)) <= tol ||
           (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b));
}

// The merging -------------------------------------------------------------------------------------

/// The neurons to merge at the tolerance: in the first bucket that holds a neuron's range, if
/// that bucket holds two neurons or more. Also the number of such buckets.
std::pair<std::vector<bool>, int64_t> aux_merge(const std::vector<double> &lo,
                                                const std::vector<double> &hi,
                                                const std::vector<double> &buckets, double tol) {
    const std::size_t n = lo.size(), nb = buckets.size();
    // the first bucket of each neuron, and the neurons in each bucket
    std::vector<int64_t> first(n, -1), count(nb, 0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t b = 0; b < nb && first[i] < 0; ++b)
            if (buckets[b] - tol <= lo[i] && hi[i] <= buckets[b] + tol) first[i] = int64_t(b);
    for (std::size_t i = 0; i < n; ++i)
        if (first[i] >= 0) ++count[std::size_t(first[i])];
    std::vector<bool> merged(n, false);
    for (std::size_t i = 0; i < n; ++i)
        merged[i] = first[i] >= 0 && count[std::size_t(first[i])] > 1;
    int64_t buckets_used = 0;
    for (const int64_t k : count) buckets_used += k > 1 ? 1 : 0;
    return {merged, buckets_used};
}

/// The rate of neurons that stay.
double aux_rate(const std::vector<bool> &merged) {
    return double(std::count(merged.begin(), merged.end(), false)) / double(merged.size());
}

/// The buckets of layer j: the activation's limits, or the neurons' centers in a shuffled order,
/// the first 100.
std::vector<double> aux_buckets(Activation a, bool dynamic, const std::vector<double> &lo,
                                const std::vector<double> &hi, uint64_t seed) {
    std::vector<double> buckets;
    if (!dynamic) {
        buckets = NeuralNetwork::merge_buckets(a);
    } else {
        // Fisher-Yates with the engine's own words (std::shuffle differs between libraries)
        for (std::size_t i = 0; i < lo.size(); ++i) buckets.push_back((lo[i] + hi[i]) * 0.5);
        std::mt19937_64 engine(seed);
        for (std::size_t i = buckets.size(); i > 1; --i)
            std::swap(buckets[i - 1], buckets[std::size_t(engine() % i)]);
    }
    buckets.resize(std::min<std::size_t>(100, buckets.size()));
    return buckets;
}

/// The values per layer: one value for all, or one per layer (checked by the caller).
std::vector<double> aux_per_layer(const std::vector<double> &values, std::size_t layers) {
    return values.size() == 1 ? std::vector<double>(layers, values[0]) : values;
}

// The reduction -----------------------------------------------------------------------------------

/// The reduction of the chain in place, the set S through it: the merging of every activation
/// that a layer follows (CORA's loop over k), then the rest of the layers.
Zonotope aux_reduce(Chain &c, Zonotope S, const ReduceOptions &options, std::vector<double> tol,
                    const std::vector<double> &rate) {
    const std::size_t m = c.size();
    Box bounds = aux_box(S);
    c.merged.assign(m, Tensor());
    for (std::size_t j = 0; j + 1 < m; ++j) {
        // the output range of the layer's neurons at the box that enters it
        const Box pre = aux_act_box(c.act[j], aux_affine_box(c, j, bounds));
        const std::vector<double> lo = pre.first.data(), hi = pre.second.data();
        const std::vector<double> buckets =
            aux_buckets(c.act[j], options.dynamic_buckets, lo, hi, options.seed);
        // the tolerance, searched by bisection if a rate is asked for
        double tol_j = tol[j], tol_min = 0.0;
        double tol_max =
            *std::max_element(hi.begin(), hi.end()) - *std::min_element(lo.begin(), lo.end());
        // the neurons merged at the tolerance, found again for each tolerance of the search
        std::vector<bool> merged;
        int64_t used = 0;
        int searches = options.min_searches;
        // bisect the tolerance until the rate is met (or once, without a rate)
        for (int round = 0;; ++round) {
            if (round >= kMaxSearches)
                CORA_FAIL(Error::not_converged, "compute_reduced_network: the rate of layer " +
                                                    std::to_string(j) + " was not reached");
            --searches;
            std::tie(merged, used) = aux_merge(lo, hi, buckets, tol_j);
            if (rate.empty()) break;
            const double want = rate[j], got = aux_rate(merged);
            if (want == got || (searches <= 0 && got <= want)) break;
            if (got < want) {
                tol_max = tol_j;
                tol_j = (tol_min + tol_j) / 2;
            } else {
                tol_min = tol_j;
                tol_j = std::isinf(tol_max) ? 10 * tol_j : (tol_j + tol_max) / 2;
            }
            if (aux_within(tol_min, tol_max, kBisectionEnd)) tol_j = tol_max;
        }
        tol[j] = tol_j;
        // the neurons that stay, as indices and as the mask of the merged ones
        std::vector<int64_t> keep;
        std::vector<double> mask;
        for (std::size_t i = 0; i < merged.size(); ++i) {
            if (!merged[i]) keep.push_back(int64_t(i));
            mask.push_back(merged[i] ? 1.0 : 0.0);
        }
        const Tensor merged_mask = Tensor::like(pre.first, mask, {int64_t(mask.size())});
        c.merged[j] = merged_mask;
        // the next layer reads the neurons that stay; the ones merged go in as an interval
        const Tensor W2 = c.W[j + 1];
        if (used > 0) {
            const Tensor mid = ((pre.first + pre.second) * 0.5).multiply(merged_mask);
            const Tensor rad = ((pre.second - pre.first) * 0.5).multiply(merged_mask);
            Tensor dc = aux_times(W2, mid), dr = aux_times(W2.abs(), rad);
            if (c.dc[j + 1].defined()) {
                dc = dc + c.dc[j + 1];
                dr = dr + c.dr[j + 1];
            }
            c.dc[j + 1] = dc;
            c.dr[j + 1] = dr;
        }
        c.W[j + 1] = W2.take(keep, 1);
        // this layer keeps its rows (and the rows of its own error interval)
        c.W[j] = c.W[j].take(keep, 0);
        c.b[j] = c.b[j].take(keep, 0);
        if (c.dc[j].defined()) {
            c.dc[j] = c.dc[j].take(keep, 0);
            c.dr[j] = c.dr[j].take(keep, 0);
        }
        // the set through the reduced layer, and its box for the next one
        S = aux_act_set(c.act[j], aux_affine_set(c, j, S));
        bounds = aux_box(S);
    }
    // the layers after the last merged activation
    S = aux_affine_set(c, m - 1, S);
    return aux_act_set(c.act[m - 1], S);
}

// The sets ----------------------------------------------------------------------------------------

/// The set through all layers of the chain, with their error intervals.
Zonotope aux_forward(const Chain &c, Zonotope S) {
    for (std::size_t j = 0; j < c.size(); ++j) S = aux_act_set(c.act[j], aux_affine_set(c, j, S));
    return S;
}

/// The neurons at the input and after each layer of the chain.
std::vector<int64_t> aux_counts(const Chain &c) {
    std::vector<int64_t> counts{c.W[0].shape()[1]};
    for (const Tensor &W : c.W) counts.push_back(W.shape()[0]);
    return counts;
}

/// The identity chain of CORA's input compression: identity, ReLU, identity on n values.
Chain aux_compression_chain(const Tensor &like, int64_t n) {
    Chain c;
    const Tensor I = like.eye_like(n), zero = like.zeros_like({n});
    c.W = {I, I};
    c.b = {zero, zero};
    c.act = {Activation::relu, Activation::identity};
    c.dc.assign(2, Tensor());
    c.dr.assign(2, Tensor());
    return c;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReducedNetwork NeuralNetwork::compute_reduced_network(const Zonotope &X,
                                                      const ReduceOptions &options) const {
    detail::Entry entry;
    CORA_CHECK(X.dim() == input_dim() && X.center().ndim() == 1, Error::dimension_mismatch, [&] {
        return "compute_reduced_network: the set must be a single zonotope of dimension " +
               std::to_string(input_dim()) + ", got dimension " + std::to_string(X.dim());
    });
    CORA_CHECK(options.min_searches >= 0, Error::wrong_value,
               [] { return std::string("compute_reduced_network: min_searches >= 0"); });
    // the normal form as a chain of dense layers
    const NeuralNetwork normal = normal_form();
    Chain c;
    for (std::size_t k = 0; k < normal.layers_.size(); ++k) {
        const Layer &L = normal.layers_[k];
        const Activation a = normal.activations_[k];
        CORA_CHECK(a == Activation::relu || a == Activation::sigmoid || a == Activation::tanh ||
                       a == Activation::exp || a == Activation::softplus ||
                       (a == Activation::identity && k + 1 == normal.layers_.size()),
                   Error::not_supported, [] {
                       return std::string("compute_reduced_network: the activations must be ReLU, "
                                          "sigmoid, tanh, exp or softplus (the last may be none)");
                   });
        c.W.push_back(L.plan ? L.dense() : L.W);
        c.b.push_back(L.b);
        c.act.push_back(a);
    }
    c.dc.assign(c.size(), Tensor());
    c.dr.assign(c.size(), Tensor());
    c.merged.assign(c.size(), Tensor());
    // the tolerance and the rate: one value for all layers or one per layer
    for (const std::vector<double> *values : {&options.bucket_tol, &options.reduction_rate})
        CORA_CHECK(values->size() <= 1 || values->size() == c.size(), Error::wrong_value, [&] {
            return "compute_reduced_network: BucketTol and ReductionRate take one value or one "
                   "per layer (" +
                   std::to_string(c.size()) + "), got " + std::to_string(values->size());
        });
    std::vector<double> tol = aux_per_layer(
        options.bucket_tol.empty() ? std::vector<double>{kBucketWidth} : options.bucket_tol,
        c.size());
    std::vector<double> rate = aux_per_layer(options.reduction_rate, c.size());
    if (options.reduction_rate.empty()) rate.clear();
    // a rate of 1 asks for no reduction: the set goes through the normal form as it is
    Zonotope S = X;
    ReduceOptions plain = options;
    plain.input_compression = false;
    std::vector<int64_t> before = aux_counts(c);
    if (!rate.empty() && *std::min_element(rate.begin(), rate.end()) == 1.0) {
        S = aux_forward(c, S);
    } else {
        if (options.input_compression) {
            // the input must be positive; the identity reduced by a ReLU's buckets, then its map
            const Box box = aux_box(S);
            CORA_CHECK(!box.first.less(box.first.zeros_like()).any(0).item(), Error::special_error,
                       [] {
                           return std::string("compute_reduced_network: the input has to be "
                                              "positive for input compression.");
                       });
            Chain comp = aux_compression_chain(S.center(), S.dim());
            aux_reduce(comp, S, plain, {tol[0], tol[0]},
                       rate.empty() ? std::vector<double>()
                                    : std::vector<double>{rate[0], rate[0]});
            // the set at the kept coordinates, and the layer that maps them back (then a ReLU)
            S = aux_affine_set(comp, 0, S);
            c.W.insert(c.W.begin(), comp.W[1]);
            c.b.insert(c.b.begin(), comp.b[1]);
            c.dc.insert(c.dc.begin(), comp.dc[1]);
            c.dr.insert(c.dr.begin(), comp.dr[1]);
            c.act.insert(c.act.begin(), Activation::relu);
            c.merged.insert(c.merged.begin(), Tensor());
            tol.insert(tol.begin(), tol[0]);
            if (!rate.empty()) rate.insert(rate.begin(), rate[0]);
            before = aux_counts(c);
        }
        S = aux_reduce(c, S, plain, tol, rate);
    }
    // the network of the reduced layers
    std::vector<Layer> layers;
    layers.reserve(c.size());
    for (std::size_t k = 0; k < c.size(); ++k) layers.push_back(Layer{c.W[k], c.b[k]});
    ReducedNetwork out{NeuralNetwork(layers, c.act), c.dc, c.dr, c.merged, {}, S};
    // the rate of neurons that stay: the target, or the ratio of the counts
    if (!rate.empty()) {
        out.reduction_rate = rate;
    } else {
        const std::vector<int64_t> after = aux_counts(c);
        for (std::size_t i = 0; i < after.size(); ++i)
            out.reduction_rate.push_back(double(after[i]) / double(before[i]));
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
