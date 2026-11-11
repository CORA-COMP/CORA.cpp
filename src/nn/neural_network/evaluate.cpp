// evaluate - the outputs of a network at points, and an enclosure of them over a set
//
// An activation y = a(x) over the range [l, u] of a neuron is enclosed by one line and an error,
// y in slope x + offset +- radius (CORA's poly_method "singh", order 1, used by every layer here).
// A ReLU is exact where l >= 0 (y = x) or u <= 0 (y = 0), else the tightest parallelogram has
// slope u / (u - l) and offset and radius mu = -slope l / 2. A sigmoid or tanh is increasing and
// S-shaped, so its derivative is at least the smaller one at the ends: slope = min(a'(l), a'(u)),
// offset = (a(u) + a(l) - slope (u + l)) / 2, radius = (a(u) - a(l) - slope (u - l)) / 2
// [Singh et al., NeurIPS 2018, Thm. 3.1 and 3.2]. The line maps (c, G, GI) exactly, so the factors
// and the exponents never change; the radius is a new independent generator per neuron, or a new
// factor of its own. evaluate_with_input carries the input rows along, so the output depends on
// the same factors as the input. All neurons are handled at once by tensor operations.
//
// Syntax:   y = nn.evaluate(x);   Y = nn.evaluate(X);   XY = nn.evaluate_with_input(X);
// Inputs:   x - points, columns (in, N)
//           X - a zonotope or polynomial zonotope of dimension in (a zonotope may be a batch)
// Outputs:  y - the outputs (out, N)
//           Y - a zonotope that contains the outputs of all points of X
//           XY - the set of the pairs (x, N(x)), dimension in + out
// See also: cora/nn/neural_network/neural_network.h, Zonotope::interval
//
// CORA: nn/@neuralNetwork/evaluate.m
// CORA: nn/@neuralNetwork/evaluate_.m
// CORA: nn/layers/nonlinear/nnReLULayer.m
// CORA: nn/layers/nonlinear/nnLeakyReLULayer.m
// CORA: nn/layers/nonlinear/nnSigmoidLayer.m
// CORA: nn/layers/nonlinear/nnTanhLayer.m
// CORA: nn/layers/nonlinear/nnSShapeLayer.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotope.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// CORA authors: Tobias Ladner, Sebastian Sigl, Lukas Koller
// Authors:       Tobias Ladner, Claude Sonnet 5.5
// Written:       02-November-2026
// Last update:   11-November-2026
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The line of an activation -----------------------------------------------------------------------

/// The line of an activation over a range: y in slope x + offset +- radius.
struct Line {
    Tensor slope, offset, radius;
};

/// The line that encloses the activation over [l, u] (elementwise; the formulas above).
Line aux_line(Activation activation, const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like(), one = l.ones_like();
    switch (activation) {
    case Activation::identity:
        return {one, zero, zero};
    case Activation::relu: {
        // a stable neuron is exact; u - l is positive where the range crosses 0
        const Tensor negative = u.less_equal(zero), positive = l.greater_equal(zero);
        const Tensor slope = u.divide((u - l).maximum(numerics::kTiny));
        const Tensor mu = slope.multiply(l) * -0.5;
        const Tensor exact = Tensor::where(negative, one, Tensor::where(positive, one, zero));
        return {Tensor::where(negative, zero, Tensor::where(positive, one, slope)),
                Tensor::where(exact, zero, mu), Tensor::where(exact, zero, mu)};
    }
    case Activation::sigmoid:
    case Activation::tanh: {
        const bool is_tanh = activation == Activation::tanh;
        const Tensor fl = is_tanh ? l.tanh() : l.sigmoid(), fu = is_tanh ? u.tanh() : u.sigmoid();
        // the derivative from the value: 1 - f^2 for tanh, f (1 - f) for the sigmoid
        const Tensor dl = is_tanh ? one - fl.multiply(fl) : fl.multiply(one - fl);
        const Tensor du = is_tanh ? one - fu.multiply(fu) : fu.multiply(one - fu);
        const Tensor slope = dl.minimum(du);
        return {slope, (fu + fl - slope.multiply(u + l)) * 0.5,
                (fu - fl - slope.multiply(u - l)) * 0.5};
    }
    }
    CORA_FAIL(Error::wrong_value, "evaluate: an unknown activation");
}

/// The values of the activation at the points.
Tensor aux_apply(Activation activation, const Tensor &x) {
    switch (activation) {
    case Activation::identity:
        return x;
    case Activation::relu:
        return x.maximum(0.0);
    case Activation::sigmoid:
        return x.sigmoid();
    case Activation::tanh:
        return x.tanh();
    }
    CORA_FAIL(Error::wrong_value, "evaluate: an unknown activation");
}

/// The rows of a set that goes through the layers: the center, the dependent generators with the
/// mask of those with only even exponents (1, h), the independent generators, and the factors that
/// the errors added (their generators are the last columns of G).
struct Rows {
    Tensor c, G, GI, even;
    int64_t new_factors = 0;
};

// The layers on a set -----------------------------------------------------------------------------

/// The bounds of every row (CORA's bound_approx).
std::pair<Tensor, Tensor> aux_bounds(const Rows &r) {
    const Tensor odd = r.even.ones_like() - r.even;
    const Tensor shift = r.G.multiply(r.even).sum(-1, true) * 0.5;
    const Tensor radius = r.G.abs().multiply(r.even).sum(-1, true) * 0.5 +
                          r.G.abs().multiply(odd).sum(-1, true) + r.GI.abs().sum(-1, true);
    const Tensor mid = r.c + shift;
    return {mid - radius, mid + radius};
}

/// The columns of diag(radius) with a positive radius (all of them: keep_zero).
Tensor aux_error_columns(const Tensor &radius, bool keep_zero) {
    const Tensor D = radius.diag();
    if (keep_zero) return D;
    std::vector<int64_t> positive;
    const std::vector<double> values = radius.data();
    for (std::size_t i = 0; i < values.size(); ++i)
        if (values[i] > 0) positive.push_back(static_cast<int64_t>(i));
    return D.take(positive, -1);
}

/// The layers of the network on the rows: an exact map, then the line and the error of the
/// activation. keep_zero keeps the (zero) generator of a neuron without error.
Rows aux_layers(const std::vector<NeuralNetwork::Layer> &layers,
                const std::vector<Activation> &activations, Rows r, bool errors_in_gi,
                bool keep_zero) {
    for (std::size_t k = 0; k < layers.size(); ++k) {
        r.c = layers[k].W.matmul(r.c) + layers[k].b;
        r.G = layers[k].W.matmul(r.G);
        r.GI = layers[k].W.matmul(r.GI);
        // the range of every neuron, the line over it, and the set it maps to
        const auto [l, u] = aux_bounds(r);
        const Line line = aux_line(activations[k], l, u);
        r.c = line.slope.multiply(r.c) + line.offset;
        r.G = line.slope.multiply(r.G);
        r.GI = line.slope.multiply(r.GI);
        // the identity has no error
        if (activations[k] == Activation::identity) continue;
        // the error, as independent generators or as new factors
        const Tensor D = aux_error_columns(line.radius, keep_zero);
        if (errors_in_gi) {
            r.GI = Tensor::concat({r.GI, D}, -1);
        } else {
            r.G = Tensor::concat({r.G, D}, -1);
            r.even = Tensor::concat({r.even, D.slice(-2, 0, 1).zeros_like()}, -1);
            r.new_factors += D.shape().back();
        }
    }
    return r;
}

/// The rows of a column set with the dependent generators G (exponents E) and the independent GI.
Rows aux_rows(const Tensor &c, const Tensor &G, const Tensor &GI, const Exponents &E) {
    std::vector<double> even;
    for (const std::vector<int> &column : E)
        even.push_back(std::all_of(column.begin(), column.end(), [](int e) { return e % 2 == 0; })
                           ? 1.0
                           : 0.0);
    return {c, G, GI, Tensor::like(c, even, {1, static_cast<int64_t>(even.size())}), 0};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Points ------------------------------------------------------------------------------------------

Tensor NeuralNetwork::evaluate(const Tensor &x) const {
    detail::Entry entry;
    CORA_CHECK(x.shape().size() == 2 && x.shape()[0] == input_dim(), Error::dimension_mismatch,
               [&] {
                   return "evaluate: the points must be columns (" + std::to_string(input_dim()) +
                          ", N), got the shape " + shape_text(x.shape());
               });
    // an affine map per layer, then its activation
    Tensor h = x;
    for (std::size_t k = 0; k < layers_.size(); ++k)
        h = aux_apply(activations_[k], layers_[k].W.matmul(h) + layers_[k].b);
    return h;
}

// Sets --------------------------------------------------------------------------------------------

Zonotope NeuralNetwork::evaluate(const Zonotope &X) const {
    detail::Entry entry;
    CORA_CHECK(X.dim() == input_dim(), Error::dimension_mismatch, [&] {
        return "evaluate: the set has the dimension " + std::to_string(X.dim()) +
               ", the network takes " + std::to_string(input_dim());
    });
    // the layers on the set, its generators independent; a zero generator stays for an exact neuron
    const Tensor none = X.generators().slice(-1, 0, 0);
    const Rows r = aux_layers(layers_, activations_, aux_rows(X.center(), none, X.generators(), {}),
                              true, true);
    return {r.c, r.GI};
}

Zonotope NeuralNetwork::evaluate_with_input(const Zonotope &X) const {
    detail::Entry entry;
    // the generators of a zonotope are independent: the polynomial zonotope without dependent ones
    const Tensor none = X.generators().slice(-1, 0, 0);
    const PolyZonotope joint =
        evaluate_with_input(PolyZonotope(X.center(), none, X.generators(), {}, {}));
    return {joint.center(), joint.independent_generators()};
}

PolyZonotope NeuralNetwork::evaluate_with_input(const PolyZonotope &X,
                                                const NnOptions &options) const {
    detail::Entry entry;
    CORA_CHECK(X.dim() == input_dim(), Error::dimension_mismatch, [&] {
        return "evaluate_with_input: the set has the dimension " + std::to_string(X.dim()) +
               ", the network takes " + std::to_string(input_dim());
    });
    const Rows r =
        aux_layers(layers_, activations_,
                   aux_rows(X.center(), X.generators(), X.independent_generators(), X.exponents()),
                   options.add_approx_error_to_gi, false);
    // the input rows keep their generators, and are zero in the generators the errors added
    const int64_t n = input_dim(), h = X.num_dependent(), q = X.independent_generators().shape()[1];
    const Tensor zeros_G = X.center().zeros_like({n, r.G.shape()[1] - h});
    const Tensor zeros_GI = X.center().zeros_like({n, r.GI.shape()[1] - q});
    // the new factors follow the old ones, with identifiers beyond all of them
    std::vector<int64_t> ids = X.ids();
    const int64_t first = ids.empty() ? 1 : *std::max_element(ids.begin(), ids.end()) + 1;
    for (int64_t j = 0; j < r.new_factors; ++j) ids.push_back(first + j);
    Exponents E = X.exponents();
    for (std::vector<int> &column : E) column.resize(ids.size(), 0);
    for (int64_t j = 0; j < r.new_factors; ++j) {
        std::vector<int> column(ids.size(), 0);
        column[X.ids().size() + std::size_t(j)] = 1;
        E.push_back(column);
    }
    return {Tensor::concat({X.center(), r.c}, 0),
            Tensor::concat({Tensor::concat({X.generators(), zeros_G}, 1), r.G}, 0),
            Tensor::concat({Tensor::concat({X.independent_generators(), zeros_GI}, 1), r.GI}, 0), E,
            ids};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
