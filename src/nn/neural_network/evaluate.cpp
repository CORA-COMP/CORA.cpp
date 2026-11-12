// evaluate - the outputs of a network at points, and an enclosure of them over a set
//
// An activation y = a(x) over the range [l, u] of a neuron is enclosed by one line and an error,
// y in slope x + offset +- radius (CORA's poly_method "singh" or "regression", order 1): singh is
// the tightest parallelogram [Singh et al., NeurIPS 2018, Thm. 3.1 and 3.2], regression the
// least-squares line through 20 equidistant points of [l, u] shifted by the middle of its
// difference to the activation, whose bounds are exact (the ends and where f'(x) = slope). The
// line maps (c, G, GI) exactly, so the factors and the exponents never change; the radius is a
// new independent generator per neuron, or a new factor. The order of the operations is that of
// CORALean's Layer.activation, bit for bit at nearest for singh (test_lean_nn_evaluate): the
// slope, the box [offset - radius, offset + radius] as a zonotope whose middle is added to the
// center and whose reach is the generator. evaluate_with_input carries the input
// rows along, so the output depends on the same factors as the input. All neurons are handled at
// once by tensor operations.
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

/// The derivative of the activation at the values y = a(x) (relu: at x, which y = max(x, 0) tells
/// apart from the sign, so the ReLU takes y > 0).
Tensor aux_derivative(Activation activation, const Tensor &y) {
    const Tensor one = y.ones_like();
    switch (activation) {
    case Activation::identity:
        return one;
    case Activation::relu:
        return Tensor::where(y.greater(y.zeros_like()), one, y.zeros_like());
    case Activation::sigmoid:
        return y.multiply(one - y);
    case Activation::tanh:
        return one - y.multiply(y);
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

/// The sigmoid's value from below, 1 - 1 / (1 + exp x), and from above, 1 / (1 + exp(-x)): the
/// enclosure of CORALean's sigmoidEnclosure (at nearest both are the machine's results).
Tensor aux_sigmoid_lo(const Tensor &x) {
    const Tensor one = x.ones_like();
    return one - one.divide((one + x.exp()).maximum(1.0));
}

Tensor aux_sigmoid_hi(const Tensor &x) {
    const Tensor one = x.ones_like();
    return one.divide((one + x.negative().exp()).maximum(1.0));
}

/// The sigmoid's line by Singh's transformer (CORALean's sigmoidSinghRelaxation, in the file's
/// docstring): the slope is the smaller lower bound sig(x) sig(-x) of the derivative at the ends.
Line aux_line_sigmoid_singh(const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like();
    auto derivative = [&](const Tensor &x) {
        return aux_sigmoid_lo(x).maximum(zero).multiply(aux_sigmoid_lo(x.negative()).maximum(zero));
    };
    const Tensor slope = derivative(l).minimum(derivative(u));
    const Tensor lo = aux_sigmoid_lo(l) - slope.multiply(l);
    const Tensor hi = aux_sigmoid_hi(u) - slope.multiply(u);
    const Tensor offset = (lo + hi) * 0.5;
    return {slope, offset, (hi - offset).maximum(offset - lo)};
}

/// The line over [l, u] by Singh's transformer (CORALean's relaxations, bit for bit at nearest).
Line aux_line_singh(Activation activation, const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like(), one = l.ones_like();
    switch (activation) {
    case Activation::identity:
        return {one, zero, zero};
    case Activation::relu: {
        // exact if l >= 0 or u <= 0, else slope u / (u - l); half the gap to relu is the line
        const Tensor negative = u.less_equal(zero), positive = zero.less_equal(l);
        const Tensor secant = u.divide((u - l).maximum(numerics::kTiny)).minimum(1.0).maximum(0.0);
        const Tensor slope = Tensor::where(negative, zero, Tensor::where(positive, one, secant));
        const Tensor gap = (u - slope.multiply(u)).maximum(slope.multiply(l).negative()) * 0.5;
        return {slope, gap, gap};
    }
    case Activation::sigmoid:
        return aux_line_sigmoid_singh(l, u);
    case Activation::tanh: {
        // the sigmoid's line at the doubled range, scaled by 4 and shifted by -1
        const Line s = aux_line_sigmoid_singh(l + l, u + u);
        const Tensor four = (s.slope + s.slope) + (s.slope + s.slope);
        return {four, (s.offset + s.offset) + (-1.0), s.radius + s.radius};
    }
    }
    CORA_FAIL(Error::wrong_value, "evaluate: an unknown activation");
}

/// The least-squares line through 20 points of [l, u] with the exact error of the fit; where
/// l == u the tangent (CORA's computeApproxPoly), and the extremes of f - p lie at the ends and at
/// f' = m in closed form (tanh as CORA, sigmoid 2 atanh(sqrt(1 - 4m)), the ReLU at 0 and the ends).
Line aux_line_regression(Activation activation, const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like(), one = l.ones_like();
    if (activation == Activation::identity) return {one, zero, zero};
    constexpr int64_t kPoints = 20; // 10 (order + 1) of CORA's computeApproxPoly
    // the points x_k = l + (u - l) k / 19, and the fit y = m x + t by the normal equations
    std::vector<double> grid(kPoints);
    for (int64_t k = 0; k < kPoints; ++k) grid[std::size_t(k)] = double(k) / double(kPoints - 1);
    const Tensor X = l + (u - l).multiply(Tensor::like(l, grid, {1, kPoints}));
    const Tensor Y = aux_apply(activation, X);
    const Tensor dx = X - X.mean(-1, true), dy = Y - Y.mean(-1, true);
    const Tensor point = u.equal(l);
    const Tensor var = Tensor::where(point, one, dx.multiply(dx).sum(-1, true));
    const Tensor m_fit = dx.multiply(dy).sum(-1, true).divide(var);
    // the tangent of a range of one point: the derivative from the value
    const Tensor fl = aux_apply(activation, l);
    const Tensor m = Tensor::where(point, aux_derivative(activation, fl), m_fit);
    const Tensor t = Tensor::where(point, fl - aux_derivative(activation, fl).multiply(l),
                                   Y.mean(-1, true) - m_fit.multiply(X.mean(-1, true)));
    // the difference between the activation and the line, at the ends and where the slope is m
    Tensor diff_max, diff_min;
    if (activation == Activation::relu) {
        // p - relu(x) of a piecewise linear function: at l, 0 and u
        const Tensor p_l = m.multiply(l) + t, p_u = m.multiply(u) + t - u;
        // outside the crossing the neuron is exact: the line is the activation itself
        const Tensor cross =
            Tensor::where(l.less(zero), Tensor::where(u.greater(zero), one, zero), zero);
        const Tensor lo = p_l.minimum(t).minimum(p_u), hi = p_l.maximum(t).maximum(p_u);
        const Tensor mid = (lo + hi) * 0.5, rad = (hi - lo) * 0.5;
        const Tensor negative = u.less_equal(zero), positive = l.greater_equal(zero);
        const Tensor exact = Tensor::where(cross.greater(zero), zero, one);
        // the line shifts by the middle of p - f, so that |f - p'| <= radius
        const Tensor slope = Tensor::where(negative, zero, Tensor::where(positive, one, m));
        const Tensor offset = Tensor::where(exact, zero, t - mid);
        return {slope, offset, Tensor::where(exact, zero, rad)};
    }
    // tanh and sigmoid: the points where f'(x) = m, +-x*, and the ends
    const bool is_tanh = activation == Activation::tanh;
    Tensor x_star;
    if (is_tanh) {
        x_star = (one - m).maximum(0.0).sqrt().atanh();
    } else {
        x_star = (one - m * 4.0).maximum(0.0).sqrt().atanh() * 2.0;
    }
    Tensor candidates[] = {l, u, x_star, x_star.negative()};
    diff_max = diff_min = Tensor();
    for (const Tensor &c : candidates) {
        // a candidate outside the range, or not finite, counts as the end l
        const Tensor x = Tensor::where(c.greater_equal(l), Tensor::where(c.less_equal(u), c, l), l);
        const Tensor d = aux_apply(activation, x) - (m.multiply(x) + t);
        diff_max = diff_max.defined() ? diff_max.maximum(d) : d;
        diff_min = diff_min.defined() ? diff_min.minimum(d) : d;
    }
    // f - p in [diff_min, diff_max]: the line moves up by the middle, the error is the radius
    const Tensor mid = (diff_max + diff_min) * 0.5, rad = (diff_max - diff_min) * 0.5;
    return {m, Tensor::where(point, t, t + mid), Tensor::where(point, zero, rad)};
}

/// The line of the activation over [l, u] by the method.
Line aux_line(Activation activation, const Tensor &l, const Tensor &u, PolyMethod method) {
    return method == PolyMethod::regression ? aux_line_regression(activation, l, u)
                                            : aux_line_singh(activation, l, u);
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
                bool keep_zero, PolyMethod method) {
    for (std::size_t k = 0; k < layers.size(); ++k) {
        r.c = layers[k].W.matmul(r.c) + layers[k].b;
        r.G = layers[k].W.matmul(r.G);
        r.GI = layers[k].W.matmul(r.GI);
        // the range of every neuron, the line over it, and the set it maps to
        const auto [l, u] = aux_bounds(r);
        const Line line = aux_line(activations[k], l, u, method);
        // CORALean's Layer.activation: the slope map (+ 0.0: -0 times a zero slope is +0), the box
        const Tensor lo = line.offset - line.radius, hi = line.offset + line.radius;
        const Tensor middle = (lo + hi) * 0.5, reach = (hi - middle).maximum(middle - lo);
        r.c = line.slope.multiply(r.c) + 0.0 + middle;
        r.G = line.slope.multiply(r.G) + 0.0;
        r.GI = line.slope.multiply(r.GI) + 0.0;
        // the identity has no error
        if (activations[k] == Activation::identity) continue;
        // the error, as independent generators or as new factors
        const Tensor D = aux_error_columns(reach, keep_zero);
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

Zonotope NeuralNetwork::evaluate(const Zonotope &X, const NnOptions &options) const {
    detail::Entry entry;
    CORA_CHECK(X.dim() == input_dim(), Error::dimension_mismatch, [&] {
        return "evaluate: the set has the dimension " + std::to_string(X.dim()) +
               ", the network takes " + std::to_string(input_dim());
    });
    // the layers on the set, its generators independent; a zero generator stays for an exact neuron
    const Tensor none = X.generators().slice(-1, 0, 0);
    const Rows r = aux_layers(layers_, activations_, aux_rows(X.center(), none, X.generators(), {}),
                              true, true, options.poly_method);
    return {r.c, r.GI};
}

Zonotope NeuralNetwork::evaluate_with_input(const Zonotope &X, const NnOptions &options) const {
    detail::Entry entry;
    // the generators of a zonotope are independent: the polynomial zonotope without dependent ones
    const Tensor none = X.generators().slice(-1, 0, 0);
    NnOptions opt = options;
    opt.add_approx_error_to_gi = true;
    const PolyZonotope joint =
        evaluate_with_input(PolyZonotope(X.center(), none, X.generators(), {}, {}), opt);
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
                   options.add_approx_error_to_gi, false, options.poly_method);
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

PolyZonotope NeuralNetwork::evaluate(const PolyZonotope &X, const NnOptions &options) const {
    detail::Entry entry;
    // the output rows of the pairs
    const PolyZonotope joint = evaluate_with_input(X, options);
    const int64_t n = input_dim(), m = output_dim();
    return {joint.center().slice(0, n, n + m), joint.generators().slice(0, n, n + m),
            joint.independent_generators().slice(0, n, n + m), joint.exponents(), joint.ids()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
