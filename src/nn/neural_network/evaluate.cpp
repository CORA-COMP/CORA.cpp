// evaluate - the outputs of a network at points, and an enclosure of them over a set
//
// An activation y = a(x) over the range [l, u] of a neuron is enclosed by one line and an error,
// y in slope x + offset +- radius (CORA's poly_method "singh" or "regression", order 1): singh is
// the tightest parallelogram [Singh et al., NeurIPS 2018, Thm. 3.1 and 3.2], regression the
// least-squares line through 20 equidistant points of [l, u] shifted by the middle of its
// difference to the activation, whose bounds are those of the ends and of the enclosed points
// where f'(x) = slope (RegressionError::extremum, CORALean's float definition) or the derivative
// bracket's (RegressionError::bracket). The line maps (c, G, GI) exactly, so the factors and the
// exponents never change; the radius is a new independent generator per neuron, or a new factor.
// The order of the operations is that of CORALean's Layer.activation, bit for bit at nearest for
// all three (test_lean_nn_evaluate): the slope, the box [offset - radius, offset + radius] as a
// zonotope whose middle is added to the center and whose reach is the generator.
// evaluate_with_input carries the input rows along. All neurons go at once by tensor operations.
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
// Last update:   13-November-2026
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

// The regression line, CORALean's ----------------------------------------------------------------

/// The least-squares slope through 20 points of [l, u] (CORALean's regressionSlope, bit for bit at
/// nearest): the step h, the points x_i = l + i h, the middles y_i of the enclosure, the weights
/// w_i = 2 i - 19, their sum with y_i from the right, the quotient; zero where u <= l.
Tensor aux_regression_slope(const Tensor &l, const Tensor &u) {
    constexpr int64_t kPoints = 20;
    const Tensor zero = l.zeros_like(), one = l.ones_like();
    std::vector<double> index(kPoints), weight(kPoints);
    for (int64_t i = 0; i < kPoints; ++i) {
        index[std::size_t(i)] = double(i);
        weight[std::size_t(i)] = double(2 * i - (kPoints - 1));
    }
    const Tensor h = (u - l).divide(one * double(kPoints - 1));
    const Tensor X = l + h.multiply(Tensor::like(l, index, {1, kPoints}));
    const Tensor Y = (aux_sigmoid_lo(X) + aux_sigmoid_hi(X)) * 0.5;
    const Tensor T = Y.multiply(Tensor::like(l, weight, {1, kPoints}));
    Tensor S = zero;
    for (int64_t i = kPoints - 1; i >= 0; --i) S = T.slice(-1, i, i + 1) + S;
    const Tensor empty = u.less_equal(l);
    const Tensor denominator = Tensor::where(empty, one, (u - l) * 70.0);
    return Tensor::where(empty, zero, S.divide(denominator));
}

/// The sigmoid's line at the clamped regression slope with the bracket's error (CORALean's
/// sigmoidRegressionRelaxation, bit for bit at nearest); the slope-zero line where it is tighter.
Line aux_line_sigmoid_bracket(const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like();
    const Tensor width = u - l;
    const Tensor m = aux_regression_slope(l, u);
    // the derivative bracket and the clamped slope
    const Tensor f_lo_l = aux_sigmoid_lo(l), f_hi_l = aux_sigmoid_hi(l);
    const Tensor d_lo = f_lo_l.maximum(zero).multiply(aux_sigmoid_lo(u.negative()).maximum(zero));
    const Tensor d_hi = aux_sigmoid_hi(u).multiply(aux_sigmoid_hi(l.negative()));
    const Tensor lambda = d_lo.maximum(d_hi.minimum(m));
    // the error of the bracket
    const Tensor lo = (f_lo_l - lambda.multiply(l)) + (d_lo - lambda).multiply(width);
    const Tensor hi = (f_hi_l - lambda.multiply(l)) + (d_hi - lambda).multiply(width);
    const Tensor offset = (lo + hi) * 0.5;
    const Tensor error = (hi - offset).maximum(offset - lo);
    // the slope-zero line
    const Tensor f_hi_u = aux_sigmoid_hi(u);
    const Tensor offset0 = (f_lo_l + f_hi_u) * 0.5;
    const Tensor error0 = (f_hi_u - offset0).maximum(offset0 - f_lo_l);
    const Tensor use = error.less_equal(error0);
    return {Tensor::where(use, lambda, zero), Tensor::where(use, offset, offset0),
            Tensor::where(use, error, error0)};
}

/// The sigmoid's line at the regression slope with the error of the extremum of sigma(t) - m t
/// (CORALean's sigmoidRegressionExactRelaxation, bit for bit at nearest): the bounds over the ends
/// and the boxes of the stationary points +-2 artanh sqrt(1 - 4 m) in [l, u].
Line aux_line_sigmoid_extremum(const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like(), one = l.ones_like();
    const Tensor m = aux_regression_slope(l, u);
    // bounds of sigma(t) - m t on [a, b]: sigma at one end, m t at the other
    const auto upper = [&](const Tensor &a, const Tensor &b) {
        return aux_sigmoid_hi(b) - (m.multiply(a)).minimum(m.multiply(b));
    };
    const auto lower = [&](const Tensor &a, const Tensor &b) {
        return aux_sigmoid_lo(a) - (m.multiply(a)).maximum(m.multiply(b));
    };
    Tensor hi = upper(l, l).maximum(upper(u, u));
    Tensor lo = lower(l, l).minimum(lower(u, u));
    // the stationary points exist for 0 < m <= 1/4 only; 4 m is the repeated sum m + (m + (m + m))
    const Tensor four_m = m + (m + (m + m));
    const Tensor exists =
        Tensor::where(m.less_equal(zero), zero, Tensor::where(one.less(four_m), zero, one));
    // artanh of the root from below (capped below one) and above, rounded the machine's way
    constexpr double kCap = 1.0 - 0x1p-52;
    // sqrtDown is the radicand over sqrtUp (the machine's root), zero where that is not positive
    const Tensor radicand = one - four_m;
    const Tensor root = radicand.sqrt();
    const Tensor root_down =
        Tensor::where(zero.less(root), radicand.maximum(0.0).divide(root), zero);
    const Tensor s_lo = root_down.maximum(0.0).minimum(kCap);
    const Tensor artanh_lo = ((one + s_lo).divide(one - s_lo)).maximum(1.0).log() * 0.5;
    const Tensor t_lo = Tensor::where(zero.less(one + s_lo), artanh_lo + artanh_lo, zero);
    const Tensor artanh_hi = ((one + root).divide(one - root)).log() * 0.5;
    const Tensor t_hi = artanh_hi + artanh_hi;
    const Tensor hi_exists = zero.less(one - root);
    // the part of [a, b] that holds t0 (the two ends in either order), folded into hi and lo
    const auto fold = [&](const Tensor &a, const Tensor &b, bool negate) {
        const Tensor lowest = t_lo.minimum(t_hi), highest = t_lo.maximum(t_hi);
        const Tensor from = Tensor::where(hi_exists, a.maximum(lowest), a.maximum(t_lo));
        const Tensor to = Tensor::where(hi_exists, b.minimum(highest), b);
        const Tensor take = Tensor::where(exists.greater(zero), from.less_equal(to), zero);
        const Tensor box_lo = negate ? to.negative() : from, box_hi = negate ? from.negative() : to;
        hi = Tensor::where(take, hi.maximum(upper(box_lo, box_hi)), hi);
        lo = Tensor::where(take, lo.minimum(lower(box_lo, box_hi)), lo);
    };
    fold(l, u, false);
    fold(u.negative(), l.negative(), true);
    const Tensor offset = (lo + hi) * 0.5;
    return {m, offset, (hi - offset).maximum(offset - lo)};
}

/// The line over [l, u] by the regression slope with CORALean's error, the extremum's
/// (Layer.sigmoidRegressionExact, Layer.tanhRegressionExact) or the bracket's
/// (Layer.sigmoidRegression, Layer.tanhRegression); the ReLU's line is Singh's.
Line aux_line_regression(Activation activation, const Tensor &l, const Tensor &u,
                         RegressionError error) {
    const auto sigmoid = [&](const Tensor &a, const Tensor &b) {
        return error == RegressionError::extremum ? aux_line_sigmoid_extremum(a, b)
                                                  : aux_line_sigmoid_bracket(a, b);
    };
    switch (activation) {
    case Activation::sigmoid:
        return sigmoid(l, u);
    case Activation::tanh: {
        const Line s = sigmoid(l + l, u + u);
        const Tensor four = (s.slope + s.slope) + (s.slope + s.slope);
        return {four, (s.offset + s.offset) + (-1.0), s.radius + s.radius};
    }
    default:
        return aux_line_singh(activation, l, u);
    }
}

/// The line of the activation over [l, u] by the method.
Line aux_line(Activation activation, const Tensor &l, const Tensor &u, const NnOptions &options) {
    if (options.poly_method == PolyMethod::singh) return aux_line_singh(activation, l, u);
    return aux_line_regression(activation, l, u, options.regression_error);
}

// The rows of a set -------------------------------------------------------------------------------

/// The rows of a set that goes through the layers: the center, the dependent generators with the
/// mask of those with only even exponents (1, h), the independent generators, and the factors that
/// the errors added (their generators are the last columns of G).
struct Rows {
    Tensor c, G, GI, even;
    int64_t new_factors = 0;
};

// The layers on a set -----------------------------------------------------------------------------

/// The bounds of every row (CORA's bound_approx), summed as CORALean's PolyZonotope.interval: the
/// center, then the reach of G, then that of GI, each rounded on its own (a column of only even
/// exponents contributes half its reach, shifted by half the column).
std::pair<Tensor, Tensor> aux_bounds(const Rows &r) {
    const Tensor odd = r.even.ones_like() - r.even;
    const Tensor shift = r.G.multiply(r.even).sum(-1, true) * 0.5;
    const Tensor reach_G =
        r.G.abs().multiply(r.even).sum(-1, true) * 0.5 + r.G.abs().multiply(odd).sum(-1, true);
    const Tensor reach_GI = r.GI.abs().sum(-1, true);
    const Tensor mid = r.c + shift;
    return {(mid - reach_G) - reach_GI, (mid + reach_G) + reach_GI};
}

/// The columns of diag(radius) with a positive radius (all of them with keep_zero).
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
                bool keep_zero, const NnOptions &options) {
    for (std::size_t k = 0; k < layers.size(); ++k) {
        r.c = layers[k].W.matmul(r.c) + layers[k].b;
        r.G = layers[k].W.matmul(r.G);
        r.GI = layers[k].W.matmul(r.GI);
        // the range of every neuron, the line over it, and the set it maps to
        const auto [l, u] = aux_bounds(r);
        const Line line = aux_line(activations[k], l, u, options);
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
                              true, true, options);
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
                   options.add_approx_error_to_gi, false, options);
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
