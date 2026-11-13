// evaluate - the outputs of a network at points, and an enclosure of them over a set
//
// An activation y = a(x) over the range [l, u] of a neuron is enclosed by one line and an error,
// y in slope x + offset +- radius (CORA's poly_method "singh" or "regression", order 1): singh is
// the tightest parallelogram [Singh et al., NeurIPS 2018, Thm. 3.1 and 3.2], regression the
// least-squares line through 20 equidistant points of [l, u] shifted by the middle of its
// difference to the activation, whose bounds are those of the ends and of the enclosed points
// where f'(x) = slope (RegressionError::extremum, CORALean's float definition) or the derivative
// bracket's (RegressionError::bracket). The line maps (c, G, GI) exactly; the radius is a new
// independent generator per neuron, or a new factor.
// The order of the operations is that of CORALean's Layer.activation, bit for bit at nearest
// (test_lean_nn_evaluate). evaluate_with_input carries the input rows along. All neurons go at
// once by tensor operations; a layer of order above 1 goes neuron by neuron (priv_poly_layer.cpp).
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
// CORA: nn/@neuralNetwork/getRefinableLayers.m
// CORA authors: Tobias Ladner, Sebastian Sigl, Lukas Koller
// Authors:       Tobias Ladner, Claude Sonnet 5.5
// Written:       02-November-2026
// Last update:   13-November-2026
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_layers.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The line of an activation -----------------------------------------------------------------------

using Line = PrivLine;

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
    if (options.poly_method == PolyMethod::regression)
        return aux_line_regression(activation, l, u, options.regression_error);
    CORA_FAIL(Error::not_supported, "evaluate: this poly_method is for polynomial zonotopes");
}

// The rows of a set -------------------------------------------------------------------------------

using Rows = PrivRows;

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

// The layers of order k, tracked rows -------------------------------------------------------------

/// The tracked exponents and identifiers after count new factors, whose generators were appended.
void aux_track_factors(Rows &r, int64_t count) {
    const std::size_t p = r.ids.size(), c = std::size_t(count);
    for (std::vector<int> &column : r.E) column.resize(p + c, 0);
    if (!r.ids.empty())
        r.last_id = std::max(r.last_id, *std::max_element(r.ids.begin(), r.ids.end()));
    for (std::size_t j = 0; j < c; ++j) {
        std::vector<int> column(p + c, 0);
        column[p + j] = 1;
        r.E.push_back(column);
        r.ids.push_back(++r.last_id);
    }
}

/// The orders of the neurons of layer k (CORA's layer.order): none given is 1, one value is that of
/// every neuron, one per neuron is taken as it is, any other count takes the largest.
std::vector<int> aux_orders(const NnOptions &options, std::size_t k, int64_t neurons) {
    const std::vector<int> given =
        k < options.orders.size() ? options.orders[k] : std::vector<int>{};
    if (given.empty()) return std::vector<int>(std::size_t(neurons), 1);
    if (int64_t(given.size()) == neurons) return given;
    return std::vector<int>(std::size_t(neurons), *std::max_element(given.begin(), given.end()));
}

/// Whether the layer is evaluated neuron by neuron: an order above 1, a limit on the generators,
/// or a poly_method that is not a line.
bool aux_is_poly_layer(Activation activation, const std::vector<int> &orders,
                       const NnOptions &options) {
    if (activation == Activation::identity) return false;
    const bool line_method =
        options.poly_method == PolyMethod::singh || options.poly_method == PolyMethod::regression;
    return !line_method || options.num_generators > 0 ||
           *std::max_element(orders.begin(), orders.end()) > 1;
}

/// The layers of the network on the rows: an exact map, then the line and the error of the
/// activation. keep_zero keeps the (zero) generator of a neuron without error. errors, if given,
/// receives per layer the radius of each neuron's error (empty for the identity).
Rows aux_layers(const std::vector<NeuralNetwork::Layer> &layers,
                const std::vector<Activation> &activations, Rows r, bool errors_in_gi,
                bool keep_zero, const NnOptions &options,
                std::vector<std::vector<double>> *errors = nullptr) {
    for (std::size_t k = 0; k < layers.size(); ++k) {
        r.c = layers[k].W.matmul(r.c) + layers[k].b;
        r.G = layers[k].W.matmul(r.G);
        r.GI = layers[k].W.matmul(r.GI);
        // a layer of order k goes neuron by neuron on tracked rows
        const std::vector<int> orders = aux_orders(options, k, layers[k].W.shape()[0]);
        if (r.track && aux_is_poly_layer(activations[k], orders, options)) {
            std::vector<double> d;
            priv_layer_poly(r, activations[k], orders, errors_in_gi, options, &d);
            if (errors) errors->push_back(d);
            continue;
        }
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
        if (activations[k] == Activation::identity) {
            if (errors) errors->emplace_back();
            continue;
        }
        if (errors) errors->push_back(reach.data());
        if (!options.use_approx_error) continue;
        // the error, as independent generators or as new factors
        const Tensor D = aux_error_columns(reach, keep_zero);
        const int64_t count = D.shape().back();
        if (errors_in_gi) {
            r.GI = Tensor::concat({r.GI, D}, -1);
            if (r.track) r.gi_src.insert(r.gi_src.end(), std::size_t(count), -1);
        } else {
            r.G = Tensor::concat({r.G, D}, -1);
            r.even = Tensor::concat({r.even, D.slice(-2, 0, 1).zeros_like()}, -1);
            r.new_factors += count;
            if (r.track) aux_track_factors(r, count);
        }
    }
    return r;
}

/// The rows of a column set with the dependent generators G (exponents E) and the independent GI.
Rows aux_rows(const Tensor &c, const Tensor &G, const Tensor &GI, const Exponents &E) {
    Rows r;
    r.c = c;
    r.G = G;
    r.GI = GI;
    r.even = priv_even_mask(c, E);
    return r;
}

/// The rows of a polynomial zonotope that track their exponents and factors, for the layers of
/// order k.
Rows aux_tracked_rows(const PolyZonotope &X) {
    Rows r = aux_rows(X.center(), X.generators(), X.independent_generators(), X.exponents());
    r.track = true;
    r.E = X.exponents();
    r.ids = X.ids();
    r.last_id = X.ids().empty() ? 0 : *std::max_element(X.ids().begin(), X.ids().end());
    for (int64_t j = 0; j < X.independent_generators().shape()[1]; ++j) r.gi_src.push_back(j);
    return r;
}

/// The set with its independent generators made dependent, each with a factor of its own.
PolyZonotope aux_restructured(const PolyZonotope &X) {
    const int64_t q = X.independent_generators().shape()[1];
    if (q == 0) return X;
    // the factors of the independent generators follow the old ones
    std::vector<int64_t> ids = X.ids();
    const int64_t first = ids.empty() ? 1 : *std::max_element(ids.begin(), ids.end()) + 1;
    for (int64_t j = 0; j < q; ++j) ids.push_back(first + j);
    Exponents E = X.exponents();
    for (std::vector<int> &column : E) column.resize(ids.size(), 0);
    for (int64_t j = 0; j < q; ++j) {
        std::vector<int> column(ids.size(), 0);
        column[X.ids().size() + std::size_t(j)] = 1;
        E.push_back(column);
    }
    return {X.center(), Tensor::concat({X.generators(), X.independent_generators()}, 1),
            X.independent_generators().slice(-1, 0, 0), E, ids};
}

/// Whether any layer needs the rows tracked (an order above 1, a limit on the generators, or a
/// poly_method of another kind than a line).
bool aux_needs_tracking(const std::vector<NeuralNetwork::Layer> &layers,
                        const std::vector<Activation> &activations, const NnOptions &options) {
    for (std::size_t k = 0; k < layers.size(); ++k)
        if (aux_is_poly_layer(activations[k], aux_orders(options, k, layers[k].W.shape()[0]),
                              options))
            return true;
    return false;
}

// The pairs ---------------------------------------------------------------------------------------

/// The rows of the pairs: the input rows of X beside the output rows of the layers, which depend
/// on the same factors; the generators with the same exponents are one column.
PolyZonotope aux_joint(const PolyZonotope &X, const Rows &r) {
    // the factors of X, then those that only the layers added
    std::vector<int64_t> ids = X.ids();
    for (const int64_t id : r.ids)
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    const auto widen = [&](const Exponents &E, const std::vector<int64_t> &own) {
        Exponents out;
        for (const std::vector<int> &column : E) {
            std::vector<int> wide(ids.size(), 0);
            for (std::size_t i = 0; i < own.size(); ++i)
                wide[std::size_t(std::find(ids.begin(), ids.end(), own[i]) - ids.begin())] =
                    column[i];
            out.push_back(wide);
        }
        return out;
    };
    Exponents E = widen(X.exponents(), X.ids());
    const Exponents Er = widen(r.E, r.ids);
    E.insert(E.end(), Er.begin(), Er.end());
    // the dependent generators side by side, each block zero in the rows of the other
    const int64_t n = X.dim(), m = r.c.shape()[0], hx = X.num_dependent(), hr = r.G.shape()[1];
    const Tensor zero_x = X.center().zeros_like({n, hr}), zero_r = r.c.zeros_like({m, hx});
    const Tensor G = Tensor::concat(
        {Tensor::concat({X.generators(), zero_x}, 1), Tensor::concat({zero_r, r.G}, 1)}, 0);
    // the independent generators: those of X with the image of each, then the ones of the layers
    const int64_t q = X.independent_generators().shape()[1], qr = r.GI.shape()[1];
    std::vector<int64_t> extra;
    std::vector<Tensor> images(std::size_t(q), r.c.zeros_like({m, 1}));
    for (int64_t j = 0; j < qr; ++j) {
        if (r.gi_src[std::size_t(j)] >= 0)
            images[std::size_t(r.gi_src[std::size_t(j)])] = r.GI.slice(-1, j, j + 1);
        else
            extra.push_back(j);
    }
    Tensor GI =
        Tensor::concat({X.independent_generators(), r.c.zeros_like({n, int64_t(extra.size())})}, 1);
    Tensor below = q > 0 ? Tensor::concat(images, 1) : r.c.zeros_like({m, 0});
    if (!extra.empty()) below = Tensor::concat({below, r.GI.take(extra, -1)}, 1);
    GI = Tensor::concat({GI, below}, 0);
    return {Tensor::concat({X.center(), r.c}, 0), G, GI, E, ids};
}

/// The options of a set evaluation: an order is at least 1.
void aux_check_options(const NnOptions &options) {
    for (const std::vector<int> &layer : options.orders)
        for (const int order : layer)
            if (order < 1)
                CORA_FAIL(Error::wrong_value,
                          "evaluate: the order of a neuron must be at least 1, got " +
                              std::to_string(order));
    if (options.num_generators < 0)
        CORA_FAIL(Error::wrong_value, "evaluate: num_generators must not be negative, got " +
                                          std::to_string(options.num_generators));
}

/// The set of the pairs (x, N(x)) of X, collecting the errors of the activations if asked.
PolyZonotope aux_evaluate(const std::vector<NeuralNetwork::Layer> &layers,
                          const std::vector<Activation> &activations, const PolyZonotope &X,
                          const NnOptions &options, std::vector<std::vector<double>> *errors) {
    // the rows that track their factors, if a layer is of order k
    if (aux_needs_tracking(layers, activations, options)) {
        const bool remove_gi = options.remove_gi.value_or(!options.add_approx_error_to_gi);
        const PolyZonotope Xs = remove_gi ? aux_restructured(X) : X;
        const Rows r = aux_layers(layers, activations, aux_tracked_rows(Xs),
                                  options.add_approx_error_to_gi, false, options, errors);
        return aux_joint(Xs, r);
    }
    const Rows r =
        aux_layers(layers, activations,
                   aux_rows(X.center(), X.generators(), X.independent_generators(), X.exponents()),
                   options.add_approx_error_to_gi, false, options, errors);
    // the input rows keep their generators, and are zero in the generators the errors added
    const int64_t n = X.dim(), h = X.num_dependent(), q = X.independent_generators().shape()[1];
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

} // namespace


// ===========================================  MAIN  =========================================== //

// The layers of order k ---------------------------------------------------------------------------

std::pair<Tensor, Tensor> priv_bounds(const PrivRows &r) {
    detail::Entry entry;
    return aux_bounds(r);
}

PrivLine priv_line(Activation activation, const Tensor &l, const Tensor &u,
                   const NnOptions &options) {
    detail::Entry entry;
    return aux_line(activation, l, u, options);
}

Tensor priv_even_mask(const Tensor &like, const Exponents &E) {
    detail::Entry entry;
    std::vector<double> even;
    for (const std::vector<int> &column : E)
        even.push_back(std::all_of(column.begin(), column.end(), [](int e) { return e % 2 == 0; })
                           ? 1.0
                           : 0.0);
    return Tensor::like(like, even, {1, static_cast<int64_t>(even.size())});
}

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
    CORA_CHECK(
        options.poly_method == PolyMethod::singh || options.poly_method == PolyMethod::regression,
        Error::not_supported, [] {
            return std::string("evaluate: a zonotope takes the poly_method singh or regression");
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
    // the options are valid, and the layers run on the set
    aux_check_options(options);
    return aux_evaluate(layers_, activations_, X, options, nullptr);
}

std::vector<std::vector<double>> NeuralNetwork::approx_errors(const PolyZonotope &X,
                                                              const NnOptions &options) const {
    detail::Entry entry;
    CORA_CHECK(X.dim() == input_dim(), Error::dimension_mismatch, [&] {
        return "approx_errors: the set has the dimension " + std::to_string(X.dim()) +
               ", the network takes " + std::to_string(input_dim());
    });
    aux_check_options(options);
    std::vector<std::vector<double>> errors;
    aux_evaluate(layers_, activations_, X, options, &errors);
    return errors;
}

std::vector<int64_t> NeuralNetwork::refinable_layers() const {
    detail::Entry entry;
    // the layers with a nonlinear activation
    std::vector<int64_t> out;
    for (std::size_t k = 0; k < activations_.size(); ++k)
        if (activations_[k] != Activation::identity) out.push_back(int64_t(k));
    return out;
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
