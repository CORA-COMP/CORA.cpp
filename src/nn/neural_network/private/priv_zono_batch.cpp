// priv_zono_batch - batched zonotopes through a dense network: slots, interval center, factors
//
// See priv_zono_batch.h. The order of the operations of a layer is CORALean's Layer.activation
// (as in evaluate.cpp): the bounds as CORALean's interval of a zonotope, the line, the slope map
// with + 0.0 (a zero slope makes -0 +0), and the box of the offset and the error.
//
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch_.m
// CORA: nn/@neuralNetwork/prepareForZonoBatchEval.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateZonotopeBatch.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// CORA authors: Lukas Koller, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "nn/neural_network/private/priv_zono_batch.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Slots and indices -------------------------------------------------------------------------------

/// The slots of a count option (CORA's inf is every one).
int64_t aux_count(double option, int64_t n) {
    return std::isinf(option) ? n : std::min<int64_t>(n, int64_t(option));
}

/// The one-hot rows (..., k, n) of the indices idx (..., k).
Tensor aux_onehot(const Tensor &idx, int64_t n) {
    return idx.expand_dims(-1).equal(idx.arange_like(n).reshape({1, 1, n}));
}

// The activations ---------------------------------------------------------------------------------

/// The activation and its derivative at the points.
Tensor aux_apply(Activation activation, const Tensor &x) {
    switch (activation) {
    case Activation::identity:
        return x;
    case Activation::relu:
        // max(x, 0), the sigmoid and tanh
        return x.maximum(0.0);
    case Activation::sigmoid:
        return x.sigmoid();
    case Activation::tanh:
        return x.tanh();
    }
    CORA_FAIL(Error::wrong_value, "verify: an unknown activation");
}

/// The derivative of the activation at the points (a ReLU's is 0 at 0, as autograd's).
Tensor aux_derivative(Activation activation, const Tensor &x) {
    switch (activation) {
    case Activation::identity:
        return x.ones_like();
    case Activation::relu:
        return x.greater(x.zeros_like());
    case Activation::sigmoid: {
        const Tensor s = x.sigmoid();
        return s.multiply(1.0 - s);
    }
    case Activation::tanh:
        return 1.0 - x.tanh().square();
    }
    CORA_FAIL(Error::wrong_value, "verify: an unknown activation");
}

/// The points where f'(x) = m (CORA's computeExtremePointsBatch): +-2 atanh sqrt(1 - 4 m) for the
/// sigmoid, +-atanh sqrt(1 - m) for tanh, the slope clamped into the range of f'.
Tensor aux_extreme_point(Activation activation, const Tensor &m) {
    const double eps = machine_eps(m.dtype());
    if (activation == Activation::sigmoid)
        return (1.0 - m.minimum(0.25) * 4.0).maximum(0.0).sqrt().atanh() * 2.0;
    return (1.0 - m.maximum(eps).minimum(1.0)).maximum(0.0).sqrt().atanh();
}

// The lines ---------------------------------------------------------------------------------------

/// CORA's poly_method "bounds" (aux_imgEncBatch): the secant slope and the least and largest
/// f(x) - m x over the extreme points inside [l, u] and the ends; that range's middle and radius.
PrivLine aux_bounds_line(Activation activation, const Tensor &l, const Tensor &u) {
    const auto f = [&](const Tensor &x) { return aux_apply(activation, x); };
    const Tensor width = u - l, center = (u + l) * 0.5;
    const double eps = machine_eps(l.dtype());
    const Tensor equal = width.less_equal(center.abs().maximum(1.0) * eps);
    const Tensor one = l.ones_like();
    const Tensor m = Tensor::where(equal, aux_derivative(activation, center),
                                   (f(u) - f(l)).divide(Tensor::where(equal, one, width)));
    const Tensor xu = aux_extreme_point(activation, m);
    // the least and the largest f(x) - m x over the candidates inside [l, u]
    Tensor lo = l.full_like(l.shape(), kInf), hi = l.full_like(l.shape(), -kInf);
    for (const Tensor &x : {xu.negative(), xu, l, u}) {
        const Tensor d = f(x) - m.multiply(x);
        const Tensor inside = x.greater_equal(l).multiply(x.less_equal(u));
        lo = Tensor::where(inside, lo.minimum(d), lo);
        hi = Tensor::where(inside, hi.maximum(d), hi);
    }
    return {m, (hi + lo) * 0.5, (hi - lo) * 0.5};
}

} // namespace


// ===========================================  MAIN  =========================================== //

PrivLine priv_verify_line(Activation activation, const Tensor &l, const Tensor &u,
                          const std::string &poly_method) {
    // a ReLU's line is CORALean's in both methods (the same line in the reals)
    if (activation == Activation::relu || poly_method == "singh")
        return priv_line(activation, l, u, NnOptions{});
    if (poly_method == "bounds") return aux_bounds_line(activation, l, u);
    CORA_FAIL(Error::wrong_field_value,
              "verify: the poly_method '" + poly_method + "' is unknown; use 'bounds' or 'singh'");
}

// The network -------------------------------------------------------------------------------------

ZonoNet::ZonoNet(const NeuralNetwork &net, double num_init_gens, double num_approx_err,
                 std::string poly_method_, std::string approx_error_order_)
    : activations(net.activations()), poly_method(std::move(poly_method_)),
      approx_error_order(std::move(approx_error_order_)) {
    n0 = net.input_dim();
    num_outputs = net.output_dim();
    q0 = aux_count(num_init_gens, n0);
    int64_t id = 1, widest = 0;
    for (std::size_t k = 0; k < net.layers().size(); ++k) {
        const NeuralNetwork::Layer &layer = net.layers()[k];
        // the transposed weights, which the rows multiply from the right
        W.push_back(layer.W.matrix_transpose());
        abs_w.push_back(W.back().abs());
        b.push_back(layer.b);
        // a slot count and the first global id per activation layer
        const int64_t n = layer.W.shape()[0];
        widths.push_back(n);
        const bool active = activations[k] != Activation::identity;
        slots.push_back(active ? aux_count(num_approx_err, n) : 0);
        first_ids.push_back(id);
        if (active) {
            id += n;
            widest = std::max(widest, n);
        }
    }
    // the factors, and whether an error has no slot
    num_neurons = id - 1;
    q = q0;
    for (const int64_t s : slots) q += s;
    interval_center = q0 < n0 || aux_count(num_approx_err, widest) < widest;
}

Tensor ZonoNet::evaluate(const Tensor &x) const {
    Tensor h = x;
    for (std::size_t k = 0; k < W.size(); ++k) h = aux_apply(activations[k], h.matmul(W[k]) + b[k]);
    return h;
}

// Points ------------------------------------------------------------------------------------------

ZonoSensitivity ZonoNet::sensitivity(const Tensor &x) const {
    // the inputs of the activations, then the chain rule from the outputs back
    std::vector<Tensor> pre;
    Tensor h = x;
    for (std::size_t k = 0; k < W.size(); ++k) {
        pre.push_back(h.matmul(W[k]) + b[k]);
        h = aux_apply(activations[k], pre.back());
    }
    const int64_t B = x.shape()[0];
    // the Jacobian of the outputs, from the last layer back
    Tensor S = x.eye_like(num_outputs).broadcast_to({B, num_outputs, num_outputs});
    ZonoSensitivity out;
    out.layer_sum.resize(W.size());
    out.layer_max.resize(W.size());
    for (std::size_t k = W.size(); k-- > 0;) {
        if (activations[k] != Activation::identity) {
            S = S.multiply(aux_derivative(activations[k], pre[k]).expand_dims(1));
            const Tensor a = S.abs();
            out.layer_sum[k] = a.sum(1);
            out.layer_max[k] = a.maximum(numerics::kSensitivityFloor).max(1);
        }
        S = S.matmul(W[k].matrix_transpose());
    }
    out.input = S.abs().maximum(numerics::kSensitivityFloor).max(1);
    out.jacobian = S;
    return out;
}

// Sets --------------------------------------------------------------------------------------------

ZonoInput ZonoNet::input_zonotope(const Tensor &x, const Tensor &r, const Tensor &dims_) const {
    const int64_t B = x.shape()[0];
    // the first q0 dimensions in order, or the given ones
    const Tensor dims =
        dims_.defined() ? dims_
                        : r.arange_like(n0).slice(0, 0, q0).reshape({1, q0}).broadcast_to({B, q0});
    const Tensor values = r.take_along_axis(dims, 1);
    const Tensor hot = aux_onehot(dims, n0);
    ZonoInput in;
    in.c = x;
    in.G = hot.multiply(values.expand_dims(-1));
    in.dims = dims;
    in.values = values;
    // the radius without a generator goes into the interval center
    if (interval_center) in.dr = r.multiply(1.0 - hot.max(1));
    return in;
}

std::pair<Tensor, Tensor> ZonoNet::clamp(int64_t first_id, const Tensor &l, const Tensor &u,
                                         const Tensor &splits) const {
    const int64_t n = l.shape()[1];
    const Tensor ids = (l.arange_like(n) + double(first_id)).reshape({1, 1, n});
    const Tensor hit = splits.abs().expand_dims(-1).equal(ids);
    const Tensor zero = l.zeros_like();
    // -id: the neuron is at least 0; +id: at most 0
    const Tensor lower = hit.multiply(splits.less(splits.zeros_like()).expand_dims(-1)).any(1);
    const Tensor upper = hit.multiply(splits.greater(splits.zeros_like()).expand_dims(-1)).any(1);
    const Tensor lc = Tensor::where(lower, l.maximum(u.minimum(zero)), l);
    const Tensor uc = Tensor::where(upper, u.minimum(lc.maximum(zero)), u);
    return {lc, uc};
}

ZonoPropagation ZonoNet::forward(const Tensor &c_, const Tensor &dr_, const Tensor &G_,
                                 const Tensor &splits, const std::vector<Tensor> *layer_sens,
                                 bool record) const {
    Tensor c = c_, dr = dr_, G = G_;
    ZonoPropagation out;
    const bool split = splits.defined() && splits.shape()[1] > 0;
    for (std::size_t k = 0; k < W.size(); ++k) {
        // the linear layer, exactly
        c = c.matmul(W[k]) + b[k];
        G = G.matmul(W[k]);
        if (dr.defined()) dr = dr.matmul(abs_w[k]);
        if (activations[k] == Activation::identity) continue;
        // the bounds of every neuron (CORALean's interval of a zonotope), then the split clamps
        const Tensor r = G.abs().sum(1);
        const Tensor mid = c + 0.0;
        Tensor l = (mid - 0.0) - r, u = (mid + 0.0) + r;
        if (dr.defined()) {
            l = l - dr;
            u = u + dr;
        }
        const auto [lc, uc] = split ? clamp(first_ids[k], l, u, splits) : std::pair{l, u};
        const PrivLine line = priv_verify_line(activations[k], lc, uc, poly_method);
        // CORALean's Layer.activation: the slope map and the box of the offset and the error
        const Tensor lo = line.offset - line.radius, hi = line.offset + line.radius;
        const Tensor middle = (lo + hi) * 0.5, reach = (hi - middle).maximum(middle - lo);
        ZonoLayerRecord rec;
        if (record) {
            rec.index = int64_t(k);
            rec.first_id = first_ids[k];
            rec.c_in = c;
            rec.dr_in = dr;
            rec.G_in = G;
            rec.l = l;
            rec.u = u;
            rec.er = reach;
        }
        c = line.slope.multiply(c) + 0.0 + middle;
        G = line.slope.expand_dims(1).multiply(G) + 0.0;
        if (dr.defined()) dr = line.slope.multiply(dr);
        // the errors: a generator per slot neuron, the rest into the interval center
        const int64_t B = reach.shape()[0], n = reach.shape()[1], dn = slots[k];
        if (dn == 0) {
            dr = dr.defined() ? dr + reach : reach;
        } else {
            Tensor neurons;
            if (dn == n || approx_error_order == "sequential") {
                neurons =
                    reach.arange_like(n).slice(0, 0, dn).reshape({1, dn}).broadcast_to({B, dn});
            } else if (approx_error_order == "length" ||
                       approx_error_order == "sensitivity*length") {
                const bool sens = approx_error_order == "sensitivity*length" && layer_sens &&
                                  (*layer_sens)[k].defined();
                const Tensor score = sens ? reach.multiply((*layer_sens)[k]) : reach;
                neurons = score.detach().argsort(1, true).slice(1, 0, dn);
            } else {
                CORA_FAIL(Error::wrong_field_value,
                          "verify: the approx_error_order '" + approx_error_order +
                              "' is unknown; use 'length', 'sensitivity*length' or 'sequential'");
            }
            // the generators of the slot neurons, the rest of the errors in dr
            const Tensor values = reach.take_along_axis(neurons, 1);
            const Tensor hot = aux_onehot(neurons, n);
            G = Tensor::concat({G, hot.multiply(values.expand_dims(-1))}, 1);
            if (dn < n) {
                const Tensor rest = reach.multiply(1.0 - hot.max(1));
                dr = dr.defined() ? dr + rest : rest;
            }
            rec.slot_neurons = neurons;
            rec.slot_values = values;
        }
        if (record) {
            // the layer's output set, for the ReLU rows
            rec.c_out = c;
            rec.dr_out = dr;
            rec.G_out = G;
            out.layers.push_back(rec);
        }
    }
    out.c = c;
    out.dr = dr;
    out.G = G;
    return out;
}

// Gradients ---------------------------------------------------------------------------------------

Tensor priv_frad_gradients(ZonoPropagation &out, const ZonoInput &in) {
    // the squared length of every generator row and the F-radius of the output
    const Tensor rows = out.G.square().sum(-1);
    const Tensor frad = (rows.sum(-1, true) + numerics::kFradSmoothing).sqrt();
    const auto gradient = [&](int64_t start, const Tensor &values) {
        const int64_t count = values.shape()[1];
        const Tensor part = rows.slice(1, start, start + count);
        const Tensor zero = values.zeros_like();
        return Tensor::where(values.equal(zero), zero, part.divide(values.multiply(frad)));
    };
    const int64_t q0 = in.values.shape()[1];
    int64_t start = q0;
    for (ZonoLayerRecord &rec : out.layers) {
        const int64_t n = rec.l.shape()[1];
        if (!rec.slot_values.defined()) {
            rec.grad = rec.l.zeros_like();
            continue;
        }
        // the gradient of each slot, scattered to its neuron
        const Tensor g = gradient(start, rec.slot_values);
        rec.grad = aux_onehot(rec.slot_neurons, n).multiply(g.expand_dims(-1)).sum(1);
        start += rec.slot_values.shape()[1];
    }
    return gradient(0, in.values);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
