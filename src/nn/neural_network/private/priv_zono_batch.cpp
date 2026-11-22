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

// The reverse pass ---------------------------------------------------------------------------

/// The neurons (B, n) of the layer from first_id that an earlier split held at least 0 (-id) and
/// at most 0 (+id).
std::pair<Tensor, Tensor> aux_split_masks(int64_t first_id, const Tensor &l, const Tensor &splits) {
    const int64_t n = l.shape()[1];
    const Tensor ids = (l.arange_like(n) + double(first_id)).reshape({1, 1, n});
    const Tensor hit = splits.abs().expand_dims(-1).equal(ids);
    const Tensor lower = hit.multiply(splits.less(splits.zeros_like()).expand_dims(-1)).any(1);
    const Tensor upper = hit.multiply(splits.greater(splits.zeros_like()).expand_dims(-1)).any(1);
    return {lower, upper};
}

/// The clamped bounds and their partial derivatives: lc = lc_l l + lc_u u, uc = uc_l l + uc_u u
/// locally (the branch each min and max takes).
struct ClampJacobian {
    Tensor lc, uc, lc_l, lc_u, uc_l, uc_u;
};

ClampJacobian aux_clamp_jacobian(int64_t first_id, const Tensor &l, const Tensor &u,
                                 const Tensor &splits) {
    const Tensor zero = l.zeros_like(), one = l.ones_like();
    if (!splits.defined() || splits.shape()[1] == 0) return {l, u, one, zero, zero, one};
    const auto [lower, upper] = aux_split_masks(first_id, l, splits);
    // lc = max(l, min(u, 0)) where held at least 0
    const Tensor inner = u.minimum(zero);
    const Tensor take_l = l.greater_equal(inner);
    const Tensor lc = Tensor::where(lower, l.maximum(inner), l);
    const Tensor lc_l = Tensor::where(lower, take_l, one);
    const Tensor lc_u = Tensor::where(lower, (1.0 - take_l).multiply(u.less(zero)), zero);
    // uc = min(u, max(lc, 0)) where held at most 0
    const Tensor inner2 = lc.maximum(zero);
    const Tensor take_u = u.less_equal(inner2);
    const Tensor uc = Tensor::where(upper, u.minimum(inner2), u);
    const Tensor via_lc = Tensor::where(upper, (1.0 - take_u).multiply(lc.greater(zero)), zero);
    return {lc,
            uc,
            lc_l,
            lc_u,
            via_lc.multiply(lc_l),
            Tensor::where(upper, take_u, one) + via_lc.multiply(lc_u)};
}

/// The partial derivatives of a line's slope m, offset o and radius r in l and u.
struct LineJacobian {
    Tensor m_l, m_u, o_l, o_u, r_l, r_u;
};

/// A ReLU's line (both methods): exact outside the crossing, else m = u / (u - l) and the offset
/// and radius -u l / (2 (u - l)).
LineJacobian aux_relu_jacobian(const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like();
    const Tensor cross = u.greater(zero).multiply(l.less(zero));
    const Tensor w2 = (u - l).maximum(numerics::kTiny).square();
    const Tensor m_l = Tensor::where(cross, u.divide(w2), zero);
    const Tensor m_u = Tensor::where(cross, l.negative().divide(w2), zero);
    const Tensor o_l = Tensor::where(cross, u.square().divide(w2) * -0.5, zero);
    const Tensor o_u = Tensor::where(cross, l.square().divide(w2) * 0.5, zero);
    return {m_l, m_u, o_l, o_u, o_l, o_u};
}

/// The first and second derivative of the sigmoid or tanh.
std::pair<Tensor, Tensor> aux_derivatives(Activation activation, const Tensor &x) {
    if (activation == Activation::sigmoid) {
        const Tensor s = x.sigmoid(), d = s.multiply(1.0 - s);
        return {d, d.multiply(1.0 - s * 2.0)};
    }
    const Tensor t = x.tanh(), d = 1.0 - t.square();
    return {d, d.multiply(t) * -2.0};
}

/// CORA's "bounds" line: m the secant (f'(c) on a point), lo and hi the least and largest f(x) - m
/// x over the candidates; at an extreme point d/dm = -x (f'(x) = m), at an end also f'(x) - m.
LineJacobian aux_bounds_jacobian(Activation activation, const Tensor &l, const Tensor &u) {
    const auto f = [&](const Tensor &x) { return aux_apply(activation, x); };
    const Tensor width = u - l, center = (u + l) * 0.5, zero = l.zeros_like(), one = l.ones_like();
    const double eps = machine_eps(l.dtype());
    const Tensor equal = width.less_equal(center.abs().maximum(1.0) * eps);
    const Tensor w = Tensor::where(equal, one, width);
    const auto [dl, ddl] = aux_derivatives(activation, l);
    const auto [du, ddu] = aux_derivatives(activation, u);
    const Tensor half_dd = aux_derivatives(activation, center).second * 0.5;
    const Tensor m =
        Tensor::where(equal, aux_derivatives(activation, center).first, (f(u) - f(l)).divide(w));
    const Tensor m_l = Tensor::where(equal, half_dd, (m - dl).divide(w));
    const Tensor m_u = Tensor::where(equal, half_dd, (du - m).divide(w));
    const Tensor xu = aux_extreme_point(activation, m);
    // the least and the largest candidate, with its x and its direct derivatives
    Tensor lo = l.full_like(l.shape(), kInf), hi = l.full_like(l.shape(), -kInf);
    Tensor lo_x = zero, hi_x = zero, lo_dl = zero, lo_du = zero, hi_dl = zero, hi_du = zero;
    const Tensor cands[] = {xu.negative(), xu, l, u};
    for (int i = 0; i < 4; ++i) {
        const Tensor &x = cands[i];
        const Tensor d = f(x) - m.multiply(x);
        const Tensor inside = x.greater_equal(l).multiply(x.less_equal(u));
        const Tensor dir_l = i == 2 ? dl - m : zero, dir_u = i == 3 ? du - m : zero;
        const Tensor below = inside.multiply(d.less(lo)), above = inside.multiply(d.greater(hi));
        lo = Tensor::where(below, d, lo);
        lo_x = Tensor::where(below, x, lo_x);
        lo_dl = Tensor::where(below, dir_l, lo_dl);
        lo_du = Tensor::where(below, dir_u, lo_du);
        hi = Tensor::where(above, d, hi);
        hi_x = Tensor::where(above, x, hi_x);
        hi_dl = Tensor::where(above, dir_l, hi_dl);
        hi_du = Tensor::where(above, dir_u, hi_du);
    }
    const Tensor lo_l = lo_dl - lo_x.multiply(m_l), lo_u = lo_du - lo_x.multiply(m_u);
    const Tensor hi_l = hi_dl - hi_x.multiply(m_l), hi_u = hi_du - hi_x.multiply(m_u);
    return {m_l,
            m_u,
            (hi_l + lo_l) * 0.5,
            (hi_u + lo_u) * 0.5,
            (hi_l - lo_l) * 0.5,
            (hi_u - lo_u) * 0.5};
}

/// The sigmoid's Singh line: m = min(f'(l), f'(u)), lo = f(l) - m l, hi = f(u) - m u.
LineJacobian aux_sigmoid_singh_jacobian(const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like();
    const auto [dl, ddl] = aux_derivatives(Activation::sigmoid, l);
    const auto [du, ddu] = aux_derivatives(Activation::sigmoid, u);
    const Tensor use_l = dl.less_equal(du);
    const Tensor m = dl.minimum(du);
    const Tensor m_l = Tensor::where(use_l, ddl, zero), m_u = Tensor::where(use_l, zero, ddu);
    const Tensor lo_l = dl - m - l.multiply(m_l), lo_u = l.multiply(m_u).negative();
    const Tensor hi_l = u.multiply(m_l).negative(), hi_u = du - m - u.multiply(m_u);
    const Tensor sign =
        Tensor::where((u.sigmoid() - m.multiply(u)).greater_equal(l.sigmoid() - m.multiply(l)),
                      l.ones_like(), l.full_like(l.shape(), -1.0));
    return {m_l,
            m_u,
            (hi_l + lo_l) * 0.5,
            (hi_u + lo_u) * 0.5,
            sign.multiply(hi_l - lo_l) * 0.5,
            sign.multiply(hi_u - lo_u) * 0.5};
}

/// The partial derivatives of priv_verify_line in l and u.
LineJacobian aux_line_jacobian(Activation activation, const Tensor &l, const Tensor &u,
                               const std::string &poly_method) {
    if (activation == Activation::relu) return aux_relu_jacobian(l, u);
    if (poly_method == "bounds") return aux_bounds_jacobian(activation, l, u);
    if (activation == Activation::sigmoid) return aux_sigmoid_singh_jacobian(l, u);
    // tanh: the sigmoid's line at the doubled range, the slope times 4, the rest times 2
    const LineJacobian s = aux_sigmoid_singh_jacobian(l + l, u + u);
    return {s.m_l * 8.0, s.m_u * 8.0, s.o_l * 4.0, s.o_u * 4.0, s.r_l * 4.0, s.r_u * 4.0};
}

/// The gradients with the slopes held: the F-radius is linear in each generator's value.
Tensor aux_held_gradients(ZonoPropagation &out, const ZonoInput &in) {
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
        WT.push_back(layer.W);
        abs_wT.push_back(layer.W.abs());
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
    const auto [lower, upper] = aux_split_masks(first_id, l, splits);
    const Tensor zero = l.zeros_like();
    // -id: the neuron is at least 0; +id: at most 0
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

Tensor priv_frad_gradients(ZonoPropagation &out, const ZonoInput &in, const ZonoNet &net,
                           const Tensor &splits, bool exact) {
    if (!exact) return aux_held_gradients(out, in);
    // d F / d G at the output, then back through every layer
    const Tensor frad = (out.G.square().sum(-1).sum(-1, true) + numerics::kFradSmoothing).sqrt();
    Tensor gG = out.G.divide(frad.expand_dims(-1));
    Tensor gc = out.c.zeros_like(), gdr = out.dr.defined() ? out.dr.zeros_like() : Tensor();
    std::size_t next = out.layers.size();
    for (std::size_t k = net.W.size(); k-- > 0;) {
        if (net.activations[k] != Activation::identity) {
            ZonoLayerRecord &rec = out.layers[--next];
            const Tensor &c1 = rec.c_in, &G1 = rec.G_in, &dr1 = rec.dr_in;
            const int64_t n = rec.l.shape()[1], qk = G1.shape()[1];
            // the errors: the slot generators' values and the rest in dr are the reach
            Tensor g_reach;
            if (rec.slot_values.defined()) {
                const Tensor &neurons = rec.slot_neurons;
                const int64_t dn = neurons.shape()[1];
                const Tensor hot = aux_onehot(neurons, n);
                const Tensor gv = gG.slice(1, qk, qk + dn)
                                      .take_along_axis(neurons.expand_dims(-1), 2)
                                      .squeeze(-1);
                rec.grad = hot.multiply(gv.expand_dims(-1)).sum(1);
                g_reach = rec.grad;
                if (dn < n && gdr.defined()) g_reach = g_reach + gdr.multiply(1.0 - hot.max(1));
                gG = gG.slice(1, 0, qk);
            } else {
                rec.grad = rec.l.zeros_like();
                g_reach = gdr.defined() ? gdr : rec.l.zeros_like();
            }
            // the line at the clamped bounds; c2 = m c1 + o, G2 = m G1, dr2 = m dr1 + reach
            const ClampJacobian cl = aux_clamp_jacobian(rec.first_id, rec.l, rec.u, splits);
            const Tensor m =
                priv_verify_line(net.activations[k], cl.lc, cl.uc, net.poly_method).slope;
            const LineJacobian J =
                aux_line_jacobian(net.activations[k], cl.lc, cl.uc, net.poly_method);
            Tensor g_m = gc.multiply(c1) + gG.multiply(G1).sum(1);
            if (dr1.defined()) g_m = g_m + gdr.multiply(dr1);
            const Tensor g_lc = g_m.multiply(J.m_l) + gc.multiply(J.o_l) + g_reach.multiply(J.r_l);
            const Tensor g_uc = g_m.multiply(J.m_u) + gc.multiply(J.o_u) + g_reach.multiply(J.r_u);
            const Tensor g_l = g_lc.multiply(cl.lc_l) + g_uc.multiply(cl.uc_l);
            const Tensor g_u = g_lc.multiply(cl.lc_u) + g_uc.multiply(cl.uc_u);
            // l = c1 - sum |G1| - dr1, u = c1 + sum |G1| + dr1
            const Tensor g_rad = g_u - g_l;
            gc = gc.multiply(m) + g_l + g_u;
            gG = m.expand_dims(1).multiply(gG) + G1.sign().multiply(g_rad.expand_dims(1));
            gdr = dr1.defined() ? gdr.multiply(m) + g_rad : Tensor();
        }
        // the linear layer; the centers and radii of the first layer's input are not needed
        gG = gG.matmul(net.WT[k]);
        if (k == 0) break;
        gc = gc.matmul(net.WT[k]);
        if (gdr.defined()) gdr = gdr.matmul(net.abs_wT[k]);
    }
    return gG.take_along_axis(in.dims.expand_dims(-1), 2).squeeze(-1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
