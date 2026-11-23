// priv_zono_batch - batched zonotopes through a dense network: slots, interval center, factors
//
// See priv_zono_batch.h. The order of the operations of a layer is CORALean's Layer.activation
// (as in evaluate.cpp): the bounds as CORALean's interval of a zonotope, the line, the slope map
// with + 0.0 (a zero slope makes -0 +0), and the box of the offset and the error.
// Without a record the error rows v_i e_j of a layer wait for the next linear map, which makes
// them v_i times the row j of W: no (B, dn, n) block and no product with it.
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Set-based training for neural network verification",
//        Transactions on Machine Learning Research, 2025.
//    [2] G. Singh, T. Gehr, M. Mirman, M. Puschel, M. Vechev. "Fast and effective robustness
//        certification", NeurIPS 2018.
//    [3] N. Kochdumper, C. Schilling, M. Althoff, S. Bak. "Open- and closed-loop neural network
//        verification using polynomial zonotopes", NFM 2023 (arXiv:2207.02715).
//
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch_.m
// CORA: nn/@neuralNetwork/prepareForZonoBatchEval.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateZonotopeBatch.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// CORA authors: Lukas Koller, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   23-November-2026
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

/// The rows G (B, q, n) times M (n, m) as one product of a (B q, n) matrix, not B of them.
Tensor aux_rows_times(const Tensor &G, const Tensor &M) {
    const int64_t B = G.shape()[0], q = G.shape()[1];
    return G.reshape({B * q, G.shape()[2]}).matmul(M).reshape({B, q, M.shape()[1]});
}

/// G (B, q, n) with zero rows up to `rows` (a graph's earlier source has fewer factors).
Tensor aux_rows_to(const Tensor &G, int64_t rows) {
    const std::vector<int64_t> s = G.shape();
    if (s[1] >= rows) return G;
    return Tensor::concat({G, G.zeros_like({s[0], rows - s[1], s[2]})}, 1);
}

/// G plus the rows R (B, r, n), r <= q: R's factors are G's first r.
Tensor aux_add_rows(const Tensor &G, const Tensor &R) {
    const int64_t q = G.shape()[1], r = R.shape()[1];
    if (r == q) return G + R;
    return Tensor::concat({G.slice(1, 0, r) + R, G.slice(1, r, q)}, 1);
}

/// The one-hot rows (..., k, n) of the indices idx (..., k).
Tensor aux_onehot(const Tensor &idx, int64_t n) {
    return idx.expand_dims(-1).equal(idx.arange_like(n).reshape({1, 1, n}));
}

// The activations ---------------------------------------------------------------------------------

/// The activation at the points (evaluate.cpp's).
Tensor aux_apply(Activation activation, const Tensor &x) { return priv_apply(activation, x); }

/// The derivative of the activation at the points as a slope (priv_derivative).
Tensor aux_derivative(Activation activation, const Tensor &x) {
    return priv_derivative(activation, x);
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

// The reverse pass --------------------------------------------------------------------------------

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
    // the slope and its derivatives in l and u
    const Tensor dl = aux_derivatives(activation, l).first;
    const Tensor du = aux_derivatives(activation, u).first;
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
        // f(x) - m x at the candidate, kept where it is inside [l, u] and the least or largest
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
    // the chain through m, then the offset and radius are the middle and half of [lo, hi]
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

/// The line of x^2 (priv_square_line): m = l + u, o = -(l u + m^2 / 4) / 2, r = (u - l)^2 / 8.
LineJacobian aux_square_jacobian(const Tensor &l, const Tensor &u) {
    const Tensor one = l.ones_like(), mid = (l + u) * 0.5, quarter = (u - l) * 0.25;
    return {one, one, (u + mid) * -0.5, (l + mid) * -0.5, quarter.negative(), quarter};
}

/// The partial derivatives of priv_verify_line in l and u.
LineJacobian aux_line_jacobian(Activation activation, const Tensor &l, const Tensor &u,
                               const std::string &poly_method) {
    if (activation == Activation::relu) return aux_relu_jacobian(l, u);
    if (activation == Activation::square) return aux_square_jacobian(l, u);
    if (activation == Activation::sign || activation == Activation::floor ||
        activation == Activation::exp || activation == Activation::reciprocal ||
        activation == Activation::product || activation == Activation::cube ||
        activation == Activation::inverse || activation == Activation::sin ||
        activation == Activation::cos || activation == Activation::softplus ||
        activation == Activation::root || activation == Activation::inv_sqrt_root) {
        // steps' lines (sign, floor) do not move with the bounds; the others' here are held
        const Tensor zero = l.zeros_like();
        return {zero, zero, zero, zero, zero, zero};
    }
    if (poly_method == "bounds") return aux_bounds_jacobian(activation, l, u);
    if (activation == Activation::sigmoid) return aux_sigmoid_singh_jacobian(l, u);
    // tanh: the sigmoid's line at the doubled range, the slope times 4, the rest times 2
    const LineJacobian s = aux_sigmoid_singh_jacobian(l + l, u + u);
    return {s.m_l * 8.0, s.m_u * 8.0, s.o_l * 4.0, s.o_u * 4.0, s.r_l * 4.0, s.r_u * 4.0};
}

/// The line of layer k over its input set (c, G, dr) at the clamped bounds [lc, uc]: a product's
/// from the set itself, the others' from the bounds.
PrivLine aux_layer_line(const ZonoNet &net, std::size_t k, const Tensor &c, const Tensor &G,
                        const Tensor &dr, const Tensor &lc, const Tensor &uc) {
    if (net.activations[k] == Activation::product) return priv_product_line(c, G, dr);
    return priv_verify_line(net.activations[k], lc, uc, net.poly_method);
}

// The boxes of the sources ------------------------------------------------------------------------

/// Whether layer k's bounds take the box of the layer it reads alone: a reciprocal (a softmax's
/// sums of exps), 1 / x (sums of cubes) or x^3 (of ReLUs).
bool aux_reads_range(const ZonoNet &net, std::size_t k) {
    const Activation a = net.activations[k];
    if (a != Activation::reciprocal && a != Activation::inverse && a != Activation::cube)
        return false;
    const int64_t j = net.sources[k];
    if (j < 0 || !net.residuals[k].empty()) return false;
    const Activation s = net.activations[std::size_t(j)];
    return s == Activation::exp || s == Activation::cube || s == Activation::relu;
}

/// The output box of a rising activation layer (exp, x^3, ReLU) at its clamped bounds [lc, uc].
std::pair<Tensor, Tensor> aux_range(Activation a, const Tensor &lc, const Tensor &uc) {
    if (a == Activation::exp) return {lc.exp(), uc.exp()};
    if (a == Activation::cube) return {lc.square().multiply(lc), uc.square().multiply(uc)};
    return {lc.maximum(0.0), uc.maximum(0.0)};
}

/// The box [lo, hi] (B, n_in) through layer k's map: its center's image +- |W| times its radius,
/// widened by the rounding (n_in + 4 ulps of the largest term).
std::pair<Tensor, Tensor> aux_box_image(const ZonoNet &net, std::size_t k, const Tensor &lo,
                                        const Tensor &hi) {
    const Tensor mid = (lo + hi) * 0.5, rad = (hi - lo) * 0.5;
    const Tensor c = net.map(k, mid) + net.b[k], r = net.map_abs(k, rad);
    const double ulps = double(net.maps[k].inputs() + 4) * machine_eps(lo.dtype());
    const Tensor w = r + (net.map_abs(k, mid.abs()) + net.b[k].abs() + r) * ulps;
    return {c - w, c + w};
}

/// The slopes' absolute values as a map of radii (the sign of a negative slope does not flip one).
Tensor aux_abs_slope(const Tensor &m) { return m.abs(); }

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

// The optimized slopes ----------------------------------------------------------------------------

/// Where a ReLU crosses zero at the (clamped) bounds.
Tensor aux_crossing(const Tensor &l, const Tensor &u) {
    const Tensor zero = l.zeros_like();
    return u.greater(zero).multiply(l.less(zero));
}

/// Where -a l is the larger end of relu(x) - a x over [l, u].
Tensor aux_alpha_left(const Tensor &l, const Tensor &u, const Tensor &a) {
    return a.multiply(l).negative().greater_equal(u - a.multiply(u));
}

/// The slopes of layer k, if alpha holds them and the layer is a ReLU.
const Tensor *aux_alpha_of(const ZonoNet &net, const std::vector<Tensor> *alpha, std::size_t k) {
    if (!alpha || k >= alpha->size() || !(*alpha)[k].defined()) return nullptr;
    return net.activations[k] == Activation::relu ? &(*alpha)[k] : nullptr;
}

/// A ReLU's line with the slope a in [0, 1] where it crosses zero: relu(x) - a x lies in [0, h]
/// over [l, u] with h = max(-a l, (1 - a) u), so the offset and the radius are h / 2.
PrivLine aux_alpha_line(const PrivLine &base, const Tensor &l, const Tensor &u, const Tensor &a) {
    const Tensor cross = aux_crossing(l, u);
    const Tensor half = a.multiply(l).negative().maximum(u - a.multiply(u)) * 0.5;
    return {Tensor::where(cross, a, base.slope), Tensor::where(cross, half, base.offset),
            Tensor::where(cross, half, base.radius)};
}

/// The partial derivatives of aux_alpha_line in l and u: the slope is held, h / 2 moves with the
/// larger end.
LineJacobian aux_alpha_jacobian(const LineJacobian &base, const Tensor &l, const Tensor &u,
                                const Tensor &a) {
    const Tensor cross = aux_crossing(l, u), left = aux_alpha_left(l, u, a);
    const Tensor zero = l.zeros_like();
    const Tensor h_l = Tensor::where(left, a.negative(), zero) * 0.5;
    const Tensor h_u = Tensor::where(left, zero, 1.0 - a) * 0.5;
    return {Tensor::where(cross, zero, base.m_l), Tensor::where(cross, zero, base.m_u),
            Tensor::where(cross, h_l, base.o_l),  Tensor::where(cross, h_u, base.o_u),
            Tensor::where(cross, h_l, base.r_l),  Tensor::where(cross, h_u, base.r_u)};
}

/// The reverse pass from the partial derivatives of the output (g_c, g_gen, g_dr) back through
/// every layer: the input's generator rows (B, q0, n0) if need_input, each record's slot gradient
/// (per neuron) into slot_grads, the slopes' gradients into g_alpha where alpha holds them.
Tensor aux_reverse(const ZonoPropagation &out, const ZonoNet &net, const Tensor &splits, Tensor g_c,
                   Tensor g_gen, Tensor g_dr, const std::vector<Tensor> *alpha,
                   std::vector<Tensor> *g_alpha, std::vector<Tensor> *slot_grads, bool need_input) {
    // the factor rows of each output (rows[j + 1]; 0: the input), from the output's back
    const std::size_t L = net.W.size();
    std::vector<int64_t> rows(L + 1);
    rows[L] = g_gen.shape()[1];
    for (std::size_t k = L, i = out.layers.size(); k-- > 0;) {
        // the slots that layer k added
        int64_t added = 0;
        if (net.activations[k] != Activation::identity) {
            const ZonoLayerRecord &rec = out.layers[--i];
            if (rec.slot_values.defined()) added = rec.slot_values.shape()[1];
        }
        rows[k] = rows[k + 1] - added;
    }
    // each output's gradients, summed over the layers that read it
    struct Grad {
        Tensor c, gen, dr;
    };
    std::vector<Grad> acc(L + 1);
    acc[L] = {std::move(g_c), std::move(g_gen), std::move(g_dr)};
    const auto add = [&](int64_t j, const Tensor &c, const Tensor &gen, const Tensor &dr) {
        Grad &a = acc[std::size_t(j + 1)];
        const auto sum = [](Tensor &t, const Tensor &v) {
            if (v.defined()) t = t.defined() ? t + v : v;
        };
        sum(a.c, c);
        sum(a.gen, gen);
        sum(a.dr, dr);
    };
    // from the last layer back: the activation, then the linear maps
    std::size_t next = out.layers.size();
    for (std::size_t k = L; k-- > 0;) {
        check_deadline(net.deadline);
        g_c = std::move(acc[k + 1].c);
        g_gen = std::move(acc[k + 1].gen);
        g_dr = std::move(acc[k + 1].dr);
        if (net.activations[k] != Activation::identity) {
            const ZonoLayerRecord &rec = out.layers[--next];
            const Tensor &c1 = rec.c_in, &G1 = rec.G_in, &dr1 = rec.dr_in;
            const int64_t n = rec.l.shape()[1], qk = G1.shape()[1];
            // the errors: the slot generators' values and the rest in dr are the reach
            Tensor g_reach;
            if (rec.slot_values.defined()) {
                const Tensor &neurons = rec.slot_neurons;
                const int64_t dn = neurons.shape()[1];
                const Tensor hot = aux_onehot(neurons, n);
                const Tensor gv = g_gen.slice(1, qk, qk + dn)
                                      .take_along_axis(neurons.expand_dims(-1), 2)
                                      .squeeze(-1);
                g_reach = hot.multiply(gv.expand_dims(-1)).sum(1);
                if (slot_grads) (*slot_grads)[next] = g_reach;
                if (dn < n && g_dr.defined()) g_reach = g_reach + g_dr.multiply(1.0 - hot.max(1));
                g_gen = g_gen.slice(1, 0, qk);
            } else {
                if (slot_grads) (*slot_grads)[next] = rec.l.zeros_like();
                g_reach = g_dr.defined() ? g_dr : rec.l.zeros_like();
            }
            // the line at the clamped bounds; c2 = m c1 + o, G2 = m G1, dr2 = |m| dr1 + reach
            const ClampJacobian cl = aux_clamp_jacobian(rec.first_id, rec.l, rec.u, splits);
            const Activation act = net.activations[k];
            PrivLine line = aux_layer_line(net, k, c1, G1, dr1, cl.lc, cl.uc);
            LineJacobian J = aux_line_jacobian(net.activations[k], cl.lc, cl.uc, net.poly_method);
            const Tensor *a = aux_alpha_of(net, alpha, k);
            if (a) {
                line = aux_alpha_line(line, cl.lc, cl.uc, *a);
                J = aux_alpha_jacobian(J, cl.lc, cl.uc, *a);
            }
            const Tensor &m = line.slope;
            if (act == Activation::product) {
                // held slopes [c_b; c_a]: c2, G2 and dr2 (by |m|) linear in c1, G1 and dr1
                g_c = priv_slope_times_t(act, m, g_c);
                g_gen = priv_slope_times_t(act, m.expand_dims(1), g_gen);
                g_dr = dr1.defined() ? priv_slope_times_t(act, aux_abs_slope(m), g_dr) : Tensor();
            } else {
                // dr2 = |m| dr1: its gradient in m has the sign of m (+ at 0, as for a ReLU)
                const Tensor sign_m =
                    Tensor::where(m.less(m.zeros_like()), -m.ones_like(), m.ones_like());
                Tensor g_m = g_c.multiply(c1) + g_gen.multiply(G1).sum(1);
                if (dr1.defined()) g_m = g_m + g_dr.multiply(dr1).multiply(sign_m);
                if (a) {
                    // the slope's own gradient: c2, G2, dr2 through m, offset and reach through h
                    const Tensor zero = rec.l.zeros_like();
                    const Tensor left = aux_alpha_left(cl.lc, cl.uc, *a);
                    const Tensor h_a = Tensor::where(left, cl.lc.negative(), cl.uc.negative());
                    (*g_alpha)[k] = Tensor::where(aux_crossing(cl.lc, cl.uc),
                                                  g_m + (g_c + g_reach).multiply(h_a) * 0.5, zero);
                }
                const Tensor g_lc =
                    g_m.multiply(J.m_l) + g_c.multiply(J.o_l) + g_reach.multiply(J.r_l);
                const Tensor g_uc =
                    g_m.multiply(J.m_u) + g_c.multiply(J.o_u) + g_reach.multiply(J.r_u);
                const Tensor g_l = g_lc.multiply(cl.lc_l) + g_uc.multiply(cl.uc_l);
                const Tensor g_u = g_lc.multiply(cl.lc_u) + g_uc.multiply(cl.uc_u);
                // l = c1 - sum |G1| - dr1, u = c1 + sum |G1| + dr1
                const Tensor g_rad = g_u - g_l;
                g_c = g_c.multiply(m) + g_l + g_u;
                g_gen = m.expand_dims(1).multiply(g_gen) + G1.sign().multiply(g_rad.expand_dims(1));
                g_dr = dr1.defined() ? g_dr.multiply(aux_abs_slope(m)) + g_rad : Tensor();
            }
        }
        // back to the inputs: a source's own factor rows (the input's center and radii unused)
        const auto back = [&](int64_t j, const auto &map, const auto &map_abs) {
            const int64_t r = rows[std::size_t(j + 1)];
            const Tensor gen = r < g_gen.shape()[1] ? g_gen.slice(1, 0, r) : g_gen;
            if (j < 0) {
                if (need_input) add(j, Tensor(), map(gen), Tensor());
                return;
            }
            add(j, map(g_c), map(gen), g_dr.defined() ? map_abs(g_dr) : Tensor());
        };
        for (std::size_t i = 0; i < net.residuals[k].size(); ++i)
            back(
                net.residuals[k][i].source,
                [&](const Tensor &x) { return net.residual(k, i, x, false, true); },
                [&](const Tensor &x) { return net.residual(k, i, x, true, true); });
        back(
            net.sources[k], [&](const Tensor &x) { return net.map_t(k, x); },
            [&](const Tensor &x) { return net.map_abs_t(k, x); });
    }
    return acc[0].gen;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The lines and boxes of a layer ------------------------------------------------------------------

Tensor priv_relu_chord_offset(const Tensor &l, const Tensor &u, const Tensor &chord) {
    // -chord l alone can fall below relu(u) = u once the slope is rounded: the larger of the two
    return chord.multiply(l).negative().maximum(u - chord.multiply(u));
}

PrivLine priv_alpha_line(const PrivLine &base, const Tensor &l, const Tensor &u, const Tensor &a) {
    return aux_alpha_line(base, l, u, a);
}

std::pair<Tensor, Tensor> priv_box_image(const ZonoNet &net, std::size_t k, const Tensor &lo,
                                         const Tensor &hi) {
    return aux_box_image(net, k, lo, hi);
}

PrivLine priv_verify_line(Activation activation, const Tensor &l, const Tensor &u,
                          const std::string &poly_method) {
    // a ReLU's line is CORALean's in both methods (the same line in the reals); x^2 has one line
    if (activation == Activation::square) return priv_square_line(l, u);
    if (activation == Activation::sign) return priv_sign_line(l, u);
    if (activation == Activation::floor) return priv_floor_line(l, u);
    if (activation == Activation::exp) return priv_exp_line(l, u);
    if (activation == Activation::reciprocal) return priv_reciprocal_line(l, u);
    if (activation == Activation::cube) return priv_cube_line(l, u);
    if (activation == Activation::inverse) return priv_inverse_line(l, u);
    if (activation == Activation::sin || activation == Activation::cos)
        return priv_trig_line(l, u, activation == Activation::cos);
    if (activation == Activation::product)
        CORA_FAIL(Error::not_supported, "verify: a product's line needs its set");
    // softplus, root and inv_sqrt_root: Singh's line is the one line of both methods
    if (activation == Activation::softplus || activation == Activation::root ||
        activation == Activation::inv_sqrt_root)
        return priv_monotone_line(activation, l, u, false);
    if (activation == Activation::group_sort)
        CORA_FAIL(Error::not_supported, "a group sort has no line (CORA throws too)");
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
        // the transposed weights for the rows; a convolution or a gather keeps its plan
        maps.push_back(layer);
        const bool dense = !layer.plan;
        W.push_back(dense ? layer.W.matrix_transpose() : Tensor());
        abs_w.push_back(dense ? W.back().abs() : Tensor());
        WT.push_back(dense ? layer.W : Tensor());
        abs_wT.push_back(dense ? layer.W.abs() : Tensor());
        b.push_back(layer.b);
        // its place in the graph
        sources.push_back(net.source(k));
        residuals.push_back(layer.residuals);
        // a slot count and the first global id per activation layer
        const int64_t n = layer.outputs();
        widths.push_back(n);
        const bool active = activations[k] != Activation::identity;
        slots.push_back(active ? aux_count(num_approx_err, n) : 0);
        first_ids.push_back(id);
        if (active) {
            id += n;
            widest = std::max(widest, n);
        }
    }
    // the graph: the outputs read by more than the next layer's own map
    chain = net.is_chain();
    kept.assign(sources.size() + 1, false);
    for (std::size_t k = 0; k < sources.size(); ++k) {
        if (sources[k] != int64_t(k) - 1) kept[std::size_t(sources[k] + 1)] = true;
        for (const NeuralNetwork::Residual &r : residuals[k])
            kept[std::size_t(r.source + 1)] = true;
    }
    // the factors, and whether an error has no slot
    num_neurons = id - 1;
    q = q0;
    for (const int64_t s : slots) q += s;
    interval_center = q0 < n0 || aux_count(num_approx_err, widest) < widest;
}

// The linear maps ---------------------------------------------------------------------------------

Tensor ZonoNet::map(std::size_t k, const Tensor &x) const {
    // a convolution or a gather by its plan, a matrix as one product of the batch's rows
    if (!W[k].defined()) return maps[k].rows(x);
    return x.shape().size() == 3 ? aux_rows_times(x, W[k]) : x.matmul(W[k]);
}

Tensor ZonoNet::map_abs(std::size_t k, const Tensor &x) const {
    if (!W[k].defined()) return maps[k].abs_rows(x);
    return x.shape().size() == 3 ? aux_rows_times(x, abs_w[k]) : x.matmul(abs_w[k]);
}

Tensor ZonoNet::map_t(std::size_t k, const Tensor &y) const {
    if (!W[k].defined()) return maps[k].rows_t(y);
    return y.shape().size() == 3 ? aux_rows_times(y, WT[k]) : y.matmul(WT[k]);
}

Tensor ZonoNet::map_abs_t(std::size_t k, const Tensor &y) const {
    if (!W[k].defined()) return maps[k].abs_rows_t(y);
    return y.shape().size() == 3 ? aux_rows_times(y, abs_wT[k]) : y.matmul(abs_wT[k]);
}

// The graph ---------------------------------------------------------------------------------------

Tensor ZonoNet::residual(std::size_t k, std::size_t i, const Tensor &x, bool absolute,
                         bool transposed) const {
    // no map: the identity
    const NeuralNetwork::Residual &r = residuals[k][i];
    if (!r.map) return x;
    if (transposed) return absolute ? r.map->abs_rows_t(x) : r.map->rows_t(x);
    return absolute ? r.map->abs_rows(x) : r.map->rows(x);
}

Tensor ZonoNet::linear(std::size_t k, const std::vector<Tensor> &outs, bool absolute) const {
    // the source's map, plus each residual's
    const Tensor &x = outs[std::size_t(sources[k] + 1)];
    Tensor z = absolute ? map_abs(k, x) : map(k, x);
    for (std::size_t i = 0; i < residuals[k].size(); ++i)
        z = z + residual(k, i, outs[std::size_t(residuals[k][i].source + 1)], absolute, false);
    return z;
}

void ZonoNet::linear_t(std::size_t k, const Tensor &g, std::vector<Tensor> &acc,
                       bool absolute) const {
    // each input's share added to what other layers gave it
    const auto add = [&](int64_t j, Tensor v) {
        Tensor &t = acc[std::size_t(j + 1)];
        t = t.defined() ? t + v : std::move(v);
    };
    for (std::size_t i = 0; i < residuals[k].size(); ++i)
        add(residuals[k][i].source, residual(k, i, g, absolute, true));
    add(sources[k], absolute ? map_abs_t(k, g) : map_t(k, g));
}

Tensor ZonoNet::evaluate(const Tensor &x) const {
    // the outputs that a later layer still reads
    std::vector<Tensor> outs{x};
    for (std::size_t k = 0; k < W.size(); ++k) {
        check_deadline(deadline);
        // the layer's linear part and activation; an output no later layer reads is dropped
        outs.push_back(aux_apply(activations[k], linear(k, outs) + b[k]));
        if (!kept[k]) outs[k] = Tensor();
    }
    return outs.back();
}

// Points ------------------------------------------------------------------------------------------

ZonoSensitivity ZonoNet::sensitivity(const Tensor &x) const {
    // the inputs of the activations, then the chain rule from the outputs back
    std::vector<Tensor> pre, outs{x};
    for (std::size_t k = 0; k < W.size(); ++k) {
        check_deadline(deadline);
        pre.push_back(linear(k, outs) + b[k]);
        outs.push_back(aux_apply(activations[k], pre.back()));
    }
    const int64_t B = x.shape()[0];
    // the Jacobian of the outputs, from the last layer back (summed over the layers that read one)
    std::vector<Tensor> acc(W.size() + 1);
    acc.back() = x.eye_like(num_outputs).broadcast_to({B, num_outputs, num_outputs});
    ZonoSensitivity out;
    out.layer_sum.resize(W.size());
    out.layer_max.resize(W.size());
    for (std::size_t k = W.size(); k-- > 0;) {
        check_deadline(deadline);
        Tensor S = std::move(acc[k + 1]);
        if (activations[k] != Activation::identity) {
            // through the activation's derivative, and the layer's sums and maxima of |S|
            S = priv_slope_times_t(activations[k],
                                   aux_derivative(activations[k], pre[k]).expand_dims(1), S);
            const Tensor a = S.abs();
            out.layer_sum[k] = a.sum(1);
            out.layer_max[k] = a.maximum(numerics::kSensitivityFloor).max(1);
        }
        linear_t(k, S, acc);
    }
    out.input = acc[0].abs().maximum(numerics::kSensitivityFloor).max(1);
    out.jacobian = acc[0];
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
                                 bool record, const std::vector<Tensor> *alpha) const {
    Tensor c = c_, dr = dr_, G = G_;
    ZonoPropagation out;
    const bool split = splits.defined() && splits.shape()[1] > 0;
    // without a record a layer's error generators stay diagonal until the next linear map
    Tensor pending_values, pending_neurons;
    bool pending_all = false;
    const auto take_pending = [&](std::size_t k) {
        const int64_t B = pending_values.shape()[0], dn = pending_values.shape()[1];
        // a convolution or a gather maps the error block (B, dn, n) itself
        if (!W[k].defined()) {
            const int64_t n = maps[k].inputs();
            return map(k, aux_onehot(pending_neurons, n).multiply(pending_values.expand_dims(-1))) +
                   0.0;
        }
        const Tensor &M = W[k];
        const Tensor rows =
            pending_all
                ? M.expand_dims(0)
                : M.take(pending_neurons.reshape({B * dn}), 0).reshape({B, dn, M.shape()[1]});
        // + 0.0: the product's zero is +0, as the sum of the block's row with zeros is
        return pending_values.expand_dims(-1).multiply(rows) + 0.0;
    };
    // a graph's outputs that more than the next layer read (states[j + 1]; 0: the input)
    struct State {
        Tensor c, dr, G;
    };
    std::vector<State> states(chain ? 0 : W.size() + 1);
    if (!chain) states[0] = {c, dr, G};
    // a layer's output box at its clamped bounds, for a layer that reads it alone (aux_reads_range)
    std::vector<std::pair<Tensor, Tensor>> ranges(W.size());
    std::vector<bool> range_read(W.size(), false);
    for (std::size_t k = 0; k < W.size(); ++k)
        if (aux_reads_range(*this, k)) range_read[std::size_t(sources[k])] = true;
    for (std::size_t k = 0; k < W.size(); ++k) {
        check_deadline(deadline);
        // the linear layer, exactly; an earlier source's rows are the first factors of all so far
        const int64_t rows = G.shape()[1];
        if (sources[k] != int64_t(k) - 1) {
            const State &from = states[std::size_t(sources[k] + 1)];
            c = from.c;
            dr = from.dr;
            G = from.G;
        }
        c = map(k, c) + b[k];
        G = aux_rows_to(map(k, G), rows);
        if (pending_values.defined()) {
            G = Tensor::concat({G, take_pending(k)}, 1);
            pending_values = Tensor();
        }
        if (dr.defined()) dr = map_abs(k, dr);
        // the residuals: their generators add to the first rows, their radii to dr
        for (std::size_t i = 0; i < residuals[k].size(); ++i) {
            const State &from = states[std::size_t(residuals[k][i].source + 1)];
            c = c + residual(k, i, from.c, false, false);
            G = aux_add_rows(G, residual(k, i, from.G, false, false));
            if (from.dr.defined()) {
                const Tensor r = residual(k, i, from.dr, true, false);
                dr = dr.defined() ? dr + r : r;
            }
        }
        if (activations[k] == Activation::identity) {
            if (!chain && kept[k + 1]) states[k + 1] = {c, dr, G};
            continue;
        }
        // the bounds of every neuron (CORALean's interval of a zonotope), then the split clamps
        const Tensor r = G.abs().sum(1);
        const Tensor mid = c + 0.0;
        Tensor l = (mid - 0.0) - r, u = (mid + 0.0) + r;
        if (dr.defined()) {
            l = l - dr;
            u = u + dr;
        }
        if (aux_reads_range(*this, k)) {
            // also in the image of the source's box (sums of exps >= 1, of cubes of ReLUs >= 0)
            const auto &range = ranges[std::size_t(sources[k])];
            const auto [lo, hi] = aux_box_image(*this, k, range.first, range.second);
            l = l.maximum(lo);
            u = u.minimum(hi).maximum(l);
        }
        const auto [lc, uc] = split ? clamp(first_ids[k], l, u, splits) : std::pair{l, u};
        if (range_read[k]) ranges[k] = aux_range(activations[k], lc, uc);
        PrivLine line = aux_layer_line(*this, k, c, G, dr, lc, uc);
        if (const Tensor *a = aux_alpha_of(*this, alpha, k))
            line = aux_alpha_line(line, lc, uc, *a);
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
        const Activation act = activations[k];
        c = priv_slope_times(act, line.slope, c) + 0.0 + middle;
        G = priv_slope_times(act, line.slope.expand_dims(1), G) + 0.0;
        if (dr.defined()) dr = priv_slope_times(act, aux_abs_slope(line.slope), dr);
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
            const Tensor values = dn == n ? reach : reach.take_along_axis(neurons, 1);
            const bool defer =
                !record && k + 1 < W.size() && sources[k + 1] == int64_t(k) && !kept[k + 1];
            const Tensor hot = defer && dn == n ? Tensor() : aux_onehot(neurons, n);
            if (defer) {
                pending_values = values;
                pending_neurons = neurons;
                pending_all = dn == n;
            } else {
                G = Tensor::concat({G, hot.multiply(values.expand_dims(-1))}, 1);
            }
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
        if (!chain && kept[k + 1]) states[k + 1] = {c, dr, G};
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
    // the reverse pass sets each record's slot gradient
    std::vector<Tensor> slot_grads(out.layers.size());
    const Tensor g_gen = aux_reverse(
        out, net, splits, out.c.zeros_like(), out.G.divide(frad.expand_dims(-1)),
        out.dr.defined() ? out.dr.zeros_like() : Tensor(), nullptr, nullptr, &slot_grads, true);
    // the records' gradients, and the input's of its generator dimensions
    for (std::size_t i = 0; i < out.layers.size(); ++i) out.layers[i].grad = slot_grads[i];
    return g_gen.take_along_axis(in.dims.expand_dims(-1), 2).squeeze(-1);
}

std::vector<Tensor> priv_alpha_gradients(const ZonoPropagation &out, const ZonoNet &net,
                                         const Tensor &splits, const std::vector<Tensor> &alpha,
                                         const Tensor &g_c, const Tensor &g_gen,
                                         const Tensor &g_dr) {
    // the reverse pass without the input's gradient
    std::vector<Tensor> g_alpha(net.W.size());
    aux_reverse(out, net, splits, g_c, g_gen, g_dr, &alpha, &g_alpha, nullptr, false);
    return g_alpha;
}

std::vector<Tensor> priv_alpha_start(const ZonoPropagation &out, const ZonoNet &net,
                                     const Tensor &splits) {
    std::vector<Tensor> alpha(net.W.size());
    for (const ZonoLayerRecord &rec : out.layers) {
        const auto k = std::size_t(rec.index);
        if (net.activations[k] != Activation::relu) continue;
        const bool split = splits.defined() && splits.shape()[1] > 0;
        const auto [lc, uc] =
            split ? net.clamp(rec.first_id, rec.l, rec.u, splits) : std::pair{rec.l, rec.u};
        // u / (u - l) where the neuron crosses zero, else a slope that is never read
        const Tensor cross = aux_crossing(lc, uc);
        alpha[k] =
            Tensor::where(cross, uc.divide((uc - lc).maximum(numerics::kTiny)), lc.zeros_like());
    }
    return alpha;
}

Tensor priv_backward_lower(const ZonoPropagation &out, const ZonoNet &net, const Tensor &splits,
                           const Tensor &x, const Tensor &rad, const Tensor &rows,
                           const std::vector<Tensor> *alpha) {
    const int64_t B = x.shape()[0], R = rows.shape()[0];
    // Lambda (B, R, n): each output's coefficients of the rows (summed over readers), the constant
    std::vector<Tensor> lams(net.W.size() + 1);
    lams.back() = rows.expand_dims(0).broadcast_to({B, R, rows.shape()[1]});
    Tensor constant = x.zeros_like({B, R});
    const bool split = splits.defined() && splits.shape()[1] > 0;
    std::size_t next = out.layers.size();
    for (std::size_t k = net.W.size(); k-- > 0;) {
        check_deadline(net.deadline);
        Tensor lam = std::move(lams[k + 1]);
        if (net.activations[k] != Activation::identity) {
            const ZonoLayerRecord &rec = out.layers[--next];
            const auto [lc, uc] =
                split ? net.clamp(rec.first_id, rec.l, rec.u, splits) : std::pair{rec.l, rec.u};
            Tensor sl, su, bl, bu;
            if (net.activations[k] == Activation::relu) {
                // the chord above, a slope in [0, 1] below (alpha, else CROWN's 0 or 1)
                const Tensor zero = lc.zeros_like(), one = lc.ones_like();
                const Tensor cross = aux_crossing(lc, uc), on = lc.greater_equal(zero);
                const Tensor chord = uc.divide((uc - lc).maximum(numerics::kTiny));
                su = Tensor::where(cross, chord, on);
                bu = Tensor::where(cross, priv_relu_chord_offset(lc, uc, chord), zero);
                const Tensor *a = aux_alpha_of(net, alpha, k);
                const Tensor below = a ? *a : uc.greater(lc.negative());
                sl = Tensor::where(cross, below, on);
                bl = zero;
            } else {
                // the parallel lines of the zonotope's line
                const PrivLine line = aux_layer_line(net, k, rec.c_in, rec.G_in, rec.dr_in, lc, uc);
                sl = su = line.slope;
                bl = line.offset - line.radius;
                bu = line.offset + line.radius;
            }
            const Tensor pos = lam.maximum(0.0), neg = lam.minimum(0.0);
            constant = constant + pos.matmul(bl.expand_dims(-1)).squeeze(-1) +
                       neg.matmul(bu.expand_dims(-1)).squeeze(-1);
            // a product's one slope: Lambda through the line's transpose
            lam = net.activations[k] != Activation::product
                      ? pos.multiply(sl.expand_dims(1)) + neg.multiply(su.expand_dims(1))
                      : priv_slope_times_t(Activation::product, sl.expand_dims(1), lam);
        }
        // the linear layer
        constant = constant + lam.multiply(net.b[k]).sum(-1);
        net.linear_t(k, lam, lams);
    }
    // the input box
    const Tensor &lam = lams[0];
    return lam.matmul(x.expand_dims(-1)).squeeze(-1) -
           lam.abs().matmul(rad.expand_dims(-1)).squeeze(-1) + constant;
}

Tensor priv_backward_alpha(const ZonoPropagation &out, const ZonoNet &net, const Tensor &splits,
                           const Tensor &x, const Tensor &rad, const Tensor &rows,
                           int64_t iterations, double step, double deadline) {
    const int64_t B = x.shape()[0], R = rows.shape()[0];
    const bool split = splits.defined() && splits.shape()[1] > 0;
    // per activation layer the clamped bounds' lines: a ReLU's chord above, its lower slope free
    struct Lines {
        Tensor sl, su, bl, bu, cross;
    };
    std::vector<Lines> lines(net.W.size());
    std::vector<Tensor> alpha(net.W.size()), m1(net.W.size()), m2(net.W.size());
    std::size_t next = out.layers.size();
    for (std::size_t k = net.W.size(); k-- > 0;) {
        if (net.activations[k] == Activation::identity) continue;
        const ZonoLayerRecord &rec = out.layers[--next];
        const auto [lc, uc] =
            split ? net.clamp(rec.first_id, rec.l, rec.u, splits) : std::pair{rec.l, rec.u};
        Lines &L = lines[k];
        if (net.activations[k] == Activation::relu) {
            // the chord above where the neuron crosses zero, the identity or zero elsewhere
            const Tensor zero = lc.zeros_like(), on = lc.greater_equal(zero);
            L.cross = aux_crossing(lc, uc);
            const Tensor chord = uc.divide((uc - lc).maximum(numerics::kTiny));
            L.su = Tensor::where(L.cross, chord, on).expand_dims(1);
            L.bu =
                Tensor::where(L.cross, priv_relu_chord_offset(lc, uc, chord), zero).expand_dims(1);
            L.sl = Tensor::where(L.cross, zero, on).expand_dims(1);
            L.bl = zero.expand_dims(1);
            // CROWN's start: 1 where u > -l, per row
            alpha[k] =
                uc.greater(lc.negative()).expand_dims(1).broadcast_to({B, R, lc.shape()[1]}) + 0.0;
            L.cross = L.cross.expand_dims(1);
        } else {
            const PrivLine line = aux_layer_line(net, k, rec.c_in, rec.G_in, rec.dr_in, lc, uc);
            L.sl = L.su = line.slope.expand_dims(1);
            L.bl = (line.offset - line.radius).expand_dims(1);
            L.bu = (line.offset + line.radius).expand_dims(1);
        }
    }
    // the rows at the output, the input box, the best bound so far and Adam's constants
    const Tensor lam0 = rows.expand_dims(0).broadcast_to({B, R, rows.shape()[1]});
    const Tensor xe = x.expand_dims(-1), re = rad.expand_dims(-1);
    Tensor best;
    constexpr double kB1 = 0.9, kB2 = 0.999;
    for (int64_t t = 0; t <= iterations; ++t) {
        // the backward pass, keeping each layer's Lambda for the reverse (acc: each output's)
        std::vector<Tensor> lams(net.W.size()), acc(net.W.size() + 1);
        acc.back() = lam0;
        Tensor constant = x.zeros_like({B, R});
        for (std::size_t k = net.W.size(); k-- > 0;) {
            // late: the best bound so far, none before the first pass
            if (best.defined() && now_seconds() >= std::min(deadline, net.deadline)) return best;
            check_deadline(net.deadline);
            Tensor lam = std::move(acc[k + 1]);
            if (net.activations[k] != Activation::identity) {
                const Lines &L = lines[k];
                lams[k] = lam;
                const Tensor sl =
                    alpha[k].defined() ? Tensor::where(L.cross, alpha[k], L.sl) : L.sl;
                const Tensor pos = lam.maximum(0.0), neg = lam.minimum(0.0);
                constant = constant + pos.multiply(L.bl).sum(-1) + neg.multiply(L.bu).sum(-1);
                lam = net.activations[k] != Activation::product
                          ? pos.multiply(sl) + neg.multiply(L.su)
                          : priv_slope_times_t(Activation::product, L.su, lam);
            }
            // the bias, then back to the inputs
            constant = constant + lam.multiply(net.b[k]).sum(-1);
            net.linear_t(k, lam, acc);
        }
        // the bound over the input box
        const Tensor &lam = acc[0];
        const Tensor value =
            lam.matmul(xe).squeeze(-1) - lam.abs().matmul(re).squeeze(-1) + constant;
        best = best.defined() ? best.maximum(value) : value;
        // (every step's bound holds: the best so far at the deadline)
        if (t == iterations || now_seconds() >= std::min(deadline, net.deadline)) break;
        // the reverse: d value / d Lambda from the input box up, the slopes' gradients on the way
        std::vector<Tensor> gs{x.expand_dims(1) - lam.sign().multiply(rad.expand_dims(1))};
        const double c1 = 1.0 - std::pow(kB1, double(t + 1)),
                     c2 = 1.0 - std::pow(kB2, double(t + 1));
        for (std::size_t k = 0; k < net.W.size(); ++k) {
            if (now_seconds() >= std::min(deadline, net.deadline)) return best;
            // through the linear layer: Lambda_k W_k, plus the bias
            Tensor g = net.linear(k, gs) + net.b[k];
            if (net.activations[k] == Activation::identity) {
                gs.push_back(g);
                continue;
            }
            const Lines &L = lines[k];
            const Tensor &lk = lams[k];
            const Tensor pos = lk.greater(lk.zeros_like()), neg = lk.less(lk.zeros_like());
            const Tensor sl = alpha[k].defined() ? Tensor::where(L.cross, alpha[k], L.sl) : L.sl;
            if (alpha[k].defined()) {
                // Adam on the slopes in [0, 1]
                const Tensor ga =
                    Tensor::where(L.cross, g.multiply(lk.maximum(0.0)), g.zeros_like());
                m1[k] = m1[k].defined() ? m1[k] * kB1 + ga * (1.0 - kB1) : ga * (1.0 - kB1);
                const Tensor g2 = ga.square() * (1.0 - kB2);
                m2[k] = m2[k].defined() ? m2[k] * kB2 + g2 : g2;
                const Tensor move =
                    (m1[k] / c1).divide((m2[k] / c2).sqrt() + numerics::kAdamEpsilon);
                alpha[k] = (alpha[k] + move * step).maximum(0.0).minimum(1.0);
            }
            if (net.activations[k] != Activation::product) {
                gs.push_back(pos.multiply(g.multiply(sl) + L.bl) +
                             neg.multiply(g.multiply(L.su) + L.bu));
            } else {
                // a product's one slope: the line's map of g, then the offset of the side
                const Tensor gm = priv_slope_times(Activation::product, L.su, g);
                gs.push_back(pos.multiply(gm + L.bl) + neg.multiply(gm + L.bu));
            }
        }
    }
    return best;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
