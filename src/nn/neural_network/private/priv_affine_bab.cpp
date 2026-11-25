// priv_affine_bab - the ReLU-split branch and bound on the root's affine form (a ReLU one slope)
//
// See priv_zono_batch.h. With one slope a in [0, 1] per crossing ReLU, below and above, a ReLU
// layer's output is a z + h + h e, e in [-1, 1] (h = max(-a l, u - a u) / 2), so the input z_j
// of every crossing neuron and every output row are affine in the factors e in [-1, 1]^n: one per
// input dimension, one per crossing neuron. priv_backward_lower gives these forms row by row
// (ZonoBackward::one_slope) at the root's bounds, without any layer's generator matrix. A node
// fixes the first D neurons of one order to a side. Inactive (z <= 0, y = 0), the neuron's factor
// is -1 - (a / h) z; active (z >= 0, y = z), ((1 - a) z - h) / h: the row's own coefficient of
// the factor is replaced by that form of z, and the side and the root's bound on it (z >= l,
// z <= u) get multipliers >= 0. The bound is the Lagrangian's minimum over the factors' box, a
// 1-norm (weak duality, docs/nn_verification.md). The nodes of one depth share the neurons' rows
// and differ in their signs only: a batch costs a few products with the (D, n) rows.
// References:
//    [1] G. Singh, T. Gehr, M. Mirman, M. Pueschel, M. Vechev. "Fast and effective robustness
//        certification", NeurIPS 2018 (the zonotope's ReLU line).
//    [2] S. Wang, H. Zhang, K. Xu, X. Lin, S. Jana, C.-J. Hsieh, J. Z. Kolter. "Beta-CROWN:
//        efficient bound propagation with per-neuron split constraints for neural network
//        robustness verification", NeurIPS 2021 (the splits' multipliers).
//
// CORA.cpp only: CORA's verify splits on the zonotope
// Authors:       Claude Fable 5.2
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "global/tensor/device_memory.h"
#include "nn/neural_network/private/priv_verify.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
/// A side no factor of the box reaches by more than this is empty (the forms' rounding is below).
constexpr double kEmptySide = numerics::kEmptySide;

// The affine rows ---------------------------------------------------------------------------------

/// The pass's bytes per row: Lambda and its lines on every layer, about.
int64_t aux_row_bytes(const ZonoNet &net) {
    int64_t width = net.n0;
    for (const int64_t w : net.widths) width += w;
    for (const NeuralNetwork::Layer &layer : net.maps) width += layer.work();
    return 8 * 32 * width;
}

/// The affine forms (R, n) and centers (R) of rows (R, n_top) times layer top's input (-1: the
/// output) in the table's factors: one backward pass with the table's slopes.
std::pair<Tensor, Tensor> aux_forms(const AffineTable &t, const ZonoPropagation &out,
                                    const ZonoNet &net, const ZonoBounds &bounds,
                                    const Tensor &rows, int64_t top, ZonoAffineRows *lines) {
    ZonoAffineRows got;
    ZonoBackward opt;
    opt.top = top;
    opt.bounds = &bounds;
    opt.one_slope = &t.slopes;
    opt.affine_at = &t.pos;
    opt.affine = &got;
    const int64_t R = rows.shape()[0];
    const Tensor value =
        priv_backward_lower(out, net, t.x.zeros_like({1, 0}), t.x, t.rad, rows, nullptr, &opt);
    // the input's factors, then each crossing neuron's: Lambda times half the chord's offset
    std::vector<Tensor> parts{got.input.squeeze(0).multiply(t.rad)};
    for (std::size_t k = 0; k < t.pos.size(); ++k) {
        if (t.pos[k].empty()) continue;
        const int64_t c = int64_t(t.pos[k].size());
        const bool read = k < got.lambda.size() && got.lambda[k].defined();
        parts.push_back(read ? got.lambda[k].squeeze(0).multiply(got.offset[k] * 0.5)
                             : t.x.zeros_like({R, c}));
    }
    const Tensor A = Tensor::concat(parts, -1);
    // the pass's value is the form's minimum over the box: its center lies |A|_1 above
    if (lines) *lines = got;
    return {A, value.squeeze(0) + A.abs().sum(-1)};
}

// The line search ---------------------------------------------------------------------------------

struct Step {
    Tensor eta;       ///< (N): the step, the Lagrangian's value at it at least its value at 0
    Tensor unbounded; ///< (N): 1 where the value still grows at the last breakpoint, no limit
};

/// A step eta in [0, upper] (N; infinite: no limit) that does not decrease the concave piecewise
/// linear c eta - |r + eta d|_1 (r, d (N, n), c (N)): `bisections` geometric bisections of its
/// derivative's sign between the first and the last breakpoint, the lower end taken.
Step aux_line_search(const Tensor &r, const Tensor &d, const Tensor &c, const Tensor &upper,
                     int64_t bisections) {
    const Tensor zero = r.zeros_like({1});
    // r + eta d = d (eta - t), t = -r / d: the derivative is c - sum_j |d_j| sign(eta - t_j)
    const Tensor w = d.abs();
    const Tensor moving = w.greater(zero);
    const Tensor t =
        Tensor::where(moving, r.negative().divide(Tensor::where(moving, d, d.ones_like())), zero);
    const Tensor top = c + w.sum(-1);
    const auto rises = [&](const Tensor &eta) {
        return (top - w.multiply(t.less(eta.expand_dims(-1))).sum(-1) * 2.0).greater(zero);
    };
    // the breakpoints t > 0: where a coefficient changes its sign
    const Tensor cross = t.greater(zero);
    const Tensor first = Tensor::where(cross, t, r.full_like({1}, kInf)).min(-1);
    const Tensor last = Tensor::where(cross, t, zero).max(-1);
    const Tensor limited = upper.isfinite();
    const Tensor any = first.isfinite();
    const Tensor end = Tensor::where(any, last, Tensor::where(limited, upper, zero)).minimum(upper);
    Tensor lo = first.minimum(end), hi = end;
    // the derivative right of 0 (linear up to the first breakpoint) and past the end
    const Tensor grows = rises(lo * 0.5);
    const Tensor open = rises(end * 2.0 + numerics::kTiny);
    for (int64_t i = 0; i < bisections; ++i) {
        const Tensor mid = lo.multiply(hi).sqrt();
        const Tensor up = rises(mid);
        lo = Tensor::where(up, mid, lo);
        hi = Tensor::where(up, hi, mid);
    }
    Step s;
    s.eta = Tensor::where(grows, Tensor::where(open, end, lo), zero);
    s.unbounded = grows.multiply(open).multiply(1.0 - limited);
    return s;
}

// The problem of one row --------------------------------------------------------------------------

/// One output row's problem in the order of its splits: the first D rows are a node's of depth D.
struct Problem {
    Tensor A, a1;         ///< (S, n), (S): the neurons' rows and their 1-norms
    Tensor c, l, u;       ///< (S): their centers and bounds
    Tensor r_off, r_on;   ///< (S): the row's coefficient of z after the substitution, per side
    Tensor s_off, s_on;   ///< (S): the row's constant's shift, per side
    Tensor d;             ///< (n): the row's coefficients
    Tensor hot;           ///< (S): each neuron's own factor
    double c0 = 0;        ///< the row's center minus its offset
    int64_t S = 0, n = 0; ///< the neurons and the factors
    std::vector<int64_t> order; ///< the table's neuron of each split
};

/// The neurons' substitution constants (U each) of the row d: inactive, the factor is
/// -1 - (a / h) z; active, ((1 - a) z - h) / h.
struct Substitution {
    Tensor r_off, r_on, s_off, s_on;
};

Substitution aux_substitution(const AffineTable &t, const Tensor &d) {
    const Tensor own = d.slice(0, t.n0, t.n0 + t.U);
    const Tensor ratio = t.slope.divide(t.half), rest = (1.0 - t.slope).divide(t.half);
    Substitution s;
    s.r_off = ratio.multiply(own).negative();
    s.s_off = (1.0 + ratio.multiply(t.c)).multiply(own).negative();
    s.r_on = rest.multiply(own);
    s.s_on = (rest.multiply(t.c) - 1.0).multiply(own);
    return s;
}

/// The order of the splits: per neuron the bound of the root with it alone split, its side's
/// multiplier by the line search; the neuron whose worse side gains most first.
std::vector<int64_t> aux_order(const AffineTable &t, const Tensor &d, double c0,
                               const Substitution &sub, int64_t chunk) {
    std::vector<double> score;
    const Tensor none = t.c.full_like({1}, kInf);
    for (int64_t at = 0; at < t.U; at += chunk) {
        const int64_t to = std::min(t.U, at + chunk);
        const Tensor A = t.A.slice(0, at, to), cz = t.c.slice(0, at, to);
        // the row without the neuron's own factor
        const Tensor own = (d.arange_like(to - at) + double(t.n0 + at))
                               .expand_dims(-1)
                               .equal(d.arange_like(t.n0 + t.U).expand_dims(0));
        const Tensor rest = d.expand_dims(0).multiply(1.0 - own);
        Tensor worse;
        for (const bool off : {true, false}) {
            const Tensor ratio = (off ? sub.r_off : sub.r_on).slice(0, at, to);
            const Tensor shift = (off ? sub.s_off : sub.s_on).slice(0, at, to);
            const Tensor r = rest + A.multiply(ratio.expand_dims(-1));
            // the side's multiplier alone: its best step from 0
            const Tensor dir = off ? A : A.negative(), dc = off ? cz : cz.negative();
            const Step s = aux_line_search(r, dir, dc, none.broadcast_to({to - at}), 30);
            const Tensor value = shift + s.eta.multiply(dc) -
                                 (r + dir.multiply(s.eta.expand_dims(-1))).abs().sum(-1);
            // a side that no factor of the box reaches: never the worse one
            const Tensor side = Tensor::where(s.unbounded, none, value + c0);
            worse = worse.defined() ? worse.minimum(side) : side;
        }
        // the neuron's score: its worse side's bound
        const std::vector<double> part = worse.data();
        score.insert(score.end(), part.begin(), part.end());
    }
    // the neurons by their score, the largest first
    std::vector<int64_t> order(std::size_t(t.U));
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = int64_t(i);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        const double sa = score[std::size_t(a)], sb = score[std::size_t(b)];
        // not a number: last
        return (sa == sa ? sa : -kInf) > (sb == sb ? sb : -kInf);
    });
    return order;
}

Problem aux_problem(const AffineTable &t, const Tensor &d, double c0, int64_t chunk) {
    const Substitution sub = aux_substitution(t, d);
    const std::vector<int64_t> order = aux_order(t, d, c0, sub, chunk);
    Problem p;
    p.S = t.U;
    p.n = t.n0 + t.U;
    p.d = d;
    p.c0 = c0;
    // the neurons' rows and constants in that order
    p.order = order;
    p.A = t.A.take(order, 0);
    p.a1 = p.A.abs().sum(-1);
    p.c = t.c.take(order, 0);
    p.l = t.l.take(order, 0);
    p.u = t.u.take(order, 0);
    // the substitutions' constants, and each neuron's own factor
    p.r_off = sub.r_off.take(order, 0);
    p.r_on = sub.r_on.take(order, 0);
    p.s_off = sub.s_off.take(order, 0);
    p.s_on = sub.s_on.take(order, 0);
    std::vector<double> hot;
    hot.reserve(order.size());
    for (const int64_t j : order) hot.push_back(double(t.n0 + j));
    p.hot = Tensor::like(d, hot, {p.S});
    return p;
}

// The node bound ----------------------------------------------------------------------------------

/// The steps on a node's multipliers: `polyak` supergradient steps of Polyak's size toward the
/// margin (the best of them kept), then `steps` steps by the line search.
struct Ascent {
    int64_t steps = 1, bisections = 8, polyak = 0;
    double margin = 0;
};

/// The lower bounds (N) of the row on the nodes of sides sg (N, D) (+1: inactive, -1: active) at
/// the multipliers p (the side), q (the root's bound) >= 0 (N, D), both moved by `steps` ascent
/// steps: every value is the Lagrangian's minimum at multipliers >= 0, the best is kept.
Tensor aux_ascent(const Problem &P, const Tensor &sg, Tensor &p, Tensor &q, const Ascent &a) {
    const int64_t steps = a.steps, bisections = a.bisections;
    const int64_t D = sg.shape()[1];
    const Tensor zero = sg.zeros_like({1}), inf = sg.full_like({1}, kInf);
    const Tensor A = P.A.slice(0, 0, D), AT = A.matrix_transpose();
    const Tensor a1 = P.a1.slice(0, 0, D), cz = P.c.slice(0, 0, D);
    const Tensor off = sg.greater(zero);
    // the row after the substitutions: the split neurons' own factors by their forms of z
    const Tensor own =
        P.hot.slice(0, 0, D).expand_dims(-1).equal(P.d.arange_like(P.n).expand_dims(0)).sum(0);
    const Tensor ratio = Tensor::where(off, P.r_off.slice(0, 0, D), P.r_on.slice(0, 0, D));
    const Tensor base = P.d.multiply(1.0 - own).expand_dims(0) + ratio.matmul(A);
    const Tensor c0 =
        Tensor::where(off, P.s_off.slice(0, 0, D), P.s_on.slice(0, 0, D)).sum(-1) + P.c0;
    // the constraints sigma z <= 0 and l <= z (inactive), z <= u (active) at the factors 0
    const Tensor b0 = sg.multiply(cz);
    const Tensor b1 = Tensor::where(off, P.l.slice(0, 0, D) - cz, cz - P.u.slice(0, 0, D));
    const Tensor empty = b0.greater(a1 + kEmptySide).maximum(b1.greater(a1 + kEmptySide)).max(-1);
    const auto value = [&](const Tensor &r) {
        return c0 + (p.multiply(b0) + q.multiply(b1)).sum(-1) - r.abs().sum(-1);
    };
    // the supergradient: the constraints at the minimizer, a multiplier at 0 not below it
    const auto ascent = [&](const Tensor &at) {
        const Tensor xm = Tensor::where(at.less(zero), at.ones_like(), at.full_like({1}, -1.0));
        const Tensor z = sg.multiply(xm.matmul(AT));
        const Tensor s0 = b0 + z, s1 = b1 - z;
        return std::pair{Tensor::where(p.less_equal(zero).multiply(s0.less(zero)), zero, s0),
                         Tensor::where(q.less_equal(zero).multiply(s1.less(zero)), zero, s1)};
    };
    Tensor r = base + sg.multiply(p - q).matmul(A);
    // Polyak's steps toward a value above the margin: the multipliers of the best value kept
    if (a.polyak > 0) {
        Tensor top, bp = p, bq = q;
        for (int64_t i = 0; i <= a.polyak; ++i) {
            const Tensor now = value(r);
            const Tensor up = top.defined() ? now.greater(top) : now.equal(now);
            top = top.defined() ? top.maximum(now) : now;
            bp = Tensor::where(up.expand_dims(-1), p, bp);
            bq = Tensor::where(up.expand_dims(-1), q, bq);
            if (i == a.polyak) break;
            // the step (target - value) / |g|^2 along the supergradient
            const auto [g0, g1] = ascent(r);
            const Tensor norm = (g0.square() + g1.square()).sum(-1);
            const Tensor gap = (a.margin + numerics::kVerifiedMargin - now).maximum(0.0);
            const Tensor eta =
                Tensor::where(norm.greater(zero), gap.divide(norm.maximum(numerics::kTiny)), zero)
                    .expand_dims(-1);
            p = (p + g0.multiply(eta)).maximum(0.0);
            q = (q + g1.multiply(eta)).maximum(0.0);
            r = base + sg.multiply(p - q).matmul(A);
        }
        p = bp;
        q = bq;
        if (steps > 0) r = base + sg.multiply(p - q).matmul(A);
        if (steps <= 0) return Tensor::where(empty, inf, top);
    }
    // without steps: these multipliers' value, and the one at 0; a step's value is no lower
    Tensor best = steps > 0 ? Tensor() : (c0 - base.abs().sum(-1)).maximum(value(r));
    for (int64_t i = 0; i < steps; ++i) {
        // along the supergradient, the step by the line search
        const auto [g0, g1] = ascent(r);
        const Tensor dir = sg.multiply(g0 - g1).matmul(A);
        const Tensor dc = (g0.multiply(b0) + g1.multiply(b1)).sum(-1);
        // the step that keeps every multiplier >= 0, at most
        const Tensor w0 =
            Tensor::where(g0.less(zero), p.negative().divide(g0.minimum(-numerics::kTiny)), inf);
        const Tensor w1 =
            Tensor::where(g1.less(zero), q.negative().divide(g1.minimum(-numerics::kTiny)), inf);
        const Tensor eta = aux_line_search(r, dir, dc, w0.min(-1).minimum(w1.min(-1)), bisections)
                               .eta.expand_dims(-1);
        p = (p + g0.multiply(eta)).maximum(0.0);
        q = (q + g1.multiply(eta)).maximum(0.0);
        r = base + sg.multiply(p - q).matmul(A);
        best = best.defined() ? best.maximum(value(r)) : value(r);
    }
    return Tensor::where(empty, inf, best);
}

/// The problem's tensors in another dtype, for the steps of aux_bound.
Problem aux_problem_to(const Problem &P, DType dtype) {
    Problem S = P;
    for (Tensor *t :
         {&S.A, &S.a1, &S.c, &S.l, &S.u, &S.r_off, &S.r_on, &S.s_off, &S.s_on, &S.d, &S.hot})
        *t = t->to(dtype);
    return S;
}

/// aux_ascent, its steps on the copy `search` of P in another dtype (if given): the nodes whose
/// bound there exceeds the margin are evaluated on P itself at the multipliers the steps ended
/// at, >= 0, and only that bound is returned; the others' is -inf (open).
Tensor aux_bound(const Problem &P, const Problem *search, const Tensor &sg, Tensor &p, Tensor &q,
                 const Ascent &a) {
    if (!search || a.steps + a.polyak <= 0) return aux_ascent(P, sg, p, q, a);
    const DType dt = search->A.dtype(), exact = P.A.dtype();
    Tensor ps = p.to(dt), qs = q.to(dt);
    const Tensor rough = aux_ascent(*search, sg.to(dt), ps, qs, a);
    p = ps.to(exact).maximum(0.0);
    q = qs.to(exact).maximum(0.0);
    // the candidates: above the margin in the search dtype (not a number: none)
    const std::vector<int64_t> above = rough.greater(rough.full_like({1}, a.margin)).nonzero();
    std::vector<double> low(std::size_t(sg.shape()[0]), -kInf);
    if (!above.empty()) {
        Tensor pa = p.take(above, 0), qa = q.take(above, 0);
        const std::vector<double> sure =
            aux_ascent(P, sg.take(above, 0), pa, qa, Ascent{0, 0, 0, a.margin}).data();
        for (std::size_t i = 0; i < above.size(); ++i) low[std::size_t(above[i])] = sure[i];
    }
    return Tensor::like(sg, low, {sg.shape()[0]});
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The table ---------------------------------------------------------------------------------------

AffineTable priv_affine_table(const ZonoPropagation &out, const ZonoNet &net,
                              const ZonoBounds &bounds, const Tensor &x, const Tensor &rad,
                              const std::vector<Tensor> &slopes, int64_t max_bytes) {
    priv_oom_site("affine_bab_table");
    AffineTable t;
    t.slopes = slopes;
    t.x = x;
    t.rad = rad;
    t.n0 = x.shape()[1];
    if (x.shape()[0] != 1) CORA_FAIL(Error::wrong_value, "verify: affine-bab bounds one box");
    // ReLU layers only: every other layer is linear
    for (const Activation a : net.activations)
        if (a != Activation::relu && a != Activation::identity) return t;
    // the crossing neurons per layer at the given bounds (else the record's)
    t.pos.resize(net.W.size());
    std::vector<Tensor> ls, us;
    for (const ZonoLayerRecord &rec : out.layers) {
        const auto k = std::size_t(rec.index);
        const bool given = k < bounds.l.size() && bounds.l[k].defined();
        const Tensor l = given ? bounds.l[k] : rec.l, u = given ? bounds.u[k] : rec.u;
        const std::vector<double> lv = l.data(), uv = u.data();
        for (std::size_t i = 0; i < lv.size(); ++i)
            if (lv[i] < 0 && uv[i] > 0) t.pos[k].push_back(int64_t(i));
        if (t.pos[k].empty()) continue;
        t.U += int64_t(t.pos[k].size());
        ls.push_back(l.squeeze(0).take(t.pos[k], 0));
        us.push_back(u.squeeze(0).take(t.pos[k], 0));
    }
    t.applicable = true;
    if (t.U == 0) return t;
    // the rows (U, n), the order's and a batch's copies within the budget
    const int64_t n = t.n0 + t.U;
    if (8 * t.U * n * 4 > max_bytes) {
        t.applicable = false;
        return t;
    }
    t.l = Tensor::concat(ls, 0);
    t.u = Tensor::concat(us, 0);
    // every crossing neuron's input: the unit rows of its layer, in chunks of the budget
    const int64_t chunk = std::max<int64_t>(1, max_bytes / aux_row_bytes(net));
    std::vector<Tensor> As, cs;
    for (std::size_t k = 0; k < t.pos.size(); ++k) {
        const int64_t width = net.widths[k];
        for (std::size_t at = 0; at < t.pos[k].size(); at += std::size_t(chunk)) {
            check_deadline(net.deadline);
            const std::size_t to = std::min(t.pos[k].size(), at + std::size_t(chunk));
            std::vector<double> ids;
            for (std::size_t i = at; i < to; ++i) ids.push_back(double(t.pos[k][i]));
            const Tensor E = Tensor::like(x, ids, {int64_t(ids.size())})
                                 .expand_dims(-1)
                                 .equal(x.arange_like(width).expand_dims(0)) +
                             0.0;
            const auto [A, c] = aux_forms(t, out, net, bounds, E, int64_t(k), nullptr);
            As.push_back(A);
            cs.push_back(c);
        }
    }
    t.A = Tensor::concat(As, 0);
    t.c = Tensor::concat(cs, 0);
    // the chords' slopes and half offsets: any output row's pass reads every layer
    ZonoAffineRows lines;
    aux_forms(t, out, net, bounds, x.ones_like({1, net.num_outputs}), -1, &lines);
    // per crossing neuron, layer by layer as the rows
    std::vector<Tensor> as, hs;
    for (std::size_t k = 0; k < t.pos.size(); ++k) {
        if (t.pos[k].empty()) continue;
        as.push_back(lines.slope[k].squeeze(0));
        hs.push_back(lines.offset[k].squeeze(0) * 0.5);
    }
    t.slope = Tensor::concat(as, 0);
    t.half = Tensor::concat(hs, 0);
    return t;
}

std::pair<Tensor, Tensor> priv_affine_rows(const AffineTable &t, const ZonoPropagation &out,
                                           const ZonoNet &net, const ZonoBounds &bounds,
                                           const Tensor &rows) {
    return aux_forms(t, out, net, bounds, rows, -1, nullptr);
}

Tensor priv_affine_node_bounds(const AffineTable &t, const ZonoPropagation &out, const ZonoNet &net,
                               const ZonoBounds &bounds, const Tensor &row, const Tensor &offset,
                               const Tensor &sides, int64_t steps, std::vector<int64_t> *order,
                               int64_t polyak) {
    // the row's problem, then the nodes' bounds from multipliers 0
    const auto [d, center] = priv_affine_rows(t, out, net, bounds, row);
    const Problem P =
        aux_problem(t, d.squeeze(0), (center - offset).item(), std::max<int64_t>(1, t.U));
    if (order) *order = P.order;
    Tensor p = sides.zeros_like(), q = sides.zeros_like();
    return aux_bound(P, nullptr, sides, p, q, Ascent{steps, 8, polyak, 0.0});
}

// The branch and bound ----------------------------------------------------------------------------

int priv_affine_bab(const AffineTable &t, const ZonoPropagation &out, const ZonoNet &net,
                    const ZonoBounds &bounds, const Tensor &row, const Tensor &offset,
                    const AffineBabOptions &o, std::map<std::string, double> &stats) {
    priv_oom_site("affine_bab");
    if (!t.applicable) return 0;
    const double started = now_seconds();
    const auto late = [&] { return now_seconds() >= std::min(o.deadline, net.deadline); };
    // the row's form: its root bound is the form's minimum over the box
    const auto [d2, center] = priv_affine_rows(t, out, net, bounds, row);
    const Tensor d = d2.squeeze(0);
    const double c0 = (center - offset).item();
    const double root = c0 - d.abs().sum().item();
    // above the margin at the root: no split
    stats["affine_bab_root_margin"] = std::min(
        stats.count("affine_bab_root_margin") ? stats["affine_bab_root_margin"] : kInf, root);
    if (root > o.margin) return 1;
    if (t.U == 0) return 0;
    // a batch: about twelve (N, n) tensors at once
    const int64_t n = t.n0 + t.U;
    const int64_t batch =
        std::max<int64_t>(2, std::min(o.batch, o.max_bytes / std::max<int64_t>(1, 8 * 12 * n)));
    // the row's problem in the order of its splits
    const Problem P = aux_problem(t, d, c0, batch);
    // the steps' copy in the search dtype
    const bool searching = o.search && o.search_dtype != d.dtype();
    const Problem Ps = searching ? aux_problem_to(P, o.search_dtype) : Problem();
    const Problem *search = searching ? &Ps : nullptr;
    stats["affine_bab_batch"] = double(batch);
    stats["affine_bab_factors"] = double(n);
    // the open nodes: parts of one depth each, the last part's children first
    struct Part {
        Tensor sg, p, q;
    };
    // the root's two children: the first neuron of the order inactive and active
    const Tensor one = d.ones_like({1, 1});
    std::vector<Part> stack{
        {Tensor::concat({one, one.negative()}, 0), d.zeros_like({2, 1}), d.zeros_like({2, 1})}};
    int64_t held = 2;
    int res = 1;
    while (!stack.empty()) {
        if (late()) {
            res = 0;
            break;
        }
        // the last part, a batch of it at most
        Part part = std::move(stack.back());
        stack.pop_back();
        const int64_t F = part.sg.shape()[0], D = part.sg.shape()[1];
        if (F > batch) {
            stack.push_back({part.sg.slice(0, 0, F - batch), part.p.slice(0, 0, F - batch),
                             part.q.slice(0, 0, F - batch)});
            part = {part.sg.slice(0, F - batch, F), part.p.slice(0, F - batch, F),
                    part.q.slice(0, F - batch, F)};
        }
        // the batch's bounds, its multipliers moved
        const int64_t N = part.sg.shape()[0];
        held -= N;
        const Tensor low = aux_bound(P, search, part.sg, part.p, part.q,
                                     Ascent{o.steps, o.bisections, o.polyak, o.margin});
        // open unless its bound is above the margin (not a number: open)
        const std::vector<int64_t> open =
            (1.0 - low.greater(low.full_like({1}, o.margin))).nonzero();
        stats["affine_bab_nodes"] += double(N);
        stats["affine_bab_depth"] = std::max(stats["affine_bab_depth"], double(D));
        if (open.empty()) continue;
        // every neuron split and still open, or too many open nodes: not decided here
        const int64_t M = int64_t(open.size());
        held += 2 * M;
        stats["affine_bab_frontier_peak"] =
            std::max(stats["affine_bab_frontier_peak"], double(held));
        if (D >= P.S || held * 24 * (D + 1) > o.frontier_bytes) {
            res = 0;
            break;
        }
        // the children: the next neuron of the order inactive and active, its multipliers 0
        const Tensor sg = part.sg.take(open, 0), p = part.p.take(open, 0), q = part.q.take(open, 0);
        const Tensor side = d.ones_like({M, 1}), none = d.zeros_like({2 * M, 1});
        stack.push_back(
            {Tensor::concat(
                 {Tensor::concat({sg, side}, 1), Tensor::concat({sg, side.negative()}, 1)}, 0),
             Tensor::concat({Tensor::concat({p, p}, 0), none}, 1),
             Tensor::concat({Tensor::concat({q, q}, 0), none}, 1)});
    }
    stats["affine_bab_seconds"] += now_seconds() - started;
    return res;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
