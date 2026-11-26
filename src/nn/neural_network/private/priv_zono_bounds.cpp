// priv_zono_bounds - CROWN's bounds of the inputs of the activation layers, for the zonotope
//
// See priv_zono_batch.h. Layer by layer from the input, the input z_k of activation layer k is
// bounded by priv_backward_lower from layer k (CROWN) through the earlier layers' relaxations at
// their bounds so far: a ReLU's chord above and a lower slope in [0, 1] below (CROWN's 0 or 1, or
// alpha-CROWN's optimized per row), another activation between the parallels of the zonotope's
// line. Each bound is intersected with the zonotope's; ZonoNet::forward then builds its lines at
// the intersection. Every bound holds for the true inputs of the boxes, so any line valid on it is.
// The optimized slopes keep each row's best bound over all steps (each step's bound holds).
// References:
//    [1] H. Zhang, T.-W. Weng, P.-Y. Chen, C.-J. Hsieh, L. Daniel. "Efficient neural network
//        robustness certification with general activation functions", NeurIPS 2018.
//    [2] K. Xu, H. Zhang, S. Wang, Y. Wang, S. Jana, X. Lin, C.-J. Hsieh. "Fast and complete:
//        enabling complete neural network verification with rapid and massively parallel
//        incomplete verifiers", ICLR 2021.
//
// CORA.cpp only: CORA's verify takes the zonotope's own bounds
// Authors:       Claude Opus 5.5
// Written:       23-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "global/tensor/device_memory.h"
#include "nn/neural_network/private/priv_verify.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The optimized slopes ----------------------------------------------------------------------------

/// The best lower bounds (B, R) of the rows (R, n) times layer top's input over `iterations`
/// projected Adam steps of size `step` on the ReLU layers' lower slopes per row from CROWN's
/// (start: per layer, (B, n)), the lines at `bounds`; none after the deadline.
Tensor aux_optimized(const ZonoPropagation &out, const ZonoNet &net, const ZonoBounds &bounds,
                     const std::vector<Tensor> &start, std::size_t top, const Tensor &x,
                     const Tensor &rad, const Tensor &rows, int64_t iterations, double step,
                     double deadline, bool softmax_lse, bool products, double product_start,
                     double steps_until) {
    // the steps' time over: plain passes, no slopes or planes per row
    if (now_seconds() >= steps_until) iterations = 0;
    const int64_t B = x.shape()[0], R = rows.shape()[0];
    std::vector<Tensor> alpha(top), m1(top), m2(top);
    if (iterations > 0)
        for (std::size_t k = 0; k < top; ++k)
            if (start[k].defined())
                alpha[k] = start[k].expand_dims(1).broadcast_to({B, R, start[k].shape()[1]}) + 0.0;
    // the earlier product layers' planes per row: from the set's line (mix 0), side 1/2
    std::vector<Tensor> mix(top), side(top), x1(top), x2(top), s1(top), s2(top);
    bool planes = false;
    for (std::size_t k = 0; k < top && products && iterations > 0; ++k) {
        if (net.activations[k] != Activation::product) continue;
        mix[k] = x.full_like({B, R, net.widths[k] / 2}, product_start);
        side[k] = mix[k] * 0.0 + 0.5;
        planes = true;
    }
    const Tensor none = x.zeros_like({B, 0});
    Tensor best;
    for (int64_t t = 0;; ++t) {
        // one backward pass (priv_backward_lower from layer top), its slopes' gradient if more
        std::vector<Tensor> g, gm, gs;
        const bool more =
            t < iterations && now_seconds() < std::min({deadline, net.deadline, steps_until});
        ZonoBackward opt;
        opt.top = int64_t(top);
        opt.bounds = &bounds;
        opt.row_alpha = iterations > 0 ? &alpha : nullptr;
        opt.g_alpha = more ? &g : nullptr;
        opt.softmax_lse = softmax_lse;
        if (planes) {
            opt.product_mix = &mix;
            opt.product_side = &side;
            opt.g_product_mix = more ? &gm : nullptr;
            opt.g_product_side = more ? &gs : nullptr;
        }
        const Tensor value = priv_backward_lower(out, net, none, x, rad, rows, nullptr, &opt);
        // every step's bound holds: the best per row
        best = best.defined() ? best.maximum(value) : value;
        if (!more) return best;
        priv_adam_slopes(alpha, m1, m2, g, t + 1, step);
        priv_adam_slopes(mix, x1, x2, gm, t + 1, step);
        priv_adam_slopes(side, s1, s2, gs, t + 1, step);
    }
}

/// Per neuron of layer k's input the first neuron with the same affine form of the earlier
/// activations' outputs and the input (a gather's copies: a product's operands); empty: not
/// looked for (the rows beyond max_bytes).
std::vector<int64_t> aux_same_forms(const ZonoNet &net, std::size_t k, const Tensor &x,
                                    int64_t max_bytes) {
    const int64_t n = net.widths[k];
    int64_t held = 8 * n * n; // the bytes of the rows so far
    if (held > max_bytes) return {};
    // the unit rows back through the linear layers, stopped at each activation's output
    std::vector<Tensor> lams(k + 1), parts;
    Tensor constant = x.zeros_like({1, n});
    const Tensor eye = x.eye_like(n).expand_dims(0);
    constant = constant + eye.multiply(net.b[k]).sum(-1);
    net.linear_t(k, eye, lams);
    for (std::size_t j = k; j-- > 0;) {
        Tensor lam = std::move(lams[j + 1]);
        if (!lam.defined()) continue;
        held += 8 * n * lam.shape()[2];
        if (held > max_bytes) return {};
        if (net.activations[j] != Activation::identity) {
            parts.push_back(lam);
            continue;
        }
        constant = constant + lam.multiply(net.b[j]).sum(-1);
        net.linear_t(j, lam, lams);
    }
    if (lams[0].defined()) parts.push_back(lams[0]);
    parts.push_back(constant.expand_dims(-1));
    // the rows' weighted sums (xorshift weights): equal sums are compared row by row below
    const Tensor rows = Tensor::concat(parts, -1).squeeze(0);
    const int64_t m = rows.shape()[1];
    std::vector<double> weights(static_cast<std::size_t>(m));
    uint64_t state = 88172645463325252ULL;
    for (double &w : weights) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        w = double(state >> 11) / 9007199254740992.0 + 0.5;
    }
    // per sum its first row
    const std::vector<double> sums =
        rows.matmul(Tensor::like(x, weights, {m, 1})).squeeze(-1).data();
    std::map<double, int64_t> first;
    std::vector<int64_t> same(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        same[std::size_t(i)] = first.emplace(sums[std::size_t(i)], i).first->second;
    // a row that differs from its sum's first keeps itself
    const std::vector<double> equal = rows.take(same, 0).equal(rows).min(-1).data();
    for (int64_t i = 0; i < n; ++i)
        if (equal[std::size_t(i)] <= 0) same[std::size_t(i)] = i;
    return same;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The slope steps ---------------------------------------------------------------------------------

void priv_adam_slopes(std::vector<Tensor> &alpha, std::vector<Tensor> &m1, std::vector<Tensor> &m2,
                      const std::vector<Tensor> &g, int64_t t, double step, double upper) {
    constexpr double kB1 = 0.9, kB2 = 0.999;
    const double c1 = 1.0 - std::pow(kB1, double(t)), c2 = 1.0 - std::pow(kB2, double(t));
    for (std::size_t k = 0; k < alpha.size(); ++k) {
        if (!alpha[k].defined() || k >= g.size() || !g[k].defined()) continue;
        // the moments, then the step projected onto [0, upper]
        m1[k] = m1[k].defined() ? m1[k] * kB1 + g[k] * (1.0 - kB1) : g[k] * (1.0 - kB1);
        const Tensor g2 = g[k].square() * (1.0 - kB2);
        m2[k] = m2[k].defined() ? m2[k] * kB2 + g2 : g2;
        const Tensor move = (m1[k] / c1).divide((m2[k] / c2).sqrt() + numerics::kAdamEpsilon);
        alpha[k] = (alpha[k] + move * step).maximum(0.0).minimum(upper);
    }
}

// The bounds of the layer inputs ------------------------------------------------------------------

ZonoBounds priv_crown_bounds(const ZonoPropagation &out, const ZonoNet &net, const Tensor &x,
                             const Tensor &rad, int64_t iterations, double step, int64_t max_bytes,
                             double deadline, bool softmax_lse, int64_t *backoffs, bool products,
                             double product_start, double steps_until) {
    const std::size_t L = net.W.size();
    int64_t steps = iterations; // 0 once out of memory at one row
    const int64_t B = x.shape()[0];
    ZonoBounds bounds;
    bounds.l.resize(L);
    bounds.u.resize(L);
    std::vector<Tensor> start(L); // a ReLU's CROWN lower slope at its bounds: 1 where u > -l
    int64_t widest = net.n0;
    for (const int64_t w : net.widths) widest = std::max(widest, w);
    bool earlier = false; // an activation before: the first one's input is exact on the box
    std::size_t next = 0;
    for (std::size_t k = 0; k < L; ++k) {
        if (net.activations[k] == Activation::identity) continue;
        const ZonoLayerRecord &rec = out.layers[next++];
        Tensor l = rec.l, u = rec.u;
        // the neurons to bound: a ReLU's crossing ones, all of another but a product (set line)
        std::vector<int64_t> idx, same;
        const int64_t n = l.shape()[1];
        if (earlier && (products || net.activations[k] != Activation::product) &&
            now_seconds() < std::min(deadline, net.deadline)) {
            const std::vector<double> open =
                net.activations[k] == Activation::relu
                    ? l.less(l.zeros_like()).multiply(u.greater(u.zeros_like())).max(0).data()
                    : std::vector<double>(std::size_t(n), 1.0);
            // a product's operands: one of the copies of a form is bounded, the others take its
            if (net.activations[k] == Activation::product)
                same = aux_same_forms(net, k, x, max_bytes);
            for (int64_t i = 0; i < n; ++i)
                if (open[std::size_t(i)] > 0 && (same.empty() || same[std::size_t(i)] == i))
                    idx.push_back(i);
        }
        // the rows in chunks: Lambda (B, 2 c, widest); with steps the reverse's and the slopes'
        int64_t stepped = 0;
        for (std::size_t j = 0; j < k && iterations > 0; ++j) {
            const Activation a = net.activations[j];
            if (a != Activation::identity) stepped += 3 * net.widths[j];
            if (a == Activation::relu || (products && a == Activation::product))
                stepped += 5 * net.widths[j];
        }
        const int64_t per_row = 8 * 2 * B * ((iterations > 0 ? 4 : 2) * widest + stepped);
        const int64_t per_plain = 8 * 2 * B * 2 * widest;
        int64_t chunk = std::max<int64_t>(1, max_bytes / std::max<int64_t>(1, per_row));
        int64_t halved = 1; // by running out of memory
        for (std::size_t at = 0; at < idx.size();) {
            // the steps' time over: plain passes, as many rows as those take
            if (steps > 0 && now_seconds() >= steps_until)
                chunk = std::max<int64_t>(1, max_bytes / std::max<int64_t>(1, per_plain) / halved);
            const std::size_t to = std::min(idx.size(), at + std::size_t(chunk));
            const std::vector<int64_t> part(idx.begin() + std::ptrdiff_t(at),
                                            idx.begin() + std::ptrdiff_t(to));
            const int64_t c = int64_t(part.size());
            // the rows e_i and -e_i: the lower bound of -z_i is minus its upper bound
            std::vector<double> part_d;
            part_d.reserve(part.size());
            for (const int64_t i : part) part_d.push_back(double(i));
            const Tensor ids = Tensor::like(l, part_d, {c});
            const Tensor E = ids.expand_dims(-1).equal(l.arange_like(n).expand_dims(0)) + 0.0;
            Tensor v;
            try {
                priv_oom_site("crown_bounds");
                const Tensor rows = Tensor::concat({E, E.negative()}, 0);
                v = aux_optimized(out, net, bounds, start, k, x, rad, rows, steps, step, deadline,
                                  softmax_lse, products, product_start, steps_until);
            } catch (const std::exception &error) {
                // out of memory: half the rows, then no slope steps, then the record's bounds
                if (!priv_is_oom(error)) std::rethrow_exception(std::current_exception());
                free_device_cache(x.device_id());
                if (backoffs) ++*backoffs;
                halved *= 2;
                if (chunk > 1)
                    chunk /= 2;
                else if (steps > 0)
                    steps = 0;
                else
                    at = to; // these neurons keep the record's bounds (they hold)
                continue;
            }
            // back to the layer's neurons (B, n): the rows' bounds where bounded, else the record's
            const Tensor hit = E.sum(0).greater(l.zeros_like({n}));
            const Tensor lo = v.slice(1, 0, c).matmul(E), hi = v.slice(1, c, 2 * c).matmul(E);
            l = Tensor::where(hit, l.maximum(lo), l);
            u = Tensor::where(hit, u.minimum(hi.negative()), u);
            at = to;
        }
        if (!same.empty()) {
            // the copies' bounds: the bounded one's, 8 ulps wider (the forms' own rounding)
            const Tensor wide = (l.abs() + u.abs()).take(same, -1) * (8.0 * machine_eps(l.dtype()));
            l = (l.take(same, -1) - wide).maximum(rec.l);
            u = (u.take(same, -1) + wide).minimum(rec.u);
        }
        // a bound beyond the float range or an empty intersection (the rounding): the record's
        const Tensor ok = l.isfinite().multiply(u.isfinite()).multiply(l.less_equal(u));
        l = Tensor::where(ok, l, rec.l);
        u = Tensor::where(ok, u, rec.u);
        bounds.l[k] = l;
        bounds.u[k] = u;
        if (net.activations[k] == Activation::relu) start[k] = u.greater(l.negative());
        earlier = true;
    }
    return bounds;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
