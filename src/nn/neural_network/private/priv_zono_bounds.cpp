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
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The optimized slopes ----------------------------------------------------------------------------

/// The best lower bounds (B, R) of the rows (R, n) times layer top's input over `iterations`
/// projected Adam steps of size `step` on the ReLU layers' lower slopes per row, from CROWN's
/// (start: per layer, (B, n)), the lines at `bounds`; none after the deadline; softmax_lse:
/// ZonoBackward's.
Tensor aux_optimized(const ZonoPropagation &out, const ZonoNet &net, const ZonoBounds &bounds,
                     const std::vector<Tensor> &start, std::size_t top, const Tensor &x,
                     const Tensor &rad, const Tensor &rows, int64_t iterations, double step,
                     double deadline, bool softmax_lse) {
    const int64_t B = x.shape()[0], R = rows.shape()[0];
    std::vector<Tensor> alpha(top), m1(top), m2(top);
    if (iterations > 0)
        for (std::size_t k = 0; k < top; ++k)
            if (start[k].defined())
                alpha[k] = start[k].expand_dims(1).broadcast_to({B, R, start[k].shape()[1]}) + 0.0;
    const Tensor none = x.zeros_like({B, 0});
    Tensor best;
    for (int64_t t = 0;; ++t) {
        // one backward pass (priv_backward_lower from layer top), its slopes' gradient if more
        std::vector<Tensor> g;
        const bool more = t < iterations && now_seconds() < std::min(deadline, net.deadline);
        ZonoBackward opt;
        opt.top = int64_t(top);
        opt.bounds = &bounds;
        opt.row_alpha = iterations > 0 ? &alpha : nullptr;
        opt.g_alpha = more ? &g : nullptr;
        opt.softmax_lse = softmax_lse;
        const Tensor value = priv_backward_lower(out, net, none, x, rad, rows, nullptr, &opt);
        // every step's bound holds: the best per row
        best = best.defined() ? best.maximum(value) : value;
        if (!more) return best;
        priv_adam_slopes(alpha, m1, m2, g, t + 1, step);
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The slope steps ---------------------------------------------------------------------------------

void priv_adam_slopes(std::vector<Tensor> &alpha, std::vector<Tensor> &m1, std::vector<Tensor> &m2,
                      const std::vector<Tensor> &g, int64_t t, double step) {
    constexpr double kB1 = 0.9, kB2 = 0.999;
    const double c1 = 1.0 - std::pow(kB1, double(t)), c2 = 1.0 - std::pow(kB2, double(t));
    for (std::size_t k = 0; k < alpha.size(); ++k) {
        if (!alpha[k].defined() || k >= g.size() || !g[k].defined()) continue;
        // the moments, then the step projected onto [0, 1]
        m1[k] = m1[k].defined() ? m1[k] * kB1 + g[k] * (1.0 - kB1) : g[k] * (1.0 - kB1);
        const Tensor g2 = g[k].square() * (1.0 - kB2);
        m2[k] = m2[k].defined() ? m2[k] * kB2 + g2 : g2;
        const Tensor move = (m1[k] / c1).divide((m2[k] / c2).sqrt() + numerics::kAdamEpsilon);
        alpha[k] = (alpha[k] + move * step).maximum(0.0).minimum(1.0);
    }
}

// The bounds of the layer inputs ------------------------------------------------------------------

ZonoBounds priv_crown_bounds(const ZonoPropagation &out, const ZonoNet &net, const Tensor &x,
                             const Tensor &rad, int64_t iterations, double step, int64_t max_bytes,
                             double deadline, bool softmax_lse) {
    const std::size_t L = net.W.size();
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
        std::vector<int64_t> idx;
        const int64_t n = l.shape()[1];
        if (earlier && net.activations[k] != Activation::product &&
            now_seconds() < std::min(deadline, net.deadline)) {
            const std::vector<double> open =
                net.activations[k] == Activation::relu
                    ? l.less(l.zeros_like()).multiply(u.greater(u.zeros_like())).max(0).data()
                    : std::vector<double>(std::size_t(n), 1.0);
            for (int64_t i = 0; i < n; ++i)
                if (open[std::size_t(i)] > 0) idx.push_back(i);
        }
        // the rows in chunks: Lambda (B, 2 c, widest) per layer, the slopes and Adam's 4 more
        const int64_t per_row = 8 * 2 * B * widest * (iterations > 0 ? 6 * int64_t(L) : 2);
        const int64_t chunk = std::max<int64_t>(1, max_bytes / std::max<int64_t>(1, per_row));
        for (std::size_t at = 0; at < idx.size(); at += std::size_t(chunk)) {
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
            const Tensor rows = Tensor::concat({E, E.negative()}, 0);
            const Tensor v = aux_optimized(out, net, bounds, start, k, x, rad, rows, iterations,
                                           step, deadline, softmax_lse);
            // back to the layer's neurons (B, n): the rows' bounds where bounded, else the record's
            const Tensor hit = E.sum(0).greater(l.zeros_like({n}));
            const Tensor lo = v.slice(1, 0, c).matmul(E), hi = v.slice(1, c, 2 * c).matmul(E);
            l = Tensor::where(hit, l.maximum(lo), l);
            u = Tensor::where(hit, u.minimum(hi.negative()), u);
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
