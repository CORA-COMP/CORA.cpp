// priv_zono_bounds - CROWN's bounds of the inputs of the activation layers, for the zonotope
//
// See priv_zono_batch.h. Layer by layer from the input, the input z_k of activation layer k is
// bounded by a backward pass (CROWN) through the earlier layers' relaxations at their bounds so
// far: a ReLU's chord above and a lower slope in [0, 1] below (CROWN's 0 or 1, or alpha-CROWN's
// optimized per row), another activation between the parallels of the zonotope's line. Each
// bound is intersected with the zonotope's; ZonoNet::forward then builds its lines at the
// intersection. Every bound holds for the true inputs of the boxes, so any line valid on it is.
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

// The relaxations ---------------------------------------------------------------------------------

/// A layer's relaxation sl z + bl <= f(z) <= su z + bu (B, n) (a product: its one line su, bl,
/// bu), cross: a ReLU's crossing neurons (its lower slope free in [0, 1]).
struct Relax {
    Tensor sl, su, bl, bu, cross;
};

/// The relaxation of layer k over [l, u]: a ReLU's chord and CROWN's lower slope, else the
/// parallels of the zonotope's line (a product's on the record's input set).
Relax aux_relax(const ZonoNet &net, std::size_t k, const ZonoLayerRecord &rec, const Tensor &l,
                const Tensor &u) {
    Relax r;
    const Activation act = net.activations[k];
    if (act == Activation::relu) {
        const Tensor zero = l.zeros_like(), on = l.greater_equal(zero);
        r.cross = u.greater(zero).multiply(l.less(zero));
        const Tensor chord = u.divide((u - l).maximum(numerics::kTiny));
        r.su = Tensor::where(r.cross, chord, on);
        r.bu = Tensor::where(r.cross, chord.multiply(l).negative(), zero);
        r.sl = Tensor::where(r.cross, u.greater(l.negative()), on);
        r.bl = zero;
        return r;
    }
    // another activation: its zonotope line's lower and upper parallels
    const PrivLine line = act == Activation::product
                              ? priv_product_line(rec.c_in, rec.G_in, rec.dr_in)
                              : priv_verify_line(act, l, u, net.poly_method);
    r.sl = r.su = line.slope;
    r.bl = line.offset - line.radius;
    r.bu = line.offset + line.radius;
    return r;
}

/// The lower slope of a relaxation per row (B, R, n): alpha where the ReLU crosses, else sl.
Tensor aux_lower_slope(const Relax &r, const Tensor &alpha) {
    const Tensor sl = r.sl.expand_dims(1);
    return alpha.defined() ? Tensor::where(r.cross.expand_dims(1), alpha, sl) : sl;
}

// The backward pass -------------------------------------------------------------------------------

/// CROWN's lower bounds (B, R) of lam_top (B, R, n) times the input of layer top over the boxes
/// (x, rad), through the relaxations rx of the earlier layers (alpha: a ReLU's lower slopes per
/// row). With g_alpha, the bound's gradient in each alpha (0 off the crossing).
Tensor aux_backward(const ZonoNet &net, const std::vector<Relax> &rx, std::size_t top,
                    const Tensor &x, const Tensor &rad, const Tensor &lam_top,
                    const std::vector<Tensor> &alpha, std::vector<Tensor> *g_alpha) {
    const int64_t B = x.shape()[0], R = lam_top.shape()[1];
    // Lambda of each output (acc[j + 1]; 0: the input), summed over the layers that read it
    std::vector<Tensor> acc(top + 1), lams(top);
    Tensor constant = lam_top.multiply(net.b[top]).sum(-1) + x.zeros_like({B, R});
    net.linear_t(top, lam_top, acc);
    for (std::size_t k = top; k-- > 0;) {
        check_deadline(net.deadline);
        Tensor lam = std::move(acc[k + 1]);
        if (!lam.defined()) continue; // an output that no layer up to top reads
        const Activation act = net.activations[k];
        if (act != Activation::identity) {
            const Relax &r = rx[k];
            lams[k] = lam;
            const Tensor pos = lam.maximum(0.0), neg = lam.minimum(0.0);
            constant = constant + pos.multiply(r.bl.expand_dims(1)).sum(-1) +
                       neg.multiply(r.bu.expand_dims(1)).sum(-1);
            lam = act != Activation::product ? pos.multiply(aux_lower_slope(r, alpha[k])) +
                                                   neg.multiply(r.su.expand_dims(1))
                                             : priv_slope_times_t(act, r.su.expand_dims(1), lam);
        }
        constant = constant + lam.multiply(net.b[k]).sum(-1);
        net.linear_t(k, lam, acc);
    }
    const Tensor &lam0 = acc[0];
    const Tensor value = lam0.matmul(x.expand_dims(-1)).squeeze(-1) -
                         lam0.abs().matmul(rad.expand_dims(-1)).squeeze(-1) + constant;
    if (!g_alpha) return value;
    // the reverse: each layer's input at the box's minimizing corner through the linear bounds
    std::vector<Tensor> gs{x.expand_dims(1) - lam0.sign().multiply(rad.expand_dims(1))};
    for (std::size_t k = 0; k < top; ++k) {
        check_deadline(net.deadline);
        const Tensor g = net.linear(k, gs) + net.b[k];
        const Activation act = net.activations[k];
        if (act == Activation::identity || !lams[k].defined()) {
            gs.push_back(g); // an output off the pass: never read by a layer on it
            continue;
        }
        const Relax &r = rx[k];
        const Tensor &lk = lams[k];
        const Tensor pos = lk.greater(lk.zeros_like()), neg = lk.less(lk.zeros_like());
        if (alpha[k].defined())
            (*g_alpha)[k] =
                Tensor::where(r.cross.expand_dims(1), g.multiply(lk.maximum(0.0)), g.zeros_like());
        const Tensor bl = r.bl.expand_dims(1), su = r.su.expand_dims(1), bu = r.bu.expand_dims(1);
        if (act != Activation::product) {
            gs.push_back(pos.multiply(g.multiply(aux_lower_slope(r, alpha[k])) + bl) +
                         neg.multiply(g.multiply(su) + bu));
        } else {
            const Tensor gm = priv_slope_times(act, su, g);
            gs.push_back(pos.multiply(gm + bl) + neg.multiply(gm + bu));
        }
    }
    return value;
}

/// The best lower bounds (B, R) of lam_top over `iterations` projected Adam steps of size `step`
/// on the ReLU layers' lower slopes per row (from CROWN's), none after the deadline.
Tensor aux_optimized(const ZonoNet &net, const std::vector<Relax> &rx, std::size_t top,
                     const Tensor &x, const Tensor &rad, const Tensor &lam_top, int64_t iterations,
                     double step, double deadline) {
    const int64_t B = x.shape()[0], R = lam_top.shape()[1];
    std::vector<Tensor> alpha(top), m1(top), m2(top);
    if (iterations > 0)
        for (std::size_t k = 0; k < top; ++k)
            if (rx[k].cross.defined())
                alpha[k] = rx[k].sl.expand_dims(1).broadcast_to({B, R, rx[k].sl.shape()[1]}) + 0.0;
    Tensor best;
    for (int64_t t = 0;; ++t) {
        std::vector<Tensor> g(top);
        const bool more = t < iterations && now_seconds() < std::min(deadline, net.deadline);
        const Tensor value =
            aux_backward(net, rx, top, x, rad, lam_top, alpha, more ? &g : nullptr);
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
                             double deadline) {
    const std::size_t L = net.W.size();
    const int64_t B = x.shape()[0];
    ZonoBounds bounds;
    bounds.l.resize(L);
    bounds.u.resize(L);
    std::vector<Relax> rx(L);
    int64_t widest = net.n0;
    for (const int64_t w : net.widths) widest = std::max(widest, w);
    bool earlier = false; // an activation before: the first one's input is exact on the box
    std::size_t next = 0;
    for (std::size_t k = 0; k < L; ++k) {
        if (net.activations[k] == Activation::identity) continue;
        const ZonoLayerRecord &rec = out.layers[next++];
        Tensor l = rec.l, u = rec.u;
        // the neurons to bound: a ReLU's crossing ones, every one of another activation
        std::vector<int64_t> idx;
        const int64_t n = l.shape()[1];
        if (earlier && now_seconds() < std::min(deadline, net.deadline)) {
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
            const Tensor ids = Tensor::like(l, std::vector<double>(part.begin(), part.end()), {c});
            const Tensor E = ids.expand_dims(-1).equal(l.arange_like(n).expand_dims(0)) + 0.0;
            const Tensor lam =
                Tensor::concat({E, E.negative()}, 0).expand_dims(0).broadcast_to({B, 2 * c, n});
            const Tensor v = aux_optimized(net, rx, k, x, rad, lam, iterations, step, deadline);
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
        rx[k] = aux_relax(net, k, rec, l, u);
        earlier = true;
    }
    return bounds;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
