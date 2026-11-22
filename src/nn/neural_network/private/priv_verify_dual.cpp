// priv_verify_dual - the dual certificate of verify's check: a Lagrangian bound under factor rows
//
// The output of a recorded propagation is y = c + G^T beta + delta, beta in [-1, 1]^q and
// |delta| <= dr. A split of a neuron and the faces of a crossing ReLU are rows C beta <= d on the
// same factors (delta taken at its worst), so the margin a^T y of a row is bounded on the part of
// the box that satisfies them. For every lambda >= 0, weak duality gives
//     min { w^T beta : beta in [-1, 1]^q, C beta <= d } >= -|w + C^T lambda|_1 - lambda^T d,
// w = G a; lambda = 0 is the zonotope's bound. lambda follows projected Adam steps on that
// concave function in the search dtype (beta-CROWN's split multipliers, applied to the factors of
// the latent space); the bound of the best lambda is evaluated again in the rows' dtype (float64),
// so it holds for any lambda the steps found.
//
// Syntax:   auto [C, d] = priv_dual_rows(out, net, splits, 32);
//           Tensor low = priv_dual_lower(w, C, d, 20, 0.1, DType::float32);
// See also: priv_zono_batch.h, priv_verify_zonotack.cpp
// References:
//    [1] S. Wang, H. Zhang, K. Xu, X. Lin, S. Jana, C.-J. Hsieh, J. Z. Kolter. "Beta-CROWN:
//        efficient bound propagation with per-neuron split constraints for neural network
//        robustness verification", NeurIPS 2021.
//    [2] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA.cpp only: the dual bound of the check (CORA bounds the factors' box by dual-iter only)
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The rows (B, k, q) of the columns j (B, k) of the generator rows G (B, qi, n), zero-padded.
Tensor aux_columns(const Tensor &G, const Tensor &j, int64_t q) {
    const Tensor R = G.take_along_axis(j.expand_dims(1), 2).permute_dims({0, 2, 1});
    const int64_t extra = q - R.shape()[2];
    if (extra == 0) return R;
    return Tensor::concat({R, R.zeros_like({R.shape()[0], R.shape()[1], extra})}, 2);
}

/// The entries (B, k) of x (B, n) at j (B, k), plus those of dr (if defined) times `sign`.
Tensor aux_at(const Tensor &x, const Tensor &dr, const Tensor &j, double sign) {
    const Tensor v = x.take_along_axis(j, 1);
    return dr.defined() ? v + dr.take_along_axis(j, 1) * sign : v;
}

/// The dual function -|w + lambda C|_1 - lambda d of lambda (B, R, p), and lambda C.
Tensor aux_dual(const Tensor &w, const Tensor &C, const Tensor &d, const Tensor &lambda,
                Tensor *sum) {
    const Tensor s = w + lambda.matmul(C);
    if (sum) *sum = s;
    return s.abs().sum(-1).negative() - lambda.multiply(d.expand_dims(1)).sum(-1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The rows ----------------------------------------------------------------------------------------

std::pair<Tensor, Tensor> priv_dual_rows(const ZonoPropagation &out, const ZonoNet &net,
                                         const Tensor &splits, int64_t num_relu) {
    const int64_t q = net.q;
    std::vector<Tensor> Cs, ds;
    // the splits: -id holds the neuron's input at least 0 (-G beta <= c + dr), +id at most 0
    const bool split = splits.defined() && splits.shape()[1] > 0;
    for (const ZonoLayerRecord &rec : out.layers) {
        if (!split) break;
        const int64_t n = rec.l.shape()[1];
        const Tensor id = splits.abs() - double(rec.first_id);
        const Tensor inside =
            id.greater_equal(id.zeros_like()).multiply(id.less(id.full_like({1}, double(n))));
        const Tensor j = id.clip(0.0, double(n - 1));
        const Tensor sign = Tensor::where(splits.less(splits.zeros_like()),
                                          splits.full_like({1}, -1.0), splits.ones_like());
        // the row of each split of this layer, zero for the others
        const Tensor hit = inside.expand_dims(-1);
        const Tensor rows = aux_columns(rec.G_in, j, q).multiply(sign.expand_dims(-1));
        const Tensor off = aux_at(rec.c_in, Tensor(), j, 1.0).multiply(sign.negative()) +
                           (rec.dr_in.defined() ? rec.dr_in.take_along_axis(j, 1) : j.zeros_like());
        Cs.push_back(Tensor::where(hit, rows, rows.zeros_like()));
        ds.push_back(Tensor::where(inside, off, off.zeros_like()));
    }
    // the three faces of the crossing ReLU neurons of the largest error, over all ReLU layers
    std::vector<const ZonoLayerRecord *> recs;
    for (const ZonoLayerRecord &rec : out.layers)
        if (net.activations[std::size_t(rec.index)] == Activation::relu) recs.push_back(&rec);
    if (num_relu > 0 && !recs.empty()) {
        std::vector<Tensor> scores;
        std::vector<std::pair<Tensor, Tensor>> bounds;
        for (const ZonoLayerRecord *rec : recs) {
            const auto [lc, uc] = split ? net.clamp(rec->first_id, rec->l, rec->u, splits)
                                        : std::pair{rec->l, rec->u};
            const Tensor cross = uc.greater(uc.zeros_like()).multiply(lc.less(lc.zeros_like()));
            scores.push_back(Tensor::where(cross, rec->er, rec->er.full_like({1}, -kInf)));
            bounds.emplace_back(lc, uc);
        }
        // the top neurons over all layers, then each layer's share of them
        const Tensor score = Tensor::concat(scores, 1);
        const int64_t k = std::min(num_relu, score.shape()[1]);
        const auto [best, top] = score.topk(k, 1);
        const Tensor live = best.isfinite();
        int64_t start = 0;
        for (std::size_t i = 0; i < recs.size(); ++i) {
            const ZonoLayerRecord &rec = *recs[i];
            const int64_t n = rec.l.shape()[1];
            const Tensor inside =
                live.multiply(top.greater_equal(top.full_like({1}, double(start))))
                    .multiply(top.less(top.full_like({1}, double(start + n))));
            const Tensor j = (top - double(start)).clip(0.0, double(n - 1));
            const Tensor go = aux_columns(rec.G_out, j, q), gi = aux_columns(rec.G_in, j, q);
            const Tensor l = bounds[i].first.take_along_axis(j, 1);
            const Tensor u = bounds[i].second.take_along_axis(j, 1);
            const Tensor s = u.divide((u - l).maximum(numerics::kTiny));
            // y >= 0, y >= x, y <= s (x - l), each with delta at its worst
            const Tensor co = rec.c_out.take_along_axis(j, 1), ci = rec.c_in.take_along_axis(j, 1);
            const Tensor ro =
                rec.dr_out.defined() ? rec.dr_out.take_along_axis(j, 1) : j.zeros_like();
            const Tensor ri =
                rec.dr_in.defined() ? rec.dr_in.take_along_axis(j, 1) : j.zeros_like();
            const Tensor face_rows[] = {go.negative(), gi - go,
                                        go - gi.multiply(s.expand_dims(-1))};
            const Tensor face_offs[] = {co + ro, co - ci + ro + ri,
                                        s.multiply(ci - l) - co + ro + s.multiply(ri)};
            // each face a row of the neurons of this layer, zero for the others
            for (int f = 0; f < 3; ++f) {
                Cs.push_back(
                    Tensor::where(inside.expand_dims(-1), face_rows[f], face_rows[f].zeros_like()));
                ds.push_back(Tensor::where(inside, face_offs[f], face_offs[f].zeros_like()));
            }
            start += n;
        }
    }
    if (Cs.empty()) return {};
    return {Tensor::concat(Cs, 1), Tensor::concat(ds, 1)};
}

// The bound ---------------------------------------------------------------------------------------

Tensor priv_dual_lower(const Tensor &w, const Tensor &C_, const Tensor &d_, int64_t iterations,
                       double step, DType dtype) {
    // the rows at unit 1-norm (a zero row stays zero), then the steps in the search dtype
    const Tensor norm = C_.abs().sum(-1);
    const Tensor scale = Tensor::where(norm.greater(norm.zeros_like()), norm, norm.ones_like());
    const Tensor C = C_.divide(scale.expand_dims(-1)), d = d_.divide(scale);
    const Tensor w32 = w.to(dtype), C32 = C.to(dtype), d32 = d.to(dtype);
    const Tensor CT32 = C32.matrix_transpose();
    const int64_t B = w.shape()[0], R = w.shape()[1], p = C.shape()[1];
    // the step of each row in units of its own coefficients
    const Tensor lr = w32.abs().max(-1, true) * step;
    Tensor lambda = w32.zeros_like({B, R, p}), m1 = lambda, m2 = lambda, best = lambda;
    Tensor best_value = aux_dual(w32, C32, d32, lambda, nullptr);
    constexpr double kB1 = 0.9, kB2 = 0.999;
    for (int64_t t = 1; t <= iterations; ++t) {
        // the supergradient -sign(w + lambda C) C^T - d, Adam, projected onto lambda >= 0
        Tensor s;
        aux_dual(w32, C32, d32, lambda, &s);
        const Tensor g = s.sign().matmul(CT32).negative() - d32.expand_dims(1);
        m1 = m1 * kB1 + g * (1.0 - kB1);
        m2 = m2 * kB2 + g.square() * (1.0 - kB2);
        const double c1 = 1.0 - std::pow(kB1, double(t)), c2 = 1.0 - std::pow(kB2, double(t));
        const Tensor move = (m1 / c1).divide((m2 / c2).sqrt() + numerics::kAdamEpsilon);
        lambda = (lambda + move.multiply(lr)).maximum(0.0);
        // the best lambda of each row so far
        const Tensor value = aux_dual(w32, C32, d32, lambda, nullptr);
        const Tensor better = value.greater(best_value);
        best_value = Tensor::where(better, value, best_value);
        best = Tensor::where(better.expand_dims(-1), lambda, best);
    }
    // the certificate: the bound of the best lambda in the rows' dtype, at least lambda = 0's
    const Tensor at_best = aux_dual(w, C, d, best.to(w.dtype()).maximum(0.0), nullptr);
    return at_best.maximum(w.abs().sum(-1).negative());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
