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
// On a frozen zonotope the nodes of one depth share their split rows (priv_dual_lower_signs).
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
// Last update:   23-November-2026
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <array>
#include <cmath>
#include <functional>
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

/// Projected Adam ascent of a concave dual on lambda >= 0, both giving its value and (with true)
/// its supergradient at lambda, steps of lr per row from lambda; the best lambda of each row.
Tensor
aux_ascent_joint(Tensor lambda, const Tensor &lr, int64_t iterations,
                 const std::function<std::pair<Tensor, Tensor>(const Tensor &, bool)> &both) {
    Tensor m1 = lambda.zeros_like(), m2 = m1, best = lambda;
    auto [best_value, g] = both(lambda, iterations > 0);
    constexpr double kB1 = 0.9, kB2 = 0.999;
    for (int64_t t = 1; t <= iterations; ++t) {
        m1 = m1 * kB1 + g * (1.0 - kB1);
        m2 = m2 * kB2 + g.square() * (1.0 - kB2);
        const double c1 = 1.0 - std::pow(kB1, double(t)), c2 = 1.0 - std::pow(kB2, double(t));
        const Tensor move = (m1 / c1).divide((m2 / c2).sqrt() + numerics::kAdamEpsilon);
        lambda = (lambda + move.multiply(lr)).maximum(0.0);
        // the best lambda of each row so far, and the next step's supergradient
        const auto [v, next] = both(lambda, t < iterations);
        const Tensor better = v.greater(best_value);
        best_value = Tensor::where(better, v, best_value);
        best = Tensor::where(better.expand_dims(-1), lambda, best);
        g = next;
    }
    return best;
}

/// aux_ascent_joint of a separate value and supergradient grad.
Tensor aux_ascent(const Tensor &lambda, const Tensor &lr, int64_t iterations,
                  const std::function<Tensor(const Tensor &)> &value,
                  const std::function<Tensor(const Tensor &)> &grad) {
    return aux_ascent_joint(lambda, lr, iterations, [&](const Tensor &l, bool with_grad) {
        return std::pair{value(l), with_grad ? grad(l) : Tensor()};
    });
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

// The table of the splits -------------------------------------------------------------------------

PrivSplitTable priv_split_table(const ZonoPropagation &out, const ZonoNet &net) {
    const int64_t q = net.q;
    PrivSplitTable t;
    std::vector<Tensor> rows, cs, drs, ers, outs, couts, drouts, ls, us;
    std::vector<double> ids, cols;
    for (const ZonoLayerRecord &rec : out.layers) {
        if (net.activations[std::size_t(rec.index)] != Activation::relu) continue;
        // the neurons whose input crosses 0, on the host: the table is made once
        const int64_t n = rec.l.shape()[1];
        const std::vector<double> l = rec.l.data(), u = rec.u.data();
        std::vector<int64_t> j;
        for (int64_t k = 0; k < n; ++k)
            if (l[std::size_t(k)] < 0 && u[std::size_t(k)] > 0) j.push_back(k);
        if (j.empty()) continue;
        const Tensor G = rec.G_in.take(j, 2).squeeze(0).matrix_transpose(); // (|j|, qi)
        const int64_t extra = q - G.shape()[1];
        rows.push_back(extra == 0 ? G
                                  : Tensor::concat({G, G.zeros_like({G.shape()[0], extra})}, 1));
        cs.push_back(rec.c_in.take(j, 1).squeeze(0));
        drs.push_back(rec.dr_in.defined() ? rec.dr_in.take(j, 1).squeeze(0)
                                          : cs.back().zeros_like());
        ers.push_back(rec.er.take(j, 1).squeeze(0));
        ls.push_back(rec.l.take(j, 1).squeeze(0));
        us.push_back(rec.u.take(j, 1).squeeze(0));
        // its output's rows, padded as its input's: one beta for both
        const Tensor Go = rec.G_out.take(j, 2).squeeze(0).matrix_transpose();
        const int64_t extra_out = q - Go.shape()[1];
        outs.push_back(extra_out == 0
                           ? Go
                           : Tensor::concat({Go, Go.zeros_like({Go.shape()[0], extra_out})}, 1));
        couts.push_back(rec.c_out.take(j, 1).squeeze(0));
        drouts.push_back(rec.dr_out.defined() ? rec.dr_out.take(j, 1).squeeze(0)
                                              : couts.back().zeros_like());
        for (const int64_t k : j) ids.push_back(double(rec.first_id + k));
        // the slots' factors follow the layer's input's (the crossing without one: dr only)
        std::vector<int64_t> slot(std::size_t(n), -1);
        if (rec.slot_neurons.defined()) {
            const std::vector<double> sn = rec.slot_neurons.data();
            for (std::size_t i = 0; i < sn.size(); ++i)
                slot[std::size_t(sn[i])] = rec.G_in.shape()[1] + int64_t(i);
        }
        for (const int64_t k : j) {
            t.slotless += slot[std::size_t(k)] < 0 ? 1 : 0;
            cols.push_back(double(slot[std::size_t(k)]));
        }
    }
    // the table, empty if no neuron crosses
    t.ids = Tensor::like(out.c, ids, {int64_t(ids.size())});
    if (ids.empty()) {
        t.rows = out.c.zeros_like({0, q});
        t.c = t.dr = t.er = t.c_out = t.dr_out = t.col = t.l = t.u = out.c.zeros_like({0});
        t.rows_out = t.rows;
        return t;
    }
    t.rows = Tensor::concat(rows, 0);
    t.c = Tensor::concat(cs, 0);
    t.dr = Tensor::concat(drs, 0);
    t.er = Tensor::concat(ers, 0);
    t.rows_out = Tensor::concat(outs, 0);
    t.c_out = Tensor::concat(couts, 0);
    t.dr_out = Tensor::concat(drouts, 0);
    t.col = Tensor::like(out.c, cols, {int64_t(cols.size())});
    t.l = Tensor::concat(ls, 0);
    t.u = Tensor::concat(us, 0);
    return t;
}

// The bound of the nodes of one depth -------------------------------------------------------------

// node n: s_j G_j beta <= -s_j c_j + dr_j, so lambda C is (lambda * s) R (nn_verification.md)

Tensor priv_dual_lower_signs(const Tensor &w, const Tensor &R_, const Tensor &c_, const Tensor &dr_,
                             const Tensor &s, int64_t iterations, double step, DType dtype,
                             const Tensor &start, Tensor *best_out) {
    const int64_t N = s.shape()[0], K = w.shape()[0], d = R_.shape()[0];
    // the rows at unit 1-norm (the signs keep it), then node n's offsets -s c + dr
    const Tensor norm = R_.abs().sum(-1);
    const Tensor scale = Tensor::where(norm.greater(norm.zeros_like()), norm, norm.ones_like());
    const Tensor R = R_.divide(scale.expand_dims(-1));
    const Tensor off = (s.multiply(c_.negative()) + dr_).divide(scale); // (N, d)
    // L(lambda) = -|w + (lambda s) R|_1 - sum lambda off, per node and row
    const auto dual = [&](const Tensor &ww, const Tensor &RR, const Tensor &oo, const Tensor &ss,
                          const Tensor &lambda, Tensor *sum) {
        const Tensor ls = lambda.multiply(ss.expand_dims(1)); // (N, K, d)
        const Tensor t = ww.expand_dims(0) + ls.matmul(RR);   // (N, K, q)
        if (sum) *sum = t;
        return t.abs().sum(-1).negative() - lambda.multiply(oo.expand_dims(1)).sum(-1);
    };
    const Tensor w32 = w.to(dtype), R32 = R.to(dtype), off32 = off.to(dtype), s32 = s.to(dtype);
    const Tensor RT32 = R32.matrix_transpose();
    const Tensor zero = w32.zeros_like({N, K, d});
    const Tensor lambda = start.defined() ? start.to(dtype).maximum(0.0) : zero;
    const Tensor lr = w32.abs().max(-1, true).expand_dims(0) * step; // (1, K, 1)
    // the supergradient -(sign(w + lambda C) R^T) s - off
    const Tensor best = aux_ascent(
        lambda, lr, iterations,
        [&](const Tensor &l) { return dual(w32, R32, off32, s32, l, nullptr); },
        [&](const Tensor &l) {
            Tensor sum;
            dual(w32, R32, off32, s32, l, &sum);
            return sum.sign().matmul(RT32).multiply(s32.expand_dims(1)).negative() -
                   off32.expand_dims(1);
        });
    if (best_out) *best_out = best;
    // the certificate: the best lambda / scale on the unscaled rows (the scaled ones are rounded)
    const Tensor unscaled = best.to(w.dtype()).maximum(0.0).divide(scale);
    const Tensor at_best =
        dual(w, R_, s.multiply(c_.negative()) + dr_, s.to(w.dtype()), unscaled, nullptr);
    return at_best.maximum(w.abs().sum(-1).negative().expand_dims(0));
}

// The bound of the nodes with the phases of their splits ------------------------------------------

// a split's halfspace and phase (y = 0 or y = z), an unsplit one's faces (nn_verification.md)

Tensor priv_dual_lower_phases(const Tensor &w, const PrivSplitTable &t, const Tensor &s,
                              int64_t iterations, double step, DType dtype, const Tensor &start,
                              Tensor *best_out, Tensor *minimizer) {
    const int64_t N = s.shape()[0], K = w.shape()[0], d = t.rows.shape()[0];
    // the states (N, d): active (s = -1), inactive (+1), unsplit (0: the faces)
    const Tensor a = s.less(s.zeros_like()), f = s.equal(s.zeros_like()), split = 1.0 - f;
    const auto row = [&](const Tensor &v) { return v.expand_dims(0).broadcast_to({N, d}); };
    // the chord's slope u / (u - l), rounded up (any larger slope is a true face too)
    Tensor sigma = t.c.zeros_like();
    if (t.l.defined()) {
        const Tensor width = (t.u - t.l).maximum(numerics::kTiny);
        sigma = t.u.maximum(0.0).divide(width) * (1.0 + 4 * machine_eps());
    }
    // block b's row alpha_b G_in + gamma_b G_out (N, d), its offset's coefficients of c, dr, ...
    const Tensor one = f.ones_like(), sg = row(sigma);
    const Tensor al[3] = {s, f - a, a - f.multiply(sg)};
    const Tensor ga[3] = {f.negative(), 1.0 - f * 2.0, f * 2.0 - 1.0};
    const Tensor k_c[3] = {split.multiply(s.negative()), a - f, f.multiply(sg) - a};
    const Tensor k_dr[3] = {split, a + f, a + f.multiply(sg)};
    const Tensor k_co[3] = {f, a.negative() + f - (split - a), (split - a) + a - f};
    const Tensor k_dro[3] = {f, one, one};
    const Tensor k_l[3] = {f.zeros_like(), f.zeros_like(), f.multiply(sg).negative()};
    // each row's 1-norm (its multiplier's unit; a zero row keeps 1)
    const auto unit = [](const Tensor &n) {
        return Tensor::where(n.greater(n.zeros_like()), n, n.ones_like());
    };
    const Tensor n_in = row(t.rows.abs().sum(-1)), n_out = row(t.rows_out.abs().sum(-1));
    const Tensor n_on = row((t.rows_out - t.rows).abs().sum(-1));
    const Tensor n_chord = row((t.rows_out - t.rows.multiply(sigma.expand_dims(-1))).abs().sum(-1));
    // the norms: G_in, G_out (y >= 0); G_out (inactive), G_out - G_in (active, y >= z), the chord
    const Tensor norms[3] = {unit(Tensor::where(f, n_out, n_in)),
                             unit(Tensor::where(split - a, n_out, n_on)),
                             unit(Tensor::where(f, n_chord, Tensor::where(a, n_on, n_out)))};
    // the data in a dtype: the steps' (the search dtype) and the certificate's (the table's)
    struct Data {
        Tensor w, Gin, Gout, al[3], ga[3], off[3], n[3];
    };
    const auto data = [&](DType to) {
        Data x;
        x.w = w.to(to);
        x.Gin = t.rows.to(to);
        x.Gout = t.rows_out.to(to);
        for (int b = 0; b < 3; ++b) {
            x.al[b] = al[b].to(to);
            x.ga[b] = ga[b].to(to);
            x.n[b] = norms[b].to(to);
            Tensor o = k_c[b].multiply(row(t.c)) + k_dr[b].multiply(row(t.dr)) +
                       k_co[b].multiply(row(t.c_out)) + k_dro[b].multiply(row(t.dr_out));
            if (t.l.defined()) o = o + k_l[b].multiply(row(t.l));
            x.off[b] = o.to(to);
        }
        return x;
    };
    // the multipliers lambda_b = nu_b / norm_b (N, K, d) of nu (N, K, 3 d)
    const auto lambdas = [&](const Data &x, const Tensor &nu) {
        std::array<Tensor, 3> l;
        for (int b = 0; b < 3; ++b)
            l[std::size_t(b)] = nu.slice(2, b * d, (b + 1) * d).divide(x.n[b].expand_dims(1));
        return l;
    };
    // L(nu) = -|w + P G_in + Q G_out|_1 - sum_b lambda_b off_b
    const auto value = [&](const Data &x, const Tensor &nu, Tensor *sum) {
        const auto l = lambdas(x, nu);
        Tensor P, Q, offs;
        for (int b = 0; b < 3; ++b) {
            const Tensor lb = l[std::size_t(b)];
            const Tensor p = lb.multiply(x.al[b].expand_dims(1)),
                         qq = lb.multiply(x.ga[b].expand_dims(1));
            const Tensor o = lb.multiply(x.off[b].expand_dims(1));
            P = P.defined() ? P + p : p;
            Q = Q.defined() ? Q + qq : qq;
            offs = offs.defined() ? offs + o : o;
        }
        const Tensor tt = x.w.expand_dims(0) + P.matmul(x.Gin) + Q.matmul(x.Gout);
        if (sum) *sum = tt;
        return tt.abs().sum(-1).negative() - offs.sum(-1);
    };
    const Data d32 = data(dtype);
    const Tensor GinT = d32.Gin.matrix_transpose(), GoutT = d32.Gout.matrix_transpose();
    // the step in units of the row's largest coefficient (1 for w = 0: Farkas' certificate)
    const Tensor wmax = d32.w.abs().max(-1, true);
    const Tensor lr =
        Tensor::where(wmax.greater(wmax.zeros_like()), wmax, wmax.ones_like()).expand_dims(0) *
        step; // (1, K, 1)
    const Tensor zero = d32.w.zeros_like({N, K, 3 * d});
    const Tensor nu0 = start.defined() ? start.to(dtype).maximum(0.0) : zero;
    const Tensor best =
        aux_ascent_joint(nu0, lr, iterations, [&](const Tensor &nu, bool with_grad) {
            // the supergradient in nu_b: (-sign(t) times its row minus its offset) over its norm
            Tensor tt;
            const Tensor v = value(d32, nu, &tt);
            if (!with_grad) return std::pair{v, Tensor()};
            const Tensor g = tt.sign().negative();
            const Tensor gi = g.matmul(GinT), go = g.matmul(GoutT); // (N, K, d)
            std::vector<Tensor> parts;
            for (int b = 0; b < 3; ++b)
                parts.push_back((gi.multiply(d32.al[b].expand_dims(1)) +
                                 go.multiply(d32.ga[b].expand_dims(1)) - d32.off[b].expand_dims(1))
                                    .divide(d32.n[b].expand_dims(1)));
            return std::pair{v, Tensor::concat(parts, 2)};
        });
    if (best_out) *best_out = best;
    // the certificate: lambda = nu / norm >= 0 on the table's unscaled rows, in its dtype (float64)
    const Data d64 = data(t.rows.dtype());
    Tensor tt;
    const Tensor at_best = value(d64, best.to(w.dtype()).maximum(0.0), &tt);
    if (minimizer) *minimizer = tt.sign().negative(); // (N, K, q): the Lagrangian's minimizer
    return at_best.maximum(w.abs().sum(-1).negative().expand_dims(0));
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
    // the supergradient -sign(w + lambda C) C^T - d
    const Tensor best = aux_ascent(
        w32.zeros_like({B, R, p}), lr, iterations,
        [&](const Tensor &l) { return aux_dual(w32, C32, d32, l, nullptr); },
        [&](const Tensor &l) {
            Tensor s;
            aux_dual(w32, C32, d32, l, &s);
            return s.sign().matmul(CT32).negative() - d32.expand_dims(1);
        });
    // the certificate on the unscaled rows (the scaled ones are rounded): lambda / scale
    return priv_dual_certificate(w, C_, d_, best.to(w.dtype()).divide(scale.expand_dims(1)));
}

// The certificate ---------------------------------------------------------------------------------

Tensor priv_dual_certificate(const Tensor &w, const Tensor &C, const Tensor &d,
                             const Tensor &lambda) {
    // weak duality at every lambda >= 0, and at least lambda = 0's bound
    const Tensor at = aux_dual(w, C, d, lambda.maximum(0.0), nullptr);
    return at.maximum(w.abs().sum(-1).negative());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
