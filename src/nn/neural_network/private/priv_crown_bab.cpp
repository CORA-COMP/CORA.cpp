// priv_crown_bab - the ReLU-split branch and bound with CROWN's node bound (beta-CROWN)
//
// See priv_zono_batch.h. A node fixes some crossing ReLU neurons of the root to a side: active
// (z >= 0, the identity) or inactive (z <= 0, zero). Its bound is priv_backward_lower over the
// root's box with the node's split signs: each split neuron's bounds clamped to its side, its line
// exact there, and the split's Lagrangian term max(beta, 0) sigma z (sigma z <= 0 on the node's
// inputs) added per row. For any lower slopes in [0, 1] and any beta >= 0 the pass's bound holds
// over the node's inputs (weak duality, docs/nn_verification.md), so each step's bound does, and a
// node keeps its best. The optimized variables live on the root's crossing neurons only (U of
// them): the slopes and multipliers (N, K, U), the signs (N, U); a layer's (N, K, n) are gathers.
// A child starts at its parent's slopes and multipliers (0 for its new split). A node branches on
// the unsplit neuron of largest chord loss (BaBSR's intercept score) in its best row.
// References:
//    [1] S. Wang, H. Zhang, K. Xu, X. Lin, S. Jana, C.-J. Hsieh, J. Z. Kolter. "Beta-CROWN:
//        efficient bound propagation with per-neuron split constraints for neural network
//        robustness verification", NeurIPS 2021.
//    [2] R. Bunel, J. Lu, I. Turkaslan, P. H. S. Torr, P. Kohli, M. P. Kumar. "Branch and bound for
//        piecewise linear neural network verification", JMLR 2020 (BaBSR).
//
// CORA.cpp only: CORA's verify splits on the zonotope
// Authors:       Claude Opus 5.5
// Written:       24-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "global/tensor/device_memory.h"
#include "nn/neural_network/private/priv_verify.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The crossing neurons ----------------------------------------------------------------------------

/// The root's crossing ReLU neurons: per layer their positions and each neuron's index among the
/// U of them (U for the others: a zero column), their bounds (U).
struct CrossingSpace {
    std::vector<std::vector<int64_t>> pos, idx; ///< per layer of the net (empty: none)
    int64_t U = 0;
    Tensor l, u;
};

CrossingSpace aux_space(const ZonoPropagation &out, const ZonoNet &net, const ZonoBounds &bounds) {
    CrossingSpace s;
    s.pos.resize(net.W.size());
    s.idx.resize(net.W.size());
    std::vector<Tensor> ls, us;
    // per ReLU layer the neurons crossing 0 at its bounds, numbered in order
    for (const ZonoLayerRecord &rec : out.layers) {
        const auto k = std::size_t(rec.index);
        if (net.activations[k] != Activation::relu) continue;
        const bool given = k < bounds.l.size() && bounds.l[k].defined();
        const Tensor l = given ? bounds.l[k] : rec.l, u = given ? bounds.u[k] : rec.u;
        const std::vector<double> lv = l.data(), uv = u.data();
        // their positions, and each neuron's number (-1: not crossing)
        s.idx[k].assign(lv.size(), -1);
        for (std::size_t i = 0; i < lv.size(); ++i)
            if (lv[i] < 0 && uv[i] > 0) s.pos[k].push_back(int64_t(i));
        for (std::size_t j = 0; j < s.pos[k].size(); ++j)
            s.idx[k][std::size_t(s.pos[k][j])] = s.U + int64_t(j);
        s.U += int64_t(s.pos[k].size());
        if (!s.pos[k].empty()) {
            ls.push_back(l.squeeze(0).take(s.pos[k], 0));
            us.push_back(u.squeeze(0).take(s.pos[k], 0));
        }
    }
    // the others read the zero column U
    for (std::vector<int64_t> &v : s.idx)
        for (int64_t &i : v)
            if (i < 0) i = s.U;
    if (!ls.empty()) {
        s.l = Tensor::concat(ls, 0);
        s.u = Tensor::concat(us, 0);
    }
    return s;
}

/// The layers' tensors (..., n) of one on the crossing neurons X (..., U): the others 0.
std::vector<Tensor> aux_expand(const CrossingSpace &s, const Tensor &X) {
    std::vector<Tensor> out(s.idx.size());
    std::vector<int64_t> shape = X.shape();
    shape.back() = 1;
    const Tensor Xe = Tensor::concat({X, X.zeros_like(shape)}, -1);
    for (std::size_t k = 0; k < s.idx.size(); ++k)
        if (!s.pos[k].empty()) out[k] = Xe.take(s.idx[k], -1);
    return out;
}

/// The crossing neurons' entries (..., U) of the layers' tensors (..., n) (undefined: 0).
Tensor aux_gather(const CrossingSpace &s, const std::vector<Tensor> &layers, const Tensor &like) {
    std::vector<Tensor> parts;
    if (s.U == 0) {
        std::vector<int64_t> shape = like.shape();
        shape.back() = 0;
        return like.zeros_like(shape);
    }
    for (std::size_t k = 0; k < s.pos.size(); ++k) {
        if (s.pos[k].empty()) continue;
        if (k < layers.size() && layers[k].defined()) {
            parts.push_back(layers[k].take(s.pos[k], -1));
        } else {
            std::vector<int64_t> shape = like.shape();
            shape.back() = int64_t(s.pos[k].size());
            parts.push_back(like.zeros_like(shape));
        }
    }
    return Tensor::concat(parts, -1);
}

/// Out of memory: the device's cache freed for a smaller retry, the count in the stats.
void aux_out_of_memory(const Tensor &like, std::map<std::string, double> &stats) {
    free_device_cache(like.device_id());
    stats["crown_bab_oom"] += 1;
}

/// One-hot rows (N, n) of the indices idx (N).
Tensor aux_onehot(const Tensor &idx, int64_t n) {
    return idx.expand_dims(-1).equal(idx.arange_like(n).expand_dims(0)) + 0.0;
}

// The nodes ---------------------------------------------------------------------------------------

/// A batch of nodes: signs (N, U), slopes and multipliers (N, K, U), the neuron each branches on
/// next (N; -1: none left) and its rows' bounds (N, K).
struct Nodes {
    Tensor S, alpha, beta, next, best;
    /// The nodes [from, to).
    Nodes slice(int64_t from, int64_t to) const {
        return {S.slice(0, from, to), alpha.slice(0, from, to), beta.slice(0, from, to),
                next.slice(0, from, to), best.slice(0, from, to)};
    }
};

/// What the bound of a batch gives: each row's best bound (N, K), its slopes and multipliers,
/// the last pass's chord losses (N, K, U) and corners (N, K, n0).
struct Bounded {
    Tensor best, alpha, beta, score, corner;
};

/// What one backward pass reads: the record, its net and bounds, the box and the rows, in one
/// dtype.
struct Pass {
    const ZonoPropagation *out = nullptr;
    const ZonoNet *net = nullptr;
    const ZonoBounds *bounds = nullptr;
    Tensor x, rad, rows, offsets;
    std::shared_ptr<ZonoLineCache> lines = std::make_shared<ZonoLineCache>(); ///< of its passes
};

// The search --------------------------------------------------------------------------------------

/// The branch and bound of one polytope.
class CrownBab {
  public:
    CrownBab(const ZonoPropagation &out, const ZonoNet &net, const ZonoBounds &bounds,
             const Tensor &x, const Tensor &rad, const Tensor &rows, const Tensor &offsets,
             const CrownBabOptions &o, std::map<std::string, double> &stats,
             const CrownBabSearch *search)
        : net_(net), x_(x), o_(o), stats_(stats), space_(aux_space(out, net, bounds)),
          K_(rows.shape()[0]) {
        exact_ = {&out, &net, &bounds, x, rad, rows, offsets};
        if (search && search->out && search->net && search->bounds) {
            // the search's copy in its dtype (its weights')
            const DType dt = search->net->b.back().dtype();
            search_ = {search->out, search->net, search->bounds, x.to(dt),
                       rad.to(dt),  rows.to(dt), offsets.to(dt)};
            searching_ = true;
        }
    }

    int run(const std::function<bool(const Tensor &)> &check);
    /// The root's best bound (K) of every row, no split.
    Tensor roots() const {
        return bound(root_nodes(), root_slopes(), root_slopes() * 0.0, o_.root_steps)
            .best.squeeze(0);
    }

    /// Per row the slopes of its affine form at the root's best slopes (priv_crown_bab_slopes):
    /// per ReLU layer (K, n), and the rows' root bounds (K).
    std::vector<Tensor> line_slopes(Tensor *root) const {
        const Bounded b = bound(root_nodes(), root_slopes(), root_slopes() * 0.0, o_.root_steps);
        if (root) *root = b.best.squeeze(0);
        // the chord (-1) where the row's Lambda is below 0: there its upper line is read
        const Tensor below = b.score.greater(b.score.zeros_like({1}));
        const Tensor a = Tensor::where(below, b.alpha.full_like({1}, -1.0), b.alpha);
        std::vector<Tensor> out(space_.idx.size());
        const Tensor chords = Tensor::concat({a.squeeze(0), a.full_like({K_, 1}, -1.0)}, -1);
        for (std::size_t k = 0; k < space_.idx.size(); ++k)
            if (!space_.pos[k].empty()) out[k] = chords.take(space_.idx[k], -1);
        return out;
    }

  private:
    bool late() const { return now_seconds() >= std::min(o_.deadline, net_.deadline); }
    Tensor evaluate(const Pass &p, const std::vector<Tensor> &signs, const Tensor &alpha,
                    const Tensor &beta, std::vector<Tensor> *ga, std::vector<Tensor> *gb,
                    std::vector<Tensor> *ic, Tensor *corner, std::vector<Tensor> *gm = nullptr,
                    std::vector<Tensor> *gs = nullptr) const;
    /// The products' planes per row, once: o_.product_steps steps at the root, the best kept.
    void tune_planes() const;
    Bounded bound(const Tensor &S, const Tensor &alpha, const Tensor &beta, int64_t steps) const;
    Tensor choose(const Tensor &S, const Bounded &b) const;
    Tensor look_ahead(const Tensor &S, const Tensor &alpha, const Tensor &beta, const Bounded &b,
                      int64_t batch) const;
    /// look_ahead's pick, BaBSR's (choose) where it runs out of memory.
    Tensor branch(const Tensor &S, const Tensor &alpha, const Tensor &beta, const Bounded &b,
                  int64_t batch) const;
    /// A node's bytes in a pass of `net` with `item` bytes an entry.
    int64_t per_node_bytes(const ZonoNet &net, int64_t item) const;
    Tensor root_nodes() const { return x_.zeros_like({1, space_.U}); }
    Tensor root_slopes() const {
        // CROWN's lower slopes: 1 where u > -l
        const int64_t U = space_.U;
        return U > 0 ? space_.u.greater(space_.l.negative())
                               .reshape({1, 1, U})
                               .broadcast_to({1, K_, U}) +
                           0.0
                     : x_.zeros_like({1, K_, 0});
    }

    const ZonoNet &net_;
    Tensor x_;
    Pass exact_, search_;
    bool searching_ = false;
    CrownBabOptions o_;
    std::map<std::string, double> &stats_;
    CrossingSpace space_;
    int64_t K_ = 0;
    /// The products' planes of every node's pass (per layer (1, K, n / 2); empty: the set's line).
    mutable std::vector<Tensor> mix_, side_;
    mutable bool tuned_ = false;
    mutable int64_t exact_chunk_ = int64_t(1) << 40; ///< nodes of one pass on the record
};

// The node bound ----------------------------------------------------------------------------------

Tensor CrownBab::evaluate(const Pass &p, const std::vector<Tensor> &signs, const Tensor &alpha,
                          const Tensor &beta, std::vector<Tensor> *ga, std::vector<Tensor> *gb,
                          std::vector<Tensor> *ic, Tensor *corner, std::vector<Tensor> *gm,
                          std::vector<Tensor> *gs) const {
    // the nodes' boxes, their slopes and multipliers per layer
    const int64_t N = alpha.shape()[0], n0 = p.x.shape()[1];
    const Tensor xs = p.x.broadcast_to({N, n0}) + 0.0, rs = p.rad.broadcast_to({N, n0}) + 0.0;
    const std::vector<Tensor> a_l = aux_expand(space_, alpha), b_l = aux_expand(space_, beta);
    // one pass with the nodes' splits (the gradients, scores and corners where asked)
    ZonoBackward opt;
    opt.bounds = p.bounds;
    opt.row_alpha = &a_l;
    opt.g_alpha = ga;
    opt.split_sign = &signs;
    opt.beta = &b_l;
    opt.g_beta = gb;
    opt.intercepts = ic;
    opt.corner = corner;
    opt.softmax_lse = o_.softmax_lse;
    opt.lines = p.lines.get();
    // the products' planes in the pass's dtype
    std::vector<Tensor> mix = mix_, side = side_;
    for (std::vector<Tensor> *v : {&mix, &side})
        for (Tensor &t : *v)
            if (t.defined()) t = t.to(p.x.dtype());
    if (!mix.empty()) {
        opt.product_mix = &mix;
        opt.product_side = &side;
        opt.g_product_mix = gm;
        opt.g_product_side = gs;
    }
    return priv_backward_lower(*p.out, *p.net, p.x.zeros_like({N, 0}), xs, rs, p.rows, nullptr,
                               &opt) -
           p.offsets;
}

void CrownBab::tune_planes() const {
    if (tuned_ || o_.product_steps <= 0) return;
    tuned_ = true;
    // the product layers: mix 0 (the set's line), side 1/2
    const std::size_t L = net_.W.size();
    std::vector<Tensor> mix(L), side(L), m_m1(L), m_m2(L), s_m1(L), s_m2(L);
    bool any = false;
    for (std::size_t k = 0; k < L; ++k) {
        if (net_.activations[k] != Activation::product) continue;
        mix[k] = x_.full_like({1, K_, net_.widths[k] / 2}, o_.product_start);
        side[k] = mix[k] * 0.0 + 0.5;
        any = true;
    }
    if (!any) return;
    const double started = now_seconds();
    const Tensor S = root_nodes(), beta = root_slopes() * 0.0;
    const std::vector<Tensor> signs = aux_expand(space_, S);
    std::vector<Tensor> a1{root_slopes()}, a_m1(1), a_m2(1), best_mix = mix, best_side = side;
    Tensor best;
    for (int64_t t = 0; t <= o_.product_steps; ++t) {
        // one root pass at these planes and slopes; each row keeps the planes of its best bound
        mix_ = mix;
        side_ = side;
        std::vector<Tensor> ga, gm, gs;
        const bool more = t < o_.product_steps && !late();
        const Tensor value = evaluate(exact_, signs, a1[0], beta, more ? &ga : nullptr, nullptr,
                                      nullptr, nullptr, more ? &gm : nullptr, more ? &gs : nullptr);
        const Tensor up = best.defined() ? value.greater(best).reshape({1, K_, 1}) : Tensor();
        if (t == 0) stats_["crown_bab_planes_before"] = value.max(1).item();
        for (std::size_t k = 0; k < L && up.defined(); ++k) {
            if (!mix[k].defined()) continue;
            best_mix[k] = Tensor::where(up, mix[k], best_mix[k]);
            best_side[k] = Tensor::where(up, side[k], best_side[k]);
        }
        best = best.defined() ? best.maximum(value) : value;
        if (!more) break;
        // projected Adam on the planes and the lower slopes, all in [0, 1]
        const std::vector<Tensor> g_a{aux_gather(space_, ga, a1[0])};
        priv_adam_slopes(a1, a_m1, a_m2, g_a, t + 1, o_.alpha_step);
        priv_adam_slopes(mix, m_m1, m_m2, gm, t + 1, o_.product_step);
        priv_adam_slopes(side, s_m1, s_m2, gs, t + 1, o_.product_step);
    }
    mix_ = best_mix;
    side_ = best_side;
    stats_["crown_bab_planes_after"] = best.max(1).item();
    stats_["crown_bab_planes_seconds"] += now_seconds() - started;
}

Bounded CrownBab::bound(const Tensor &S, const Tensor &alpha, const Tensor &beta,
                        int64_t steps) const {
    priv_oom_site("crown_bab_bound");
    tune_planes();
    const double started = now_seconds();
    // the steps in the search's dtype (or the record's), every step's bound per row the best
    const Pass &p = searching_ ? search_ : exact_;
    const DType dt = p.x.dtype();
    const std::vector<Tensor> signs = aux_expand(space_, S.to(dt));
    std::vector<Tensor> a1{alpha.to(dt)}, a_m1(1), a_m2(1), b1{beta.to(dt)}, b_m1(1), b_m2(1);
    Bounded r;
    for (int64_t t = 0;; ++t) {
        // one pass: its bound, and its gradients if another step follows
        const bool more = t < steps && !late();
        std::vector<Tensor> ga, gb, ic;
        Tensor corner;
        const Tensor value =
            evaluate(p, signs, a1[0], b1[0], more ? &ga : nullptr, more ? &gb : nullptr,
                     searching_ ? nullptr : &ic, searching_ ? nullptr : &corner);
        // every step's bound holds: per row the best, with its slopes and multipliers
        if (!r.best.defined()) {
            r.best = value;
            r.alpha = a1[0];
            r.beta = b1[0];
        } else {
            const Tensor up = value.greater(r.best).expand_dims(-1);
            r.best = r.best.maximum(value);
            r.alpha = Tensor::where(up, a1[0], r.alpha);
            r.beta = Tensor::where(up, b1[0], r.beta);
        }
        // in the record's dtype: the last pass's scores and corners
        if (!searching_) {
            r.score = aux_gather(space_, ic, a1[0]);
            r.corner = corner;
        }
        if (!more) break;
        // all decided: no more steps
        if (r.best.max(1).min().item() > o_.margin) break;
        // projected Adam: the slopes in [0, 1], the multipliers at least 0
        const std::vector<Tensor> g_a{aux_gather(space_, ga, a1[0])},
            g_b{aux_gather(space_, gb, b1[0])};
        priv_adam_slopes(a1, a_m1, a_m2, g_a, t + 1, o_.alpha_step);
        priv_adam_slopes(b1, b_m1, b_m2, g_b, t + 1, o_.beta_step, kInf);
    }
    if (searching_) {
        // the best slopes and multipliers once more in the record's dtype: only this bound decides
        const DType de = exact_.x.dtype();
        r.alpha = r.alpha.to(de).maximum(0.0).minimum(1.0);
        r.beta = r.beta.to(de).maximum(0.0);
        // in chunks of what the record's passes hold (the search's batch may be larger)
        const int64_t N = S.shape()[0], chunk = std::max<int64_t>(1, exact_chunk_);
        const double exact_started = now_seconds();
        std::vector<Tensor> bests, scores, corners;
        for (int64_t at = 0; at < N; at += chunk) {
            // a chunk's bound, scores and corners on the record
            const int64_t to = std::min(N, at + chunk);
            std::vector<Tensor> ic;
            Tensor corner;
            bests.push_back(evaluate(exact_, aux_expand(space_, S.slice(0, at, to)),
                                     r.alpha.slice(0, at, to), r.beta.slice(0, at, to), nullptr,
                                     nullptr, &ic, &corner));
            scores.push_back(aux_gather(space_, ic, r.alpha.slice(0, at, to)));
            corners.push_back(corner);
        }
        r.best = Tensor::concat(bests, 0);
        r.score = Tensor::concat(scores, 0);
        r.corner = Tensor::concat(corners, 0);
        stats_["crown_bab_exact_seconds"] += now_seconds() - exact_started;
    }
    stats_["crown_bab_bound_seconds"] += now_seconds() - started;
    return r;
}

// The branching -----------------------------------------------------------------------------------

Tensor CrownBab::choose(const Tensor &S, const Bounded &b) const {
    // the best row's chord losses on the unsplit neurons; none above 0: the widest unsplit one
    const Tensor hot = aux_onehot(b.best.argmax(1), K_).expand_dims(-1);
    const Tensor score = b.score.multiply(hot).sum(1);
    const Tensor free = S.equal(S.zeros_like());
    const Tensor masked = Tensor::where(free, score, score.full_like({1}, -1.0));
    const Tensor width = space_.l.negative().minimum(space_.u).expand_dims(0);
    const Tensor wide =
        Tensor::where(free, width.broadcast_to(S.shape()) + 0.0, score.full_like({1}, -1.0));
    const Tensor pick = Tensor::where(masked.max(1).greater(masked.zeros_like({1})),
                                      masked.argmax(1), wide.argmax(1));
    // every neuron split: none left
    return Tensor::where(free.any(1), pick, pick.full_like({1}, -1.0));
}

Tensor CrownBab::look_ahead(const Tensor &S, const Tensor &alpha, const Tensor &beta,
                            const Bounded &b, int64_t batch) const {
    // BaBSR's pick, and its best `fsb` candidates (filtered smart branching)
    const Tensor pick = choose(S, b);
    const int64_t N = S.shape()[0], U = space_.U;
    const int64_t k = std::min<int64_t>(o_.fsb, U);
    if (k <= 1 || N == 0) return pick;
    const Tensor hot = aux_onehot(b.best.argmax(1), K_).expand_dims(-1);
    const Tensor score = b.score.multiply(hot).sum(1);
    const Tensor free = S.equal(S.zeros_like());
    const Tensor masked = Tensor::where(free, score, score.full_like({1}, -1.0));
    const auto [vals, cand] = masked.topk(k, 1); // (N, k)
    // each candidate's two children, by one pass at the parent's slopes and multipliers
    std::vector<Tensor> parts;
    for (int64_t c = 0; c < k; ++c) {
        const Tensor h = aux_onehot(cand.slice(1, c, c + 1).squeeze(1), U);
        parts.push_back(S + h);
        parts.push_back(S - h);
    }
    const Tensor Sc = Tensor::concat(parts, 0);
    std::vector<Tensor> as(std::size_t(2 * k), alpha), bs(std::size_t(2 * k), beta);
    const Tensor Ac = Tensor::concat(as, 0), Bc = Tensor::concat(bs, 0);
    std::vector<Tensor> lows;
    const int64_t total = 2 * k * N;
    for (int64_t at = 0; at < total; at += batch) {
        const int64_t to = std::min(total, at + batch);
        lows.push_back(
            bound(Sc.slice(0, at, to), Ac.slice(0, at, to), Bc.slice(0, at, to), 0).best.max(1));
    }
    // per candidate its worse child (k, N); the best candidate among those of a positive score
    const Tensor low = Tensor::concat(lows, 0).reshape({k, 2, N});
    const Tensor worse = low.min(1);
    const Tensor real = vals.matrix_transpose().greater(vals.zeros_like({1}));
    const Tensor ranked = Tensor::where(real, worse, worse.full_like({1}, -kInf));
    const Tensor best_c = ranked.argmax(0);                                           // (N)
    const Tensor chosen = cand.take_along_axis(best_c.expand_dims(-1), 1).squeeze(1); // (N)
    const Tensor any = real.any(0);
    return Tensor::where(any.multiply(pick.greater_equal(pick.zeros_like())), chosen, pick);
}

Tensor CrownBab::branch(const Tensor &S, const Tensor &alpha, const Tensor &beta, const Bounded &b,
                        int64_t batch) const {
    if (std::min<int64_t>(o_.fsb, space_.U) <= 1) return choose(S, b);
    try {
        priv_oom_site("crown_bab_look_ahead");
        return look_ahead(S, alpha, beta, b, batch);
    } catch (const std::exception &error) {
        // out of memory in the children's passes: the score's pick (any neuron splits soundly)
        if (!priv_is_oom(error)) std::rethrow_exception(std::current_exception());
        aux_out_of_memory(x_, stats_);
        return choose(S, b);
    }
}

int64_t CrownBab::per_node_bytes(const ZonoNet &net, int64_t item) const {
    // per row and neuron of every layer: Lambda, the lines, the reverse and the slopes' steps
    int64_t width = net.n0;
    for (const int64_t w : net.widths) width += w;
    for (const NeuralNetwork::Layer &layer : net.maps) width += layer.work();
    return item * K_ * (16 * width + 12 * space_.U);
}

// The branch and bound ----------------------------------------------------------------------------

int CrownBab::run(const std::function<bool(const Tensor &)> &check) {
    const int64_t U = space_.U;
    stats_["crown_bab_crossing"] = std::max(stats_["crown_bab_crossing"], double(U));
    // the root: no split, CROWN's lower slopes (1 where u > -l) and no multiplier
    const Tensor S0 = root_nodes();
    const Tensor a0 = root_slopes();
    Bounded root;
    try {
        root = bound(S0, a0, a0.zeros_like(), o_.root_steps);
    } catch (const std::exception &error) {
        // out of memory at the root: no bound, the polytope stays open
        if (!priv_is_oom(error)) std::rethrow_exception(std::current_exception());
        aux_out_of_memory(x_, stats_);
        return 0;
    }
    const double margin = root.best.max(1).item();
    stats_["crown_bab_root_margin"] = std::min(
        stats_.count("crown_bab_root_margin") ? stats_["crown_bab_root_margin"] : kInf, margin);
    if (margin > o_.margin) return 1;
    if (check(root.corner.reshape({K_, x_.shape()[1]}))) return 2;
    if (U == 0) return 0; // no neuron to split: the lines are exact up to the slopes
    // the open nodes as a stack of batches, the last batch's children first
    const int64_t node_bytes = 8 * (U + 2 * K_ * U + 2);
    // a batch: what the steps' passes hold (the search's, if any; the record's passes in chunks)
    const int64_t exact = per_node_bytes(net_, 8);
    const int64_t per =
        searching_ ? per_node_bytes(*search_.net, search_.x.dtype() == x_.dtype() ? 8 : 4) : exact;
    exact_chunk_ = std::max<int64_t>(1, o_.max_bytes / std::max<int64_t>(1, exact));
    int64_t batch =
        std::max<int64_t>(1, std::min(o_.batch, o_.max_bytes / std::max<int64_t>(1, per)));
    stats_["crown_bab_batch"] = double(batch);
    std::vector<Nodes> stack{
        {S0, root.alpha, root.beta, branch(S0, root.alpha, root.beta, root, batch), root.best}};
    int64_t held = 1;
    while (!stack.empty()) {
        if (late()) return 0;
        // up to half a batch of parents: two children each
        Nodes p = stack.back();
        stack.pop_back();
        const int64_t F = p.S.shape()[0], take = std::max<int64_t>(1, batch / 2);
        if (F > take) {
            stack.push_back(p.slice(0, F - take));
            p = p.slice(F - take, F);
        }
        const int64_t M = p.S.shape()[0];
        held -= M;
        // a node with every crossing neuron split and still open: the search cannot decide it
        if (p.next.min().item() < 0) return 0;
        // the children: +1 inactive (z <= 0), -1 active (z >= 0) on the parent's neuron
        const Tensor hot = aux_onehot(p.next, U);
        const Tensor S2 = Tensor::concat({p.S + hot, p.S - hot}, 0);
        const Tensor A2 = Tensor::concat({p.alpha, p.alpha}, 0);
        const Tensor B2 = Tensor::concat({p.beta, p.beta}, 0);
        Bounded b;
        try {
            b = bound(S2, A2, B2, o_.steps);
        } catch (const std::exception &error) {
            // out of memory: the parents back on the stack, half the batch (one node: open)
            if (!priv_is_oom(error)) std::rethrow_exception(std::current_exception());
            aux_out_of_memory(x_, stats_);
            if (batch == 1) return 0;
            batch = (batch + 1) / 2;
            stats_["crown_bab_batch"] = double(batch);
            held += M;
            stack.push_back(p);
            continue;
        }
        // a parent's bound holds on its children's inputs too (a subset of its own)
        b.best = b.best.maximum(Tensor::concat({p.best, p.best}, 0));
        const Tensor low = b.best.max(1);
        const std::vector<int64_t> open = (low.less_equal(low.full_like({1}, o_.margin))).nonzero();
        stats_["crown_bab_nodes"] += double(2 * M);
        stats_["crown_bab_verified_nodes"] += double(2 * M - int64_t(open.size()));
        const double depth = p.S.not_equal(p.S.zeros_like()).sum(1).max().item() + 1;
        stats_["crown_bab_depth"] = std::max(stats_["crown_bab_depth"], depth);
        if (open.empty()) continue;
        Nodes next{S2.take(open, 0), b.alpha.take(open, 0), b.beta.take(open, 0), Tensor(),
                   b.best.take(open, 0)};
        const Bounded kept{next.best, Tensor(), Tensor(), b.score.take(open, 0), Tensor()};
        next.next = branch(next.S, next.alpha, next.beta, kept, batch);
        const Tensor next_low = next.best.max(1);
        stats_["crown_bab_open_margin"] = next_low.min().item();
        // the corners of the weakest open nodes: every row's minimizer, checked on the network
        if (o_.witnesses > 0) {
            const int64_t W = std::min<int64_t>(o_.witnesses, int64_t(open.size()));
            const Tensor pick = next_low.negative().topk(W, 0).second;
            const Tensor corners = b.corner.take(open, 0).take(pick, 0);
            if (check(corners.reshape({W * K_, x_.shape()[1]}))) return 2;
        }
        held += next.S.shape()[0];
        stack.push_back(next);
        stats_["crown_bab_frontier_peak"] =
            std::max(stats_["crown_bab_frontier_peak"], double(held));
        if (held * node_bytes > o_.frontier_bytes) {
            stats_["crown_bab_frontier_cut"] = 1;
            return 0;
        }
    }
    return 1;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The entries -------------------------------------------------------------------------------------

int priv_crown_bab(const ZonoPropagation &out, const ZonoNet &net, const ZonoBounds &bounds,
                   const Tensor &x, const Tensor &rad, const Tensor &rows, const Tensor &offsets,
                   const CrownBabOptions &o, const std::function<bool(const Tensor &)> &check,
                   std::map<std::string, double> &stats, const CrownBabSearch *search) {
    if (x.shape()[0] != 1) CORA_FAIL(Error::wrong_value, "verify: crown-bab bounds one box");
    // one polytope's search
    CrownBab bab(out, net, bounds, x, rad, rows, offsets, o, stats, search);
    return bab.run(check);
}

Tensor priv_crown_bab_roots(const ZonoPropagation &out, const ZonoNet &net,
                            const ZonoBounds &bounds, const Tensor &x, const Tensor &rad,
                            const Tensor &rows, const Tensor &offsets, const CrownBabOptions &o,
                            std::map<std::string, double> &stats, const CrownBabSearch *search) {
    if (x.shape()[0] != 1) CORA_FAIL(Error::wrong_value, "verify: crown-bab bounds one box");
    const CrownBab bab(out, net, bounds, x, rad, rows, offsets, o, stats, search);
    return bab.roots();
}

std::vector<Tensor> priv_crown_bab_slopes(const ZonoPropagation &out, const ZonoNet &net,
                                          const ZonoBounds &bounds, const Tensor &x,
                                          const Tensor &rad, const Tensor &rows,
                                          const Tensor &offsets, const CrownBabOptions &o,
                                          std::map<std::string, double> &stats,
                                          const CrownBabSearch *search, Tensor *root) {
    if (x.shape()[0] != 1) CORA_FAIL(Error::wrong_value, "verify: crown-bab bounds one box");
    // every row takes all the steps: none stops them as a polytope's verified row would
    CrownBabOptions all = o;
    all.margin = kInf;
    const CrownBab bab(out, net, bounds, x, rad, rows, offsets, all, stats, search);
    return bab.line_slopes(root);
}

// The search dtype --------------------------------------------------------------------------------

ZonoNet priv_zono_net_to(const ZonoNet &net, DType dtype, bool native) {
    ZonoNet copy = net;
    const auto to = [&](std::vector<Tensor> &ts) {
        for (Tensor &t : ts)
            if (t.defined()) t = t.to(dtype);
    };
    to(copy.W);
    to(copy.b);
    to(copy.abs_w);
    to(copy.WT);
    to(copy.abs_wT);
    // the maps (a residual's too), native where asked
    for (NeuralNetwork::Layer &layer : copy.maps)
        layer = native ? layer.to(dtype).native_convolution() : layer.to(dtype);
    for (std::vector<NeuralNetwork::Residual> &rs : copy.residuals)
        for (NeuralNetwork::Residual &r : rs)
            if (r.map)
                r.map = std::make_shared<const NeuralNetwork::Layer>(
                    native ? r.map->to(dtype).native_convolution() : r.map->to(dtype));
    return copy;
}

ZonoPropagation priv_record_to(const ZonoPropagation &out, DType dtype) {
    // what priv_backward_lower reads: each activation layer's input set and bounds
    ZonoPropagation copy;
    const auto to = [&](const Tensor &t) { return t.defined() ? t.to(dtype) : t; };
    for (const ZonoLayerRecord &rec : out.layers) {
        // the layer, its input set and its bounds
        ZonoLayerRecord r;
        r.index = rec.index;
        r.first_id = rec.first_id;
        r.c_in = to(rec.c_in);
        r.dr_in = to(rec.dr_in);
        r.G_in = to(rec.G_in);
        r.l = to(rec.l);
        r.u = to(rec.u);
        copy.layers.push_back(std::move(r));
    }
    return copy;
}

ZonoBounds priv_bounds_to(const ZonoBounds &bounds, DType dtype) {
    ZonoBounds copy = bounds;
    for (std::vector<Tensor> *ts : {&copy.l, &copy.u})
        for (Tensor &t : *ts)
            if (t.defined()) t = t.to(dtype);
    return copy;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
