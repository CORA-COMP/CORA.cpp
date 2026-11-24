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
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "nn/neural_network/private/priv_verify.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

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
    for (const ZonoLayerRecord &rec : out.layers) {
        const auto k = std::size_t(rec.index);
        if (net.activations[k] != Activation::relu) continue;
        const bool given = k < bounds.l.size() && bounds.l[k].defined();
        const Tensor l = given ? bounds.l[k] : rec.l, u = given ? bounds.u[k] : rec.u;
        const std::vector<double> lv = l.data(), uv = u.data();
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

/// One-hot rows (N, n) of the indices idx (N).
Tensor aux_onehot(const Tensor &idx, int64_t n) {
    return idx.expand_dims(-1).equal(idx.arange_like(n).expand_dims(0)) + 0.0;
}

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

/// The branch and bound of one polytope.
class CrownBab {
  public:
    CrownBab(const ZonoPropagation &out, const ZonoNet &net, const ZonoBounds &bounds,
             const Tensor &x, const Tensor &rad, const Tensor &rows, const Tensor &offsets,
             const CrownBabOptions &o, std::map<std::string, double> &stats)
        : out_(out), net_(net), bounds_(bounds), x_(x), rad_(rad), rows_(rows), offsets_(offsets),
          o_(o), stats_(stats), space_(aux_space(out, net, bounds)), K_(rows.shape()[0]) {}

    int run(const std::function<bool(const Tensor &)> &check);

  private:
    bool late() const { return now_seconds() >= std::min(o_.deadline, net_.deadline); }
    Bounded bound(const Tensor &S, Tensor alpha, Tensor beta, int64_t steps) const;
    Tensor choose(const Tensor &S, const Bounded &b) const;
    Tensor look_ahead(const Tensor &S, const Tensor &alpha, const Tensor &beta, const Bounded &b,
                      int64_t batch) const;
    int64_t per_node_bytes() const;

    const ZonoPropagation &out_;
    const ZonoNet &net_;
    const ZonoBounds &bounds_;
    Tensor x_, rad_, rows_, offsets_;
    CrownBabOptions o_;
    std::map<std::string, double> &stats_;
    CrossingSpace space_;
    int64_t K_ = 0;
};

Bounded CrownBab::bound(const Tensor &S, Tensor alpha, Tensor beta, int64_t steps) const {
    const double started = now_seconds();
    const int64_t N = S.shape()[0], n0 = x_.shape()[1];
    const Tensor xs = x_.broadcast_to({N, n0}) + 0.0, rs = rad_.broadcast_to({N, n0}) + 0.0;
    const Tensor none = x_.zeros_like({N, 0});
    const std::vector<Tensor> signs = aux_expand(space_, S);
    std::vector<Tensor> a1{std::move(alpha)}, a_m1(1), a_m2(1), b1{std::move(beta)}, b_m1(1),
        b_m2(1);
    Bounded r;
    for (int64_t t = 0;; ++t) {
        const bool more = t < steps && !late();
        const std::vector<Tensor> a_l = aux_expand(space_, a1[0]), b_l = aux_expand(space_, b1[0]);
        std::vector<Tensor> ga, gb, ic;
        Tensor corner;
        ZonoBackward opt;
        opt.bounds = &bounds_;
        opt.row_alpha = &a_l;
        opt.g_alpha = more ? &ga : nullptr;
        opt.split_sign = &signs;
        opt.beta = &b_l;
        opt.g_beta = more ? &gb : nullptr;
        opt.intercepts = &ic;
        opt.corner = &corner;
        opt.softmax_lse = o_.softmax_lse;
        const Tensor value =
            priv_backward_lower(out_, net_, none, xs, rs, rows_, nullptr, &opt) - offsets_;
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
        r.score = aux_gather(space_, ic, a1[0]);
        r.corner = corner;
        if (!more) break;
        // all decided: no more steps
        if (r.best.max(1).min().item() > o_.margin) break;
        // projected Adam: the slopes in [0, 1], the multipliers at least 0
        const std::vector<Tensor> g_a{aux_gather(space_, ga, a1[0])},
            g_b{aux_gather(space_, gb, b1[0])};
        priv_adam_slopes(a1, a_m1, a_m2, g_a, t + 1, o_.alpha_step);
        priv_adam_slopes(b1, b_m1, b_m2, g_b, t + 1, o_.beta_step, kInf);
    }
    stats_["crown_bab_bound_seconds"] += now_seconds() - started;
    return r;
}

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

int64_t CrownBab::per_node_bytes() const {
    // per row and neuron of every layer: Lambda, the lines, the reverse and the slopes' steps
    int64_t width = net_.n0;
    for (const int64_t w : net_.widths) width += w;
    for (const NeuralNetwork::Layer &layer : net_.maps) width += layer.work();
    return 8 * K_ * (16 * width + 12 * space_.U);
}

int CrownBab::run(const std::function<bool(const Tensor &)> &check) {
    const int64_t U = space_.U;
    stats_["crown_bab_crossing"] = std::max(stats_["crown_bab_crossing"], double(U));
    // the root: no split, CROWN's lower slopes (1 where u > -l) and no multiplier
    const Tensor S0 = x_.zeros_like({1, U});
    const Tensor a0 =
        U > 0 ? space_.u.greater(space_.l.negative()).reshape({1, 1, U}).broadcast_to({1, K_, U}) +
                    0.0
              : x_.zeros_like({1, K_, 0});
    const Bounded root = bound(S0, a0, a0.zeros_like(), o_.root_steps);
    const double margin = root.best.max(1).item();
    stats_["crown_bab_root_margin"] = std::min(
        stats_.count("crown_bab_root_margin") ? stats_["crown_bab_root_margin"] : kInf, margin);
    if (margin > o_.margin) return 1;
    if (check(root.corner.reshape({K_, x_.shape()[1]}))) return 2;
    if (U == 0) return 0; // no neuron to split: the lines are exact up to the slopes
    // the open nodes as a stack of batches, the last batch's children first
    const int64_t node_bytes = 8 * (U + 2 * K_ * U + 2);
    const int64_t per = per_node_bytes();
    int64_t batch =
        std::max<int64_t>(1, std::min(o_.batch, o_.max_bytes / std::max<int64_t>(1, per)));
    stats_["crown_bab_batch"] = double(batch);
    std::vector<Nodes> stack{
        {S0, root.alpha, root.beta, look_ahead(S0, root.alpha, root.beta, root, batch), root.best}};
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
            // out of memory: the parents back on the stack, half the batch
            if (!priv_is_oom(error) || batch == 1) std::rethrow_exception(std::current_exception());
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
        next.next = look_ahead(next.S, next.alpha, next.beta, kept, batch);
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

int priv_crown_bab(const ZonoPropagation &out, const ZonoNet &net, const ZonoBounds &bounds,
                   const Tensor &x, const Tensor &rad, const Tensor &rows, const Tensor &offsets,
                   const CrownBabOptions &o, const std::function<bool(const Tensor &)> &check,
                   std::map<std::string, double> &stats) {
    if (x.shape()[0] != 1) CORA_FAIL(Error::wrong_value, "verify: crown-bab bounds one box");
    CrownBab bab(out, net, bounds, x, rad, rows, offsets, o, stats);
    return bab.run(check);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
