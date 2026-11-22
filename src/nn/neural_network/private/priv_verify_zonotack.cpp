// priv_verify_zonotack - CORA's verify (ICLR 2026, Algorithm 1): latent-space zonotopes, batched
//
// The bounding "conzonotope". Per batch of input boxes: 1. the output zonotope excludes a polytope
// when a row's lower bound exceeds verified_margin (Eq. 8); 2. bab_falsification_method: per row
// of each open polytope the input point of the output's vertex least in it, beta = -sign(A_i G_y)
// on the input factors ("zonotack", Eq. 9), "center" or "fgsm"; 3. input splits, neuron splits and
// the ReLU rows ReLU(x) >= 0, ReLU(x) >= x on the factors, chosen by the heuristics, every
// combination of pieces a child; 4. Prop. 2 per child: the unsafe outputs and the child's rows as
// factor constraints, bounded by Fourier-Motzkin, the hull over the polytopes, the input rescaled
// to that box and propagated again (refinement_max_iter - 1 times). An empty child is verified.
// The three repairs of CORA's unsound steps are options (on): project_stale_factors,
// drop_dummy_duplicates, remember_point_splits_only (docs/nn_verification.md). Running out of
// memory takes fewer ReLU rows, then half the batch (CORA's catch). Phase 2 adds to the check
// zonotope-alpha's slopes, CROWN's bound and the dual certificate (priv_verify_dual.cpp).
//
// See also: priv_zono_batch.h, priv_verify.h, docs/nn_verification.md
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   22-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/error.h"
#include "nn/neural_network/private/priv_verify.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <cmath>
#include <exception>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Indices and rows --------------------------------------------------------------------------------

/// Host indices 0..n-1 not in `taken` (which is sorted).
std::vector<int64_t> aux_rest(int64_t n, const std::vector<int64_t> &taken) {
    std::vector<int64_t> rest;
    std::size_t j = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (j < taken.size() && taken[j] == i) {
            ++j;
            continue;
        }
        rest.push_back(i);
    }
    return rest;
}

/// The tensor x with its rows at the sorted indices `rows` replaced by those of v.
Tensor aux_put_rows(const Tensor &x, const std::vector<int64_t> &rows, const Tensor &v) {
    const std::vector<int64_t> others = aux_rest(x.shape()[0], rows);
    std::vector<int64_t> order = rows;
    order.insert(order.end(), others.begin(), others.end());
    std::vector<int64_t> inverse(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) inverse[std::size_t(order[i])] = int64_t(i);
    return Tensor::concat({v, x.take(others, 0)}, 0).take(inverse, 0);
}

/// The flat indices of the set entries of a 0/1 tensor.
std::vector<int64_t> aux_where(const Tensor &mask) { return mask.nonzero(); }

/// The rows (..., k) of a (..., n) tensor at the indices (..., k), and the columns j of the
/// generator rows (B, q, n) as rows (B, k, q).
Tensor aux_at(const Tensor &x, const Tensor &j) { return x.take_along_axis(j, 1); }
Tensor aux_columns(const Tensor &G, const Tensor &j) {
    return G.take_along_axis(j.expand_dims(1), 2).permute_dims({0, 2, 1});
}

/// The rows (B, k, qi) padded with zero columns to q.
Tensor aux_pad(const Tensor &R, int64_t q) {
    const int64_t extra = q - R.shape()[2];
    if (extra == 0) return R;
    return Tensor::concat({R, R.zeros_like({R.shape()[0], R.shape()[1], extra})}, 2);
}

/// The one-hot rows (..., k, n) of the indices idx (..., k).
Tensor aux_onehot(const Tensor &idx, int64_t n) {
    return idx.expand_dims(-1).equal(idx.arange_like(n).reshape({1, 1, n}));
}

/// The rows A (B, p, q) <= b, per member its nonzero rows first, cut to the largest count of them
/// (a zero row bounds no factor); undefined if no row is left.
std::pair<Tensor, Tensor> aux_compact(const Tensor &A, const Tensor &b) {
    const Tensor nonzero = A.not_equal(A.zeros_like()).any(2);
    const int64_t keep = int64_t(nonzero.sum(1).max().item());
    if (keep == 0) return {};
    const Tensor order = (1.0 - nonzero).argsort(1, false).slice(1, 0, keep);
    return {A.take_along_axis(order.expand_dims(-1), 1), b.take_along_axis(order, 1)};
}

// Split ids and pieces ----------------------------------------------------------------------------

/// Split-id rows: b's rows below a's, or (side_by_side) b's columns after a's; nonzero ids first in
/// each row and the columns that are zero everywhere dropped.
Tensor aux_cat_splits(const Tensor &a, const Tensor &b, bool side_by_side) {
    Tensor joined;
    if (side_by_side) {
        joined = Tensor::concat({a, b}, 1);
    } else {
        const int64_t width = std::max(a.shape()[1], b.shape()[1]);
        const auto pad = [&](const Tensor &t) {
            if (t.shape()[1] == width) return t;
            return Tensor::concat({t, t.zeros_like({t.shape()[0], width - t.shape()[1]})}, 1);
        };
        joined = Tensor::concat({pad(a), pad(b)}, 0);
    }
    // nonzero ids first, then the columns that hold one
    if (joined.shape()[0] == 0 || joined.shape()[1] == 0)
        return joined.zeros_like({joined.shape()[0], 0});
    const Tensor zero = joined.equal(joined.zeros_like());
    joined = joined.take_along_axis(zero.argsort(1, false), 1);
    const std::vector<double> used = joined.not_equal(joined.zeros_like()).any(0).data();
    int64_t width = 0;
    for (std::size_t i = 0; i < used.size(); ++i)
        if (used[i] > 0) width = int64_t(i) + 1;
    return joined.slice(1, 0, width);
}

/// CORA's aux_neuronConstraints: the offsets (..., pieces - 1), relative to the center, of a
/// neuron split into pieces, around zero ("zero") or the middle of [lo, hi] ("middle").
Tensor aux_pieces_offsets(const Tensor &lo, const Tensor &hi, const Tensor &center, int64_t n,
                          const std::string &position) {
    std::vector<double> enumerated;
    if (position == "middle") {
        for (int64_t i = 1; i < n; ++i) enumerated.push_back(-1.0 + 2.0 * double(i) / double(n));
        const Tensor e = Tensor::like(lo, enumerated, {1, 1, n - 1});
        return e.multiply(((hi - lo) * 0.5).expand_dims(-1));
    }
    if (position != "zero")
        CORA_FAIL(Error::wrong_field_value, "verify: the neuron_split_position '" + position +
                                                "' is unknown; use 'zero' or 'middle'");
    // the pieces of [lo, 0] and of [0, hi], with zero itself for an even count
    const int64_t half = (n - 1) / 2;
    std::vector<Tensor> parts;
    for (int64_t i = half; i >= 1; --i)
        parts.push_back(lo.expand_dims(-1) * (double(i) / double(half + 1)));
    if (n % 2 == 0) parts.push_back(lo.zeros_like().expand_dims(-1));
    for (int64_t i = 1; i <= half; ++i)
        parts.push_back(hi.expand_dims(-1) * (double(i) / double(half + 1)));
    return Tensor::concat(parts, -1) - center.expand_dims(-1);
}

// The interval gradient ---------------------------------------------------------------------------

/// The gradient of the output box's F-radius with respect to the input radius r (B, n0), through
/// the interval propagation of the network (CORA's aux_updateGradients of an interval).
Tensor aux_interval_gradient(const ZonoNet &net, const Tensor &x, const Tensor &r) {
    // the boxes' centers and radii of each output (mids[j + 1]; 0: the input)
    std::vector<Tensor> lo{x - r}, hi{x + r}, plo, phi, mids, rads;
    for (std::size_t k = 0; k < net.W.size(); ++k) {
        mids.push_back((lo.back() + hi.back()) * 0.5);
        rads.push_back((hi.back() - lo.back()) * 0.5);
        const Tensor mid = net.linear(k, mids) + net.b[k];
        const Tensor rad = net.linear(k, rads, true);
        plo.push_back(mid - rad);
        phi.push_back(mid + rad);
        // the box after the activation
        lo.push_back(net.activations[k] == Activation::identity  ? plo.back()
                     : net.activations[k] == Activation::relu    ? plo.back().maximum(0.0)
                     : net.activations[k] == Activation::sigmoid ? plo.back().sigmoid()
                                                                 : plo.back().tanh());
        hi.push_back(net.activations[k] == Activation::identity  ? phi.back()
                     : net.activations[k] == Activation::relu    ? phi.back().maximum(0.0)
                     : net.activations[k] == Activation::sigmoid ? phi.back().sigmoid()
                                                                 : phi.back().tanh());
    }
    // the F-radius of the output box, back through the layers
    const Tensor width = hi.back() - lo.back();
    const Tensor frad = (width.square().sum(1, true) + numerics::kFradSmoothing).sqrt();
    Tensor g_hi = width.divide(frad), g_lo = g_hi.negative();
    // the gradients of each output's center and radius, summed over the layers that read it
    std::vector<Tensor> g_mids(net.W.size() + 1), g_rads(net.W.size() + 1);
    for (std::size_t k = net.W.size(); k-- > 0;) {
        // the bounds' gradients of output k (the last one's: the seeds)
        if (k + 1 < net.W.size()) {
            const Tensor g_mid = g_mids[k + 1] * 0.5, g_rad = g_rads[k + 1] * 0.5;
            g_lo = g_mid - g_rad;
            g_hi = g_mid + g_rad;
        }
        const auto slope = [&](const Tensor &z) {
            switch (net.activations[k]) {
            case Activation::identity:
                return z.ones_like();
            case Activation::relu:
                return z.greater(z.zeros_like());
            case Activation::sigmoid:
                return z.sigmoid().multiply(1.0 - z.sigmoid());
            case Activation::tanh:
                return 1.0 - z.tanh().square();
            }
            CORA_FAIL(Error::wrong_value, "verify: an unknown activation");
        };
        g_hi = g_hi.multiply(slope(phi[k]));
        g_lo = g_lo.multiply(slope(plo[k]));
        // lo = mid - rad, hi = mid + rad, mid and rad of the layer's input box
        net.linear_t(k, g_lo + g_hi, g_mids);
        net.linear_t(k, g_hi - g_lo, g_rads, true);
    }
    const Tensor g_mid = g_mids[0] * 0.5, g_rad = g_rads[0] * 0.5;
    return (g_mid + g_rad) - (g_mid - g_rad);
}

// The parts of a batch ----------------------------------------------------------------------------

/// The children of a batch: the boxes, their cases and their split ids.
struct Children {
    Tensor x, r, cases, splits;
};

/// The constraints of the splits: rows (B, ps, q), offsets (B, ps, pieces - 1), the signed-to-be
/// neuron ids (B, ps; 0 for none) and the dummy flags (B, ps).
struct SplitRows {
    Tensor rows, offsets, ids, dummy;
};

/// What one batch needs: the sensitivity, the input zonotope, the recorded propagation.
struct Artifacts {
    bool has_sens = false;
    ZonoSensitivity sens;
    ZonoInput in;
    ZonoPropagation out;
    Tensor input_grad;
};

/// Adds the seconds of its scope to a stat (the time of each step of a batch).
class StepTimer {
  public:
    StepTimer(std::map<std::string, double> *stats, const char *name)
        : stats_(stats), name_(name), started_(now_seconds()) {}
    ~StepTimer() {
        if (stats_) (*stats_)[name_] += now_seconds() - started_;
    }
    StepTimer(const StepTimer &) = delete;
    StepTimer &operator=(const StepTimer &) = delete;

  private:
    std::map<std::string, double> *stats_;
    const char *name_;
    double started_;
};

// The bounding ------------------------------------------------------------------------------------

/// CORA's verify: the zonotopes of priv_zono_batch, refined in the latent space.
class ConZonotopeBounding : public Bounding {
  public:
    /// alpha: the check also optimizes the ReLU slopes (the bounding "zonotope-alpha").
    explicit ConZonotopeBounding(const NnVerifyOptions &o, bool alpha = false)
        : o_(o), alpha_(alpha) {}

    bool prepare(NnProblem &problem) override;
    std::string branch_and_bound(NnProblem &problem, double deadline,
                                 std::map<std::string, double> &stats,
                                 AttackResult &found) override;

    /// The zonotack candidates of the boxes (x, r) of the cases, unrefined (the attack).
    bool zonotack_attack(NnProblem &problem, AttackResult &found);

  private:
    struct Step {
        bool found = false;
        Children children;
        int64_t verified = 0;
    };

    // the steps of one batch
    Step aux_iteration(const Tensor &xi, const Tensor &ri, const Tensor &ci, const Tensor &si);
    Tensor aux_margins(const ZonoPropagation &out) const;
    Tensor aux_excluded(const ZonoPropagation &out, const Tensor &cases) const;
    Tensor aux_excluded_lower(const Tensor &lower, const Tensor &cases) const;
    Tensor aux_dual_margins(const ZonoPropagation &out, const Tensor &si) const;
    Tensor aux_crown_alpha_excluded(const Tensor &xi, const Tensor &ri, const Tensor &ci,
                                    const Tensor &si, const Tensor &excluded) const;
    std::vector<int64_t> aux_near(const std::vector<int64_t> &open, const Tensor &gap) const;
    Tensor aux_gap(const ZonoPropagation &out, const Tensor &lower, const Tensor &excluded) const;
    Tensor aux_alpha_excluded(const Tensor &xi, const Tensor &ri, const Tensor &ci,
                              const Tensor &si, const Tensor &excluded) const;
    Artifacts aux_artifacts(const Tensor &xi, const Tensor &ri, const Tensor &si) const;
    Tensor aux_input_dims(const Tensor &xi, const Tensor &ri, const Artifacts &art) const;
    std::pair<Tensor, Tensor> aux_zonotack_candidates(const Tensor &cx, const Tensor &Gx,
                                                      const ZonoPropagation &out,
                                                      const Tensor &cases,
                                                      const Tensor &excluded) const;
    bool aux_falsify(const Tensor &xi, const Tensor &ri, const Tensor &ci, const Artifacts &art,
                     const Tensor &excluded);
    // the constraints and the refinement
    SplitRows aux_split_constraints(const Tensor &xi, const Tensor &ri, const Tensor &ci,
                                    const Tensor &si, const Artifacts &art) const;
    Tensor aux_look_ahead(const Tensor &xi, const Tensor &ri, const Tensor &ci, const Tensor &si,
                          const Artifacts &art, const Tensor &score, int64_t k) const;
    std::pair<Tensor, Tensor> aux_relu_constraints(const Tensor &si, const Artifacts &art) const;
    std::pair<Children, int64_t> aux_refine(const Tensor &ci, const Tensor &si,
                                            const Artifacts &art, const Tensor &excluded,
                                            const SplitRows &split, const Tensor &At,
                                            const Tensor &bt);
    std::tuple<Tensor, Tensor, Tensor>
    aux_bound_children(const Tensor &cy, const Tensor &Gy, const Tensor &dry, const Tensor &C,
                       const Tensor &d, const std::vector<int64_t> &member, int64_t children);
    Children aux_naive_children(const Tensor &xi, const Tensor &ri, const Tensor &ci,
                                const Tensor &si, const Artifacts &art) const;
    std::vector<int64_t> aux_pop(int64_t size) const;
    bool aux_back_off();
    /// Whether the deadline passed; it stays passed (the batch then stays open).
    bool aux_late() {
        timed_out_ = timed_out_ || now_seconds() >= deadline_;
        return timed_out_;
    }
    std::string aux_heuristic_or_none(const std::string &name, bool used) const {
        return used && name != "none" ? name : "";
    }

    NnVerifyOptions o_;
    bool alpha_ = false;
    // the problem, the network and the polytopes as rows
    NnProblem *problem_ = nullptr;
    std::unique_ptr<ZonoNet> net_;
    Tensor A_, b_, AT_, abs_at_, abs_a_, real_row_, poly_case_;
    int64_t P_ = 0, K_ = 0, n_relu_ = 0, batch_ = 1;
    double slot_budget_ = -1;     ///< the error slots per layer the memory budget left (-1: all)
    double init_gen_budget_ = -1; ///< the input generators it left (-1: as the options say)
    double deadline_ = kInf;
    bool timed_out_ = false;
    AttackResult *found_ = nullptr;
    std::map<std::string, double> *stats_ = nullptr;
};

bool ConZonotopeBounding::prepare(NnProblem &problem) {
    problem_ = &problem;
    net_ = std::make_unique<ZonoNet>(problem.net(), o_.num_init_gens, o_.num_approx_err,
                                     o_.poly_method, o_.approx_error_order);
    A_ = problem.A;
    b_ = problem.b;
    P_ = A_.shape()[0];
    K_ = A_.shape()[1];
    abs_a_ = A_.abs();
    // the rows as one axis (P K), transposed for the rows of the sets
    AT_ = A_.reshape({P_ * K_, A_.shape()[2]}).matrix_transpose();
    abs_at_ = AT_.abs();
    poly_case_ = problem.poly_case;
    real_row_ = b_.isfinite();
    // the ReLU rows: every ReLU neuron for inf
    n_relu_ = 0;
    if (std::isinf(o_.num_relu_constraints)) {
        for (std::size_t k = 0; k < net_->W.size(); ++k)
            if (net_->activations[k] == Activation::relu) n_relu_ += net_->widths[k];
    } else {
        n_relu_ = int64_t(o_.num_relu_constraints);
    }
    // the entries of one set per generator: every layer's values, and a convolution's patches
    int64_t elements = net_->n0, work = 0;
    for (const int64_t w : net_->widths) elements += w;
    for (const NeuralNetwork::Layer &layer : net_->maps) work = std::max(work, layer.work());
    elements += work;
    // the slot budget: fewer error slots per layer until one set fits (docs/nn_verification.md)
    double slots = o_.num_approx_err;
    const auto bytes = [&](const ZonoNet &net) { return 8 * 4 * net.q * elements; };
    while (bytes(*net_) > problem.memory_budget && net_->q > net_->q0) {
        int64_t widest = 0;
        for (const int64_t s : net_->slots) widest = std::max(widest, s);
        const int64_t half = widest / 2;
        slots = double(half);
        net_ = std::make_unique<ZonoNet>(problem.net(), o_.num_init_gens, slots, o_.poly_method,
                                         o_.approx_error_order);
    }
    // then fewer input generators, the rest of the input box in the interval center too
    double gens = o_.num_init_gens;
    while (bytes(*net_) > problem.memory_budget && net_->q0 > 1) {
        const int64_t half = net_->q0 / 2;
        gens = double(half);
        net_ = std::make_unique<ZonoNet>(problem.net(), gens, slots, o_.poly_method,
                                         o_.approx_error_order);
    }
    if (slots != o_.num_approx_err) slot_budget_ = slots;
    if (gens != o_.num_init_gens) init_gen_budget_ = gens;
    // CORA's batch size, at most what the memory budget holds
    batch_ =
        std::min(o_.mini_batch_size, std::max<int64_t>(1, problem.memory_budget / bytes(*net_)));
    return true;
}

// The check ---------------------------------------------------------------------------------------

Tensor ConZonotopeBounding::aux_margins(const ZonoPropagation &out) const {
    const int64_t B = out.c.shape()[0];
    // the lower bounds of A y - b over the zonotope (CORA's logit difference)
    const Tensor Ac = out.c.matmul(AT_);
    const Tensor AG = out.G.matmul(AT_).abs().sum(1);
    Tensor lower = Ac - AG - b_.reshape({P_ * K_});
    if (out.dr.defined()) lower = lower - out.dr.matmul(abs_at_);
    return lower.reshape({B, P_, K_});
}

Tensor ConZonotopeBounding::aux_excluded(const ZonoPropagation &out, const Tensor &cases) const {
    return aux_excluded_lower(aux_margins(out), cases);
}

Tensor ConZonotopeBounding::aux_excluded_lower(const Tensor &lower, const Tensor &cases) const {
    const Tensor decided = lower.greater(lower.full_like({1}, o_.verified_margin)).any(2);
    const Tensor other = cases.expand_dims(1).not_equal(poly_case_.expand_dims(0));
    return decided.maximum(other);
}

Tensor ConZonotopeBounding::aux_dual_margins(const ZonoPropagation &out, const Tensor &si) const {
    const int64_t B = out.c.shape()[0];
    const auto [C, d] = priv_dual_rows(out, *net_, si, o_.dual_relu_rows);
    if (!C.defined()) return aux_margins(out);
    // the rows' centers and radii of the interval center, and their generator coefficients w
    Tensor base = out.c.matmul(AT_) - b_.reshape({P_ * K_});
    if (out.dr.defined()) base = base - out.dr.matmul(abs_at_);
    const Tensor w = out.G.matmul(AT_).permute_dims({0, 2, 1});
    const Tensor dual =
        priv_dual_lower(w, C, d, o_.dual_iterations, o_.dual_step_size, problem_->options().dtype);
    return (base + dual).reshape({B, P_, K_});
}

Tensor ConZonotopeBounding::aux_crown_alpha_excluded(const Tensor &xi, const Tensor &ri,
                                                     const Tensor &ci, const Tensor &si,
                                                     const Tensor &excluded) const {
    // CROWN's bound with the lower ReLU slopes optimized per row, over the zonotope's bounds
    const ZonoInput in = net_->input_zonotope(xi, ri);
    const ZonoPropagation out = net_->forward(in.c, in.dr, in.G, si, nullptr, true);
    const int64_t B = xi.shape()[0];
    const Tensor crown = priv_backward_alpha(out, *net_, si, xi, ri, AT_.matrix_transpose(),
                                             o_.alpha_iterations, o_.alpha_step_size, deadline_) -
                         b_.reshape({P_ * K_});
    const Tensor lower = aux_margins(out).maximum(crown.reshape({B, P_, K_}));
    return excluded.maximum(aux_excluded_lower(lower, ci));
}

std::vector<int64_t> ConZonotopeBounding::aux_near(const std::vector<int64_t> &open,
                                                   const Tensor &gap) const {
    // the open sets whose gap is at most optimize_gap (all without a gap)
    if (!gap.defined()) return open;
    const std::vector<double> g = gap.data();
    std::vector<int64_t> near;
    for (const int64_t i : open)
        if (g[std::size_t(i)] <= o_.optimize_gap) near.push_back(i);
    return near;
}

Tensor ConZonotopeBounding::aux_gap(const ZonoPropagation &out, const Tensor &lower,
                                    const Tensor &excluded) const {
    // a row's gap: how far its lower bound is below 0, a share of the row's width (B, P, K)
    const int64_t B = out.c.shape()[0];
    Tensor radius = out.G.matmul(AT_).abs().sum(1);
    if (out.dr.defined()) radius = radius + out.dr.matmul(abs_at_);
    const Tensor share =
        lower.negative().divide(radius.reshape({B, P_, K_}).maximum(numerics::kTiny) * 2.0);
    // per open polytope its best row's, per set the largest of them
    const Tensor best = share.min(2);
    return Tensor::where(excluded.greater(excluded.zeros_like()), best.full_like({1}, -kInf), best)
        .max(1);
}

// The optimized slopes ----------------------------------------------------------------------------

Tensor ConZonotopeBounding::aux_alpha_excluded(const Tensor &xi_, const Tensor &ri_,
                                               const Tensor &ci_, const Tensor &si_,
                                               const Tensor &excluded_) const {
    const ZonoNet &net = *net_;
    const Tensor A = A_.reshape({P_ * K_, A_.shape()[2]}), abs_a = A.abs();
    // the open sets; a set leaves the loop once every polytope of its case is excluded
    Tensor ci = ci_, si = si_, excluded = excluded_;
    std::vector<int64_t> rows(static_cast<std::size_t>(xi_.shape()[0]));
    for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = int64_t(i);
    Tensor result = excluded_;
    ZonoInput in = net.input_zonotope(xi_, ri_);
    ZonoPropagation out = net.forward(in.c, in.dr, in.G, si, nullptr, true);
    std::vector<Tensor> alpha = priv_alpha_start(out, net, si), m1(alpha.size()), m2(alpha.size());
    constexpr double kB1 = 0.9, kB2 = 0.999;
    // the steps stop after alpha_patience of them without a newly excluded polytope
    double count = excluded.sum().item();
    int64_t gained = 0;
    for (int64_t t = 1; t <= o_.alpha_iterations && now_seconds() < deadline_; ++t) {
        if (t - gained > o_.alpha_patience) break;
        // the ascent of the sum over the open polytopes of their best row's lower bound
        const Tensor lower = aux_margins(out);
        const int64_t B = lower.shape()[0];
        const Tensor best =
            aux_onehot(lower.argmax(2), K_).multiply((1.0 - excluded).expand_dims(-1));
        const Tensor w = best.reshape({B, P_ * K_});
        const Tensor S = out.G.matmul(AT_).sign().multiply(w.expand_dims(1));
        const std::vector<Tensor> g =
            priv_alpha_gradients(out, net, si, alpha, w.matmul(A), S.matmul(A).negative(),
                                 out.dr.defined() ? w.matmul(abs_a).negative() : Tensor());
        // Adam, projected onto [0, 1]
        const double c1 = 1.0 - std::pow(kB1, double(t)), c2 = 1.0 - std::pow(kB2, double(t));
        for (std::size_t k = 0; k < alpha.size(); ++k) {
            if (!g[k].defined()) continue;
            m1[k] = m1[k].defined() ? m1[k] * kB1 + g[k] * (1.0 - kB1) : g[k] * (1.0 - kB1);
            const Tensor g2 = g[k].square() * (1.0 - kB2);
            m2[k] = m2[k].defined() ? m2[k] * kB2 + g2 : g2;
            const Tensor step = (m1[k] / c1).divide((m2[k] / c2).sqrt() + numerics::kAdamEpsilon);
            alpha[k] = (alpha[k] + step * o_.alpha_step_size).maximum(0.0).minimum(1.0);
        }
        // every bound is sound: a polytope excluded once stays excluded
        out = net.forward(in.c, in.dr, in.G, si, nullptr, true, &alpha);
        excluded = excluded.maximum(aux_excluded(out, ci));
        result = aux_put_rows(result, rows, excluded);
        const double now = excluded.sum().item();
        if (now > count) gained = t;
        count = now;
        const std::vector<int64_t> open = aux_where(1.0 - excluded.min(1));
        if (open.empty()) break;
        if (int64_t(open.size()) == B) continue;
        // the open sets only, from here
        std::vector<int64_t> kept;
        kept.reserve(open.size());
        for (const int64_t i : open) kept.push_back(rows[std::size_t(i)]);
        rows = kept;
        excluded = excluded.take(open, 0);
        count = excluded.sum().item();
        ci = ci.take(open, 0);
        si = si.take(open, 0);
        in = {in.c.take(open, 0), in.dr.defined() ? in.dr.take(open, 0) : Tensor(),
              in.G.take(open, 0), in.dims.take(open, 0), in.values.take(open, 0)};
        for (std::size_t k = 0; k < alpha.size(); ++k) {
            if (!alpha[k].defined()) continue;
            alpha[k] = alpha[k].take(open, 0);
            if (m1[k].defined()) m1[k] = m1[k].take(open, 0);
            if (m2[k].defined()) m2[k] = m2[k].take(open, 0);
        }
        out = net.forward(in.c, in.dr, in.G, si, nullptr, true, &alpha);
    }
    return result;
}

// The queue ---------------------------------------------------------------------------------------

std::vector<int64_t> ConZonotopeBounding::aux_pop(int64_t size) const {
    const int64_t take = std::min(batch_, size);
    std::vector<int64_t> idx(static_cast<std::size_t>(take));
    for (int64_t i = 0; i < take; ++i) idx[std::size_t(i)] = i;
    // half from the front and half from the back
    if (o_.verify_dequeue_type == "half-half") {
        const int64_t start = (take + 1) / 2;
        for (int64_t i = start; i < take; ++i) idx[std::size_t(i)] += size - take;
    } else if (o_.verify_dequeue_type != "front") {
        CORA_FAIL(Error::wrong_field_value, "verify: the verify_dequeue_type '" +
                                                o_.verify_dequeue_type +
                                                "' is unknown; use 'front' or 'half-half'");
    }
    return idx;
}

bool ConZonotopeBounding::aux_back_off() {
    // CORA's answer to running out of memory: fewer ReLU rows, then half the batch
    if (n_relu_ > 0) {
        n_relu_ = n_relu_ > 100 ? 100 : n_relu_ / 10;
        return true;
    }
    if (batch_ > 1) {
        batch_ = (batch_ + 1) / 2;
        return true;
    }
    return false;
}

// The branch and bound ----------------------------------------------------------------------------

std::string ConZonotopeBounding::branch_and_bound(NnProblem &problem, double deadline,
                                                  std::map<std::string, double> &stats,
                                                  AttackResult &found) {
    deadline_ = deadline;
    found_ = &found;
    stats_ = &stats;
    if (slot_budget_ >= 0) stats["slot_budget"] = slot_budget_;
    if (init_gen_budget_ >= 0) stats["init_gen_budget"] = init_gen_budget_;
    Tensor xs = (problem.outer_lo + problem.outer_hi) * 0.5;
    Tensor rs = (problem.outer_hi - problem.outer_lo) * 0.5;
    Tensor cases = problem.box_case;
    Tensor splits = xs.zeros_like({xs.shape()[0], 0});
    int64_t iteration = 0, verified = 0;
    double last = 0;
    while (xs.shape()[0] > 0) {
        // a batch starts only if the last one's time still fits before the deadline
        if (iteration >= o_.max_verif_iter || now_seconds() + last >= deadline) break;
        const double started = now_seconds();
        // the next batch out of the queue
        const std::vector<int64_t> take = aux_pop(xs.shape()[0]);
        const std::vector<int64_t> keep = aux_rest(xs.shape()[0], take);
        const Tensor xi = xs.take(take, 0), ri = rs.take(take, 0), ci = cases.take(take, 0),
                     si = splits.take(take, 0);
        xs = xs.take(keep, 0);
        rs = rs.take(keep, 0);
        cases = cases.take(keep, 0);
        splits = splits.take(keep, 0);
        // one batch; out of memory, fewer ReLU rows or a smaller batch
        Step step;
        try {
            step = aux_iteration(xi, ri, ci, si);
        } catch (const std::exception &error) {
            if (!priv_is_oom(error) || !aux_back_off())
                std::rethrow_exception(std::current_exception());
            xs = Tensor::concat({xi, xs}, 0);
            rs = Tensor::concat({ri, rs}, 0);
            cases = Tensor::concat({ci, cases}, 0);
            splits = aux_cat_splits(si, splits, false);
            continue;
        }
        if (timed_out_) {
            xs = Tensor::concat({xi, xs}, 0); // the batch stays open: unknown
            break;
        }
        stats["boxes"] += double(xi.shape()[0]);
        verified += step.verified;
        ++iteration;
        last = now_seconds() - started;
        if (step.found) {
            stats["iterations"] = double(iteration);
            return "sat";
        }
        // the children before (prepend) or after (append) the queue
        const Children &c = step.children;
        if (o_.verify_enqueue_type == "append") {
            xs = Tensor::concat({xs, c.x}, 0);
            rs = Tensor::concat({rs, c.r}, 0);
            cases = Tensor::concat({cases, c.cases}, 0);
            splits = aux_cat_splits(splits, c.splits, false);
        } else if (o_.verify_enqueue_type == "prepend") {
            xs = Tensor::concat({c.x, xs}, 0);
            rs = Tensor::concat({c.r, rs}, 0);
            cases = Tensor::concat({c.cases, cases}, 0);
            splits = aux_cat_splits(c.splits, splits, false);
        } else {
            CORA_FAIL(Error::wrong_field_value, "verify: the verify_enqueue_type '" +
                                                    o_.verify_enqueue_type +
                                                    "' is unknown; use 'prepend' or 'append'");
        }
    }
    stats["iterations"] = double(iteration);
    // the verdict: no set left is unsat
    stats["verified_sets"] = double(verified);
    return xs.shape()[0] == 0 ? "unsat" : "unknown";
}

// One batch ---------------------------------------------------------------------------------------

ConZonotopeBounding::Step ConZonotopeBounding::aux_iteration(const Tensor &xi_, const Tensor &ri_,
                                                             const Tensor &ci_, const Tensor &si_) {
    const ZonoNet &net = *net_;
    Step step;
    // 1. the check on the whole batch
    Tensor excluded_all, gap;
    std::vector<int64_t> open;
    {
        const StepTimer timer(stats_, "check_seconds");
        // the output zonotope, recorded for the backward bound
        const ZonoInput in = net.input_zonotope(xi_, ri_);
        const ZonoPropagation out = net.forward(in.c, in.dr, in.G, si_, nullptr, o_.backward_bound);
        Tensor lower = aux_margins(out);
        if (o_.backward_bound) {
            // the better of the zonotope's and CROWN's lower bound of every row
            const int64_t B = xi_.shape()[0];
            const Tensor crown =
                priv_backward_lower(out, net, si_, xi_, ri_, AT_.matrix_transpose()) -
                b_.reshape({P_ * K_});
            lower = lower.maximum(crown.reshape({B, P_, K_}));
        }
        excluded_all = aux_excluded_lower(lower, ci_);
        open = aux_where(1.0 - excluded_all.min(1));
        if ((alpha_ || o_.dual_certificate) && std::isfinite(o_.optimize_gap))
            gap = aux_gap(out, lower, excluded_all);
    }
    // 1b. the open sets again with optimized ReLU slopes ("zonotope-alpha")
    if (alpha_ && !open.empty() && o_.alpha_iterations > 0) {
        // the sets whose gap is at most optimize_gap (all for inf)
        const std::vector<int64_t> near = aux_near(open, gap);
        if (!near.empty()) {
            const StepTimer timer(stats_, "alpha_seconds");
            (*stats_)["alpha_sets"] += double(near.size());
            // the slopes of the zonotope's lines or CROWN's backward pass
            const Tensor xn = xi_.take(near, 0), rn = ri_.take(near, 0), cn = ci_.take(near, 0),
                         sn = si_.take(near, 0), en = excluded_all.take(near, 0);
            const Tensor ex = o_.alpha_method == "crown"
                                  ? aux_crown_alpha_excluded(xn, rn, cn, sn, en)
                                  : aux_alpha_excluded(xn, rn, cn, sn, en);
            excluded_all = aux_put_rows(excluded_all, near, ex);
            // the sets the slopes verified
            const std::size_t before = open.size();
            open = aux_where(1.0 - excluded_all.min(1));
            (*stats_)["alpha_verified"] += double(before - open.size());
        }
    }
    // 1c. the dual certificate on the open sets near enough
    if (o_.dual_certificate) {
        const std::vector<int64_t> near = aux_near(open, gap);
        if (!near.empty()) {
            // the propagation of these sets, recorded, and their rows' dual bounds
            const StepTimer timer(stats_, "dual_seconds");
            const Tensor ci = ci_.take(near, 0), si = si_.take(near, 0);
            const ZonoInput in = net.input_zonotope(xi_.take(near, 0), ri_.take(near, 0));
            const ZonoPropagation out = net.forward(in.c, in.dr, in.G, si, nullptr, true);
            const Tensor ex = aux_excluded_lower(aux_dual_margins(out, si), ci)
                                  .maximum(excluded_all.take(near, 0));
            excluded_all = aux_put_rows(excluded_all, near, ex);
            // the sets the dual bound verified
            const std::size_t before = open.size();
            open = aux_where(1.0 - excluded_all.min(1));
            (*stats_)["dual_verified"] += double(before - open.size());
        }
    }
    step.verified = xi_.shape()[0] - int64_t(open.size());
    const auto empty = [&] {
        return Children{xi_.slice(0, 0, 0), ri_.slice(0, 0, 0), ci_.slice(0, 0, 0),
                        si_.slice(0, 0, 0)};
    };
    if (open.empty()) {
        step.children = empty();
        return step;
    }
    const Tensor xi = xi_.take(open, 0), ri = ri_.take(open, 0), ci = ci_.take(open, 0),
                 si = si_.take(open, 0), excluded = excluded_all.take(open, 0);
    // the artifacts of the open sets, then 2. falsification
    Artifacts art;
    {
        const StepTimer timer(stats_, "artifacts_seconds");
        art = aux_artifacts(xi, ri, si);
    }
    // 2. falsification at the candidates of the sets
    bool falsified = false;
    {
        const StepTimer timer(stats_, "falsify_seconds");
        falsified = aux_falsify(xi, ri, ci, art, excluded);
    }
    if (falsified) {
        step.found = true;
        return step;
    }
    if (aux_late()) return step;
    if (o_.refinement_method == "naive") {
        step.children = aux_naive_children(xi, ri, ci, si, art);
        return step;
    }
    if (o_.refinement_method != "zonotack")
        CORA_FAIL(Error::wrong_field_value, "verify: the refinement_method '" +
                                                o_.refinement_method +
                                                "' is unknown; use 'zonotack' or 'naive'");
    // 3. the split and ReLU constraints, 4. the refinement of every child
    SplitRows split;
    std::pair<Tensor, Tensor> relu_rows;
    {
        const StepTimer timer(stats_, "constraints_seconds");
        split = aux_split_constraints(xi, ri, ci, si, art);
        relu_rows = aux_relu_constraints(si, art);
    }
    auto [children, n_empty] =
        aux_refine(ci, si, art, excluded, split, relu_rows.first, relu_rows.second);
    if (timed_out_) return step;
    step.verified += n_empty;
    // the point sets are checked and dropped
    const std::vector<int64_t> point = aux_where(children.r.equal(children.r.zeros_like()).all(1));
    if (!point.empty()) {
        if (priv_check_points(*problem_, children.x.take(point, 0), children.cases.take(point, 0),
                              *found_)) {
            step.found = true;
            return step;
        }
        step.verified += int64_t(point.size());
        const std::vector<int64_t> rest = aux_rest(children.x.shape()[0], point);
        children = {children.x.take(rest, 0), children.r.take(rest, 0),
                    children.cases.take(rest, 0), children.splits.take(rest, 0)};
    }
    step.children = children;
    return step;
}

// The artifacts -----------------------------------------------------------------------------------

Artifacts ConZonotopeBounding::aux_artifacts(const Tensor &xi, const Tensor &ri,
                                             const Tensor &si) const {
    const ZonoNet &net = *net_;
    // which heuristics run, and what they read
    std::vector<std::string> names = {
        aux_heuristic_or_none(o_.input_split_heuristic, o_.num_dimensions > 0),
        aux_heuristic_or_none(o_.neuron_split_heuristic, o_.num_neuron_splits > 0),
        aux_heuristic_or_none(o_.relu_constraint_heuristic, n_relu_ > 0),
        aux_heuristic_or_none(o_.input_generator_heuristic, net.q0 < net.n0)};
    bool need_sens = o_.bab_falsification_method == "fgsm" ||
                     (o_.approx_error_order == "sensitivity*length" && net.interval_center);
    bool need_grad = false;
    for (const std::string &name : names) {
        if (name.empty()) continue;
        need_sens = need_sens || heuristic(name).sens;
        need_grad = need_grad || heuristic(name).grad;
    }
    // the sensitivity, the input zonotope, the recorded propagation, the gradients
    Artifacts art;
    if (need_sens) {
        art.sens = net.sensitivity(xi);
        art.has_sens = true;
    }
    const Tensor dims = net.q0 < net.n0 ? aux_input_dims(xi, ri, art) : Tensor();
    art.in = net.input_zonotope(xi, ri, dims);
    art.out = net.forward(art.in.c, art.in.dr, art.in.G, si,
                          art.has_sens ? &art.sens.layer_sum : nullptr, true);
    if (need_grad) {
        const StepTimer timer(stats_, "gradient_seconds");
        art.input_grad = priv_frad_gradients(art.out, art.in, net, si, o_.exact_backprop);
    }
    return art;
}

Tensor ConZonotopeBounding::aux_input_dims(const Tensor &xi, const Tensor &ri,
                                           const Artifacts &art) const {
    // CORA's input_generator_heuristic: the q0 input dimensions that get a generator
    const std::string &name = o_.input_generator_heuristic;
    HeuristicInputs in{xi - ri, xi + ri, ri, art.has_sens ? art.sens.input : Tensor(), Tensor()};
    if (heuristic(name).grad) in.grad = aux_interval_gradient(*net_, xi, ri);
    const Tensor h = priv_compute_heuristic(name, in, Tensor(), 0, false);
    return h.argsort(1, true).slice(1, 0, net_->q0);
}

std::pair<Tensor, Tensor>
ConZonotopeBounding::aux_zonotack_candidates(const Tensor &cx, const Tensor &Gx,
                                             const ZonoPropagation &out, const Tensor &cases,
                                             const Tensor &excluded) const {
    const int64_t B = cx.shape()[0], q0 = net_->q0, n0 = net_->n0;
    // beta = -sign(A_i G_y) on the input factors, the input point c_x + G_x beta per row
    const Tensor AG = out.G.slice(1, 0, q0).matmul(AT_).permute_dims({0, 2, 1});
    const Tensor x = cx.expand_dims(1) + AG.sign().negative().matmul(Gx);
    const Tensor use = (1.0 - excluded).expand_dims(-1).multiply(real_row_.expand_dims(0));
    const std::vector<int64_t> rows = aux_where(use);
    return {x.reshape({B * P_ * K_, n0}).take(rows, 0), cases.take(priv_rows_of(rows, P_ * K_), 0)};
}

// Falsification -----------------------------------------------------------------------------------

bool ConZonotopeBounding::aux_falsify(const Tensor &xi, const Tensor &ri, const Tensor &ci,
                                      const Artifacts &art, const Tensor &excluded) {
    const std::string &method = o_.bab_falsification_method;
    if (method == "zonotack") {
        const auto [x, cases] = aux_zonotack_candidates(art.in.c, art.in.G, art.out, ci, excluded);
        return priv_check_points(*problem_, x, cases, *found_);
    }
    if (method == "center") return priv_check_points(*problem_, xi, ci, *found_);
    if (method == "fgsm") {
        // CORA's 'fgsm': per row of the case's polytopes the corner xi + ri sign(-A_i S)
        const int64_t B = xi.shape()[0], n0 = net_->n0;
        const Tensor dy = AT_.matrix_transpose().matmul(art.sens.jacobian).negative();
        const Tensor x = xi.expand_dims(1) + ri.expand_dims(1).multiply(dy.sign());
        const Tensor mine = ci.expand_dims(1).equal(poly_case_.expand_dims(0));
        const std::vector<int64_t> rows =
            aux_where(mine.expand_dims(-1).multiply(real_row_.expand_dims(0)));
        return priv_check_points(*problem_, x.reshape({B * P_ * K_, n0}).take(rows, 0),
                                 ci.take(priv_rows_of(rows, P_ * K_), 0), *found_);
    }
    CORA_FAIL(Error::wrong_field_value, "verify: the bab_falsification_method '" + method +
                                            "' is unknown; use 'zonotack', 'center' or 'fgsm'");
}

// The constraints ---------------------------------------------------------------------------------

SplitRows ConZonotopeBounding::aux_split_constraints(const Tensor &xi, const Tensor &ri,
                                                     const Tensor &ci, const Tensor &si,
                                                     const Artifacts &art) const {
    const ZonoNet &net = *net_;
    const int64_t B = xi.shape()[0], q = net.q, pieces = o_.num_splits;
    std::vector<Tensor> rows, offsets, ids, dummy;
    // the neuron splits: the best k neurons of all layers by the heuristic
    if (o_.num_neuron_splits > 0 && pieces > 1) {
        const int64_t k = o_.num_neuron_splits;
        std::vector<Tensor> scores;
        for (const ZonoLayerRecord &rec : art.out.layers) {
            const HeuristicInputs in{
                rec.l, rec.u, rec.er,
                art.has_sens ? art.sens.layer_max[std::size_t(rec.index)] : Tensor(), rec.grad};
            scores.push_back(
                priv_compute_heuristic(o_.neuron_split_heuristic, in, si, rec.first_id, true));
        }
        scores.push_back(xi.full_like({B, k}, -1.0));
        const Tensor all = Tensor::concat(scores, 1);
        const Tensor top = heuristic(o_.neuron_split_heuristic).look_ahead
                               ? aux_look_ahead(xi, ri, ci, si, art, all, k)
                               : all.topk(k, 1).second;
        // the generator column, the pieces and the id of each chosen neuron, layer by layer
        Tensor R = xi.zeros_like({B, k, q}), off = xi.zeros_like({B, k, pieces - 1});
        Tensor nid = xi.zeros_like({B, k});
        int64_t start = 0;
        for (const ZonoLayerRecord &rec : art.out.layers) {
            const int64_t n = rec.l.shape()[1];
            const Tensor inside = top.greater_equal(top.full_like({1}, double(start)))
                                      .multiply(top.less(top.full_like({1}, double(start + n))));
            const Tensor j = (top - double(start)).clip(0.0, double(n - 1));
            R = Tensor::where(inside.expand_dims(-1), aux_pad(aux_columns(rec.G_in, j), q), R);
            const Tensor pieces_at =
                aux_pieces_offsets(aux_at(rec.l, j), aux_at(rec.u, j), aux_at(rec.c_in, j), pieces,
                                   o_.neuron_split_position);
            off = Tensor::where(inside.expand_dims(-1), pieces_at, off);
            // the bound is remembered for two pieces, at a point center only (the repair)
            Tensor point = inside;
            if (rec.dr_in.defined() && o_.remember_point_splits_only)
                point = point.multiply(aux_at(rec.dr_in, j).equal(j.zeros_like()));
            if (pieces == 2) nid = Tensor::where(point, j + double(rec.first_id), nid);
            start += n;
        }
        rows.push_back(R);
        offsets.push_back(off);
        ids.push_back(nid);
        dummy.push_back(top.greater_equal(top.full_like({1}, double(start))));
    }
    // the input splits: the best input generators, cut at equal distances in [-1, 1]
    if (o_.num_dimensions > 0 && pieces > 1) {
        const int64_t nd = std::min(o_.num_dimensions, net.q0);
        const Tensor &dims = art.in.dims;
        const Tensor x = aux_at(xi, dims), r = aux_at(ri, dims);
        const HeuristicInputs in{x - r, x + r, r,
                                 art.has_sens ? aux_at(art.sens.input, dims) : Tensor(),
                                 art.input_grad};
        const Tensor top = priv_compute_heuristic(o_.input_split_heuristic, in, Tensor(), 0, false)
                               .topk(nd, 1)
                               .second;
        // the cuts of [-1, 1] into equal pieces
        std::vector<double> cuts;
        for (int64_t i = 1; i < pieces; ++i)
            cuts.push_back(-1.0 + double(i) * (2.0 / double(pieces)));
        rows.push_back(aux_onehot(top, q));
        offsets.push_back(
            Tensor::like(xi, cuts, {1, 1, pieces - 1}).broadcast_to({B, nd, pieces - 1}));
        ids.push_back(xi.zeros_like({B, nd}));
        dummy.push_back(xi.zeros_like({B, nd}));
    }
    // no split: no rows
    if (rows.empty())
        return {xi.zeros_like({B, 0, q}), xi.zeros_like({B, 0, std::max<int64_t>(pieces - 1, 1)}),
                xi.zeros_like({B, 0}), xi.zeros_like({B, 0})};
    return {Tensor::concat(rows, 1), Tensor::concat(offsets, 1), Tensor::concat(ids, 1),
            Tensor::concat(dummy, 1)};
}

Tensor ConZonotopeBounding::aux_look_ahead(const Tensor &xi, const Tensor &ri, const Tensor &ci,
                                           const Tensor &si, const Artifacts &art,
                                           const Tensor &score, int64_t k) const {
    const int64_t B = xi.shape()[0], cols = score.shape()[1];
    const int64_t C = std::min(std::max(k, o_.look_ahead_candidates), cols);
    const Tensor cand = score.topk(C, 1).second;
    // the neuron id of every column of the scores (0 for the padding columns)
    std::vector<Tensor> id_cols;
    id_cols.reserve(art.out.layers.size() + 1);
    for (const ZonoLayerRecord &rec : art.out.layers)
        id_cols.push_back(rec.l.arange_like(rec.l.shape()[1]) + double(rec.first_id));
    id_cols.push_back(xi.zeros_like({k}));
    const Tensor all_ids = Tensor::concat(id_cols, 0).reshape({1, cols}).broadcast_to({B, cols});
    const Tensor ids = all_ids.take_along_axis(cand, 1);
    // each candidate's children, at least 0 (-id) and at most 0 (+id), other bounds held (FSB)
    const Tensor sides = Tensor::concat({ids.negative(), ids}, 1).reshape({B * 2 * C, 1});
    std::vector<int64_t> each;
    for (int64_t i = 0; i < B; ++i)
        for (int64_t j = 0; j < 2 * C; ++j) each.push_back(i);
    ZonoPropagation light;
    for (const ZonoLayerRecord &rec : art.out.layers) {
        ZonoLayerRecord r;
        r.index = rec.index;
        r.first_id = rec.first_id;
        r.l = rec.l.take(each, 0);
        r.u = rec.u.take(each, 0);
        light.layers.push_back(r);
    }
    const Tensor child_splits = Tensor::concat({si.take(each, 0), sides}, 1);
    const Tensor lower = (priv_backward_lower(light, *net_, child_splits, xi.take(each, 0),
                                              ri.take(each, 0), AT_.matrix_transpose()) -
                          b_.reshape({P_ * K_}))
                             .reshape({B, 2 * C, P_, K_});
    // a child's objective: its worst polytope of the case, by the polytope's best row
    const Tensor other = ci.expand_dims(1).not_equal(poly_case_.expand_dims(0)).expand_dims(1);
    const Tensor best = Tensor::where(other, lower.full_like({1}, kInf), lower.max(3));
    const Tensor child = best.min(2);
    Tensor value = child.slice(1, 0, C).minimum(child.slice(1, C, 2 * C));
    // a padding column or a neuron the filter ruled out is never preferred
    const Tensor allowed =
        score.take_along_axis(cand, 1).isfinite().multiply(ids.not_equal(ids.zeros_like()));
    value = Tensor::where(allowed, value, value.full_like({1}, -kInf));
    const Tensor order = value.argsort(1, true).slice(1, 0, k);
    return cand.take_along_axis(order, 1);
}

std::pair<Tensor, Tensor> ConZonotopeBounding::aux_relu_constraints(const Tensor &si,
                                                                    const Artifacts &art) const {
    const ZonoNet &net = *net_;
    std::vector<const ZonoLayerRecord *> recs;
    for (const ZonoLayerRecord &rec : art.out.layers)
        if (net.activations[std::size_t(rec.index)] == Activation::relu) recs.push_back(&rec);
    if (n_relu_ == 0 || recs.empty()) return {};
    const int64_t B = art.out.c.shape()[0], q = net.q;
    // the top neurons of the heuristic over all ReLU layers
    std::vector<Tensor> scores;
    for (const ZonoLayerRecord *rec : recs) {
        const HeuristicInputs in{
            rec->l, rec->u, rec->er,
            art.has_sens ? art.sens.layer_max[std::size_t(rec->index)] : Tensor(), rec->grad};
        scores.push_back(
            priv_compute_heuristic(o_.relu_constraint_heuristic, in, si, rec->first_id, true));
    }
    scores.push_back(art.out.c.full_like({B, n_relu_}, -1.0));
    const Tensor score = Tensor::concat(scores, 1);
    const int64_t k = std::min(n_relu_, score.shape()[1]);
    const Tensor top = score.topk(k, 1).second;
    Tensor A1 = top.zeros_like({B, k, q}), A2 = A1, b1 = top.zeros_like({B, k}), b2 = b1;
    // relu_constraint_faces "triangle" adds the chord above each crossing neuron
    const bool triangle = o_.relu_constraint_faces == "triangle";
    Tensor A3 = A1, b3 = b1;
    int64_t start = 0;
    for (const ZonoLayerRecord *rec : recs) {
        const int64_t n = rec->l.shape()[1];
        const Tensor inside = top.greater_equal(top.full_like({1}, double(start)))
                                  .multiply(top.less(top.full_like({1}, double(start + n))));
        const Tensor j = (top - double(start)).clip(0.0, double(n - 1));
        // ReLU(x) >= 0 and ReLU(x) >= x as rows on the factors of the output and the input
        const Tensor go = aux_pad(aux_columns(rec->G_out, j).negative(), q);
        const Tensor gi = aux_pad(aux_columns(rec->G_in, j), q);
        const Tensor co_u = rec->dr_out.defined() ? aux_at(rec->c_out, j) + aux_at(rec->dr_out, j)
                                                  : aux_at(rec->c_out, j);
        const Tensor ci_l = rec->dr_in.defined() ? aux_at(rec->c_in, j) - aux_at(rec->dr_in, j)
                                                 : aux_at(rec->c_in, j);
        const Tensor hit = inside.expand_dims(-1);
        A1 = Tensor::where(hit, go, A1);
        A2 = Tensor::where(hit, go + gi, A2);
        b1 = Tensor::where(inside, co_u, b1);
        b2 = Tensor::where(inside, co_u - ci_l, b2);
        if (triangle) {
            // ReLU(x) <= s (x - l), s = u / (u - l), on a crossing neuron (the chord, the cut)
            const auto [lc, uc] = net.clamp(rec->first_id, rec->l, rec->u, si);
            const Tensor l = aux_at(lc, j), u = aux_at(uc, j);
            const Tensor cross =
                inside.multiply(l.less(l.zeros_like())).multiply(u.greater(u.zeros_like()));
            const Tensor slope = u.divide((u - l).maximum(numerics::kTiny));
            const Tensor gs = aux_pad(aux_columns(rec->G_in, j), q).multiply(slope.expand_dims(-1));
            Tensor b = slope.multiply(aux_at(rec->c_in, j) - l) - aux_at(rec->c_out, j);
            if (rec->dr_out.defined()) b = b + aux_at(rec->dr_out, j);
            if (rec->dr_in.defined()) b = b + slope.multiply(aux_at(rec->dr_in, j));
            A3 = Tensor::where(cross.expand_dims(-1), go.negative() - gs, A3);
            b3 = Tensor::where(cross, b, b3);
        }
        start += n;
    }
    if (triangle)
        return aux_compact(Tensor::concat({A1, A2, A3}, 1), Tensor::concat({b1, b2, b3}, 1));
    return aux_compact(Tensor::concat({A1, A2}, 1), Tensor::concat({b1, b2}, 1));
}

// The refinement ----------------------------------------------------------------------------------

std::pair<Children, int64_t> ConZonotopeBounding::aux_refine(const Tensor &ci, const Tensor &si,
                                                             const Artifacts &art,
                                                             const Tensor &excluded,
                                                             const SplitRows &split,
                                                             const Tensor &At, const Tensor &bt) {
    const ZonoNet &net = *net_;
    const int64_t B = ci.shape()[0], ps = split.rows.shape()[1], q = net.q, q0 = net.q0;
    const int64_t pieces = split.offsets.shape()[2] + 1;
    // every combination of pieces (the last split fastest): its rows, ids and duplicate flag
    int64_t H = 1;
    for (int64_t s = 0; s < ps; ++s) H *= pieces;
    const Tensor nan = split.offsets.full_like({B, ps, 1}, kNaN);
    const Tensor padded = Tensor::concat({nan, split.offsets, nan}, 2);
    std::vector<Tensor> Cs, ds, idss, dups;
    for (int64_t h = 0; h < H; ++h) {
        std::vector<double> t(static_cast<std::size_t>(ps), 0.0), t1 = t, side = t;
        for (int64_t s = ps - 1, rest = h; s >= 0; --s, rest /= pieces)
            t[std::size_t(s)] = double(rest % pieces);
        for (int64_t s = 0; s < ps; ++s) {
            t1[std::size_t(s)] = t[std::size_t(s)] + 1;
            side[std::size_t(s)] = t[std::size_t(s)] == 0 ? 1.0 : -1.0;
        }
        // the bounds of the piece: x >= lower and x <= upper (NaN: none)
        const Tensor lower = padded.take_along_axis(Tensor::like(padded, t, {1, ps, 1}), 2);
        const Tensor upper = padded.take_along_axis(Tensor::like(padded, t1, {1, ps, 1}), 2);
        const Tensor zero = split.rows.zeros_like();
        const Tensor lo_row = Tensor::where(lower.isnan(), zero, split.rows.negative());
        const Tensor up_row = Tensor::where(upper.isnan(), zero, split.rows);
        Cs.push_back(Tensor::stack({lo_row, up_row}, 2).reshape({B, 2 * ps, q}));
        const Tensor none = lower.zeros_like();
        ds.push_back(
            Tensor::stack({Tensor::where(lower.isnan(), none, lower.negative()).squeeze(-1),
                           Tensor::where(upper.isnan(), none, upper).squeeze(-1)},
                          2)
                .reshape({B, 2 * ps}));
        // the signed ids of the two-piece splits, and the duplicates of dummy ones
        const Tensor sides = Tensor::like(split.ids, side, {1, ps});
        idss.push_back(split.ids.multiply(sides));
        const Tensor later =
            Tensor::like(split.dummy, t, {1, ps}).greater(split.dummy.zeros_like({1, ps}));
        dups.push_back(ps > 0 ? split.dummy.multiply(later).any(1) : split.dummy.zeros_like({B}));
    }
    const Tensor C_all = Tensor::stack(Cs, 1).reshape({B * H, 2 * ps, q});
    const Tensor d_all = Tensor::stack(ds, 1).reshape({B * H, 2 * ps});
    const Tensor ids_all = Tensor::stack(idss, 1).reshape({B * H, ps});
    // the children kept: all but a dummy split's duplicate (CORA: none of a dummy split's)
    const Tensor dropped = o_.drop_dummy_duplicates
                               ? Tensor::stack(dups, 1)
                               : (ps > 0 ? split.dummy.any(1) : split.dummy.zeros_like({B}))
                                     .expand_dims(1)
                                     .broadcast_to({B, H});
    const std::vector<int64_t> keep = aux_where((1.0 - dropped).reshape({B * H}));
    const std::vector<int64_t> parent = priv_rows_of(keep, H);
    Tensor C = C_all.take(keep, 0), d = d_all.take(keep, 0);
    if (At.defined()) {
        C = Tensor::concat({C, At.take(parent, 0)}, 1);
        d = Tensor::concat({d, bt.take(parent, 0)}, 1);
    }
    // the input and output sets, cases, splits and sensitivities of the children
    const int64_t n_ch = int64_t(parent.size());
    Tensor cx = art.in.c.take(parent, 0), Gx = art.in.G.take(parent, 0);
    const Tensor drx = art.in.dr.defined() ? art.in.dr.take(parent, 0) : Tensor();
    const Tensor cases = ci.take(parent, 0);
    const Tensor splits = aux_cat_splits(si.take(parent, 0), ids_all.take(keep, 0), true);
    std::vector<Tensor> layer_sens;
    if (art.has_sens)
        for (const Tensor &s : art.sens.layer_sum)
            layer_sens.push_back(s.defined() ? s.take(parent, 0) : Tensor());
    Tensor cy = art.out.c.take(parent, 0), Gy = art.out.G.take(parent, 0);
    Tensor dry = art.out.dr.defined() ? art.out.dr.take(parent, 0) : Tensor();
    // the members: (child, polytope) for the polytopes the parent did not exclude
    const std::vector<int64_t> member = aux_where(1.0 - excluded.take(parent, 0));
    Tensor empty = cx.zeros_like({n_ch});
    const bool has_rows = C.shape()[1] > 0;
    for (int64_t it = 1; it < o_.refinement_max_iter; ++it) {
        if (aux_late()) return {};
        std::tuple<Tensor, Tensor, Tensor> bounds;
        {
            const StepTimer timer(stats_, "fourier_motzkin_seconds");
            bounds = aux_bound_children(cy, Gy, dry, C, d, member, n_ch);
        }
        const auto &[bl, bu, child_empty] = bounds;
        if (timed_out_) return {};
        empty = empty.maximum(child_empty);
        const Tensor bc = (bu + bl) * 0.5, br = (bu - bl) * 0.5;
        // the input set and the rows to the new box (aux_scaleAndOffsetZonotope)
        cx = cx + bc.slice(1, 0, q0).expand_dims(1).matmul(Gx).squeeze(1);
        Gx = Gx.multiply(br.slice(1, 0, q0).expand_dims(-1));
        if (has_rows) {
            d = d - C.matmul(bc.expand_dims(-1)).squeeze(-1);
            C = C.multiply(br.expand_dims(1));
            if (it == 1 && o_.project_stale_factors && q > q0) {
                // the repair: the rows keep the input factors, the error factors' share in d
                d = d + C.slice(2, q0, q).abs().sum(2);
                C = Tensor::concat({C.slice(2, 0, q0), C.slice(2, q0, q).zeros_like()}, 2);
            }
        }
        // the refined input set through the network again
        if (it + 1 < o_.refinement_max_iter) {
            const StepTimer timer(stats_, "refine_forward_seconds");
            const ZonoPropagation prop =
                net.forward(cx, drx, Gx, splits, art.has_sens ? &layer_sens : nullptr, false);
            cy = prop.c;
            Gy = prop.G;
            dry = prop.dr;
        }
    }
    // the boxes of the refined input sets; an empty or crossed one is verified
    Tensor rad = Gx.abs().sum(1);
    if (drx.defined()) rad = rad + drx;
    const Tensor lo = cx - rad, hi = cx + rad;
    empty =
        empty.maximum(lo.greater(hi).any(1)).maximum(lo.isnan().any(1)).maximum(hi.isnan().any(1));
    const std::vector<int64_t> live = aux_where(1.0 - empty);
    const int64_t n_empty = n_ch - int64_t(live.size());
    Children out{((hi + lo) * 0.5).take(live, 0), ((hi - lo) * 0.5).take(live, 0),
                 cases.take(live, 0), splits.take(live, 0)};
    return {out, n_empty};
}

std::tuple<Tensor, Tensor, Tensor>
ConZonotopeBounding::aux_bound_children(const Tensor &cy, const Tensor &Gy, const Tensor &dry,
                                        const Tensor &C, const Tensor &d,
                                        const std::vector<int64_t> &member, int64_t children) {
    const int64_t q = net_->q, rows = K_ + C.shape()[1], M = int64_t(member.size());
    const FmOrder order =
        o_.fourier_motzkin_order == "coralean" ? FmOrder::coralean : FmOrder::cora;
    // the members in chunks that the memory budget holds
    const int64_t chunk = std::max<int64_t>(1, problem_->memory_budget / (8 * 8 * rows * q));
    std::vector<Tensor> lows, ups;
    for (int64_t s = 0; s < M; s += chunk) {
        if (aux_late()) return {};
        std::vector<int64_t> mc, mp;
        for (int64_t i = s; i < std::min(M, s + chunk); ++i) {
            mc.push_back(member[std::size_t(i)] / P_);
            mp.push_back(member[std::size_t(i)] % P_);
        }
        // the unsafe outputs A y <= b of the polytope as rows on the factors (Prop. 2)
        const Tensor Ap = A_.take(mp, 0), cym = cy.take(mc, 0);
        const Tensor Cu = Ap.matmul(Gy.take(mc, 0).matrix_transpose());
        Tensor du =
            b_.take(mp, 0) - Ap.matmul(cym.expand_dims(-1)).squeeze(-1) + o_.verified_margin;
        if (dry.defined())
            du = du + abs_a_.take(mp, 0).matmul(dry.take(mc, 0).expand_dims(-1)).squeeze(-1);
        du = Tensor::where(real_row_.take(mp, 0), du, du.full_like(du.shape(), kInf));
        const Tensor Am = Tensor::concat({Cu, C.take(mc, 0)}, 1);
        const Tensor bm = Tensor::concat({du, d.take(mc, 0)}, 1);
        const int64_t m = int64_t(mc.size());
        auto [bl, bu] =
            priv_fourier_motzkin(Am, bm, Am.full_like({m, q}, -1.0), Am.full_like({m, q}, 1.0),
                                 o_.polytope_bound_approx_max_iter, order, 0.0);
        const Tensor crossed = bl.greater(bu).any(1, true);
        lows.push_back(Tensor::where(crossed, bl.full_like(bl.shape(), kInf), bl));
        ups.push_back(Tensor::where(crossed, bu.full_like(bu.shape(), -kInf), bu));
    }
    // the hull of each child's members, an absent member as the empty box
    const Tensor &like = cy;
    lows.push_back(like.full_like({1, q}, kInf));
    ups.push_back(like.full_like({1, q}, -kInf));
    std::vector<int64_t> slot(std::size_t(children * P_), M);
    for (int64_t i = 0; i < M; ++i) slot[std::size_t(member[std::size_t(i)])] = i;
    const Tensor bl = Tensor::concat(lows, 0).take(slot, 0).reshape({children, P_, q}).min(1);
    const Tensor bu = Tensor::concat(ups, 0).take(slot, 0).reshape({children, P_, q}).max(1);
    const Tensor empty = bl.greater(bu).any(1);
    const Tensor hit = empty.expand_dims(-1);
    return {Tensor::where(hit, bl.zeros_like(), bl), Tensor::where(hit, bu.zeros_like(), bu),
            empty};
}

// The naive refinement ----------------------------------------------------------------------------

Children ConZonotopeBounding::aux_naive_children(const Tensor &xi, const Tensor &ri,
                                                 const Tensor &ci, const Tensor &si,
                                                 const Artifacts &art) const {
    // CORA's 'naive': the boxes split along num_dimensions dimensions into num_splits pieces each
    Tensor x = xi, r = ri, cases = ci;
    Tensor sens = art.has_sens ? art.sens.input : Tensor(), grad;
    if (art.input_grad.defined())
        grad = aux_onehot(art.in.dims, net_->n0).multiply(art.input_grad.expand_dims(-1)).sum(1);
    const int64_t k = o_.num_splits, n0 = net_->n0;
    for (int64_t step = 0; step < std::min(o_.num_dimensions, n0); ++step) {
        const HeuristicInputs in{x - r, x + r, r, sens, grad};
        const Tensor dim = priv_compute_heuristic(o_.input_split_heuristic, in, Tensor(), 0, false)
                               .argmax(1, true);
        const Tensor hot = aux_onehot(dim, n0).squeeze(1);
        const Tensor base = x - r, width = r / double(k);
        std::vector<Tensor> xs, rs;
        for (int64_t p = 0; p < k; ++p) {
            xs.push_back(Tensor::where(hot, base + width * double(2 * p + 1), x));
            rs.push_back(Tensor::where(hot, width, r));
        }
        // the pieces of each box next to each other
        const int64_t N = x.shape()[0];
        std::vector<int64_t> each;
        for (int64_t i = 0; i < N; ++i)
            for (int64_t p = 0; p < k; ++p) each.push_back(i);
        x = Tensor::stack(xs, 1).reshape({N * k, n0});
        r = Tensor::stack(rs, 1).reshape({N * k, n0});
        cases = cases.take(each, 0);
        if (sens.defined()) sens = sens.take(each, 0);
        if (grad.defined()) grad = grad.take(each, 0);
    }
    const int64_t n = x.shape()[0] / std::max<int64_t>(1, xi.shape()[0]);
    std::vector<int64_t> each;
    for (int64_t i = 0; i < xi.shape()[0]; ++i)
        for (int64_t p = 0; p < n; ++p) each.push_back(i);
    return {x, r, cases, si.take(each, 0)};
}

// The attack --------------------------------------------------------------------------------------

bool ConZonotopeBounding::zonotack_attack(NnProblem &problem, AttackResult &found) {
    prepare(problem);
    found_ = &found;
    const Tensor x = (problem.outer_lo + problem.outer_hi) * 0.5;
    const Tensor r = (problem.outer_hi - problem.outer_lo) * 0.5;
    const Tensor cases = problem.box_case;
    const ZonoInput in = net_->input_zonotope(x, r);
    const ZonoPropagation out = net_->forward(in.c, in.dr, in.G, Tensor(), nullptr, false);
    const auto [cand, ccase] =
        aux_zonotack_candidates(in.c, in.G, out, cases, aux_excluded(out, cases));
    found.iterations = 1;
    found.evaluations = cand.shape()[0];
    return priv_check_points(problem, cand, ccase, found);
}

/// CORA's falsification_method 'zonotack' as an attack: the unrefined candidates, one round.
class ZonotackAttack : public Attack {
  public:
    explicit ZonotackAttack(const NnVerifyOptions &o) : o_(o) {}
    AttackResult run(NnProblem &problem, double, int64_t) override {
        AttackResult result;
        ConZonotopeBounding(o_).zonotack_attack(problem, result);
        return result;
    }

  private:
    NnVerifyOptions o_;
};

} // namespace


// ===========================================  MAIN  =========================================== //

void priv_register_zonotack() {
    register_bounding("conzonotope", [](const NnVerifyOptions &o) {
        return std::make_unique<ConZonotopeBounding>(o);
    });
    register_bounding("zonotope-alpha", [](const NnVerifyOptions &o) {
        return std::make_unique<ConZonotopeBounding>(o, true);
    });
    register_attack("zonotack",
                    [](const NnVerifyOptions &o) { return std::make_unique<ZonotackAttack>(o); });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
