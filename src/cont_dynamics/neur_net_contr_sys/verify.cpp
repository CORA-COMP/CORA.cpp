// verify - verifies a specification of a neural network controlled system, as CORA's verify
//
// First random runs are simulated; if one violates the specification, the reach is computed from
// its start point alone (without the specification) and the run is FALSIFIED when that reach does
// not verify the specification either, as in CORA. Otherwise the reach from the whole initial set
// stops at the first violating step; VERIFIED if it completes and the specification holds on its
// sets, else the initial set is split recursively (splitR0): the box of the set is cut in the
// middle of the dimension where the radius times the mean absolute sensitivity of the network at
// the center is largest, and every piece is reached on its own. The verdict is VERIFIED when
// every leaf verifies, UNKNOWN when a piece fails after depth - 1 splits. CORA runs the levels
// one after the other; the pieces here are a tree on a pool of threads that stops at the first
// piece that cannot be split further, which gives the same verdict, as the split of a piece
// does not depend on the others.
//
// Syntax:   v = sys.verify(X0, U, t_final, options, spec, points, rng, {.depth = 10});
// Inputs:   X0 - the initial set (Zonotope or PolyZonotope), U - the external input set (optional)
//           spec - a timed specification or a list of them, points - the number of random runs
//           split - the depth of the splitting and the number of threads
// Outputs:  v - the verdict, the start of a violating run, the number of runs and of pieces
// See also: reach, simulate_random
//
// CORA: contDynamics/@neurNetContrSys/verify.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/neur_net_contr_sys/private/priv_holds.h"
#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"
#include "cora/global/threads.h"
#include "global/format.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The first run that violates a specification at a sampled time: its index (-1 if none) and
/// the sample.
std::pair<int64_t, std::size_t> aux_violating_run(const TimedSpecifications &specs,
                                                  const ClosedLoopTrajectories &traj) {
    for (std::size_t k = 0; k < traj.x.size(); ++k) {
        const Tensor &x = traj.x[k];
        for (const TimedSpecification &spec : specs) {
            if (!priv_in_window(spec, traj.time[k], traj.time[k])) continue;
            for (int64_t j = 0; j < x.shape()[1]; ++j)
                if (!spec.spec.check(
                        Zonotope(x.slice(1, j, j + 1), x.zeros_like({x.shape()[0], 0}))))
                    return {j, k};
        }
    }
    return {-1, 0};
}

// The sets of a run -------------------------------------------------------------------------------

/// The enclosing zonotope of a set.
Zonotope aux_zonotope(const Zonotope &Z) { return Z; }
Zonotope aux_zonotope(const PolyZonotope &P) { return P.zonotope(); }

/// The box of a set, and the set of a box of the same kind.
Interval aux_box(const Zonotope &Z) { return Z.interval(); }
Interval aux_box(const PolyZonotope &P) { return P.interval(); }

Zonotope aux_set(const Zonotope &, const Interval &box) {
    const Tensor c = (box.inf() + box.sup()) * 0.5;
    return {c, box.rad().diag()};
}
PolyZonotope aux_set(const PolyZonotope &, const Interval &box) { return PolyZonotope(box); }

/// Whether a run of the reach completed and its sets satisfy the specifications.
template <class S>
bool aux_verified(const TimedSpecifications &specs, const ClosedLoopReach<S> &R) {
    return R.completed && priv_holds_reach(specs, R, [](const S &s) { return aux_zonotope(s); });
}

// The splitting ----------------------------------------------------------------------------------

/// The two halves of the box cut at the middle of dimension n.
std::pair<Interval, Interval> aux_halves(const Interval &box, int64_t n) {
    const Tensor mid = (box.inf().slice(0, n, n + 1) + box.sup().slice(0, n, n + 1)) * 0.5;
    const auto cut = [&](const Tensor &bound, const Tensor &value) {
        // the bound with row n replaced
        return Tensor::concat(
            {bound.slice(0, 0, n), value, bound.slice(0, n + 1, bound.shape()[0])}, 0);
    };
    return {Interval(box.inf(), cut(box.sup(), mid)), Interval(cut(box.inf(), mid), box.sup())};
}

/// The dimension in which a piece is cut: CORA's, or the one whose radius is largest relative to
/// the initial set's.
template <class S>
int64_t aux_cut(const NeurNetContrSys &sys, const S &X0, const Interval &box,
                const VerifySplit &split) {
    if (split.rule == VerifySplit::Rule::sensitivity) return sys.split_dimension(box);
    const std::vector<double> rad = box.rad().data(), rad0 = aux_box(X0).rad().data();
    int64_t best = 0;
    double best_ratio = -1;
    for (std::size_t i = 0; i < rad.size(); ++i) {
        const double ratio = rad0[i] > 0 ? rad[i] / rad0[i] : 0.0;
        if (ratio > best_ratio) best_ratio = ratio, best = int64_t(i);
    }
    return best;
}

/// A queue of pieces, taken by a pool of threads; a piece can add its two halves.
struct Tree {
    struct Piece {
        Interval box;
        int splits;
    };
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<Piece> queue;
    std::size_t running = 0;
    bool stop = false;
    std::exception_ptr failure = nullptr;
    int reached = 0, verified = 0, max_splits = 0;
    bool failed_at_depth = false;
};

/// The pieces of the tree: reached on `threads` threads, each failing piece split until depth - 1.
template <class S>
void aux_run_tree(const NeurNetContrSys &sys, const S &X0, const std::optional<Zonotope> &U,
                  double t_final, const ClosedLoopOptions &options,
                  const TimedSpecifications &specs, const VerifySplit &split, Tree &tree) {
    const auto work = [&] {
        for (;;) {
            Tree::Piece piece{aux_box(X0), 0};
            {
                std::unique_lock lock(tree.mutex);
                tree.wake.wait(
                    lock, [&] { return tree.stop || !tree.queue.empty() || tree.running == 0; });
                if (tree.stop || (tree.queue.empty() && tree.running == 0)) return;
                // the deepest piece first, as the tree is searched depth first
                piece = tree.queue.back();
                tree.queue.pop_back();
                ++tree.running;
            }
            try {
                const ClosedLoopReach<S> R =
                    sys.reach(aux_set(X0, piece.box), U, t_final, options, specs);
                const bool ok = aux_verified(specs, R);
                std::unique_lock lock(tree.mutex);
                --tree.running;
                ++tree.reached;
                tree.max_splits = std::max(tree.max_splits, piece.splits);
                if (ok) {
                    ++tree.verified;
                } else if (piece.splits >= split.depth - 1) {
                    // no split left: the verdict is UNKNOWN, the others need not finish
                    tree.failed_at_depth = true;
                    tree.stop = true;
                } else {
                    const auto [lo, hi] = aux_halves(piece.box, aux_cut(sys, X0, piece.box, split));
                    tree.queue.push_back({hi, piece.splits + 1});
                    tree.queue.push_back({lo, piece.splits + 1});
                }
                tree.wake.notify_all();
            } catch (...) {
                std::unique_lock lock(tree.mutex);
                --tree.running;
                if (!tree.failure) tree.failure = std::current_exception();
                tree.stop = true;
                tree.wake.notify_all();
            }
        }
    };
    // CORA does not reach the whole set again at level 1: it is split at once
    const auto [lo, hi] = aux_halves(aux_box(X0), aux_cut(sys, X0, aux_box(X0), split));
    tree.queue = {{hi, 1}, {lo, 1}};
    const int threads = std::max(1, split.max_threads);
    // the pieces are the parallelism: the pools inside an operation stay at one thread
    const int previous = max_threads();
    if (threads > 1) set_num_threads(1);
    std::vector<std::thread> pool;
    pool.reserve(std::size_t(threads));
    for (int t = 0; t < threads; ++t) pool.emplace_back(work);
    for (std::thread &th : pool) th.join();
    if (threads > 1) set_num_threads(previous);
    if (tree.failure) std::rethrow_exception(tree.failure);
}

// The verdict -------------------------------------------------------------------------------------

template <class S>
ClosedLoopVerdict aux_verify(const NeurNetContrSys &sys, const S &X0, const Zonotope &sim_set,
                             const std::optional<Zonotope> &U, double t_final,
                             const ClosedLoopOptions &options, const TimedSpecifications &specs,
                             int64_t points, Rng &rng, const VerifySplit &split) {
    ClosedLoopVerdict out;
    out.simulations = static_cast<int>(points);
    // random runs, sampled at the steps of the reach
    const ClosedLoopTrajectories traj = sys.simulate_random(
        sim_set, U, t_final, std::min(sys.dt(), options.plant.time_step), points, 0.5, rng);
    const auto [violating, sample] = aux_violating_run(specs, traj);
    if (violating >= 0) {
        out.violation_time = traj.time[sample];
        out.violating_state = traj.x[sample].slice(1, violating, violating + 1);
        // the reach from the start of the violating run, without the specification
        const Tensor start = traj.x[0].slice(1, violating, violating + 1);
        out.violating_start = start;
        const PolyZonotope point{Zonotope(start, start.zeros_like({sys.dim(), 0}))};
        const ClosedLoopReach<PolyZonotope> R = sys.reach(point, U, t_final, options);
        out.verdict = aux_verified(specs, R) ? Verdict::verified : Verdict::falsified;
        return out;
    }
    // the whole initial set, stopping at a violation
    const ClosedLoopReach<S> R = sys.reach(X0, U, t_final, options, specs);
    out.pieces_reached = 1;
    out.time_reached = R.time.back();
    out.exploded = R.exploded;
    out.violated = R.violated;
    if (aux_verified(specs, R)) {
        out.verdict = Verdict::verified;
        out.pieces_verified = 1;
        return out;
    }
    // split recursively (CORA's aux_VerifyRecursively)
    if (split.depth > 1) {
        Tree tree;
        aux_run_tree(sys, X0, U, t_final, options, specs, split, tree);
        out.pieces_reached += tree.reached;
        out.pieces_verified = tree.verified;
        out.max_splits = tree.max_splits;
        out.verdict = tree.stop ? Verdict::unknown : Verdict::verified;
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The pieces --------------------------------------------------------------------------------------

int64_t NeurNetContrSys::split_dimension(const Interval &box) const {
    detail::Entry entry;
    // the box must be a column of the system's dimension
    CORA_CHECK(box.inf().shape() == std::vector<int64_t>({dim(), 1}), Error::dimension_mismatch,
               [&] {
                   return std::string(
                       "split_dimension: the box must be a column of the system's dimension");
               });
    // the radius of each dimension times the mean absolute sensitivity of the outputs to it
    const Tensor center = (box.inf() + box.sup()) * 0.5;
    const Tensor S = nn_.sensitivity(center);
    const Tensor weight = S.abs().mean(0, true).matrix_transpose();
    const std::vector<double> score = box.rad().multiply(weight).data();
    return std::max_element(score.begin(), score.end()) - score.begin();
}

std::vector<Interval> NeurNetContrSys::split_pieces(const Interval &box, int levels) const {
    detail::Entry entry;
    // every level cuts each piece in two
    std::vector<Interval> pieces{box};
    for (int level = 0; level < levels; ++level) {
        std::vector<Interval> next;
        for (const Interval &piece : pieces) {
            const auto [lo, hi] = aux_halves(piece, split_dimension(piece));
            next.push_back(lo);
            next.push_back(hi);
        }
        pieces = std::move(next);
    }
    return pieces;
}

// The verdicts ------------------------------------------------------------------------------------

ClosedLoopVerdict NeurNetContrSys::verify(const PolyZonotope &X0, const std::optional<Zonotope> &U,
                                          double t_final, const ClosedLoopOptions &options,
                                          const TimedSpecifications &specs, int64_t points,
                                          Rng &rng, const VerifySplit &split) const {
    detail::Entry entry;
    CORA_CHECK(points >= 1, Error::wrong_value,
               [] { return std::string("verify: needs at least one simulated run"); });
    return aux_verify(*this, X0, X0.zonotope(), U, t_final, options, specs, points, rng, split);
}

ClosedLoopVerdict NeurNetContrSys::verify(const Zonotope &X0, const std::optional<Zonotope> &U,
                                          double t_final, const ClosedLoopOptions &options,
                                          const TimedSpecifications &specs, int64_t points,
                                          Rng &rng, const VerifySplit &split) const {
    detail::Entry entry;
    // the same for a zonotope, which is its own set of start points
    CORA_CHECK(points >= 1, Error::wrong_value,
               [] { return std::string("verify: needs at least one simulated run"); });
    return aux_verify(*this, X0, X0, U, t_final, options, specs, points, rng, split);
}

ClosedLoopVerdict NeurNetContrSys::verify(const PolyZonotope &X0, const std::optional<Zonotope> &U,
                                          double t_final, const ClosedLoopOptions &options,
                                          const TimedSpecification &spec, int64_t points, Rng &rng,
                                          const VerifySplit &split) const {
    detail::Entry entry;
    // one specification is a list of one
    return verify(X0, U, t_final, options, TimedSpecifications{spec}, points, rng, split);
}

ClosedLoopVerdict NeurNetContrSys::verify(const Zonotope &X0, const std::optional<Zonotope> &U,
                                          double t_final, const ClosedLoopOptions &options,
                                          const TimedSpecification &spec, int64_t points, Rng &rng,
                                          const VerifySplit &split) const {
    detail::Entry entry;
    // one specification is a list of one
    return verify(X0, U, t_final, options, TimedSpecifications{spec}, points, rng, split);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
