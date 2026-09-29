// Reachability of the linear system `x' = A x`, as in CORA's `linearSys`.
//
// Written against CoraTensor only, so it runs on whichever backend `A` and the initial set
// `X0` come from, and batches wherever that backend does: libtorch broadcasts the leading
// dimensions of `A` `(..., n, n)` and of the set, so one call covers a batch of sets, of
// systems, or both.

#pragma once

#include "contSet/zonotope/zonotope.h"

#include <vector>

namespace cora::ct {

enum class Algorithm {
    /// Every step propagates the time-point set and encloses it again.
    Standard,
    /// One step's enclosure is computed once and mapped forward with `e^{A k Δt}`.
    WrappingFree,
};

/// The reachable set per step: `time_int[k]` covers `[kΔt, (k+1)Δt]`, `time_point[k]` is
/// the set at `kΔt`.
struct Reach {
    std::vector<Zonotope> time_int, time_point;
};

class LinearSys {
  public:
    explicit LinearSys(Tensor A) : A_(std::move(A)) {}

    /// The reachable sets for `ceil(t_final / time_step)` steps from `X0`.
    Reach reach(const Zonotope &X0, double time_step, double t_final, int taylor_terms,
                Algorithm algorithm = Algorithm::Standard) const;

    /// `F(A, Δt, η)`: the interval matrix whose product with a time-point set encloses the
    /// curvature of the trajectories between two time points — the Taylor terms `i >= 2`,
    /// each weighted by where `t^i - t` is extremal on `[0, Δt]`, plus the series' remainder
    /// past order `η`.
    Interval correction_matrix_state(double time_step, int taylor_terms) const;

  private:
    Tensor A_;
};

} // namespace cora::ct
