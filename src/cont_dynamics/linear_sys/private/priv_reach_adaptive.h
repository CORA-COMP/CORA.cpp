// priv_reach_adaptive - the types of the adaptive reachability algorithm of linear_sys
//
// The algorithm of CORA's priv_reach_adaptive [1] computes outer approximations of the reachable
// output sets whose Hausdorff distance to the exact sets stays below a given error, choosing the
// time step size of every step itself. Here: the system in canonical form x' = A x + u with u in
// u_trans + U (U centered at the origin, given by its generators), y = C x, and what the
// verification loop hands from one run to the next.
//
// Syntax:   AdaptiveResult R = priv_reach_adaptive(taylor, params, spec_times, error, verify, sd);
// Inputs:   see the declaration below
// Outputs:  the output sets per step with their errors, the step sizes
// References:
//    [1] M. Wetzlinger et al. "Fully automated verification of linear systems using inner- and
//        outer-approximations of reachable sets", TAC, 2023.
// See also: priv_verify_ra_zonotope

#pragma once

#include "cont_dynamics/linear_sys/private/priv_taylor_lin_sys.h"
#include "cont_dynamics/linear_sys/private/priv_verify_time.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The model parameters in canonical form.
struct AdaptiveParams {
    Zonotope R0;                ///< initial set
    Tensor GU;                  ///< generators of the input set U (n, mU), centered; mU may be 0
    Tensor u_trans;             ///< constant input offset (n, 1)
    std::optional<Tensor> C;    ///< output matrix (p, n); empty: y = x
    double t_final;             ///< time horizon (from time 0)
};

/// What a run leaves for the next one: the step sizes used and what was computed for them.
/// Vectors over the used step sizes; entries of quantities that were not computed are empty.
struct AdaptiveSaveData {
    std::vector<double> time_step;
    std::vector<char> fullcomp;
    std::vector<double> e_g;
    std::vector<std::optional<Tensor>> GuTrans_center, GuTrans_Gbox, Pu;
    std::vector<std::optional<Tensor>> G_PU_zero, G_PU_infty, PU_A_sum_error;
    std::optional<double> reductionerrorpercentage;
};

/// The output sets per step: time_interval[k] over [time[k], time[k+1]] and time_point[k] at
/// time[k]; a step where every specification is verified has no sets (error NaN), and time_point[0]
/// is the output set of the initial set.
struct AdaptiveResult {
    std::vector<std::optional<Zonotope>> time_interval, time_point;
    std::vector<double> time_interval_error, time_point_error;
    std::vector<double> time;       ///< the time points 0, t_1, ..., t_K
    bool res = true;                ///< false: a specification is violated (not used: always true)
    bool internal_checks_ok = true; ///< the error bookkeeping was consistent
};

/// The reachable output sets with errors below `error` (Hausdorff distance to the exact
/// output sets). With verify, the time horizon ends at the last time where some specification
/// is unverified (spec_times, one VerifyTime per specification) and steps where all are verified
/// are skipped (no sets). savedata is read at the start and extended.
AdaptiveResult priv_reach_adaptive(TaylorLinSys &taylor, const AdaptiveParams &params,
                                   const std::vector<VerifyTime> &spec_times, double error,
                                   bool verify, AdaptiveSaveData &savedata);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
