// priv_reach_from_start - the algorithms "from_start" and "decomposition" of linear_sys.reach
//
// The first step is solved once (one_step): the sets Htp_0, Hti_0 of the homogeneous solution and
// the input solutions PU_0, Pu_0. Step k maps them forward by Q = e^{A dt}^(k-1), so the error of
// the sets does not accumulate over the steps:
//    Htp = Q Htp_0,  Hti = Q Hti_0,  PU <- reduce(PU + Q PU_0 + P Pu_0),  P = e^{A dt}^(k-2)
//    time_interval_k = Hti + PU + C_input,  time_point_k = Htp + PU
// With a partition of the state each set is a list of blocks, mapped block by block.
//
// CORA drops C_input after the first step and PU (with the constant input) when U is the origin;
// both under-approximate (docs/cora-differences.md). Here C_input is added at every step and PU
// accumulates whenever there is an input set.
//
// Syntax:   ReachSet R = priv_reach_from_start(sys, X0, U, dt, eta, order, blocks, steps);
// Inputs:   sys - system
//           X0 - initial set
//           U - input set or none
//           dt - step size
//           eta - Taylor order
//           order - zonotope order of the input solution
//           blocks - partition of the state
//           steps - number of steps
// Outputs:  R - the reachable sets, see LinearSys::reach
// See also: priv_reach_standard, one_step
//
// CORA: contDynamics/@linearSys/private/priv_reach_fromStart.m
// CORA: contDynamics/@linearSys/private/priv_reach_decomp.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cont_dynamics/linear_sys/private/priv_steps.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The blocks of PU reduced to `order`, as one set so that the blocks keep their coupling.
BlockSets aux_reduce(const BlockSets &PU, const Blocks &blocks, int order) {
    if (PU.size() == 1) return {PU[0].reduce(order)};
    return priv_decompose(priv_recompose(PU).reduce(order), blocks);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachSet priv_reach_from_start(const LinearSys &sys, const Zonotope &X0,
                               const std::optional<Zonotope> &input_set, double time_step,
                               int taylor_terms, int zonotope_order, const Blocks &blocks,
                               int steps) {
    // the constant input comes from the center of the input set, as in canonical_form
    const Zonotope origin{X0.center().zeros_like(), X0.center().zeros_like({sys.dim(), 0})};
    const bool has_input = input_set.has_value();
    Zonotope U = origin, u = origin;
    if (has_input) {
        const CanonicalForm cf = sys.canonical_form(*input_set);
        U = cf.U;
        u = Zonotope(cf.u, cf.u.zeros_like({sys.dim(), 0}));
    }
    TaylorLinSys taylor(sys.A());
    const Zonotope X = X0.linear_map(sys.A().eye_like()); // the identity broadcasts a batch of sets
    if (steps < 1) return {{}, {X}};
    BlockStep first = priv_one_step(sys, taylor, blocks, X, U, u, time_step, taylor_terms);

    ReachSet R;
    R.time_point.push_back(X);
    R.time_interval.push_back(priv_recompose(first.time_interval));
    R.time_point.push_back(priv_recompose(first.time_point));
    const Tensor e_adt = taylor.e_adt(time_step);
    Tensor P = sys.A().eye_like(), Q = e_adt;
    BlockSets PU = first.time_varying;
    for (int k = 2; k <= steps; ++k) {
        BlockSets Hti = priv_block_mtimes(Q, blocks, first.homogeneous_time_interval);
        BlockSets Htp = priv_block_mtimes(Q, blocks, first.homogeneous_time_point);
        if (has_input) {
            // the time-varying part of the new step and the constant input of the one before
            PU = priv_block_plus(PU, priv_block_mtimes(Q, blocks, first.time_varying));
            PU = priv_block_plus(PU, priv_block_mtimes(P, blocks, first.constant));
            PU = aux_reduce(PU, blocks, zonotope_order);
            Hti = priv_block_plus(Hti, priv_block_plus(PU, first.input_curvature));
            Htp = priv_block_plus(Htp, PU);
        }
        R.time_interval.push_back(priv_recompose(Hti));
        R.time_point.push_back(priv_recompose(Htp));
        P = Q;
        Q = Q.matmul(e_adt);
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
