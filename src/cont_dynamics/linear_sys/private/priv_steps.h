// priv_steps - the building blocks of one step of reach on a partitioned state
//
// The solutions of a step, as CORA's homogeneousSolution, particularSolution_timeVarying,
// particularSolution_constant and oneStep compute them: per block of the state, the public
// operations of LinearSys call them with the one block that spans the state.
//
// Syntax:   BlockStep s = priv_one_step(sys, taylor, blocks, X, U, u, time_step, taylor_terms);
// Inputs:   taylor - the powers of A and e^{A dt}
//           blocks - the partition of the state
//           X - set of the state space
//           U - input set of the state space
//           u - the constant input as a point (n, 1)
// Outputs:  the sets of one step, one list entry per block
// See also: priv_block.h, linear_sys.h
//
// CORA.cpp only: CORA calls these through getTaylor and the block_* helpers.
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/linear_sys/private/priv_block.h"
#include "cont_dynamics/linear_sys/private/priv_taylor_lin_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A list of zonotopes, one per block.
using BlockSets = std::vector<Zonotope>;

struct BlockHomogeneous {
    BlockSets time_point, state_curvature;
};

struct BlockConstant {
    BlockSets time_point, input_curvature, time_interval;
};

struct BlockStep {
    BlockSets time_point, time_interval, homogeneous_time_point, homogeneous_time_interval,
        time_varying, constant, state_curvature, input_curvature;
};

/// The options of a step operation that init_reach and post share, from the algorithm.
struct StepAlgorithm {
    double time_step;
    int taylor_terms, zonotope_order;
    bool wrapping_free;
};

/// The step options of "standard" or "wrapping_free"; the others (their steps are not those of
/// init_reach and post) are refused with Error::not_supported, naming the operation.
StepAlgorithm priv_step_algorithm(const char *op, const alg::LinearReach &alg);

/// dt E, the remainder of the series of e^{A dt} that enters the input solutions.
Interval priv_remainder(const LinearSys &sys, double time_step, int taylor_terms);

/// The series sum_{eta=0}^{order} A^eta dt^(eta+1) / (eta+1)! as one matrix.
Tensor priv_series(const LinearSys &sys, TaylorLinSys &taylor, double time_step, int taylor_terms);

/// e^{A dt} X and F X for the blocks Xb of a set.
BlockHomogeneous priv_homogeneous(const LinearSys &sys, TaylorLinSys &taylor, const Blocks &blocks,
                                  const BlockSets &Xb, double time_step, int taylor_terms);

/// The enclosure of the step: enclose(Xb, Htp) + state_curvature.
BlockSets priv_homogeneous_interval(const BlockSets &Xb, const BlockSets &Htp,
                                    const BlockSets &state_curvature);

/// The solution at the end of the step of the input u(t) in U with the terms of A kept apart.
BlockSets priv_particular_time_varying(const LinearSys &sys, TaylorLinSys &taylor,
                                       const Blocks &blocks, const Zonotope &U, double time_step,
                                       int taylor_terms);

/// The solution of the constant input in U (a point for one input) and its curvature.
BlockConstant priv_particular_constant(const LinearSys &sys, TaylorLinSys &taylor,
                                       const Blocks &blocks, const Zonotope &U, double time_step,
                                       int taylor_terms);

/// One step from X with the input set U and the constant input u.
BlockStep priv_one_step(const LinearSys &sys, TaylorLinSys &taylor, const Blocks &blocks,
                        const Zonotope &X, const Zonotope &U, const Zonotope &u, double time_step,
                        int taylor_terms);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
