// priv_steps - the building blocks of one step of reach on a partitioned state
//
// See also: priv_steps.h
//
// CORA.cpp only: CORA calls these through getTaylor and the block_* helpers.
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// dt E, the remainder of the series of e^{A dt} that enters the input solutions.
Interval aux_remainder(const LinearSys &sys, double time_step, int taylor_terms) {
    return time_step * sys.expm_remainder(time_step, taylor_terms);
}

/// The series sum_{eta=0}^{order} A^eta dt^(eta+1) / (eta+1)! as one matrix.
Tensor aux_series(const LinearSys &sys, TaylorLinSys &taylor, double time_step, int taylor_terms) {
    Tensor sum = sys.A().eye_like() * time_step;
    for (int eta = 1; eta <= taylor_terms; ++eta)
        sum = sum + taylor.Apower(eta) * taylor.dtoverfac(time_step, eta + 1);
    return sum;
}

/// conv(0, Ptp) for every block.
BlockSets aux_hull_with_origin(const BlockSets &Ptp) {
    BlockSets out;
    for (const Zonotope &P : Ptp) out.push_back(P.convex_hull(P.center().zeros_like()));
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Options -----------------------------------------------------------------------------------------

StepAlgorithm priv_step_algorithm(const char *op, const alg::LinearReach &alg) {
    if (const auto *a = std::get_if<alg::Standard>(&alg))
        return {a->time_step, a->taylor_terms, a->zonotope_order, false};
    // CORA reduces by options.zonotopeOrder in the wrapping-free algorithm too
    if (const auto *a = std::get_if<alg::WrappingFree>(&alg))
        return {a->time_step, a->taylor_terms, alg::Standard{}.zonotope_order, true};
    CORA_FAIL(Error::not_supported, std::string(op) +
                                        ": it serves the algorithms standard and "
                                        "wrapping_free, not " +
                                        alg::name_of(alg));
}

// Homogeneous solution ----------------------------------------------------------------------------

BlockHomogeneous priv_homogeneous(const LinearSys &sys, TaylorLinSys &taylor, const Blocks &blocks,
                                  const BlockSets &Xb, double time_step, int taylor_terms) {
    const Interval F = sys.correction_matrix_state(time_step, taylor_terms);
    return {priv_block_mtimes(taylor.e_adt(time_step), blocks, Xb),
            priv_block_mtimes(F, blocks, Xb)};
}

BlockSets priv_homogeneous_interval(const BlockSets &Xb, const BlockSets &Htp,
                                    const BlockSets &state_curvature) {
    return priv_block_plus(priv_block_enclose(Xb, Htp), state_curvature);
}

// Input solutions ---------------------------------------------------------------------------------

// the terms of the series stay apart, each with its own generators
BlockSets priv_particular_time_varying(const LinearSys &sys, TaylorLinSys &taylor,
                                       const Blocks &blocks, const Zonotope &U, double time_step,
                                       int taylor_terms) {
    // a batch has no single answer to representsa; its zero members give zero sets anyway
    if (U.center().ndim() == 2 && U.representsa("origin"))
        return priv_block_origin(U.center(), blocks);
    const BlockSets Ub = priv_decompose(U, blocks);
    // dt I U: a batch of systems batches the first term as well, like all the others
    BlockSets P = priv_block_mtimes(sys.A().eye_like() * time_step, blocks, Ub);
    // every term A^eta dt^(eta+1)/(eta+1)! is kept as generators of its own
    for (int eta = 1; eta <= taylor_terms; ++eta) {
        const Tensor term = taylor.Apower(eta) * taylor.dtoverfac(time_step, eta + 1);
        P = priv_block_plus(P, priv_block_mtimes(term, blocks, Ub));
    }
    const Interval E = aux_remainder(sys, time_step, taylor_terms);
    return priv_block_plus(P, priv_block_mtimes(E, blocks, Ub));
}

BlockConstant priv_particular_constant(const LinearSys &sys, TaylorLinSys &taylor,
                                       const Blocks &blocks, const Zonotope &U, double time_step,
                                       int taylor_terms) {
    if (U.center().ndim() == 2 && U.representsa("origin")) {
        const BlockSets zero = priv_block_origin(U.center(), blocks);
        return {zero, zero, zero};
    }
    const BlockSets Ub = priv_decompose(U, blocks);
    BlockSets Ptp;
    if (const std::optional<Tensor> Ainv = taylor.Ainv()) {
        // an invertible A integrates in closed form: A^-1 (e^{A dt} - I) U
        const Tensor M = Ainv->matmul(taylor.e_adt(time_step) - sys.A().eye_like());
        Ptp = priv_block_mtimes(M, blocks, Ub);
    } else {
        const Tensor Asum = aux_series(sys, taylor, time_step, taylor_terms);
        const Interval E = aux_remainder(sys, time_step, taylor_terms);
        Ptp =
            priv_block_plus(priv_block_mtimes(Asum, blocks, Ub), priv_block_mtimes(E, blocks, Ub));
    }
    const Interval G = sys.correction_matrix_input(time_step, taylor_terms);
    BlockSets C_input = priv_block_mtimes(G, blocks, Ub);
    BlockSets Pti = priv_block_plus(aux_hull_with_origin(Ptp), C_input);
    return {std::move(Ptp), std::move(C_input), std::move(Pti)};
}

// One step ----------------------------------------------------------------------------------------

// the solutions of the input and of the state, then the sets of the step
BlockStep priv_one_step(const LinearSys &sys, TaylorLinSys &taylor, const Blocks &blocks,
                        const Zonotope &X, const Zonotope &U, const Zonotope &u, double time_step,
                        int taylor_terms) {
    const BlockSets PU =
        priv_particular_time_varying(sys, taylor, blocks, U, time_step, taylor_terms);
    BlockConstant Pu = priv_particular_constant(sys, taylor, blocks, u, time_step, taylor_terms);
    const BlockSets Xb = priv_decompose(X, blocks);
    BlockHomogeneous H = priv_homogeneous(sys, taylor, blocks, Xb, time_step, taylor_terms);
    // the constant input shifts the end of the step, then the step is enclosed
    const BlockSets Htp = priv_block_plus(H.time_point, Pu.time_point);
    const BlockSets Hti = priv_homogeneous_interval(Xb, Htp, H.state_curvature);
    BlockSets Rtp = priv_block_plus(Htp, PU);
    BlockSets Rti = priv_block_plus(Hti, priv_block_plus(PU, Pu.input_curvature));
    return {std::move(Rtp),
            std::move(Rti),
            Htp,
            Hti,
            PU,
            std::move(Pu.time_point),
            std::move(H.state_curvature),
            std::move(Pu.input_curvature)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
