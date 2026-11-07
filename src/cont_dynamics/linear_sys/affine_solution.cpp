// affine_solution - the solution of x' = A x + u over one step, as CORA's affineSolution
//
// The set at the end of the step is e^{A dt} X plus the solution of the constant input u; the set
// over the step encloses X and that set, with the curvatures of the state and of the input.
//
// Syntax:   AffineSolution S = sys.affine_solution(X, u, time_step, taylor_terms);
// Inputs:   X - initial set of the step (zonotope of the dimension of A)
//           u - constant input in the state space (n, 1)
//           time_step - step size, positive
//           taylor_terms - order of the series, at least 1
// Outputs:  S - time_point (Htp), constant (Pu), time_interval (Hti), state_curvature,
// input_curvature See also: homogeneous_solution, particular_solution_constant, one_step
//
// CORA: contDynamics/@linearSys/affineSolution.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

AffineSolution LinearSys::affine_solution(const Zonotope &X, const Tensor &u, double time_step,
                                          int taylor_terms) const {
    detail::Entry entry;
    if (entry.check()) validate_step("affine_solution", X, time_step, taylor_terms);
    CORA_CHECK(u.shape() == std::vector<int64_t>({dim(), 1}), Error::dimension_mismatch, [&] {
        return "affine_solution: u must be a column (" + std::to_string(dim()) + ", 1), got " +
               shape_text(u.shape());
    });
    CORA_CHECK(A_.ndim() == 2, Error::not_supported, [] {
        return std::string("affine_solution: it supports a single system, not a batch of "
                           "matrices A");
    });
    TaylorLinSys taylor(A_);
    const Blocks blocks = priv_one_block(dim());
    const BlockSets Xb{X};
    // the constant input is a point of the state space
    const Zonotope point(u, u.zeros_like({dim(), 0}));
    BlockConstant Pu =
        priv_particular_constant(*this, taylor, blocks, point, time_step, taylor_terms);
    BlockHomogeneous H = priv_homogeneous(*this, taylor, blocks, Xb, time_step, taylor_terms);
    const BlockSets Htp = priv_block_plus(H.time_point, Pu.time_point);
    // the step is enclosed, then both curvatures are added together
    const BlockSets curvature = priv_block_plus(H.state_curvature, Pu.input_curvature);
    const BlockSets Hti = priv_block_plus(priv_block_enclose(Xb, Htp), curvature);
    return {Htp[0], Pu.time_point[0], Hti[0], H.state_curvature[0], Pu.input_curvature[0]};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
