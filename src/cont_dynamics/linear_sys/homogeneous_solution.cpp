// homogeneous_solution - the solution of x' = A x over one step, as CORA's homogeneousSolution
//
// The set at the end of the step is e^{A dt} X; the set over the step encloses X and that set and
// adds the curvature enlargement F X (see correction_matrix_state).
//
// Syntax:   HomogeneousSolution H = sys.homogeneous_solution(X, time_step, taylor_terms);
// Inputs:   X - initial set of the step (zonotope of the dimension of A)
//           time_step - step size, positive
//           taylor_terms - order of the series of F, at least 1
// Outputs:  H - time_point (Htp), time_interval (Hti) and state_curvature (C_state)
// See also: affine_solution, one_step, correction_matrix_state
//
// CORA: contDynamics/@linearSys/homogeneousSolution.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

HomogeneousSolution LinearSys::homogeneous_solution(const Zonotope &X, double time_step,
                                                    int taylor_terms) const {
    detail::Entry entry;
    if (entry.check()) validate_step("homogeneous_solution", X, time_step, taylor_terms);
    TaylorLinSys taylor(A_);
    // the whole state is one block
    const BlockSets Xb{X};
    BlockHomogeneous H =
        priv_homogeneous(*this, taylor, priv_one_block(dim()), Xb, time_step, taylor_terms);
    const BlockSets Hti = priv_homogeneous_interval(Xb, H.time_point, H.state_curvature);
    return {H.time_point[0], Hti[0], H.state_curvature[0]};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
