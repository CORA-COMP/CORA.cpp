// canonical_form - the system and its input in canonical form, as CORA's canonicalForm
//
// The input matrix becomes the identity: x' = A x + u' with u' in B (U - c) and the constant
// input B (c + u_vec), where c is the center of U. Reach works on this form.
//
// Syntax:   CanonicalForm cf = sys.canonical_form(U);   sys.canonical_form(U, u_vec);
// Inputs:   U - input set (zonotope of the dimension m of the columns of B; needs B)
//           u_vec - constant input per step (m, steps), default: zero
// Outputs:  cf - sys (B = I, C unchanged), U (centered, in the state space), u (n, steps)
// See also: is_canonical_form, one_step, reach
//
// CORA: contDynamics/@linearSys/canonicalForm.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CanonicalForm LinearSys::canonical_form(const Zonotope &U,
                                        const std::optional<Tensor> &u_vec) const {
    detail::Entry entry;
    CORA_CHECK(B_.has_value(), Error::wrong_value,
               [] { return std::string("canonical_form: needs B; use LinearSys(A, B)"); });
    CORA_CHECK(U.dim() == B_->shape().back(), Error::dimension_mismatch, [&] {
        return "canonical_form: U must have the dimension " + std::to_string(B_->shape().back()) +
               " of the columns of B, got " + std::to_string(U.dim());
    });
    CORA_CHECK(!u_vec ||
                   (u_vec->ndim() == 2 && u_vec->shape()[0] == U.dim() && u_vec->shape()[1] >= 1),
               Error::dimension_mismatch, [&] {
                   return "canonical_form: u_vec must be a matrix (m, steps) with m = " +
                          std::to_string(U.dim()) + " rows, got " + shape_text(u_vec->shape());
               });
    const Tensor &B = *B_;
    const Tensor &center = U.center();
    // the center of U moves to the constant input, the rest stays as the input set
    const Tensor u_input = u_vec ? *u_vec : center.zeros_like();
    const Tensor u = B.matmul(u_input) + B.matmul(center);
    const Zonotope U_centered{center.zeros_like(), U.generators()};
    return {LinearSys(A_, A_.eye_like(), C_), U_centered.linear_map(B), u};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
