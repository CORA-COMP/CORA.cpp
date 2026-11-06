// priv_is_identity - whether a matrix is the identity
//
// Syntax:   bool res = priv_is_identity(M);
// Inputs:   M - matrix
// Outputs:  res - whether M is square and every entry equals the identity's
// See also: is_canonical_form, nonlinear_sys
//
// CORA.cpp only: CORA writes all(M == eye(n),'all').
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_is_identity(const Tensor &M) {
    const std::vector<int64_t> s = M.shape();
    return s.size() == 2 && s[0] == s[1] && M.equal(M.eye_like()).min().item() != 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
