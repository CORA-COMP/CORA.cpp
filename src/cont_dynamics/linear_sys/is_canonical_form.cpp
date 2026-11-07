// is_canonical_form - whether the system is in canonical form, as CORA's isCanonicalForm
//
// Canonical form means B = I and C = I (CORA also needs c, k, D, E to vanish, which CORA.cpp's
// systems do not have). Without B the system has no inputs: not canonical; without C, C = I.
//
// Syntax:   bool res = sys.is_canonical_form();
// Inputs:   sys - linear system
// Outputs:  res - whether B and C are identity matrices (compared exactly)
// See also: canonical_form
//
// CORA: contDynamics/@linearSys/isCanonicalForm.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool LinearSys::is_canonical_form() const {
    detail::Entry entry;
    // the input matrix is n x n, so the system has as many inputs as states
    if (!B_ || !priv_is_identity(*B_) || B_->shape()[0] != dim()) return false;
    return !C_ || (priv_is_identity(*C_) && C_->shape()[0] == dim());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
