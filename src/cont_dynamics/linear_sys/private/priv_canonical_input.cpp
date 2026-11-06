// priv_canonical_input - the input of a linear system in canonical form, as CORA's canonicalForm
//
// x' = A x + B u with u in U becomes x' = A x + u' with u' in u_trans + GU [-1, 1]^k.
//
// Syntax:   CanonicalInput in = priv_canonical_input(sys, U);
// Inputs:   sys - linear system
//           U - input zonotope (checked by verify; without B it is zero)
// Outputs:  in - u_trans = B center(U) (n, 1); GU = B generators(U), (n, 0) if all are zero
// See also: priv_verify_ra_support_function, priv_verify_ra_zonotope
//
// CORA.cpp only: the part of canonicalForm that verify uses, without building the system
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CanonicalInput priv_canonical_input(const LinearSys &sys, const Zonotope &U) {
    const Tensor &A = sys.A();
    const int64_t n = A.shape()[0];
    CanonicalInput in{A.zeros_like({n, 1}), Tensor::like(A, {}, {n, 0}), n,
                      sys.C() ? sys.C()->shape()[0] : n};
    if (!sys.B()) return in;
    const Tensor &B = *sys.B();
    in.u_trans = B.matmul(U.center());
    if (U.generators().shape()[1] == 0) return in;
    const Tensor GU = B.matmul(U.generators());
    // zero generators add no uncertainty; dropping them keeps the path without an input set
    if (GU.abs().sum().item() != 0) in.GU = GU;
    return in;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
