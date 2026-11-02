// priv_canonical_input - the input of a linear system in canonical form, as CORA's canonicalForm
//
// x' = A x + B u with u in U becomes x' = A x + u' with u' in u_trans + GU [-1, 1]^k.
//
// Syntax:   CanonicalInput in = priv_canonical_input(sys, U);
// Inputs:   sys - linear system;  U - input zonotope (checked by verify; without B it is zero)
// Outputs:  in - u_trans = B center(U) (n, 1); GU = B generators(U), (n, 0) if all are zero
// See also: priv_verify_ra_support_function, priv_verify_ra_zonotope
//
// CORA: contDynamics/@linearSys/canonicalForm.m

#include "cont_dynamics/linear_sys/private/priv_verify.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CanonicalInput priv_canonical_input(const LinearSys &sys, const Zonotope &U) {
    const Tensor &A = sys.A();
    const int64_t n = A.shape()[0];
    CanonicalInput in{Tensor::like(A, std::vector<double>(n, 0.0), {n, 1}), Tensor::like(A, {}, {n, 0})};
    if (!sys.B()) return in;
    const Tensor &B = *sys.B();
    in.u_trans = B.matmul(U.center());
    if (U.generators().shape()[1] == 0) return in;
    const Tensor GU = B.matmul(U.generators());
    // zero generators add no uncertainty; dropping them keeps the path without an input set
    for (const double v : GU.data())
        if (v != 0) {
            in.GU = GU;
            break;
        }
    return in;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
