// representsa - whether a matrix zonotope is also a set of another type, as CORA's representsa
//
// Syntax:   bool b = A.representsa("point");   bool b = A.representsa("origin", tol);
// Inputs:   type - a CORA set type, as the zonotope's representsa;  tol - as there, default the
//           repository tolerance
// Outputs:  b - whether the stacked zonotope of A represents that type (A with a single point and
//           that point the zero matrix for "origin"); CORA's second output is not returned
// See also: zonotope, Zonotope::representsa
//
// CORA: matrixSet/@matZonotope/representsa.m

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool MatZonotope::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, C_.dtype(), "representsa");
    CORA_CHECK(C_.ndim() == 2, Error::not_supported,
               [] { return std::string("representsa: not supported for a batch of sets"); });
    return zonotope().representsa(type, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
