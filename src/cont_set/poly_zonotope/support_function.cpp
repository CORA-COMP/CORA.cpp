// support_function - an upper bound of the support function of a polynomial zonotope
//
// The value for the box method of CORA's supportFunc_ with type upper: d'c plus the sum of the
// absolute values of d' times the generators of the enclosing zonotope.
//
// Syntax:   v = pZ.support_function(d);
// Inputs:   d - direction (n, 1)
// Outputs:  v - the bound (1, 1)
// See also: zonotope, interval
//
// CORA: contSet/@polyZonotope/supportFunc_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor PolyZonotope::support_function_impl(const Tensor &d) const {
    return zonotope().support_function(d);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
