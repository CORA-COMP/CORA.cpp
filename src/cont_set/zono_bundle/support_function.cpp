// support_function - the support function of a zonotope bundle, as CORA's zonoBundle.supportFunc
//
// The value of the constrained zonotope of the bundle: a linear program over the factors of all
// members. The empty bundle has -Inf. Not a batch.
//
// Syntax:   rho = bundle.support_function(d);
// Inputs:   d - direction, a column (n, 1)
// Outputs:  rho - max_{x in bundle} d'x, shape (1, 1)
// See also: con_zonotope, ConZonotope::support_function
//
// CORA: contSet/@zonoBundle/supportFunc_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ZonoBundle::support_function_impl(const Tensor &d) const {
    return con_zonotope().support_function(d);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
