// priv_linear_programs - the linear programs of a zonotope bundle on the host
//
// Syntax:   bool empty = priv_bundle_is_empty(bundle);
//           bool in = priv_con_zonotope_in_zonotope(A, B, tol);
// Inputs:   bundle - a bundle
//           A, B - a constrained zonotope and a zonotope (single sets)
// Outputs:  R - whether the bundle has no point, whether A lies in B
// See also: ZonoBundle::con_zonotope, ConZonotope
//
// CORA.cpp only: CORA asks the polytope of the bundle (isempty, contains); here the constrained
// zonotope of the bundle and the polytopes of its members answer
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_bundle_is_empty(const ZonoBundle &bundle) {
    return bundle.con_zonotope().factor_programs().is_empty();
}

bool priv_con_zonotope_in_zonotope(const ConZonotope &A, const Zonotope &B, double tol) {
    const ConZonotope inner = A.compact();
    if (inner.factor_programs().is_empty()) return true;
    // the largest value of A along every facet normal of B stays below the facet's offset
    const Polytope P = B.polytope();
    const Tensor &C = P.A();
    const std::vector<double> d = P.b().data();
    for (std::size_t k = 0; k < d.size(); ++k) {
        const Tensor row = C.slice(-2, int64_t(k), int64_t(k) + 1).matrix_transpose();
        const double largest = inner.support_function_range(row).sup().item();
        if (!(largest <= d[k] || within_tol(largest, d[k], tol))) return false;
    }
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
