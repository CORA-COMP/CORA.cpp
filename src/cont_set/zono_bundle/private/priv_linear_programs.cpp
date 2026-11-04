// priv_linear_programs - the linear programs of a zonotope bundle on the host
//
// Syntax:   bool empty = priv_bundle_is_empty(bundle);   bool in = priv_con_zonotope_in_zonotope(A,
// B, tol); Inputs:   bundle - a bundle;  A, B - a constrained zonotope and a zonotope (single sets)
// Outputs:  whether the bundle has no point, whether A lies in B
// See also: ZonoBundle::con_zonotope, ConZonotope
//
// CORA.cpp only: CORA asks the polytope of the bundle (isempty, contains); there is no Polytope
// here
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zono_bundle/private/priv.h"
#include "cont_set/zonotope/private/priv.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_bundle_is_empty(const ZonoBundle &bundle) {
    return priv_is_empty(priv_host_con_zonotope(bundle.con_zonotope()));
}

bool priv_con_zonotope_in_zonotope(const ConZonotope &A, const Zonotope &B, double tol) {
    const HostConZonotope inner = priv_host_con_zonotope(priv_compact(A));
    if (priv_is_empty(inner)) return true;
    const auto [normals, offsets] = priv_zonotope_halfspaces(B);
    const std::vector<double> C = normals.data(), d = offsets.data();
    for (std::size_t k = 0; k < d.size(); ++k) {
        const std::vector<double> row(C.begin() + int64_t(k) * inner.n,
                                      C.begin() + int64_t(k + 1) * inner.n);
        const double largest = priv_range(inner, row)->second;
        if (!(largest <= d[k] || within_tol(largest, d[k], tol))) return false;
    }
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
