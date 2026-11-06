// con_intersect - the points of a constrained zonotope whose image lies in another one
//
// Syntax:   cZres = cz.con_intersect(cz2, M);
// Inputs:   cz2 - constrained zonotope (n2 dimensions)
//           M - the matrix (n2, n)
// Outputs:  cZres - exact: the set {x in cz | M x in cz2}; the factors of both sets stay, each with
//           its constraints, and one equality per dimension of cz2 makes M x and the point agree
// See also: intersect, Zonotope::linear_map
//
// CORA: contSet/@conZonotope/conIntersect.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::con_intersect(const ConZonotope &cz2, const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.shape().back() == dim() && M.shape()[M.ndim() - 2] == cz2.dim(),
               Error::dimension_mismatch, [&] {
                   return "con_intersect: M must be (" + std::to_string(cz2.dim()) + ", " +
                          std::to_string(dim()) + "), got the shape " + shape_text(M.shape());
               });
    const auto [Mc, MG] = Tensor::zonotope_linear_map(M, c_, G_);
    return priv_agreement(*this, cz2, Mc, MG);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
