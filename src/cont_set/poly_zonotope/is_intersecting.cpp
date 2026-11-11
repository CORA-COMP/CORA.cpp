// is_intersecting - whether a polynomial zonotope may meet a polytope, as CORA's isIntersecting_
//
// CORA has no exact test and answers the type "approx": the set is excluded if some constraint of
// the polytope is violated by every point of the zonotope that encloses the set (a bound of
// A_i x from below, as supportFunc_ with type lower and method interval), or if an equality
// c'x = d lies outside the range of c'x. Otherwise the answer is true; false is certain.
//
// Syntax:   res = pZ.is_intersecting(P, tol);
// Inputs:   P - a polytope of the same dimension
//           tol - added to the range for the equalities
// Outputs:  res - false only if the sets do not intersect
// See also: zonotope, Specification::check
//
// CORA: contSet/@polyZonotope/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/cont_set/polytope/polytope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool PolyZonotope::is_intersecting(const Polytope &P, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_intersecting");
    CORA_CHECK(P.dim() == dim(), Error::dimension_mismatch, [&] {
        return "is_intersecting: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(P.dim());
    });
    // the range of every row a'x over the enclosing zonotope: a'c -+ the sum of |a'G|
    const Zonotope Z = zonotope();
    const auto range = [&](const Tensor &A) {
        const Tensor mid = A.matmul(Z.center()), rad = A.matmul(Z.generators()).abs().sum(-1, true);
        return std::pair{(mid - rad).data(), (mid + rad).data()};
    };
    if (P.A().shape()[0] > 0) {
        const auto [lo, hi] = range(P.A());
        const std::vector<double> b = P.b().data();
        for (std::size_t i = 0; i < b.size(); ++i)
            if (b[i] < lo[i]) return false;
    }
    if (P.Ae().shape()[0] > 0) {
        const auto [lo, hi] = range(P.Ae());
        const std::vector<double> be = P.be().data();
        for (std::size_t i = 0; i < be.size(); ++i)
            if (be[i] < lo[i] - tol || be[i] > hi[i] + tol) return false;
    }
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
