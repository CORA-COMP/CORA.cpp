// intersect - the intersection of a polytope with a polytope, a box or a point, as CORA's and_
//
// Syntax:   Pres = P.intersect(P2);   Pres = P.intersect(I);   Pres = P.intersect(p, tol);
// Inputs:   P2, I - a polytope or a box of the dimension of P;  p - a point (n, 1);  tol - how far
//           the point may be outside P; default: the repository tolerance (docs/architecture.md)
// Outputs:  Pres - exact, as halfspaces: the inequalities of both sets stacked, and the
//           equalities of both stacked (a vertex polytope is converted by constraints first, not
//           in a batch). R^n is neutral. A point gives itself as a vertex polytope if P holds it,
//           else the empty set
// See also: cart_prod, Polytope::contains
//
// CORA: contSet/@polytope/and_.m
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Shapes and refusals -----------------------------------------------------------------------------

/// The tensor that holds the data of the polytope: its halfspaces' matrix or its vertices.
const Tensor &aux_data(const Polytope &P) { return P.is_h_rep() ? P.A() : P.V(); }

/// Why the intersection is refused, if it is: the sets must share a dimension and a batch shape.
detail::Refusal aux_refusal(const Polytope &P1, const Polytope &P2) {
    if (P1.dim() != P2.dim())
        return std::pair{Error::dimension_mismatch, "intersect: the sets differ in dimension, " +
                                                        std::to_string(P1.dim()) + " and " +
                                                        std::to_string(P2.dim())};
    const std::vector<int64_t> a = aux_data(P1).shape(), b = aux_data(P2).shape();
    if (a.size() != b.size() || !std::equal(a.begin(), a.end() - 2, b.begin()))
        return std::pair{Error::dimension_mismatch,
                         "intersect: the sets must have the same batch shape, got " +
                             shape_text(a) + " and " + shape_text(b)};
    return std::nullopt;
}

/// The intersection of two polytopes: R^n is neutral, else the rows stack.
Polytope aux_intersect(const Polytope &P, const Polytope &S) {
    if (priv_is_fullspace(S)) return P;
    if (priv_is_fullspace(P)) return S;
    const Polytope::Constraints C1 = P.constraints(), C2 = S.constraints();
    return {Tensor::concat({C1.A, C2.A}, -2), Tensor::concat({C1.b, C2.b}, -2),
            Tensor::concat({C1.Ae, C2.Ae}, -2), Tensor::concat({C1.be, C2.be}, -2)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The intersections -------------------------------------------------------------------------------

Polytope Polytope::intersect(const Polytope &S) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_intersect(*this, S);
}

Polytope Polytope::intersect(const Interval &I) const {
    // a box is the polytope of its bounds
    detail::Entry entry;
    const Polytope S = priv_polytope_of_interval(I);
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_intersect(*this, S);
}

Polytope Polytope::intersect(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, aux_data(*this).dtype(), "intersect");
    const bool column = p.defined() && p.ndim() == 2 && p.shape()[0] == dim() && p.shape()[1] == 1;
    CORA_CHECK(column, Error::dimension_mismatch, [&] {
        return "intersect: the point must be a column (" + std::to_string(dim()) +
               ", 1), got the shape " + shape_text(p.shape());
    });
    // the intersection with a point is the point if it is in the set, else nothing
    if (contains(p, "exact", tol)) return Polytope(p);
    return empty(dim());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
