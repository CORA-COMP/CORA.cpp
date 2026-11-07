// less_equal - whether a polytope is contained in a set, as CORA's polytope.le
//
// P <= S is S.contains(P). The polytope is tested by its vertices (an unbounded one is in no
// bounded set, the empty set is in every set), so S is a polytope, a box, a zonotope, a constrained
// zonotope, a bundle, the whole space or the empty set. CORA's default tolerance is eps.
//
// Syntax:   bool res = P.less_equal(S);   bool res = P.less_equal(S, tol);
// Inputs:   S - the set that should contain P
//           tol - how far P may reach out of S; default: the repository tolerance
// Outputs:  res - whether S contains P
// See also: Polytope::contains, Polytope::isequal
//
// CORA: contSet/@polytope/le.m
// CORA authors: Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the set S contains the (nonempty, bounded) polytope P, by its vertices.
bool aux_contains_vertices(const ContSet &S, const Polytope &P, double tol) {
    const Tensor &like = P.is_h_rep() ? P.A() : P.V();
    const Tensor points = priv_points_tensor(like, P.dim(), priv_polytope_vertices(P));
    switch (S.class_id()) {
    case ClassId::polytope:
        return static_cast<const Polytope &>(S).contains(points, "exact", tol);
    case ClassId::interval:
        return static_cast<const Interval &>(S).contains(points, tol);
    case ClassId::zonotope:
        return static_cast<const Zonotope &>(S).contains(points, "exact", tol);
    case ClassId::con_zonotope:
        return static_cast<const ConZonotope &>(S).contains(points, "exact", tol);
    case ClassId::zono_bundle:
        return static_cast<const ZonoBundle &>(S).contains(points, "exact", tol);
    default:
        CORA_FAIL(Error::not_supported,
                  "less_equal: no containment of a polytope in a " + S.class_name() + " yet");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Polytope::less_equal(const ContSet &S, Tol tol_arg) const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    const double tol = resolve_tol(entry, tol_arg, like.dtype(), "less_equal");
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "less_equal: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    CORA_CHECK(like.ndim() == 2, Error::not_supported, [] {
        return std::string("less_equal: it supports a single polytope, not a batch");
    });
    // the whole space holds every set, the empty set only the empty set
    if (S.class_id() == ClassId::fullspace) return true;
    if (S.class_id() == ClassId::empty_set) return priv_polytope_is_empty(*this);
    if (priv_polytope_is_empty(*this)) return true;
    if (S.class_id() == ClassId::polytope)
        return static_cast<const Polytope &>(S).contains(*this, "exact", tol);
    // an unbounded set is in no bounded set
    if (!priv_polytope_is_bounded(*this)) return false;
    return aux_contains_vertices(S, *this, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
