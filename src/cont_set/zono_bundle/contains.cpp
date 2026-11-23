// contains - whether a zonotope bundle contains points, a zonotope, a box or a bundle
//
// Points: "exact" is a linear program per point on the constrained zonotope of the bundle,
// "exact:zonotope" tests the members one by one. A zonotope or a box is inside if every member
// holds it; a bundle if each member holds it by its facets. CORA asks the polytope of the bundle.
//
// "exact:polytope" asks the polytope of the bundle, as CORA does. "approx" is "exact" for points;
// it takes no set.
//
// Syntax:   res = bundle.contains(p);   res = bundle.contains(S, "exact", tol);
// Inputs:   S - points (..., n), one per row,, a zonotope, a box or a bundle of the dimension of
// bundle
//           method - "exact" (default), "exact:zonotope", "exact:polytope" or "approx"
//           tol - relaxes the norm of the points and the facet tests; default: the repository one
// Outputs:  res - true if all points (S) are in the bundle; not for a batch
// See also: is_intersecting, Zonotope::contains, ConZonotope::contains
//
// CORA: contSet/@zonoBundle/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Checks ------------------------------------------------------------------------------------------

/// Why the call is refused, if it is: a batch, a method that is unknown, needs Polytope, or is
/// not supported for a set.
detail::Refusal aux_refusal(const ZonoBundle &bundle, const std::string &method, bool points) {
    if (detail::Refusal batch =
            priv_batch_refusal((bundle.zonotopes().front()).center().expand_dims(-1), "contains"))
        return batch;
    const bool known = method == "exact" || method == "exact:zonotope" ||
                       method == "exact:polytope" || method == "approx";
    if (!known)
        return std::pair{Error::no_specific_alg,
                         "contains: no algorithm '" + method + "' for this set"};
    if (method == "approx" && !points)
        return std::pair{Error::not_supported,
                         std::string("contains: the method 'approx' is not supported for a set; "
                                     "use 'exact'")};
    return std::nullopt;
}

/// Why the points are refused, if they are: not (..., n).
detail::Refusal aux_points_refusal(const ZonoBundle &bundle, const Tensor &p) {
    if (is_vector(p, bundle.dim())) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "contains: the points must be (..., " + std::to_string(bundle.dim()) +
                         "), one per row, got the shape " + shape_text(p.shape())};
}

// Members -----------------------------------------------------------------------------------------

/// The answer of a member's test as one bool: a mask of points is "every point".
bool aux_every(bool answer) { return answer; }
bool aux_every(const Mask &answer) { return answer.all(); }

/// Whether every member contains the set S (by the member's own exact test).
template <class S>
bool aux_all_members(const ZonoBundle &bundle, const S &set, double tol) {
    for (const Zonotope &Z : bundle.zonotopes())
        if (!aux_every(Z.contains(set, "exact", tol))) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Points ------------------------------------------------------------------------------------------

bool ZonoBundle::contains(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "contains");
    detail::Refusal r = aux_points_refusal(*this, p);
    if (!r) r = aux_refusal(*this, method, true);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (method == "exact:zonotope") return aux_all_members(*this, p, tol);
    if (method == "exact:polytope") return polytope().contains(p, "exact", tol);
    return con_zonotope().contains(p, "exact", tol);
}

// Sets --------------------------------------------------------------------------------------------

bool ZonoBundle::contains(const Zonotope &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "contains");
    detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.center().expand_dims(-1), "contains");
    if (!r) r = aux_refusal(*this, method, false);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (method == "exact:polytope") return polytope().contains(S, "exact", tol);
    return aux_all_members(*this, S, tol);
}

bool ZonoBundle::contains(const Interval &I, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "contains");
    detail::Refusal r = priv_operand_refusal(*this, I.dim(), I.inf().expand_dims(-1), "contains");
    if (!r) r = aux_refusal(*this, method, false);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // the empty box is in every set
    if (I.is_empty()) return true;
    if (method == "exact:polytope") return polytope().contains(I, "exact", tol);
    return aux_all_members(*this, I, tol);
}

bool ZonoBundle::contains(const ZonoBundle &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "contains");
    detail::Refusal r = priv_operand_refusal(
        *this, S.dim(), (S.zonotopes().front()).center().expand_dims(-1), "contains");
    if (!r) r = aux_refusal(*this, method, false);
    if (!r) r = priv_batch_refusal((S.zonotopes().front()).center().expand_dims(-1), "contains");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (method == "exact:polytope") return polytope().contains(S, "exact", tol);
    const ConZonotope inner = S.con_zonotope();
    for (const Zonotope &Z : Z_)
        if (!priv_con_zonotope_in_zonotope(inner, Z, tol)) return false;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
