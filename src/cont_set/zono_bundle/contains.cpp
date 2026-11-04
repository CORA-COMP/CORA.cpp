// contains - whether a zonotope bundle contains points, a zonotope, a box or a bundle
//
// Syntax:   res = zB.contains(p);   res = zB.contains(S, "exact", tol);
// Inputs:   p, S, I - points (n, N), a zonotope, a box or a bundle of the dimension of zB
//           method - "exact" (default), "exact:zonotope" and "approx" (which CORA takes for
//                    "exact"); CORA's "exact:polytope" needs Polytope. Both exact methods decide
//           tol - relaxes the norm of the points and the facet tests (as within_tol); default: the
//                 repository tolerance (docs/architecture.md)
// Outputs:  res - true if all points (S) are in the bundle. Points: "exact" is one linear program
//           per point on the constrained zonotope of the bundle, "exact:zonotope" tests the members
//           one by one. A zonotope or a box is in the bundle if every member contains it
//           (Zonotope::contains); a bundle S if each member contains the constrained zonotope of
//           S, by its support function along the member's facet normals. CORA asks the polytope
//           of the bundle; "approx" for a set is not supported (ConZonotope has none). Not a batch.
//           CORA's second and third outputs are not returned
// See also: is_intersecting, Zonotope::contains, ConZonotope::contains
//
// CORA: contSet/@zonoBundle/contains_.m

#include "cont_set/zono_bundle/private/priv.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the call is refused, if it is: a batch, a method that is unknown, needs Polytope, or is
/// not supported for a set.
detail::Refusal aux_refusal(const ZonoBundle &zB, const std::string &method, bool points) {
    if (detail::Refusal batch = priv_batch_refusal(zB.zonotopes().front().center(), "contains")) return batch;
    const bool known = method == "exact" || method == "exact:zonotope" ||
                       method == "exact:polytope" || method == "approx";
    if (!known)
        return std::pair{Error::no_specific_alg, "contains: no algorithm '" + method +
                                                     "' for this set"};
    if (method == "exact:polytope")
        return std::pair{Error::not_supported,
                         std::string("contains: the method 'exact:polytope' needs the class "
                                     "Polytope")};
    if (method == "approx" && !points)
        return std::pair{Error::not_supported,
                         std::string("contains: the method 'approx' is not supported for a set; "
                                     "use 'exact'")};
    return std::nullopt;
}

/// Why the points are refused, if they are: not a matrix (n, N) of columns.
detail::Refusal aux_points_refusal(const ZonoBundle &zB, const Tensor &p) {
    if (p.ndim() == 2 && p.shape()[0] == zB.dim()) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "contains: the points must be a matrix (" + std::to_string(zB.dim()) +
                         ", N) of columns, got the shape " + shape_text(p.shape())};
}

/// Whether every member contains the set S (by the member's own exact test).
template <class S>
bool aux_all_members(const ZonoBundle &zB, const S &set, double tol) {
    for (const Zonotope &Z : zB.zonotopes())
        if (!Z.contains(set, "exact", tol)) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool ZonoBundle::contains(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "contains");
    detail::Refusal r = aux_points_refusal(*this, p);
    if (!r) r = aux_refusal(*this, method, true);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (method == "exact:zonotope") return aux_all_members(*this, p, tol);
    return con_zonotope().contains(p, "exact", tol);
}

bool ZonoBundle::contains(const Zonotope &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "contains");
    detail::Refusal r = priv_operand_refusal(*this, S.dim(), S.center(), "contains");
    if (!r) r = aux_refusal(*this, method, false);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return aux_all_members(*this, S, tol);
}

bool ZonoBundle::contains(const Interval &I, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "contains");
    detail::Refusal r = priv_operand_refusal(*this, I.dim(), I.inf(), "contains");
    if (!r) r = aux_refusal(*this, method, false);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // the empty box is in every set
    return I.is_empty() || aux_all_members(*this, I, tol);
}

bool ZonoBundle::contains(const ZonoBundle &S, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "contains");
    detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "contains");
    if (!r) r = aux_refusal(*this, method, false);
    if (!r) r = priv_batch_refusal(S.zonotopes().front().center(), "contains");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    const ConZonotope inner = S.con_zonotope();
    for (const Zonotope &Z : Z_)
        if (!priv_con_zonotope_in_zonotope(inner, Z, tol)) return false;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
