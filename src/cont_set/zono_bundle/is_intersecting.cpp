// is_intersecting - whether a zonotope bundle and a set or points intersect
//
// "exact" asks whether the bundle of both is empty (a linear program on its constrained
// zonotope); "approx" tests the boxes around the sets, so it can say true for disjoint sets.
//
// Syntax:   res = bundle.is_intersecting(S);   res = bundle.is_intersecting(S, "approx", tol);
// Inputs:   S - a bundle, a zonotope, a box, or points (..., n) of the dimension of bundle
//           method - "exact" (default) or "approx"
//           tol - points: the tolerance of contains; "approx": enlarges the box of S
// Outputs:  res - true if the sets intersect (all points must lie in the bundle); not for a batch
// See also: contains, intersect, Interval::is_intersecting
//
// CORA: contSet/@zonoBundle/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the method is refused, if it is, or the call is for a batch (this bundle or the other set,
/// whose tensors are like `other`).
detail::Refusal aux_refusal(const ZonoBundle &bundle, const Tensor &other,
                            const std::string &method) {
    if (method != "exact" && method != "approx")
        return std::pair{Error::wrong_value,
                         "is_intersecting: the method must be 'exact' or 'approx', got '" + method +
                             "'"};
    if (const detail::Refusal batch = priv_batch_refusal(
            (bundle.zonotopes().front()).center().expand_dims(-1), "is_intersecting"))
        return batch;
    return priv_batch_refusal(other, "is_intersecting");
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool ZonoBundle::is_intersecting(const ZonoBundle &S, const std::string &method,
                                 Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "is_intersecting");
    // the operand's first member is the shape to compare with
    const Tensor other = (S.zonotopes().front()).center().expand_dims(-1);
    detail::Refusal r = priv_operand_refusal(*this, S.dim(), other, "is_intersecting");
    if (!r) r = aux_refusal(*this, other, method);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (method == "approx") return interval().is_intersecting(S.interval(), tol).all();
    return !priv_bundle_is_empty(intersect(S));
}

bool ZonoBundle::is_intersecting(const Zonotope &Z, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "is_intersecting");
    detail::Refusal r =
        priv_operand_refusal(*this, Z.dim(), Z.center().expand_dims(-1), "is_intersecting");
    if (!r) r = aux_refusal(*this, Z.center().expand_dims(-1), method);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // the boxes, or the emptiness of the bundle of both
    if (method == "approx") return interval().is_intersecting(Z.interval(), tol).all();
    return !priv_bundle_is_empty(intersect(Z));
}

bool ZonoBundle::is_intersecting(const Interval &I, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "is_intersecting");
    detail::Refusal r =
        priv_operand_refusal(*this, I.dim(), I.inf().expand_dims(-1), "is_intersecting");
    if (!r) r = aux_refusal(*this, I.inf().expand_dims(-1), method);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (method == "approx") return interval().is_intersecting(I, tol).all();
    // the empty box meets nothing; else as for a zonotope
    return !I.is_empty() && !priv_bundle_is_empty(intersect(I));
}

bool ZonoBundle::is_intersecting(const Tensor &p, const std::string &method, Tol tol_arg) const {
    detail::Entry entry;
    const detail::Refusal r = aux_refusal(*this, Z_.front().center().expand_dims(-1), method);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // CORA tests the points with contains_, whatever the method
    return contains(p, "exact", tol_arg);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
