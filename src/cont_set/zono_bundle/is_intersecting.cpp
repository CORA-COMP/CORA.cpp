// is_intersecting - whether a zonotope bundle and a set or points intersect
//
// "exact" asks whether the bundle of both is empty (a linear program on its constrained zonotope);
// "approx" tests the boxes around the sets, so it can say true for disjoint sets.
//
// Syntax:   res = bundle.is_intersecting(S);   res = bundle.is_intersecting(S, "approx", tol);
// Inputs:   S - a bundle, a zonotope, a box, or points (n, N) of the dimension of bundle
//           type - "exact" (default) or "approx"
//           tol - points: the tolerance of contains; "approx": enlarges the box of S; default: the
//           repository one
// Outputs:  res - true if the sets intersect (all points must lie in the bundle); not for a batch
// See also: contains, intersect, Interval::is_intersecting
//
// CORA: contSet/@zonoBundle/isIntersecting_.m
//
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the type is refused, if it is, or the call is for a batch (this bundle or the other set,
/// whose tensors are like `other`).
detail::Refusal aux_refusal(const ZonoBundle &bundle, const Tensor &other,
                            const std::string &type) {
    if (type != "exact" && type != "approx")
        return std::pair{Error::wrong_value,
                         "is_intersecting: the type must be 'exact' or 'approx', got '" + type +
                             "'"};
    if (const detail::Refusal batch =
            priv_batch_refusal(bundle.zonotopes().front().center(), "is_intersecting"))
        return batch;
    return priv_batch_refusal(other, "is_intersecting");
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool ZonoBundle::is_intersecting(const ZonoBundle &S, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "is_intersecting");
    detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "is_intersecting");
    if (!r) r = aux_refusal(*this, S.zonotopes().front().center(), type);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (type == "approx") return interval().is_intersecting(S.interval(), tol);
    return !priv_bundle_is_empty(intersect(S));
}

bool ZonoBundle::is_intersecting(const Zonotope &Z, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "is_intersecting");
    detail::Refusal r = priv_operand_refusal(*this, Z.dim(), Z.center(), "is_intersecting");
    if (!r) r = aux_refusal(*this, Z.center(), type);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // the boxes, or the emptiness of the bundle of both
    if (type == "approx") return interval().is_intersecting(Z.interval(), tol);
    return !priv_bundle_is_empty(intersect(Z));
}

bool ZonoBundle::is_intersecting(const Interval &I, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "is_intersecting");
    detail::Refusal r = priv_operand_refusal(*this, I.dim(), I.inf(), "is_intersecting");
    if (!r) r = aux_refusal(*this, I.inf(), type);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (type == "approx") return interval().is_intersecting(I, tol);
    // the empty box meets nothing; else as for a zonotope
    return !I.is_empty() && !priv_bundle_is_empty(intersect(I));
}

bool ZonoBundle::is_intersecting(const Tensor &p, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const detail::Refusal r = aux_refusal(*this, Z_.front().center(), type);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // CORA tests the points with contains_, whatever the type
    return contains(p, "exact", tol_arg);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
