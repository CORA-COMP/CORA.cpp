// isequal - whether two zonotope bundles are the same set
//
// False for other dimensions; true if the members agree one by one (Zonotope::isequal with tol,
// where CORA uses its default); else each bundle must contain the other.
//
// Syntax:   res = bundle.isequal(S);   res = bundle.isequal(S, tol);
// Inputs:   S - a bundle
//           tol - default: the repository tolerance (docs/architecture.md)
// Outputs:  res - whether the bundles are equal; not for a batch
// See also: contains, Zonotope::isequal
//
// CORA: contSet/@zonoBundle/isequal.m
// CORA authors: Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the bundles have as many members and every pair of the same position is equal.
bool aux_members_equal(const ZonoBundle &bundle, const ZonoBundle &S, double tol) {
    if (bundle.parallel_sets() != S.parallel_sets()) return false;
    for (std::size_t i = 0; i < bundle.zonotopes().size(); ++i)
        if (!bundle.zonotopes()[i].isequal(S.zonotopes()[i], tol)) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool ZonoBundle::isequal(const ZonoBundle &S, Tol tol_arg) const {
    detail::Entry entry;
    // layout: #163 step 2
    const double tol =
        resolve_tol(entry, tol_arg, detail::cora_center(Z_.front()).dtype(), "isequal");
    // layout: #163 step 2
    detail::Refusal r = priv_batch_refusal(detail::cora_center(Z_.front()), "isequal");
    // layout: #163 step 2
    if (!r) r = priv_batch_refusal(detail::cora_center(S.zonotopes().front()), "isequal");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (S.dim() != dim()) return false;
    return aux_members_equal(*this, S, tol) ||
           (contains(S, "exact", tol) && S.contains(*this, "exact", tol));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
