// priv - the private functions of zono_bundle, as CORA's @zonoBundle/private
//
// Not for callers of the class: the operations of the class use them. The linear programs run on
// the host, on one (unbatched) set.
//
// Syntax:   Refusal r = priv_members_refusal(Zs);   bool empty = priv_bundle_is_empty(zB);
// See also: zono_bundle.h

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Refusals ----------------------------------------------------------------------------------------

/// Why the list is no bundle, if it is none: it is empty, or the zonotopes differ in dimension or
/// batch shape. The constructor raises it.
detail::Refusal priv_members_refusal(const std::vector<Zonotope> &zonotopes);

/// Why `zB` and a set of dimension n whose tensors are like `like` cannot be combined by the
/// operation `who`, if they cannot: the dimension (unless `same_dimension` is false, for a
/// product) or the batch shape differs.
detail::Refusal priv_operand_refusal(const ZonoBundle &zB, int64_t n, const Tensor &like,
                                     const char *who, bool same_dimension = true);

/// Why a linear program per set cannot run on a set whose tensors are like `like` (a batch), if it
/// cannot: Error::not_supported.
detail::Refusal priv_batch_refusal(const Tensor &like, const char *who);

// Linear programs ---------------------------------------------------------------------------------

/// Whether no point lies in all members of the bundle (CORA's representsa_(zB, 'emptySet')).
bool priv_bundle_is_empty(const ZonoBundle &zB);

/// Whether the constrained zonotope A lies in the zonotope B (plus tol, as within_tol): the
/// support function of A along every facet normal of B stays below the facet's offset. An empty A
/// lies in everything.
bool priv_con_zonotope_in_zonotope(const ConZonotope &A, const Zonotope &B, double tol);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
