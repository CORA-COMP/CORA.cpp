// is_full_dim - whether a zonotope bundle has a nonempty interior
//
// Syntax:   res = zB.is_full_dim();   res = zB.is_full_dim(tol);
// Inputs:   tol - the rank tolerance of the members and of the intersection; default: the
//           repository tolerance (docs/architecture.md)
// Outputs:  res - true if every member is full-dimensional and so is the intersection, which is
//           decided on its constrained zonotope (CORA decides it on the polytope of the bundle;
//           there is no Polytope yet). Not a batch
// See also: con_zonotope, Zonotope::is_full_dim, ConZonotope::is_full_dim
//
// CORA: contSet/@zonoBundle/isFullDim.m

#include "cont_set/zono_bundle/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool ZonoBundle::is_full_dim(Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Z_.front().center().dtype(), "is_full_dim");
    const detail::Refusal batch = priv_batch_refusal(Z_.front().center(), "is_full_dim");
    CORA_CHECK(!batch, batch->first, [&] { return batch->second; });
    for (const Zonotope &Z : Z_)
        if (!Z.is_full_dim(tol)) return false;
    return con_zonotope().is_full_dim(tol);
}

bool ZonoBundle::is_full_dim_impl(Tol tol) const { return is_full_dim(tol); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
