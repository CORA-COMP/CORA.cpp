// cart_prod - the Cartesian product of a polynomial zonotope with a zonotope, as CORA's cartProd_
//
// The centers stack, the dependent generators get zero rows below, and the generators of the
// zonotope are independent, on the block diagonal of the independent generators.
//
// Syntax:   S = pZ.cart_prod(Z);
// Inputs:   Z - a zonotope or a box of any dimension
// Outputs:  S - the product, of the dimension n + dim(Z)
// See also: plus
//
// CORA: contSet/@polyZonotope/cartProd_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::cart_prod(const Zonotope &Z) const {
    detail::Entry entry;
    // layout: #163 step 2
    CORA_CHECK(detail::cora_center(Z).ndim() == 2, Error::dimension_mismatch,
               [] { return std::string("cart_prod: the zonotope must be a single set"); });
    // layout: #163 step 2
    const int64_t n = dim(), n2 = Z.dim(), q = GI_.shape()[1],
                  q2 = detail::cora_generators(Z).shape()[1];
    // zeros below the dependent generators; the independent ones on the block diagonal
    const Tensor G = Tensor::concat({G_, G_.zeros_like({n2, G_.shape()[1]})}, -2);
    const Tensor top = Tensor::concat({GI_, GI_.zeros_like({n, q2})}, -1);
    // layout: #163 step 2
    const Tensor bottom = Tensor::concat({GI_.zeros_like({n2, q}), detail::cora_generators(Z)}, -1);
    // layout: #163 step 2
    return {Tensor::concat({c_, detail::cora_center(Z)}, -2),
            G,
            Tensor::concat({top, bottom}, -2),
            E_,
            id_,
            detail::Normalized{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
