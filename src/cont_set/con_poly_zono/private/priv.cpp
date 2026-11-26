// priv - the block layouts of the constrained polynomial zonotope's operations
//
// See also: priv.h
//
// CORA.cpp only: the block layouts of CORALean's ConPolyZono (genBlockDiag, the identifiers)
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"

#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<int64_t> priv_ids(int64_t p) {
    std::vector<int64_t> id(static_cast<std::size_t>(p));
    for (int64_t i = 0; i < p; ++i) id[std::size_t(i)] = i + 1;
    return id;
}

Exponents priv_identity_exponents(int64_t h) {
    // one factor per generator
    Exponents E(std::size_t(h), std::vector<int>(std::size_t(h), 0));
    for (int64_t j = 0; j < h; ++j) E[std::size_t(j)][std::size_t(j)] = 1;
    return E;
}

ConPolyZono priv_set(const Tensor &c, const Tensor &G, const Tensor &GI, Exponents E,
                     PolyLevelSet cons) {
    const int64_t p = cons.num_vars();
    return {PolyZonotope(c, G, GI, std::move(E), priv_ids(p), detail::Verbatim{}), std::move(cons)};
}

Tensor priv_block_rows(const Tensor &A, const Tensor &B) {
    // A with zeros on the right, above zeros with B
    const int64_t h1 = A.shape()[0], n1 = A.shape()[1], h2 = B.shape()[0], n2 = B.shape()[1];
    const Tensor top = Tensor::concat({A, A.zeros_like({h1, n2})}, 1);
    return Tensor::concat({top, Tensor::concat({A.zeros_like({h2, n1}), B}, 1)}, 0);
}

Zonotope priv_box_zonotope(const Interval &I) {
    // the center, then the radius on the diagonal
    const Tensor c = (I.inf() + I.sup()) * 0.5;
    return {c, (I.sup() - c).maximum(c - I.inf()).expand_dims(-1).diag()};
}

Tensor priv_plus_zero(const Tensor &x) { return x + 0.0; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
