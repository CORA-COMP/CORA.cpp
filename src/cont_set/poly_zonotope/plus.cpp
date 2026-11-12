// plus - the Minkowski sum of a polynomial zonotope with a zonotope, a box, a point or another set
//
// A zonotope, a box or a point joins the independent part. A polynomial zonotope adds its
// dependent generators on new factors (the block diagonal of the exponents): its factors are not
// those of this set; exact_plus shares equal identifiers.
//
// Syntax:   S = pZ.plus(Z);   S = pZ.plus(pZ2);
// Inputs:   Z, I, v, pZ2 - a zonotope, a box, a point (n, 1) or a polynomial zonotope
// Outputs:  S - the sum
// See also: exact_plus
//
// CORA: contSet/@polyZonotope/plus.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::plus(const Zonotope &Z) const {
    detail::Entry entry;
    // a single set of the dimension of this one
    CORA_CHECK(Z.dim() == dim() && Z.center().ndim() == 2, Error::dimension_mismatch, [&] {
        return "plus: the zonotope must be a single set of the dimension " + std::to_string(dim());
    });
    return {c_ + Z.center(),     G_, Tensor::concat({GI_, Z.generators()}, -1), E_, id_,
            detail::Normalized{}};
}

// A box is a zonotope.
PolyZonotope PolyZonotope::plus(const Interval &I) const { return plus(I.zonotope()); }

PolyZonotope PolyZonotope::plus(const Tensor &v) const {
    detail::Entry entry;
    // a point moves the center
    CORA_CHECK(v.ndim() == 2 && v.shape()[0] == dim() && v.shape()[1] == 1,
               Error::dimension_mismatch, [&] {
                   return "plus: the point must be a column of the dimension " +
                          std::to_string(dim());
               });
    return {c_ + v, G_, GI_, E_, id_, detail::Normalized{}};
}

PolyZonotope PolyZonotope::plus(const PolyZonotope &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    // the factors of S get the identifiers above those of this set, as blkdiag of the exponents
    const int64_t shift = id_.empty() ? 0 : *std::max_element(id_.begin(), id_.end());
    std::vector<int64_t> id = id_;
    for (const int64_t v : S.id_) id.push_back(shift + v);
    // the exponents of this set, then of S, each over all identifiers (0 for those of the other)
    Exponents E;
    for (const std::vector<int> &col : E_) {
        std::vector<int> c = col;
        c.resize(id.size(), 0);
        E.push_back(std::move(c));
    }
    for (const std::vector<int> &col : S.E_) {
        std::vector<int> c(id_.size(), 0);
        c.insert(c.end(), col.begin(), col.end());
        E.push_back(std::move(c));
    }
    // the generators side by side, the exponents of the two sets stay distinct
    return {c_ + S.c_,
            Tensor::concat({G_, S.G_}, -1),
            Tensor::concat({GI_, S.GI_}, -1),
            std::move(E),
            std::move(id),
            detail::Normalized{}};
}

Set PolyZonotope::translate_impl(const Tensor &v) const { return plus(v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
