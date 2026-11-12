// zonotope - a zonotope that contains a polynomial zonotope, as CORA's polyZonotope.zonotope
//
// A generator whose exponents are all even only reaches [0, 1] of its length: half of it moves
// to the center, the other half stays a generator. The independent generators stay as they are.
//
// Syntax:   Z = pZ.zonotope();
// Inputs:   -
// Outputs:  Z - the enclosing zonotope, with the center c + half the sum of the even generators
// See also: interval, reduce
//
// CORA: contSet/@polyZonotope/zonotope.m
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

Zonotope PolyZonotope::zonotope() const {
    detail::Entry entry;
    const int64_t h = num_dependent();
    // a constant generator joins the center, an even one is split into center and half generator
    std::vector<int64_t> keep, even, constant;
    for (int64_t j = 0; j < h; ++j) {
        const std::vector<int> &col = E_[std::size_t(j)];
        if (std::all_of(col.begin(), col.end(), [](int e) { return e == 0; }))
            constant.push_back(j);
        else
            (priv_is_even(col) ? even : keep).push_back(j);
    }
    if (even.empty() && constant.empty()) return {c_, Tensor::concat({G_, GI_}, -1)};
    Tensor c = c_;
    if (!constant.empty()) c = c + priv_columns(G_, constant).sum(-1, true);
    const Tensor G_even = even.empty() ? G_.zeros_like({dim(), 0}) : priv_columns(G_, even) * 0.5;
    if (!even.empty()) c = c + G_even.sum(-1, true);
    return {c, Tensor::concat({priv_columns(G_, keep), G_even, GI_}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
