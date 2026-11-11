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
    // the even generators are split into center and half generator, the odd ones stay
    std::vector<int64_t> even, odd;
    for (int64_t j = 0; j < h; ++j) (priv_is_even(E_[std::size_t(j)]) ? even : odd).push_back(j);
    if (even.empty()) return {c_, Tensor::concat({G_, GI_}, -1)};
    const Tensor G_even = priv_columns(G_, even) * 0.5;
    const Tensor c = c_ + G_even.sum(-1, true);
    return {c, Tensor::concat({priv_columns(G_, odd), G_even, GI_}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
