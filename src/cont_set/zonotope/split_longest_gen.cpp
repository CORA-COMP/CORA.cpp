// split_longest_gen - the two halves along the longest generator
//
// Syntax:   [Z1, Z2] = Z.split_longest_gen();
// Inputs:   -
// Outputs:  Z1, Z2 - the halves along the generator g_j of largest squared length (the first of
//           equal ones), the centers c + g_j/2 and c - g_j/2 in CORA's order, g_j halved in both:
//           they cover Z. Without generators both are Z. Not a batch
// See also: split, split_first_gen
//
// CORA: contSet/@zonotope/splitLongestGen.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<Zonotope, Zonotope> Zonotope::split_longest_gen() const {
    detail::Entry entry;
    CORA_CHECK(G_.ndim() == 2, Error::not_supported, [] {
        return std::string("split_longest_gen: a generator per zonotope: it supports a single "
                           "zonotope, not a batch");
    });
    if (G_.shape().back() == 0) return {*this, *this};
    const Tensor length = G_.detach().square().sum(-2); // the choice carries no gradient
    const auto [lower, upper] = priv_split_generator(*this, int64_t(length.argmax(0).item()));
    return {upper, lower};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
