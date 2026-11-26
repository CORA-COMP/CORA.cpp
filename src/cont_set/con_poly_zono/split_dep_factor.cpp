// split_dep_factor - the halves of the set along a factor, as CORA's splitDepFactor
//
// The factor id restricted to [-1, 0] and to [0, 1] (get_subset at the centers -1/2 and 1/2 of
// radius 1/2, CORALean's splitDepFactor): the halves cover the set.
//
// Syntax:   [S1, S2] = cPZ.split_dep_factor(id);
// Inputs:   id - an identifier of the set
// Outputs:  S1, S2 - the halves
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/splitDepFactor.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<ConPolyZono, ConPolyZono> ConPolyZono::split_dep_factor(int64_t id) const {
    detail::Entry entry;
    CORA_CHECK(priv_find_id(ids(), id) >= 0, Error::wrong_value, [&] {
        return "split_dep_factor: " + std::to_string(id) + " is not an identifier of the set";
    });
    const Tensor lo = c().full_like({1}, -1.0), mid = c().zeros_like({1}), hi = c().ones_like({1});
    return {get_subset(id, Interval(lo, mid)), get_subset(id, Interval(mid, hi))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
