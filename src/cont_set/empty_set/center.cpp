// center - the center of the empty set, as CORA's emptySet.center
//
// Syntax:   Tensor c = E.center();
// Inputs:   -
// Outputs:  c - NaN (n): no point is the center, as the empty box (CORA's is (n, 0))
// See also: Interval::center, rand_point
//
// CORA: contSet/@emptySet/center.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

#include <cstddef>
#include <limits>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Tensor EmptySet::center_impl() const {
    const std::size_t n = static_cast<std::size_t>(n_);
    return Tensor::from_data(std::vector<double>(n, std::numeric_limits<double>::quiet_NaN()),
                             {n_});
}


} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
