// empty - the empty polytope of n dimensions, as CORA's polytope.empty
//
// Syntax:   Polytope P = Polytope::empty(n);
// Inputs:   n - dimension, n >= 0 (default 0)
// Outputs:  P - flagged in the empty mask: the row 0 <= -1 (none for n = 0), no vertices (0, n)
// See also: inf, Polytope::Polytope
//
// CORA: contSet/@polytope/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Polytope Polytope::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "empty: the dimension must be nonnegative, got " + std::to_string(n);
    });
    const int64_t rows = std::min<int64_t>(n, 1);
    return Polytope(Parts{Tensor::zeros({rows, n}), Tensor::zeros({rows}) - 1.0,
                          Tensor::zeros({0, n}), Tensor::zeros({0}), Tensor::zeros({0, n}),
                          EmptyMask::all_members()});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
