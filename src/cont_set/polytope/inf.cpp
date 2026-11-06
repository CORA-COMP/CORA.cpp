// inf - the polytope R^n of n dimensions, as CORA's polytope.Inf
//
// CORA also stores its 2^n vertices at infinity for n <= 8; vertices are not ported, so none are
// held.
//
// Syntax:   Polytope P = Polytope::inf(n);
// Inputs:   n - dimension, n >= 0 (default 0)
// Outputs:  P - the polytope without constraints, which every point satisfies
// See also: empty, Polytope::box
//
// CORA: contSet/@polytope/Inf.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Polytope Polytope::inf(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value,
               [&] { return "inf: the dimension must be nonnegative, got " + std::to_string(n); });
    return Polytope(Parts{Tensor::zeros({0, n}), Tensor::zeros({0, 1}), Tensor::zeros({0, n}),
                          Tensor::zeros({0, 1}), Tensor()});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
