// dim - the dimension of a polytope, as CORA's polytope.dim
//
// Syntax:   int64_t n = P.dim();
// Inputs:   -
// Outputs:  n - the columns of the rows A, or the rows of the vertices V
// See also: ContSet::dim
//
// CORA: contSet/@polytope/dim.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

int64_t Polytope::dim_impl() const {
    return is_h_rep() ? A_.shape().back() : V_.shape()[V_.ndim() - 2];
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
