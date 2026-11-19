// origin - the polytope that is the origin, as CORA's polytope.origin
//
// It holds both forms: the equalities x = 0 and the vertex 0. CORA's halfspaces are the simplex
// x >= 0, sum x <= 0, which is the same set; the equalities are CORALean's form.
//
// Syntax:   Polytope P = Polytope::origin(n);
// Inputs:   n - dimension, an integer >= 1
// Outputs:  P - {0} in R^n
// See also: Polytope::empty, Polytope::inf
//
// CORA: contSet/@polytope/origin.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Polytope Polytope::origin(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value, [&] {
        return "origin: the dimension must be a positive integer, got " + std::to_string(n);
    });
    return Polytope(Parts{Tensor::zeros({0, n}), Tensor::zeros({0}), Tensor::eye(n),
                          Tensor::zeros({n}), Tensor::zeros({1, n})});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
