// generator_length - the Euclidean length of every generator
//
// Syntax:   len = Z.generator_length();
// Inputs:   -
// Outputs:  len - a row (..., 1, m): the 2-norm of each column of the generators
// See also: compact, reduce
//
// CORA: contSet/@zonotope/generatorLength.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::generator_length() const {
    detail::Entry entry;
    // layout: #163 step 2 - a row (..., 1, m), as before
    return G_.vector_norm(Tensor::Norm::two, -1).expand_dims(-2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
