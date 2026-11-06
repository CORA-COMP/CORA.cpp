// generator_length - the Euclidean length of every generator
//
// Syntax:   len = Z.generator_length();
// Inputs:   -
// Outputs:  len - a row (..., 1, m): the 2-norm of each column of the generators
// See also: compact, reduce
//
// CORA: contSet/@zonotope/generatorLength.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::generator_length() const {
    detail::Entry entry;
    return G_.vector_norm(Tensor::Norm::two, -2, true);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
