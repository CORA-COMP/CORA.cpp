// priv_gram_schmidt - an orthonormal basis that starts with a given direction
//
// Syntax:   std::vector<double> Q = priv_gram_schmidt(v);
// Inputs:   v - a nonzero vector (p entries)
// Outputs:  Q - the basis as columns, row-major (p, p); the first column is v / |v|
// See also: ConZonotope::zono_bundle, Zonotope::simplex
//
// CORA.cpp only: CORA's gramSchmidt is a helper of its global functions
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

std::vector<double> priv_gram_schmidt(const std::vector<double> &v);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
