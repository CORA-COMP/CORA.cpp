// gram_schmidt - an orthonormal basis that starts with a given direction, as CORA's gramSchmidt
//
// The sets build frames from it: the zonotope simplex, the bundle of a constrained zonotope or a
// polytope, the polytope volume and the flow basis of a location.
//
// Syntax:   std::vector<std::vector<double>> Q = gram_schmidt(a);
// See also: global/blocks.h
//
// CORA.cpp only: the declaration of gram_schmidt (gram_schmidt.cpp tags CORA's gramSchmidt)
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#pragma once

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The orthonormal basis (n vectors of n numbers) that Gram-Schmidt makes of a and the axes without
/// the one a is most aligned with (CORA's gramSchmidt of a column): the first is a / |a|.
std::vector<std::vector<double>> gram_schmidt(const std::vector<double> &a);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
