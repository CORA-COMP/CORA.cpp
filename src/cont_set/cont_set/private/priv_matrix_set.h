// priv_matrix_set - whether a set holds matrix-shaped points
//
// Syntax:   bool m = priv_is_matrix_set(S);
// Inputs:   S - set
// Outputs:  m - whether S is an interval matrix or a matrix zonotope (elements of rank 2)
// See also: ContSet::project, ContSet::vertices, ContSet::vertices_3d
//
// CORA.cpp only: CORA keeps matrix sets in classes of their own
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_is_matrix_set(const ContSet &S);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
