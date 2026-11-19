// priv_matrix_set - whether a set holds matrix-shaped points
//
// Only Interval (Interval::matrix) and Zonotope (Zonotope::matrix) hold matrices; the centers of
// the other sets are vectors (..., n).
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

#include "cont_set/cont_set/private/priv_matrix_set.h"

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_is_matrix_set(const ContSet &S) {
    if (S.class_id() == ClassId::interval) return static_cast<const Interval &>(S).elem_rank() > 1;
    if (S.class_id() == ClassId::zonotope) return static_cast<const Zonotope &>(S).elem_rank() > 1;
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
