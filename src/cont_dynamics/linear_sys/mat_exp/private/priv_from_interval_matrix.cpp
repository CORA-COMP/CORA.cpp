// priv_from_interval_matrix - an interval matrix as a matrix zonotope
//
// See also: priv_from_interval_matrix.h
//
// CORA.cpp only: a helper of the matrix zonotope constructor from an interval matrix.
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_from_interval_matrix.h"
#include "cont_set/zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope priv_from_interval_matrix(const Interval &I) {
    const int64_t n = I.inf().size(-2), m = I.inf().size(-1);
    const Tensor radius = I.half_width();
    // generator j holds the radius of the entry j (in column order) at that entry
    const Tensor diagonal = priv_stack_columns(radius).diag();
    const Tensor transposed = diagonal.reshape(priv_replace_tail(diagonal, 2, {n * m, m, n}));
    return Zonotope::matrix(I.center(), transposed.matrix_transpose());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
