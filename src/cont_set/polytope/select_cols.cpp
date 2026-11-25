// select_cols - the polytope in some of its coordinates, as CORALean's selectCols
//
// Column cols[l] of every row becomes column l, the other columns go, the offsets stay. For cols
// that hold each position once this is the section of the set at 0 in the other coordinates; a
// position may also repeat (the coordinates are then tied). A polytope in vertex form is
// converted by constraints first (not a batch). The empty-set flag is kept.
//
// Syntax:   Pres = P.select_cols(cols);
// Inputs:   P - polytope of dimension n
//           cols - positions in [0, n), 0-based
// Outputs:  Pres - {y in R^|cols| | x in P for x_cols[l] = y_l and the other x_j = 0}
// See also: select_rows, pad_left, Polytope::project
//
// CORA.cpp only: CORALean's HPolytope.selectCols, the factor rows of a constrained zonotope
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::select_cols(const std::vector<int64_t> &cols) const {
    detail::Entry entry;
    const int64_t n = dim();
    CORA_CHECK(std::all_of(cols.begin(), cols.end(), [n](int64_t j) { return j >= 0 && j < n; }),
               Error::wrong_value, [&] {
                   return "select_cols: the positions must be in [0, " + std::to_string(n) + ")";
               });
    const Constraints C = constraints();
    return Polytope(
        Parts{select_columns(C.A, cols), C.b, select_columns(C.Ae, cols), C.be, Tensor(), empty_});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
