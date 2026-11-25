// select_rows - some rows of a polytope, as CORALean's restrict
//
// The inequality rows at `rows` and the equality rows at `eq_rows`, in the order given; the others
// go. A polytope in vertex form is converted by constraints first (not a batch). The empty-set
// flag stays if every row is kept (a dropped row may be what makes the set empty).
//
// Syntax:   Pres = P.select_rows(rows, eq_rows);
// Inputs:   P - polytope with m inequalities and me equalities
//           rows - positions in [0, m), eq_rows - positions in [0, me), 0-based
// Outputs:  Pres - the polytope of the kept rows, which contains P
// See also: select_cols, stack_rows, compact
//
// CORA.cpp only: CORALean's HPolytope.restrict, the factor rows of a constrained zonotope
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether every position is in [0, count).
bool aux_in_range(const std::vector<int64_t> &positions, int64_t count) {
    return std::all_of(positions.begin(), positions.end(),
                       [count](int64_t i) { return i >= 0 && i < count; });
}

/// The rows of M (..., rows, n) and of its offsets o (..., rows) at the positions.
std::pair<Tensor, Tensor> aux_rows(const Tensor &M, const Tensor &o,
                                   const std::vector<int64_t> &positions) {
    if (positions.empty()) return {M.slice(-2, 0, 0), o.slice(-1, 0, 0)};
    return {M.take(positions, -2), o.take(positions, -1)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::select_rows(const std::vector<int64_t> &rows,
                               const std::vector<int64_t> &eq_rows) const {
    detail::Entry entry;
    const Constraints C = constraints();
    const int64_t m = C.b.size(-1), me = C.be.size(-1);
    CORA_CHECK(aux_in_range(rows, m) && aux_in_range(eq_rows, me), Error::wrong_value, [&] {
        return "select_rows: the positions must be in [0, " + std::to_string(m) + ") and [0, " +
               std::to_string(me) + ")";
    });
    auto [A, b] = aux_rows(C.A, C.b, rows);
    auto [Ae, be] = aux_rows(C.Ae, C.be, eq_rows);
    // a dropped row may be the one that makes the set empty
    const bool all = int64_t(rows.size()) == m && int64_t(eq_rows.size()) == me;
    return Polytope(Parts{std::move(A), std::move(b), std::move(Ae), std::move(be), Tensor(),
                          all ? empty_ : EmptyMask()});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
