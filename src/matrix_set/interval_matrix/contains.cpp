// contains - whether a matrix is in an interval matrix, as CORA's intervalMatrix.contains
//
// Syntax:   bool b = A.contains(M);   bool b = A.contains(M, "exact", tol);
// Inputs:   M - matrix (..., rows, cols);  type - "exact" or "approx", the same for numbers
//           tol - how far an entry may lie outside, absolutely or relative to the bound (as
//                 within_tol); CORA's default is 1e-12, here the repository tolerance
// Outputs:  b - true if every entry satisfies inf - tol <= M <= sup + tol; for a batch, if all do
// See also: Interval::contains
//
// CORA: matrixSet/@intervalMatrix/contains.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/within_tol.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// 1 where a < b + slack, or a and b agree up to tol.
Tensor aux_below(const Tensor &a, const Tensor &b, double slack, double tol) {
    return a.less(b + slack).maximum(within_tol(a, b, tol));
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool IntervalMatrix::contains(const Tensor &M, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, int_.inf().dtype(), "contains");
    CORA_CHECK(type == "exact" || type == "approx", Error::wrong_value, [&] {
        return "contains: the type must be 'exact' or 'approx', got '" + type + "'";
    });
    CORA_CHECK(M.ndim() >= 2 && M.size(-2) == rows() && M.size(-1) == cols(),
               Error::dimension_mismatch, [&] {
                   return "contains: the matrix must have " + std::to_string(rows()) +
                          " rows and " + std::to_string(cols()) + " columns, got the shape " +
                          shape_text(M.shape());
               });
    // the lower bound is below M (+ tol) and the upper one above M (- tol), entry by entry
    const Tensor above = aux_below(int_.inf(), M, tol, tol);
    const Tensor below = aux_below(-int_.sup(), -M, tol, tol);
    return above.minimum(below).min().item() != 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
