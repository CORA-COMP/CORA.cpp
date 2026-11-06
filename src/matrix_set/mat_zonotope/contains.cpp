// contains - whether matrices are in a matrix zonotope, as CORA's matZonotope.contains
//
// CORA's default tol is 1e-12, here the repository tolerance.
//
// Syntax:   bool b = A.contains(M);   bool b = A.contains(M, "exact", tol);
// Inputs:   M - a matrix (rows, cols) or p matrices (p, rows, cols)
//           type - "exact" or "approx"
//           tol - how far a point may lie outside (the zonotope's contains of the stacked matrices)
// Outputs:  b - true if every matrix is in the set
// See also: zonotope, Zonotope::contains
//
// CORA: matrixSet/@matZonotope/contains.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool MatZonotope::contains(const Tensor &M, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, C_.dtype(), "contains");
    CORA_CHECK(type == "exact" || type == "approx", Error::wrong_value, [&] {
        return "contains: the type must be 'exact' or 'approx', got '" + type + "'";
    });
    CORA_CHECK(C_.ndim() == 2, Error::not_supported,
               [] { return std::string("contains: not supported for a batch of sets"); });
    CORA_CHECK((M.ndim() == 2 || M.ndim() == 3) && M.numel() > 0 && M.size(-2) == rows() &&
                   M.size(-1) == cols(),
               Error::dimension_mismatch, [&] {
                   return "contains: the matrices must be (" + std::to_string(rows()) + ", " +
                          std::to_string(cols()) + ") or (p, " + std::to_string(rows()) + ", " +
                          std::to_string(cols()) + "), got the shape " + shape_text(M.shape());
               });
    // the points are the columns of the stacked matrices
    const Tensor matrices = M.ndim() == 2 ? M.expand_dims(0) : M;
    const Tensor points = priv_stack_columns(matrices).squeeze(-1).matrix_transpose();
    return zonotope().contains(points, type, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
