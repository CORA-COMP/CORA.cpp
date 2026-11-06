// reduce - a matrix zonotope with fewer generators, as CORA's matZonotope.reduce
//
// The zonotope of the stacked matrices is reduced (Zonotope::reduce) and read back as matrices of
// rows x cols. CORA's returns column matrices (rows * cols, 1), which loses the shape
// (docs/cora-differences.md). Girard's method only; not a batch.
//
// Syntax:   MatZonotope R = M.reduce(order);   MatZonotope R = M.reduce(order, "girard");
// Inputs:   order - generators per entry, at least 1
//           method - "girard"
// Outputs:  R - a matrix zonotope that contains M
// See also: zonotope, Zonotope::reduce
//
// CORA: matrixSet/@matZonotope/reduce.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope MatZonotope::reduce(int order, const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the order must be at least 1, got " + std::to_string(order);
    });
    CORA_CHECK(method == "girard", Error::not_supported, [&] {
        return "reduce: the method '" + method + "' is not supported; use \"girard\"";
    });
    CORA_CHECK(C_.ndim() == 2, Error::not_supported,
               [] { return std::string("reduce: it supports a single set, not a batch"); });
    const Zonotope Z = zonotope().reduce(order);
    // back to rows x cols: the columns of the stacked vectors are matrices read column by column
    const int64_t h = Z.generators().shape().back();
    return {Z.center().reshape({cols(), rows()}).matrix_transpose(),
            Z.generators().matrix_transpose().reshape({h, cols(), rows()}).matrix_transpose()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
