// support_function - the support function of a box, as CORA's interval.supportFunc
//
// Syntax:   rho = I.support_function(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in I} d'x = d'c + |d|'r, shape (..., 1, 1)
// See also: Zonotope::support_function
//
// CORA: contSet/@interval/supportFunc_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::support_function_impl(const Tensor &d) const {
    // the direction is a column; the box is one too
    const Tensor c = center().expand_dims(-1), r = half_width().expand_dims(-1);
    return d.matrix_transpose().matmul(c) + d.abs().matrix_transpose().matmul(r);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
