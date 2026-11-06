// norm - the norm of the largest member of a box, as CORA's interval.norm_
//
// For type two (default) the 2-norm of max(|inf|, |sup|), the spectral norm for a matrix; else
// that norm of |center| + rad. CORA's mode argument is ignored by intervals and not offered.
//
// Syntax:   Tensor v = I.norm();   Tensor v = I.norm(type)   with type one, two, inf or fro
// Inputs:   -
// Outputs:  v - (..., 1, 1); -Inf for an empty interval
// See also: rad, center
//
// CORA: contSet/@interval/norm_.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner, Claude Sonnet 5.5
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::norm(Tensor::Norm type) const {
    detail::Entry entry;
    if (is_empty()) return inf_.full_like({1, 1}, -std::numeric_limits<double>::infinity());
    if (type != Tensor::Norm::two) return (center().abs() + rad()).matrix_norm(type);
    const Tensor largest = inf_.abs().maximum(sup_.abs());
    // a column is a vector: its 2-norm, not the singular value of a matrix
    if (largest.size(-1) == 1) return largest.vector_norm(Tensor::Norm::two, -2, true);
    return largest.matrix_norm(Tensor::Norm::two);
}

Tensor Interval::norm_impl(Tensor::Norm type, const std::string &) const { return norm(type); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
