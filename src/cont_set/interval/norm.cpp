// norm - the norm of the largest member of a box, as CORA's interval.norm_
//
// For type two (default) the 2-norm of max(|inf|, |sup|) of a column; a matrix is refused, as by
// CORA's intervalMatrix (CORA's interval takes the spectral norm). Else that norm of
// |center| + rad. CORA's mode argument is ignored by intervals and not offered.
//
// Syntax:   Tensor v = I.norm();   Tensor v = I.norm(type)   with type one, two, inf or fro
// Inputs:   -
// Outputs:  v - (...); -Inf for an empty interval
// See also: rad, center
//
// CORA: contSet/@interval/norm_.m
// CORA: matrixSet/@intervalMatrix/norm.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::norm(Tensor::Norm type) const {
    detail::Entry entry;
    const bool column = elem_rank_ == 1 || inf_.size(-1) == 1;
    CORA_CHECK(type != Tensor::Norm::two || column, Error::wrong_value, [] {
        return std::string("norm: the 2-norm of an interval matrix is not supported, "
                           "use 1, Inf or 'fro'");
    });
    if (is_empty()) return inf_.full_like(batch_shape(), -std::numeric_limits<double>::infinity());
    if (type != Tensor::Norm::two) {
        // the matrix norm of the bounds as matrices (a box is a column)
        const Tensor largest = center().abs() + half_width();
        const Tensor M = elem_rank_ == 1 ? largest.expand_dims(-1) : largest;
        return M.matrix_norm(type).squeeze(-1).squeeze(-1);
    }
    // a column is a vector: its 2-norm
    const Tensor largest = inf_.abs().maximum(sup_.abs());
    if (elem_rank_ == 1) return largest.vector_norm(Tensor::Norm::two, -1);
    return largest.vector_norm(Tensor::Norm::two, -2).squeeze(-1);
}

Tensor Interval::norm_impl(Tensor::Norm type, const std::string &) const { return norm(type); }


} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
