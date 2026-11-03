// linear_map - the image of the whole space under a matrix, as CORA's fullspace.mtimes
//
// Syntax:   Set R = F.linear_map(M);
// Inputs:   M - a matrix (m, n), or a 1 x 1 matrix (a scalar, whatever n)
// Outputs:  R - a scalar: the origin {0} (an Interval) for 0, else F. A matrix: the box with the
//           zero rows of M at 0 and the other coordinates unbounded if M has a zero row, else the
//           whole space of dimension m. This is an enclosure, and exact only for M of full row
//           rank, as CORA. Error::not_supported in R^0; an Interval factor (CORA: an
//           intervalMatrix) is Error::noops
// See also: Interval::linear_map
//
// CORA: contSet/@fullspace/mtimes.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Fullspace::linear_map(const Tensor &M) const {
    detail::Entry entry;
    const bool scalar = M.numel() == 1;
    CORA_CHECK(M.ndim() == 2 && (scalar || M.size(-1) == n_), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix must have " + std::to_string(n_) +
               " columns, got the shape " + shape_text(M.shape());
    });
    CORA_CHECK(n_ > 0, Error::not_supported,
               [] { return std::string("linear_map: the linear map of R^0 is not supported"); });
    if (scalar) {
        if (!within_tol(M.item(), 0.0, machine_eps(M.dtype()))) return *this;
        const Tensor origin = M.zeros_like({n_, 1});
        return Interval(origin, origin);
    }
    // rows that are entirely zero stay at 0, all others are unbounded
    const int64_t m = M.size(-2);
    const Tensor zero_row = M.abs().max(-1, true).equal(M.zeros_like({m, 1}));
    if (zero_row.max().item() == 0) return m == n_ ? Set(*this) : Set(Fullspace(m));
    const double kInf = std::numeric_limits<double>::infinity();
    const Tensor zero = M.zeros_like({m, 1});
    return Interval(Tensor::where(zero_row, zero, zero.full_like({m, 1}, -kInf)),
                    Tensor::where(zero_row, zero, zero.full_like({m, 1}, kInf)));
}

Set Fullspace::linear_map(const Interval &) const {
    detail::Entry entry;
    CORA_FAIL(Error::noops, "linear_map: the product of an interval matrix and the whole space "
                            "is not implemented (CORA: CORA:noops)");
}

Set Fullspace::linear_map_impl(const Tensor &M) const { return linear_map(M); }

/// A translation leaves the whole space as it is.
Set Fullspace::translate_impl(const Tensor &) const { return *this; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
