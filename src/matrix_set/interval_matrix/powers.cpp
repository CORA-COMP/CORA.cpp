// powers - the powers of an interval matrix from a given order, as CORA's intervalMatrix.powers
//
// Syntax:   std::vector<IntervalMatrix> P = A.powers(N);
//           P = A.powers(N, initial_order, initial_power);
// Inputs:   N - the last order;  initial_order >= 1, initial_power - the power of that order
//           (default: 1 and A)
// Outputs:  P - P[k] is the power of order initial_order + k, up to N (at least the first):
//           the initial power, then each one the previous times A, from the right (CORA's
//           pow{i} = pow{i-1} * A; matrix_power multiplies from the left), every product the
//           interval matrix product
// See also: matrix_power, matmul, expm
//
// CORA: matrixSet/@intervalMatrix/powers.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// a forwarding body: the three-argument form is the checking entry
std::vector<IntervalMatrix> IntervalMatrix::powers(int64_t m) const { return powers(m, 1, *this); }

std::vector<IntervalMatrix> IntervalMatrix::powers(int64_t max_order, int64_t initial_order,
                                                   const IntervalMatrix &initial_power) const {
    detail::Entry entry;
    CORA_CHECK(initial_order >= 1, Error::wrong_value, [&] {
        return "powers: the initial order must be at least 1, got " + std::to_string(initial_order);
    });
    CORA_CHECK(max_order <= initial_order || (rows() == cols() && initial_power.cols() == rows()),
               Error::dimension_mismatch, [&] {
                   return "powers: A must be square and the initial power must have " +
                          std::to_string(rows()) + " columns, got " +
                          shape_text(int_.inf().shape()) + " and " +
                          shape_text(initial_power.int_.inf().shape());
               });
    std::vector<IntervalMatrix> pow{initial_power};
    for (int64_t order = initial_order + 1; order <= max_order; ++order)
        pow.push_back(pow.back().matmul(*this));
    return pow;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
