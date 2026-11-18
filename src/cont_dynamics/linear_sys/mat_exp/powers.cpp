// powers - the powers of an interval matrix from a given order, as CORA's intervalMatrix.powers
//
// The initial power, then each one the previous times A, from the right (CORA's pow{i} = pow{i-1}
// * A; matrix_power multiplies from the left), every product the interval matrix product.
//
// Syntax:   std::vector<Interval> P = mat_exp::powers(A, N);
//           P = mat_exp::powers(A, N, initial_order, initial_power);
// Inputs:   N - the last order
//           initial_order >= 1, initial_power - the power of that order (default: 1 and A)
// Outputs:  P - P[k] is the power of order initial_order + k, up to N (at least the first)
// See also: matrix_power, matmul, expm
//
// CORA: matrixSet/@intervalMatrix/powers.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// a forwarding body: the three-argument form is the checking entry
std::vector<Interval> mat_exp::powers(const Interval &A, int64_t m) { return powers(A, m, 1, A); }

std::vector<Interval> mat_exp::powers(const Interval &A, int64_t max_order, int64_t initial_order,
                                      const Interval &initial_power) {
    detail::Entry entry;
    CORA_CHECK(initial_order >= 1, Error::wrong_value, [&] {
        return "powers: the initial order must be at least 1, got " + std::to_string(initial_order);
    });
    CORA_CHECK(max_order <= initial_order || (A.inf().size(-2) == A.inf().size(-1) &&
                                              initial_power.inf().size(-1) == A.inf().size(-2)),
               Error::dimension_mismatch, [&] {
                   return "powers: A must be square and the initial power must have " +
                          std::to_string(A.inf().size(-2)) + " columns, got " +
                          shape_text(A.inf().shape()) + " and " +
                          shape_text(initial_power.inf().shape());
               });
    std::vector<Interval> pow{initial_power};
    for (int64_t order = initial_order + 1; order <= max_order; ++order)
        pow.push_back(pow.back().matmul(A));
    return pow;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
