// powers - the powers of a square matrix zonotope from an initial one, as CORA's powers
//
// The initial_power, then every next one the previous times A from the right (just initial_power
// if max_order is below the initial order, as CORA's cell).
//
// Syntax:   std::vector<Zonotope> P = mat_exp::powers(A, max_order);
//           P = mat_exp::powers(A, max_order, initial_order, initial_power);
// Inputs:   max_order - the last order
//           initial_order - the order of initial_power (at least 1);
//           initial_power - a matrix zonotope of A's shape (default: A, with initial_order 1)
// Outputs:  P - P[k] is the power of order initial_order + k, up to max_order
// See also: matrix_power, matmul
//
// CORA: matrixSet/@matZonotope/powers.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Zonotope> mat_exp::powers(const Zonotope &A, int64_t max_order) {
    detail::Entry entry;
    CORA_CHECK(A.center().size(-2) == A.center().size(-1), Error::dimension_mismatch, [&] {
        return "powers: the matrix zonotope must be square, got the shape " +
               shape_text(A.center().shape());
    });
    return powers(A, max_order, 1, A);
}

// the form with an initial power checks that it is a square set of the same size
std::vector<Zonotope> mat_exp::powers(const Zonotope &A, int64_t max_order, int64_t initial_order,
                                      const Zonotope &initial_power) {
    detail::Entry entry;
    CORA_CHECK(A.center().size(-2) == A.center().size(-1) &&
                   initial_power.center().size(-2) == A.center().size(-2) &&
                   initial_power.center().size(-1) == A.center().size(-1),
               Error::dimension_mismatch, [&] {
                   return "powers: the matrix zonotope and the initial power must be square of one "
                          "size, got " +
                          shape_text(A.center().shape()) + " and " +
                          shape_text(initial_power.center().shape());
               });
    CORA_CHECK(initial_order >= 1, Error::wrong_value, [&] {
        return "powers: the initial order must be at least 1, got " + std::to_string(initial_order);
    });
    // each power is the previous one times A from the right, as CORA's loop
    std::vector<Zonotope> result{initial_power};
    for (int64_t order = initial_order + 1; order <= max_order; ++order)
        result.push_back(result.back().matmul(A));
    return result;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
