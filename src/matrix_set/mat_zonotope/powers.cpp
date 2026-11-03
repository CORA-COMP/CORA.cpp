// powers - the powers of a square matrix zonotope from an initial one, as CORA's powers
//
// Syntax:   std::vector<MatZonotope> P = A.powers(max_order);
//           P = A.powers(max_order, initial_order, initial_power);
// Inputs:   max_order - the last order;  initial_order - the order of initial_power (at least 1);
//           initial_power - a matrix zonotope of A's shape (default: A, with initial_order 1)
// Outputs:  P - P[k] is the power of order initial_order + k: initial_power, then every next one
//           the previous times A from the right, up to max_order (just initial_power if max_order
//           is below the initial order, as CORA's cell)
// See also: matrix_power, matmul
//
// CORA: matrixSet/@matZonotope/powers.m

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<MatZonotope> MatZonotope::powers(int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "powers: the matrix zonotope must be square, got the shape " +
               shape_text(C_.shape());
    });
    return powers(max_order, 1, *this);
}

// the form with an initial power checks that it is a square set of the same size
std::vector<MatZonotope> MatZonotope::powers(int64_t max_order, int64_t initial_order,
                                             const MatZonotope &initial_power) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols() && initial_power.rows() == rows() && initial_power.cols() == cols(),
               Error::dimension_mismatch, [&] {
                   return "powers: the matrix zonotope and the initial power must be square of one "
                          "size, got " +
                          shape_text(C_.shape()) + " and " + shape_text(initial_power.C_.shape());
               });
    CORA_CHECK(initial_order >= 1, Error::wrong_value, [&] {
        return "powers: the initial order must be at least 1, got " + std::to_string(initial_order);
    });
    // each power is the previous one times A from the right, as CORA's loop
    std::vector<MatZonotope> result{initial_power};
    for (int64_t order = initial_order + 1; order <= max_order; ++order)
        result.push_back(result.back().matmul(*this));
    return result;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
