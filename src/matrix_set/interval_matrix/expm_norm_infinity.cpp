// expm_norm_infinity - CORA's bound of the norm of the mapping mimicking exp, expmNormInf
//
// Syntax:   Tensor v = A.expm_norm_infinity(t);
// Inputs:   t >= 0 - the time
// Outputs:  v - (..., 1, 1): s2 + s1 with s2 = rS t / (1 - rAS t / 2) and
//           s1 = (rA rS t^2 / 2) / (rA rAS t^2 / 4 - (rA + rAS) t / 2 + 1), the infinity norms
//           rA of the center, rS of the radius and rAS of |center| + radius; CORA's formula,
//           which is not checked for a vanishing denominator
// See also: expm_absolute_bound, infinity_norm
//
// CORA: matrixSet/@intervalMatrix/expmNormInf.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::expm_norm_infinity(double t) const {
    detail::Entry entry;
    CORA_CHECK(t >= 0, Error::wrong_value, [&] {
        return "expm_norm_infinity: the time must be nonnegative, got " + std::to_string(t);
    });
    const Tensor C = center(), S = rad();
    const Tensor norm_c = C.matrix_norm(Tensor::Norm::inf);
    const Tensor norm_s = S.matrix_norm(Tensor::Norm::inf);
    const Tensor norm_cs = (C.abs() + S).matrix_norm(Tensor::Norm::inf);
    const double j = 2.0;
    // the quotient of the two terms, with CORA's order of operations
    const Tensor numerator = norm_c.multiply(norm_s) * (t * t) / j;
    const Tensor denominator =
        norm_c.multiply(norm_cs) * (t * t) / (j * j) - (norm_c + norm_cs) * t / j + 1.0;
    const Tensor summand1 = numerator.divide(denominator);
    const Tensor one_minus = (norm_cs * t / j).scalar_op(Tensor::Binary::subtract, 1.0, true);
    return summand1 + (norm_s * t).divide(one_minus);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
