// expm_taylor_horner_scaling - the matrix exponential by Taylor, Horner or scaling and squaring
//
// A is square. Each product is the interval matrix product; for style 2 the result is raised to
// the power 2^p. f = (a^(N+1) / (N+1)!) / (1 - a / (N + 2)) with a = norm(A, Inf); nothing
// (CORA's []) where a >= N + 2 (for scaling: where a >= (N + 2) 2^p), for a batch where it holds
// for any member. CORA's box for a point matrix is the point f, which excludes exp(A); here it is
// [-f, f] always (docs/cora-differences.md). Another style: refused.
//
// Syntax:   std::optional<Interval> E = mat_exp::expm_taylor_horner_scaling(A, N, style);
//           E = mat_exp::expm_taylor_horner_scaling(A, N, 2, scaling_exponent);
// Inputs:   N - the order of the series (>= 1 for the Horner styles)
//           style - 0 the Taylor series, 1 Horner's scheme, 2 Horner on [A] / 2^p
//           scaling_exponent - p >= 0, default 1, for style 2
// Outputs:  E - the series to order N plus the box [-f, f]; nothing where a >= N + 2
// See also: expm_different_algorithms, expm, expm_independent
//
// CORA: matrixSet/@intervalMatrix/expmTaylorHornerScaling.m
// CORA: matrixSet/@intervalMatrix/private/priv_taylorSeries.m
// CORA: matrixSet/@intervalMatrix/private/priv_hornerTaylorSeries.m
// CORA: matrixSet/@intervalMatrix/private/priv_scalingSquaringHornerTaylorSeries.m
// CORA: matrixSet/@intervalMatrix/private/priv_intervalMatrixRemainder.m
// CORA authors: Ivan Brkan
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_expm_series.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Helpers -----------------------------------------------------------------------------------------

/// Whether the series of order N converges for the norm bound a (CORA: a < N + 2) in every member.
bool aux_converges(const Tensor &a, int64_t N) { return a.max().item() < double(N + 2); }

/// The Taylor remainder box [-f, f] (CORA's priv_intervalMatrixRemainder) for the norm bound a.
Interval aux_remainder(const Interval &A, const Tensor &a, int64_t N) {
    Tensor factor = a.ones_like();
    for (int64_t i = 1; i <= N + 1; ++i) factor = factor.multiply(a / double(i));
    factor = factor.divide((a / double(N + 2)).scalar_op(Tensor::Binary::subtract, 1.0, true));
    const Tensor zero = A.inf().zeros_like();
    // the norm bound is one number per batch member, (...); the matrix is (..., n, n)
    return Interval::from_center_delta(
        zero, zero.ones_like().multiply(factor.expand_dims(-1).expand_dims(-1)));
}

/// The Taylor series: sum of the terms [A]^i / i!, i <= N, each from the previous by one product.
Interval aux_taylor(const Interval &A, const Tensor &a, int64_t N) {
    const Tensor eye = A.inf().eye_like(A.inf().size(-2));
    const Tensor zeros = eye.zeros_like();
    Interval taylor = Interval::matrix(zeros, zeros), pow = Interval::matrix(eye, eye);
    for (int64_t i = 1; i <= N; ++i) {
        taylor = taylor.plus(pow);
        // the first product has the matrix of numbers I on the left, as CORA's
        pow = (i == 1 ? A.linear_map(eye) : pow.matmul(A)) * (1.0 / double(i));
    }
    return taylor.plus(pow).plus(aux_remainder(A, a, N));
}

/// Horner's scheme, I + A/N (I + A/(N-1) (... (I + A))) from the inside out.
Interval aux_horner(const Interval &A, const Tensor &a, int64_t N) {
    const Tensor eye = A.inf().eye_like(A.inf().size(-2));
    Interval horner = (A * (1.0 / double(N))) + eye;
    for (int64_t i = 1; i < N; ++i) horner = (A.matmul(horner) * (1.0 / double(N - i))) + eye;
    return horner.plus(aux_remainder(A, a, N));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Series ------------------------------------------------------------------------------------------

std::optional<Interval> mat_exp::expm_taylor_horner_scaling(const Interval &A, int64_t max_order,
                                                            int64_t style,
                                                            int64_t scaling_exponent) {
    detail::Entry entry;
    if (entry.check()) mat_exp::check_series_input(A, max_order, style, scaling_exponent, false);
    const Tensor a = A.norm(Tensor::Norm::inf);
    if (style == 0 || style == 1) {
        if (!aux_converges(a, max_order)) return std::nullopt;
        return style == 0 ? aux_taylor(A, a, max_order) : aux_horner(A, a, max_order);
    }
    // scaling and squaring: Horner of [A] / 2^p, to the power 2^p
    const double potential = std::ldexp(1.0, int(scaling_exponent));
    if (a.max().item() >= double(max_order + 2) * potential) return std::nullopt;
    const Interval scaled = A * (1.0 / potential);
    const Interval horner = aux_horner(scaled, scaled.norm(Tensor::Norm::inf), max_order);
    return priv_power(horner, int64_t(potential));
}

void mat_exp::check_series_input(const Interval &A, int64_t max_order, int64_t style,
                                 int64_t scaling_exponent, bool allow_expm) {
    detail::Validation entry;
    CORA_CHECK(A.inf().size(-2) == A.inf().size(-1), Error::dimension_mismatch, [&] {
        return "expm: the matrix must be square, got the shape " + shape_text(A.inf().shape());
    });
    // the styles of CORA's switch: 4 only where expm is one of them
    CORA_CHECK(style == 0 || style == 1 || style == 2 || (allow_expm && style == 4),
               Error::wrong_value, [&] {
                   return "expm: the style must be 0 (Taylor), 1 (Horner), 2 (scaling)" +
                          std::string(allow_expm ? " or 4 (expm)" : "") + ", got " +
                          std::to_string(style);
               });
    // Horner divides by the order
    const int64_t lowest = style == 1 || style == 2 ? 1 : 0;
    CORA_CHECK(max_order >= lowest, Error::wrong_value, [&] {
        return "expm: the order must be at least " + std::to_string(lowest) + ", got " +
               std::to_string(max_order);
    });
    CORA_CHECK(scaling_exponent >= 0 && scaling_exponent <= 20, Error::wrong_value, [&] {
        return "expm: the scaling exponent must be in 0..20, got " +
               std::to_string(scaling_exponent);
    });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
