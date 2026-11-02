// within_tol - whether two numbers agree up to a tolerance, absolutely or relatively
//
// The one comparison behind every `tol` argument, as CORA's withinTol: a and b agree if
// |a - b| <= tol, or |a - b| / min(|a|, |b|) <= tol, or both are the same infinity.
//
// Syntax:   bool same = within_tol(a, b);   within_tol(a, b, 1e-12);
// Inputs:   a, b - numbers;  tol - nonnegative tolerance (CORA's default 1e-8)
// Outputs:  res - true if a and b agree up to tol
// See also: cont_set/interval/interval.h (contains)
//
// CORA: global/functions/matlab/validate/check/withinTol.m

#pragma once

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// |a - b| <= tol, absolutely or relative to the smaller magnitude; equal infinities agree.
inline bool within_tol(double a, double b, double tol = 1e-8) noexcept {
    const double difference = std::abs(a - b);
    if (difference <= tol) return true;
    if (difference <= tol * std::fmin(std::abs(a), std::abs(b))) return true;
    return std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
