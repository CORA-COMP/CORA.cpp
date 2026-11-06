// within_tol - whether numbers agree up to a tolerance, absolutely or relatively
//
// The one comparison behind every `tol` argument, as CORA's withinTol: a and b agree if
// |a - b| <= tol, or |a - b| <= tol * min(|a|, |b|), or both are the same infinity. Opposite
// infinities and NaN never agree.
//
// Syntax:   bool same = within_tol(a, b);   within_tol(a, b, 1e-12);
//           Tensor same = within_tol(A, B);   (elementwise, broadcast; 1 where they agree, else 0)
// Inputs:   a, b - numbers, or tensors of one dtype
//           tol - nonnegative, default: tolerance.h
// Outputs:  same - whether (where) a and b agree up to tol
// See also: global/tolerance.h, cont_set/interval/interval.h (contains)
//
// CORA: global/functions/matlab/validate/check/withinTol.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tolerance.h"
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// |a - b| <= tol, absolutely or relative to the smaller magnitude; equal infinities agree.
inline bool within_tol(double a, double b, Tol tol = std::nullopt) noexcept {
    if (a == b) return true; // includes the equal infinities
    if (std::isinf(a) || std::isinf(b)) return false;
    const double t = resolve_tol(tol);
    const double difference = std::abs(a - b);
    return difference <= t || difference <= t * std::fmin(std::abs(a), std::abs(b));
}

/// within_tol of every entry, as a 0/1 mask; tol resolves for the dtype of a.
Tensor within_tol(const Tensor &a, const Tensor &b, Tol tol = std::nullopt);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
