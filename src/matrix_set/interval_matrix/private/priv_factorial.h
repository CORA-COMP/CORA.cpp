// priv_factorial - n! as a double
//
// Syntax:   double f = priv_factorial(5);
// Inputs:   n - a nonnegative integer
// Outputs:  f - the product 1 * 2 * ... * n; Inf from n = 171, as MATLAB's factorial
// See also: expm, exponential_remainder
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <cstdint>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

inline double priv_factorial(int64_t n) {
    double f = 1.0;
    for (int64_t i = 2; i <= n; ++i) f *= static_cast<double>(i);
    return f;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
