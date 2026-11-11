// power - the integer power of a Taylor model, as CORA's taylm.power and mpower
//
// x^n by repeated products (squaring), so the dependency of the factors stays: (a - a)^2 is
// exactly 0 up to the remainder. n = 0 gives 1.
//
// Syntax:   T = A.power(n);
// Inputs:   n - an integer exponent, n >= 0
// Outputs:  T - the model
// See also: times
//
// CORA: contSet/@taylm/power.m
// CORA: contSet/@taylm/mpower.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Taylm Taylm::power(int n) const {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value,
               [&] { return "power: the exponent must be nonnegative, got " + std::to_string(n); });
    Taylm result = constant(1.0, vars_, max_order_), base = *this;
    // binary exponentiation, one multiplication per bit
    for (int k = n; k > 0; k >>= 1) {
        if (k & 1) result = result.times(base);
        if (k > 1) base = base.times(base);
    }
    return result;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
