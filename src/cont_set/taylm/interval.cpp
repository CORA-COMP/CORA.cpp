// interval - the range of a Taylor model, as CORA's taylm.interval with method "int"
//
// The remainder plus, monomial by monomial, the coefficient times the bound of the monomial over
// [-1, 1]^k: [0, 1] for even exponents, [-1, 1] otherwise. The bounds do not cancel between
// monomials, so a - a^2 is bounded by [-2, 1], as CORA does.
//
// Syntax:   [lo, hi] = T.bounds();  I = T.interval();
// Inputs:   -
// Outputs:  lo, hi - the bounds
//           I - the box (1, 1)
// See also: evaluate
//
// CORA: contSet/@taylm/interval.m
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

std::array<double, 2> Taylm::bounds() const {
    detail::Entry entry;
    const std::array<double, 2> p = polynomial_bounds();
    return {rem_lo_ + p[0], rem_hi_ + p[1]};
}

CORA_FROM_NUMBERS Interval Taylm::interval() const {
    detail::Entry entry;
    const std::array<double, 2> b = bounds();
    return {Tensor::from_data({b[0]}, {1, 1}), Tensor::from_data({b[1]}, {1, 1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
