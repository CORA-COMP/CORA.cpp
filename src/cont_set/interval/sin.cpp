// sin - the sine of a box, as CORA's interval.sin
//
// Syntax:   Interval R = I.sin();
// Inputs:   -
// Outputs:  R - cos(I - pi/2), with the case analysis of cos on the multiples of 2 pi. Nothing
//           rounds outward, as in CORA
// See also: cos, sinh
//
// CORA: contSet/@interval/sin.m

#include "cora/cont_set/interval/interval.h"

#include <numbers>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::sin() const {
    detail::Entry entry;
    constexpr double kHalfPi = std::numbers::pi / 2;
    return Interval(inf_ - kHalfPi, sup_ - kHalfPi).cos();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
