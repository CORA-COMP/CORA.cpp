// interval_matrix - the interval matrix of the center and radius, as CORA's interval.intervalMatrix
//
// Syntax:   Interval M = I.interval_matrix();
// Inputs:   -
// Outputs:  M - [c - r, c + r] with the center c and the radius r of I; equal to I up to rounding
//           (an Interval of matrix bounds is CORA's intervalMatrix)
// See also: center, rad
//
// CORA: contSet/@interval/intervalMatrix.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::interval_matrix() const {
    detail::Entry entry;
    const Tensor c = center_impl(), r = rad();
    return {c - r, c + r};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
