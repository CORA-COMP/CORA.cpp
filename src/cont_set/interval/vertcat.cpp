// vertcat - intervals stacked, as CORA's interval.vertcat
//
// Syntax:   Interval I = Interval::vertcat({I1, I2, ...});
// Inputs:   Is - the intervals, which agree in the number of columns
// Outputs:  I - the bounds stacked as rows; nothing rounds (a number is a point interval, made
//           by the caller)
// See also: horzcat, cat (which checks the shapes)
//
// CORA: contSet/@interval/vertcat.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::vertcat(const std::vector<Interval> &Is) {
    detail::Entry entry;
    return cat(1, Is);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
