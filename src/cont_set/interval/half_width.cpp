// half_width - the half-width of each coordinate of a box, as CORA's interval.rad
//
// Syntax:   Tensor r = I.half_width();
// Inputs:   -
// Outputs:  r - (sup - inf) / 2 elementwise, of the shape of the bounds
// See also: center, ball_radius
//
// CORA: contSet/@interval/rad.m
// CORA: matrixSet/@intervalMatrix/rad.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::half_width() const {
    detail::Entry entry;
    return (sup_ - inf_) * 0.5;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
