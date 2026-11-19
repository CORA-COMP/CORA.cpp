// ball_radius - the Euclidean radius of a box, as CORA's interval.radius
//
// The radius of the smallest ball about the center that holds the box.
//
// Syntax:   Tensor r = I.ball_radius();
// Inputs:   -
// Outputs:  r - the 2-norm of half_width(I) over all entries, (batch...)
// See also: half_width, norm
//
// CORA: contSet/@interval/radius.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::ball_radius() const {
    detail::Entry entry;
    const Tensor squares = half_width().square().sum(-1);
    return (elem_rank_ == 2 ? squares.sum(-1) : squares).sqrt();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
