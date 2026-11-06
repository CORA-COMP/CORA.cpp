// radius - the Euclidean radius of a box, as CORA's interval.radius
//
// The radius of the smallest ball about the center that holds the box.
//
// Syntax:   Tensor r = I.radius();
// Inputs:   -
// Outputs:  r - the 2-norm of rad(I) over all entries, (..., 1, 1)
// See also: rad, norm
//
// CORA: contSet/@interval/radius.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::radius() const {
    detail::Entry entry;
    const Tensor r = rad();
    return r.square().sum(-1, true).sum(-2, true).sqrt();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
