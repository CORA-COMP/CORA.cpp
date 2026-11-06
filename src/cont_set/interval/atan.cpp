// atan - the arctangent of a box, as CORA's interval.atan
//
// Syntax:   Interval R = I.atan();
// Inputs:   -
// Outputs:  R - [atan(inf), atan(sup)]
// See also: atan2, Tensor::atan
//
// CORA: contSet/@interval/atan.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::atan() const {
    detail::Entry entry;
    return {inf_.atan(), sup_.atan()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
