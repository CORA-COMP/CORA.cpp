// exp - the exponential of a box, as CORA's interval.exp
//
// Syntax:   Interval R = I.exp();
// Inputs:   -
// Outputs:  R - [exp(inf), exp(sup)]
// See also: cos, cosh
//
// CORA: contSet/@interval/exp.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::exp() const {
    detail::Entry entry;
    return {inf_.exp(), sup_.exp()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
