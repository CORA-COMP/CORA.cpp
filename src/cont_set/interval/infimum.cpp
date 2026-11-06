// infimum - the lower bound of a box, as CORA's interval.infimum
//
// Syntax:   Tensor l = I.infimum();
// Inputs:   -
// Outputs:  l - inf, of the shape of the bounds
// See also: Interval::inf
//
// CORA: contSet/@interval/infimum.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::infimum() const {
    detail::Entry entry;
    return inf_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
