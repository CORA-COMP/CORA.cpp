// atanh - the inverse hyperbolic tangent of a box, as CORA's interval.atanh
//
// Syntax:   Interval R = I.atanh();
// Inputs:   -
// Outputs:  R - [atanh(inf), atanh(sup)]; Error::out_of_domain unless -1 <= inf and sup <= 1
// See also: atan2, Tensor::atanh
//
// CORA: contSet/@interval/atanh.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_check_domain.h"
#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::atanh() const {
    detail::Entry entry;
    priv_check_domain("atanh", inf_, sup_, -1, 1);
    return {inf_.atanh(), sup_.atanh()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
