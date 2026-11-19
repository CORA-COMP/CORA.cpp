// acos - the arccosine of a box, as CORA's interval.acos
//
// Syntax:   Interval R = I.acos();
// Inputs:   -
// Outputs:  R - [acos(sup), acos(inf)] (acos decreases); Error::out_of_domain unless -1 <= inf and
// sup <= 1 See also: atan2, Tensor::acos
//
// CORA: contSet/@interval/acos.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_check_domain.h"
#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::acos() const {
    detail::Entry entry;
    priv_check_domain("acos", *this, -1, 1);
    return with(sup_.acos(), inf_.acos());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
