// acos - the arccosine of a box, as CORA's interval.acos
//
// Syntax:   Interval R = I.acos();
// Inputs:   -
// Outputs:  R - [acos(sup), acos(inf)] (acos decreases); Error::out_of_domain unless -1 <= inf and
// sup <= 1 See also: atan2, Tensor::acos
//
// CORA: contSet/@interval/acos.m

#include "cont_set/interval/private/priv_check_domain.h"
#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::acos() const {
    detail::Entry entry;
    priv_check_domain("acos", inf_, sup_, -1, 1);
    return {sup_.acos(), inf_.acos()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
