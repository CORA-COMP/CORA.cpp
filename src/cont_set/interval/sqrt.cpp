// sqrt - the square root of a box, as CORA's interval.sqrt
//
// Syntax:   Interval R = I.sqrt();
// Inputs:   -
// Outputs:  R - [sqrt(inf), sqrt(sup)] elementwise (sqrt increases); nothing rounds outward;
//           Error::out_of_domain if some bound is negative or NaN
// See also: power, log
//
// CORA: contSet/@interval/sqrt.m

#include "cont_set/interval/private/priv_check_domain.h"
#include "cora/cont_set/interval/interval.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::sqrt() const {
    detail::Entry entry;
    priv_check_domain("sqrt", inf_, sup_, 0, std::numeric_limits<double>::infinity());
    return {inf_.sqrt(), sup_.sqrt()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
