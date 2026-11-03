// log - the natural logarithm of a box, as CORA's interval.log
//
// Syntax:   Interval R = I.log();
// Inputs:   -
// Outputs:  R - [log(inf), log(sup)] (log increases), log(0) = -Inf;
//           Error::out_of_domain if some bound is negative or NaN
// See also: exp, Tensor::log
//
// CORA: contSet/@interval/log.m

#include "cont_set/interval/private/priv_check_domain.h"
#include "cora/cont_set/interval/interval.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::log() const {
    detail::Entry entry;
    priv_check_domain("log", inf_, sup_, 0, std::numeric_limits<double>::infinity());
    return {inf_.log(), sup_.log()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
