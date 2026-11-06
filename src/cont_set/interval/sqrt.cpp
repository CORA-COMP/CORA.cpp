// sqrt - the square root of a box, as CORA's interval.sqrt
//
// Error::out_of_domain if some bound is negative or NaN.
//
// Syntax:   Interval R = I.sqrt();
// Inputs:   -
// Outputs:  R - [sqrt(inf), sqrt(sup)] elementwise (sqrt increases); nothing rounds outward
// See also: power, log
//
// CORA: contSet/@interval/sqrt.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

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
