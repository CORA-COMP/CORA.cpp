// asin - the arcsine of a box, as CORA's interval.asin
//
// Syntax:   Interval R = I.asin();
// Inputs:   -
// Outputs:  R - [asin(inf), asin(sup)]; Error::out_of_domain unless -1 <= inf and sup <= 1
// See also: atan2, Tensor::asin
//
// CORA: contSet/@interval/asin.m
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

Interval Interval::asin() const {
    detail::Entry entry;
    priv_check_domain("asin", *this, -1, 1);
    return with(inf_.asin(), sup_.asin());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
