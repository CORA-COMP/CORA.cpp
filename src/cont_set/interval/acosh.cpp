// acosh - the inverse hyperbolic cosine of a box, as CORA's interval.acosh
//
// Syntax:   Interval R = I.acosh();
// Inputs:   -
// Outputs:  R - [acosh(inf), acosh(sup)]; Error::out_of_domain unless inf >= 1
// See also: atan2, Tensor::acosh
//
// CORA: contSet/@interval/acosh.m
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

Interval Interval::acosh() const {
    detail::Entry entry;
    priv_check_domain("acosh", *this, 1, std::numeric_limits<double>::infinity());
    return with(inf_.acosh(), sup_.acosh());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
