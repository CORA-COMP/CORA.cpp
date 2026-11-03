// uplus - the unary plus of a box, as CORA's interval.uplus
//
// Syntax:   Interval R = I.uplus();   Interval R = +I;
// Inputs:   -
// Outputs:  R - the box itself
// See also: uminus
//
// CORA: contSet/@interval/uplus.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::uplus() const {
    detail::Entry entry;
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
