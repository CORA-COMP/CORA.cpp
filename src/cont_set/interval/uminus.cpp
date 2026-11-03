// uminus - the negated box, as CORA's interval.uminus
//
// Syntax:   Interval R = I.uminus();   Interval R = -I;
// Inputs:   -
// Outputs:  R - [-sup, -inf]; nothing rounds
// See also: uplus, minus
//
// CORA: contSet/@interval/uminus.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::uminus() const {
    detail::Entry entry;
    return {-sup_, -inf_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
