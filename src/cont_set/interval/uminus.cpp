// uminus - the negated box, as CORA's interval.uminus
//
// Syntax:   Interval R = I.uminus();   Interval R = -I;
// Inputs:   -
// Outputs:  R - [-sup, -inf]; nothing rounds
// See also: uplus, minus
//
// CORA: contSet/@interval/uminus.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::uminus() const {
    detail::Entry entry;
    return {-sup_, -inf_};
}

Set Interval::uminus_impl() const { return uminus(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
