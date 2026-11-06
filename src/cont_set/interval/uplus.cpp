// uplus - the unary plus of a box, as CORA's interval.uplus
//
// Syntax:   Interval R = I.uplus();   Interval R = +I;
// Inputs:   -
// Outputs:  R - the box itself
// See also: uminus
//
// CORA: contSet/@interval/uplus.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::uplus() const {
    detail::Entry entry;
    return *this;
}

Set Interval::uplus_impl() const { return uplus(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
