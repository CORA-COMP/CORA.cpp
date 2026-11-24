// until - the formula where phi holds until psi does, within the window
//
// Syntax:   Stl res = phi.until(psi, time);
// Inputs:   other - the formula that is checked after / together with this one
//           time - the window, not empty
// Outputs:  res - the formula
// See also: stl.h
//
// CORA: specification/@stl/until.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

#include "specification/stl/private/priv_temporal.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Stl Stl::until(const Stl &other, const StlInterval &time) const {
    detail::Entry entry;
    CORA_CHECK(!time.is_empty_object(), Error::wrong_value,
               [] { return std::string("until: the time window must not be empty"); });
    return priv_temporal(StlKind::until, *this, &other, time);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
