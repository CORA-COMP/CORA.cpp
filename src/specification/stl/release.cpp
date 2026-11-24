// release - the formula where psi holds until phi releases it, within the window
//
// Syntax:   Stl res = phi.release(psi, time);
// Inputs:   other - the formula that is checked after / together with this one
//           time - the window, not empty
// Outputs:  res - the formula
// See also: stl.h
//
// CORA: specification/@stl/release.m
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

Stl Stl::release(const Stl &other, const StlInterval &time) const {
    detail::Entry entry;
    CORA_CHECK(!time.is_empty_object(), Error::wrong_value,
               [] { return std::string("release: the time window must not be empty"); });
    return priv_temporal(StlKind::release, *this, &other, time);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
