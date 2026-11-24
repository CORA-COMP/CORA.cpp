// finally - the formula where phi holds at some time of the window
//
// Syntax:   Stl res = phi.finally(time);
// Inputs:   time - the window, not empty
// Outputs:  res - the formula
// See also: stl.h
//
// CORA: specification/@stl/finally.m
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

Stl Stl::finally(const StlInterval &time) const {
    detail::Entry entry;
    CORA_CHECK(!time.is_empty_object(), Error::wrong_value,
               [] { return std::string("finally: the time window must not be empty"); });
    return priv_temporal(StlKind::finally, *this, nullptr, time);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
