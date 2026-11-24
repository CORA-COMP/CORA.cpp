// next - the formula that holds a given time later (X[a] phi)
//
// Syntax:   Stl res = phi.next(a);
// Inputs:   a - the time, finite and non-negative
// Outputs:  res - the formula
// See also: stl.h
//
// CORA: specification/@stl/next.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

#include "specification/stl/private/priv_temporal.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Stl Stl::next(double a) const {
    detail::Entry entry;
    CORA_CHECK(std::isfinite(a) && a >= 0, Error::wrong_input_in_constructor,
               [] { return std::string("next: the time must be finite and non-negative"); });
    return priv_temporal(StlKind::next, *this, nullptr, StlInterval(a));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
