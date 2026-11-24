// is_empty - whether a list of specifications is empty, as CORA's isempty
//
// Syntax:   res = is_empty(specs);
// Inputs:   specs - a list of specifications
// Outputs:  res - whether the list holds none
// See also: add
//
// CORA: specification/@specification/isempty.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/specification.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// CORA also asks for a set that is the empty matrix [], which a Specification never holds.
bool is_empty(const std::vector<Specification> &specs) {
    detail::Entry entry;
    return specs.empty();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
