// add - the specifications of two lists in one, as CORA's add
//
// Syntax:   all = add(specs1, specs2);
// Inputs:   specs1, specs2 - lists of specifications
// Outputs:  all - the specifications of both, in order
// See also: is_empty, inverse
//
// CORA: specification/@specification/add.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/specification.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Specification> add(const std::vector<Specification> &specs1,
                               const std::vector<Specification> &specs2) {
    detail::Entry entry;
    std::vector<Specification> all = specs1;
    all.insert(all.end(), specs2.begin(), specs2.end());
    return all;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
