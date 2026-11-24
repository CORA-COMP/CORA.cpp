// split_logic - the specifications of a list apart into those of sets and those of logic
//
// Syntax:   auto [non_logic, logic] = split_logic(specs);
// Inputs:   specs - a list of specifications
// Outputs:  non_logic - the specifications that are no temporal logic, in order
//           logic - the temporal logic specifications, in order
// See also: add, Specification::logic
//
// CORA: specification/@specification/splitLogic.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/specification.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<std::vector<Specification>, std::vector<Specification>>
split_logic(const std::vector<Specification> &specs) {
    detail::Entry entry;
    std::vector<Specification> non_logic, logic;
    // each specification goes to the list of its kind
    for (const Specification &spec : specs)
        (spec.type() == SpecType::logic ? logic : non_logic).push_back(spec);
    return {non_logic, logic};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
