// specification - the constructors of a specification, as CORA's specification(set, type)
//
// Syntax:   Specification::safe_set({h1, h2, ...});   Specification::unsafe_set({h1, h2, ...});
//           Specification::custom(holds);
// Inputs:   halfspaces - {x | a'x <= b}
//           holds - a predicate on the set, true where the specification holds
// Outputs:  spec - the specification; an unsafe set of several halfspaces is their intersection
// See also: check
//
// CORA: specification/@specification/specification.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/specification/specification.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Specification Specification::safe_set(std::vector<Halfspace> halfspaces) {
    detail::Entry entry;
    CORA_CHECK(!halfspaces.empty(), Error::wrong_input_in_constructor,
               [] { return std::string("safe_set: give at least one halfspace"); });
    return {SpecType::safe_set, std::move(halfspaces)};
}

Specification Specification::unsafe_set(std::vector<Halfspace> halfspaces) {
    detail::Entry entry;
    CORA_CHECK(!halfspaces.empty(), Error::wrong_input_in_constructor,
               [] { return std::string("unsafe_set: give at least one halfspace"); });
    return {SpecType::unsafe_set, std::move(halfspaces)};
}

Specification Specification::custom(std::function<bool(const ContSet &)> holds) {
    detail::Entry entry;
    CORA_CHECK(static_cast<bool>(holds), Error::wrong_input_in_constructor,
               [] { return std::string("custom: give the predicate on the set"); });
    return {SpecType::custom, {}, std::move(holds)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
