// specification - the constructors of a specification, as CORA's specification(set, type)
//
// Syntax:   Specification::safe_set({h1, h2, ...});   Specification::unsafe_set({h1, h2, ...});
// Inputs:   halfspaces {x | a'x <= b}
// Outputs:  the specification; an unsafe set of several halfspaces is their intersection
// See also: check
//
// CORA: specification/@specification/specification.m

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

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
