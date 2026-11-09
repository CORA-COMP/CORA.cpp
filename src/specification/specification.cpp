// specification - the constructors of a specification, as CORA's specification(set, type)
//
// Syntax:   Specification::safe_set(S);   Specification::unsafe_set(S);
// Specification::custom(holds); Inputs:   S - the set the reachable sets must lie in (safe), or
// must not touch (unsafe),
//               typically a Polytope
//           holds - a predicate on the set, true where the specification holds
// Outputs:  spec - the specification
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

Specification Specification::safe_set(const Set &S) {
    detail::Entry entry;
    return {SpecType::safe_set, S};
}

Specification Specification::unsafe_set(const Set &S) {
    detail::Entry entry;
    return {SpecType::unsafe_set, S};
}

Specification Specification::custom(std::function<bool(const ContSet &)> holds) {
    detail::Entry entry;
    CORA_CHECK(static_cast<bool>(holds), Error::wrong_input_in_constructor,
               [] { return std::string("custom: give the predicate on the set"); });
    return {SpecType::custom, std::nullopt, std::move(holds)};
}

const Set &Specification::set() const {
    detail::Entry entry;
    CORA_CHECK(set_.has_value(), Error::wrong_value,
               [] { return std::string("set: a custom specification has no set"); });
    return *set_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
