// specification - the constructors of a specification, as CORA's specification(set, type)
//
// Syntax:   Specification::safe_set(S);   Specification::unsafe_set(S);   spec.with_time(t0, t1);
// Inputs:   S - the set: to lie in (safe), to avoid (unsafe), or to meet (invariant)
//           holds - a predicate on the set, true where the specification holds
// Outputs:  spec - the specification
// See also: check, with_time, with_locations
//
// CORA: specification/@specification/specification.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/specification/specification.h"

#include <cmath>

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

Specification Specification::invariant(const Set &S) {
    detail::Entry entry;
    return {SpecType::invariant, S};
}

// the window and the locations
Specification Specification::with_time(double t0, double t1) const {
    detail::Entry entry;
    CORA_CHECK(std::isfinite(t0) && std::isfinite(t1) && t0 <= t1,
               Error::wrong_input_in_constructor, [&] {
                   return "with_time: the window [t0, t1] needs finite t0 <= t1, got [" +
                          std::to_string(t0) + ", " + std::to_string(t1) + "]";
               });
    // the same specification with the window
    Specification out = *this;
    out.time_ = TimeSpan{t0, t1};
    return out;
}

Specification Specification::with_locations(std::vector<int64_t> locations) const {
    detail::Entry entry;
    for (const int64_t l : locations)
        CORA_CHECK(l >= 0, Error::wrong_input_in_constructor, [&] {
            return "with_locations: locations are 0-based and nonnegative, got " +
                   std::to_string(l);
        });
    // the same specification with the locations
    Specification out = *this;
    out.locations_ = std::move(locations);
    return out;
}

// custom predicates and logic formulas have no set
Specification Specification::custom(std::function<bool(const ContSet &)> holds) {
    detail::Entry entry;
    CORA_CHECK(static_cast<bool>(holds), Error::wrong_input_in_constructor,
               [] { return std::string("custom: give the predicate on the set"); });
    return {SpecType::custom, std::nullopt, std::move(holds)};
}

Specification Specification::logic(const Stl &formula) {
    detail::Entry entry;
    // the formula is all it holds
    return {SpecType::logic, formula};
}

const Stl &Specification::formula() const {
    detail::Entry entry;
    CORA_CHECK(formula_.has_value(), Error::wrong_value,
               [] { return std::string("formula: only a logic specification has a formula"); });
    return *formula_;
}

// the set of a set specification
const Set &Specification::set() const {
    detail::Entry entry;
    CORA_CHECK(set_.has_value(), Error::wrong_value,
               [] { return std::string("set: a custom or logic specification has no set"); });
    return *set_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
