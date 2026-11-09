// check - whether a set satisfies a specification, as CORA's specification.check
//
// A safe set S holds for R iff S.contains(R); an unsafe set S holds iff !S.is_intersecting(R).
// Both are the set classes' own operations through the binary registry (cora::contains,
// cora::is_intersecting), so the exactness is theirs: a Polytope decides a zonotope, a box, a
// constrained zonotope or a bundle exactly by support functions or a linear program. A pair of
// classes without an operation is Error::noops. A custom specification asks its predicate, once
// for the whole set. A batch of sets is not supported.
//
// Syntax:   ok = spec.check(R);   ok = spec.check(sets);
// Inputs:   R - any set
//           sets - a sequence of sets such as ReachSet::time_interval
// Outputs:  ok - whether R (every set of the sequence) satisfies the specification
// See also: Specification::first_violation, Polytope::contains, Polytope::is_intersecting
//
// CORA: specification/@specification/check.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "cora/specification/specification.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The refusal of the operands of the operation (they differ in dimension), if any; the entry of
/// check raises it, since a nested entry does not check.
detail::Refusal aux_refusal(BinaryOp op, const ContSet &S, const ContSet &R) {
    return detail::refusal(op, S, R, std::nullopt);
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Specification::check(const ContSet &R) const {
    detail::Entry entry;
    if (type_ == SpecType::custom) return custom_(R);
    // the set classes decide, through the unchecked forms of the registry
    const ContSet &S = set_->get();
    const BinaryOp op =
        type_ == SpecType::safe_set ? BinaryOp::contains : BinaryOp::is_intersecting;
    const detail::Refusal refusal = aux_refusal(op, S, R);
    CORA_CHECK(!refusal, refusal->first, [&] { return "check: " + refusal->second; });
    if (type_ == SpecType::safe_set) return detail::contains_impl(S, R, std::nullopt);
    return !detail::is_intersecting_impl(S, R, std::nullopt);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
