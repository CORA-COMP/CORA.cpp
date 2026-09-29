// check - whether a set satisfies a specification, as CORA's specification.check
//
// Syntax:   ok = spec.check(S);   each = spec.holds(S);
// Inputs:   S - any set; a batch of sets gives one answer per member in holds
// Outputs:  ok - whether all members satisfy the specification; each - one flag per member
// See also: Specification::firstViolation, ContSet::supportFunc

#include "specification/specification.h"

#include <algorithm>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The direction whose support decides a halfspace: a for a safe set (the largest a'x must be at
/// most b), -a for an unsafe one (the smallest a'x, minus the support along -a, must exceed b).
Tensor aux_direction(SpecType type, const Tensor &a) {
    if (type == SpecType::SafeSet) return a;
    if (type == SpecType::UnsafeSet) return a * -1.0;
    throw std::invalid_argument("Specification: unknown type; use SafeSet or UnsafeSet");
}

/// One halfspace: `extreme` is the support in the direction of aux_direction.
bool aux_satisfied(SpecType type, double extreme, double b) {
    if (type == SpecType::SafeSet) return extreme <= b;
    if (type == SpecType::UnsafeSet) return -extreme > b;
    throw std::invalid_argument("Specification: unknown type; use SafeSet or UnsafeSet");
}

} // namespace

// ===========================================  MAIN  =========================================== //

std::vector<bool> Specification::holds(const ContSet &S) const {
    std::vector<bool> ok;
    for (const Halfspace &h : halfspaces_) {
        const std::vector<double> extreme = S.supportFunc(aux_direction(type_, h.a)).data();
        if (ok.empty()) ok.assign(extreme.size(), true);
        for (std::size_t i = 0; i < extreme.size(); ++i)
            if (!aux_satisfied(type_, extreme[i], h.b)) ok[i] = false;
    }
    return ok;
}

bool Specification::check(const ContSet &S) const {
    const std::vector<bool> ok = holds(S);
    return std::all_of(ok.begin(), ok.end(), [](bool v) { return v; });
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
