// check - whether a set satisfies a specification, as CORA's specification.check
//
// Syntax:   ok = spec.check(S);   each = spec.holds(S);
// Inputs:   S - any set; a batch of sets gives one answer per member in holds
// Outputs:  ok - whether all members satisfy the specification; each - one flag per member
// See also: Specification::firstViolation, ContSet::supportFunc

#include "specification/specification.h"

#include <algorithm>

namespace cora::ct {

namespace {
bool aux_satisfied(SpecType type, double extreme, double b);
} // namespace

std::vector<bool> Specification::holds(const ContSet &S) const {
    std::vector<bool> ok;
    for (const Halfspace &h : halfspaces_) {
        // Safe: the largest a'x of the set is at most b. Unsafe: its smallest a'x is above b,
        // and the smallest is minus the support function along -a.
        const bool safe = type_ == SpecType::SafeSet;
        const std::vector<double> extreme = S.supportFunc(safe ? h.a : h.a * -1.0).data();
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

// --------------------------- auxiliary functions --------------------------------

namespace {

/// One halfspace: `extreme` is the support along a (safe) or along -a (unsafe).
bool aux_satisfied(SpecType type, double extreme, double b) {
    return type == SpecType::SafeSet ? extreme <= b : -extreme > b;
}

} // namespace

} // namespace cora::ct
