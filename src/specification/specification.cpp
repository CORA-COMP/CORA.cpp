#include "specification/specification.h"

#include <algorithm>
#include <stdexcept>

namespace cora::ct {

Specification Specification::safe_set(std::vector<Halfspace> halfspaces) {
    if (halfspaces.empty()) throw std::invalid_argument("Specification: no halfspace");
    return {SpecType::SafeSet, std::move(halfspaces)};
}

Specification Specification::unsafe_set(const Halfspace &halfspace) {
    return {SpecType::UnsafeSet, {halfspace}};
}

std::vector<bool> Specification::holds(const ContSet &S) const {
    std::vector<bool> ok;
    for (const Halfspace &h : halfspaces_) {
        // Safe: the set's largest aᵀx is at most b. Unsafe: its smallest aᵀx is above b, so
        // the two do not touch; the smallest is minus the support along -a.
        const bool safe = type_ == SpecType::SafeSet;
        std::vector<double> extreme = S.support_func(safe ? h.a : h.a * -1.0).data();
        if (ok.empty()) ok.assign(extreme.size(), true);
        for (std::size_t i = 0; i < extreme.size(); ++i)
            if (!(safe ? extreme[i] <= h.b : -extreme[i] > h.b)) ok[i] = false;
    }
    return ok;
}

bool Specification::check(const ContSet &S) const {
    const std::vector<bool> ok = holds(S);
    return std::all_of(ok.begin(), ok.end(), [](bool v) { return v; });
}

} // namespace cora::ct
