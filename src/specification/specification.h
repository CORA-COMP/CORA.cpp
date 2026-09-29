// A specification on reachable sets, as CORA's `specification`, for halfspaces
// `{x | aᵀx <= b}`. It is checked with the support function alone, so it applies to every
// `ContSet`, on every backend.
//
//     auto spec = Specification::safe_set(Tensor({1, 0}), 3.0);   // stay in x1 <= 3
//     bool ok = spec.check(R.time_int);                          // R from LinearSys::reach
//
// `safe_set`: the set must lie in the halfspaces (several form a polytope).
// `unsafe_set`: the set must not touch the halfspace; one halfspace, since deciding that
// for an intersection of halfspaces takes an LP.

#pragma once

#include "contSet/contSet.h"

#include <vector>

namespace cora::ct {

/// `{x | aᵀx <= b}` with `a` a column `(n, 1)`.
struct Halfspace {
    Tensor a;
    double b;
};

enum class SpecType { SafeSet, UnsafeSet };

class Specification {
  public:
    static Specification safe_set(std::vector<Halfspace> halfspaces);
    static Specification safe_set(const Tensor &a, double b) { return safe_set({{a, b}}); }
    static Specification unsafe_set(const Halfspace &halfspace);
    static Specification unsafe_set(const Tensor &a, double b) { return unsafe_set({a, b}); }

    SpecType type() const { return type_; }
    const std::vector<Halfspace> &halfspaces() const { return halfspaces_; }

    /// Per batch element of `S`, whether it satisfies the specification.
    std::vector<bool> holds(const ContSet &S) const;

    /// Whether every batch element of `S` satisfies the specification.
    bool check(const ContSet &S) const;

    /// Whether every set of a reachable-set sequence does.
    template <class S>
    bool check(const std::vector<S> &sets) const {
        return first_violation(sets) < 0;
    }

    /// The index of the first set that violates the specification, or -1.
    template <class S>
    int first_violation(const std::vector<S> &sets) const {
        for (std::size_t k = 0; k < sets.size(); ++k)
            if (!check(sets[k])) return static_cast<int>(k);
        return -1;
    }

  private:
    Specification(SpecType type, std::vector<Halfspace> halfspaces)
        : type_(type), halfspaces_(std::move(halfspaces)) {}

    SpecType type_;
    std::vector<Halfspace> halfspaces_;
};

} // namespace cora::ct
