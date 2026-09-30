// specification - a specification on reachable sets, as CORA's specification (halfspaces)
//
// A halfspace is {x | a'x <= b}. A safe set must contain the reachable sets (several
// halfspaces form a polytope); an unsafe set must not touch them (one halfspace: for an
// intersection of halfspaces the check needs an LP). It is checked with the support function
// alone, so it applies to every ContSet on every backend.
//
// Syntax:     Specification spec = Specification::safeSet(a, b);   spec.check(R.timeInt);
// Operations: check, holds (check.cpp); safeSet, unsafeSet (specification.cpp)
// See also:   contSet/contSet.h, contDynamics/linearSys/linearSys.h

#pragma once

#include "contSet/contSet.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The halfspace {x | a'x <= b}; a is a column (n, 1).
struct Halfspace {
    Tensor a;
    double b;
};

enum class SpecType { SafeSet, UnsafeSet };

class Specification {
  public:
    /// The sets must lie in all of the halfspaces.
    static Specification safeSet(std::vector<Halfspace> halfspaces);
    static Specification safeSet(const Tensor &a, double b) { return safeSet({{a, b}}); }

    /// The sets must not touch the halfspace.
    static Specification unsafeSet(const Halfspace &halfspace);
    static Specification unsafeSet(const Tensor &a, double b) { return unsafeSet({a, b}); }

    SpecType type() const { return type_; }
    const std::vector<Halfspace> &halfspaces() const { return halfspaces_; }

    /// Per batch element of S, whether it satisfies the specification.
    std::vector<bool> holds(const ContSet &S) const;

    /// Whether every batch element of S satisfies the specification.
    bool check(const ContSet &S) const;

    /// Whether every set of a sequence (such as Reach::timeInt) does.
    template <class S>
    bool check(const std::vector<S> &sets) const {
        return firstViolation(sets) < 0;
    }

    /// The index of the first set that violates the specification, or -1.
    template <class S>
    int firstViolation(const std::vector<S> &sets) const {
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

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
