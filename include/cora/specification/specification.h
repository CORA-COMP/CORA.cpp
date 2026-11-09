// specification - a specification on reachable sets, as CORA's specification (halfspaces)
//
// A halfspace is {x | a'x <= b}. A safe set must contain the reachable sets (several
// halfspaces form a polytope); an unsafe set must not touch them. An unsafe set of several
// halfspaces is their intersection, a polytope. One halfspace is checked with the support
// function alone, so it applies to every ContSet on every backend; for a polytope a zonotope is
// decided exactly by a linear program, any other set conservatively (see check.cpp). A custom
// specification is a predicate on the set (CORA's 'custom'), such as an interval test.
//
// Syntax:     Specification spec = Specification::safe_set(a, b);   spec.check(R.time_interval);
// Operations: check, holds (check.cpp); safe_set, unsafe_set, custom (specification.cpp)
// See also:   cora/cont_set/cont_set.h, cora/cont_dynamics/linear_sys/linear_sys.h
//
// CORA: specification/@specification/specification.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"

#include <functional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The halfspace {x | a'x <= b}; a is a column (n, 1).
struct Halfspace {
    Tensor a;
    double b;
};

enum class SpecType { safe_set, unsafe_set, custom };

class Specification {
  public:
    /// The sets must lie in all of the halfspaces.
    static Specification safe_set(std::vector<Halfspace> halfspaces);
    static Specification safe_set(const Tensor &a, double b) { return safe_set({{a, b}}); }

    /// The sets must not touch the halfspace, or the polytope that is the intersection of several.
    static Specification unsafe_set(const Halfspace &halfspace) {
        return unsafe_set(std::vector<Halfspace>{halfspace});
    }
    static Specification unsafe_set(std::vector<Halfspace> halfspaces);
    static Specification unsafe_set(const Tensor &a, double b) {
        return unsafe_set(Halfspace{a, b});
    }

    /// A predicate on the set (CORA's specification(func, 'custom')): it holds where `holds`
    /// returns true; a batch is one set for it. It has no halfspaces.
    static Specification custom(std::function<bool(const ContSet &)> holds);

    SpecType type() const { return type_; }
    const std::vector<Halfspace> &halfspaces() const { return halfspaces_; }

    /// Per batch element of S, whether it satisfies the specification.
    std::vector<bool> holds(const ContSet &S) const;

    /// Whether every batch element of S satisfies the specification.
    bool check(const ContSet &S) const;

    /// Whether every set of a sequence (such as ReachSet::time_interval) does.
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
    Specification(SpecType type, std::vector<Halfspace> halfspaces,
                  std::function<bool(const ContSet &)> custom = nullptr)
        : type_(type), halfspaces_(std::move(halfspaces)), custom_(std::move(custom)) {}

    SpecType type_;
    std::vector<Halfspace> halfspaces_;
    std::function<bool(const ContSet &)> custom_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
