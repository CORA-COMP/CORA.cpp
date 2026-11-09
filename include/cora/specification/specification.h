// specification - a specification on reachable sets, as CORA's specification(set, type)
//
// A safe set S must contain the reachable sets (S.contains(R)); an unsafe set S must not touch
// them (!S.is_intersecting(R)). S is a set of any class, typically a Polytope {x | A x <= b}
// with one row per halfspace; the check asks the set's own operations (cora::contains and
// cora::is_intersecting), so a pair of classes without one is Error::noops. Whether they decide
// exactly or conservatively is documented on the set classes. A custom specification is a
// predicate on the set (CORA's 'custom'), such as an interval test. A batch of sets is not
// supported: the polytope operations take one set.
//
// Syntax:     Specification spec = Specification::safe_set(Polytope(A, b));   spec.check(R);
// Operations: check (check.cpp); safe_set, unsafe_set, custom (specification.cpp)
// See also:   cora/cont_set/registry.h, cora/cont_set/polytope/polytope.h
//
// CORA: specification/@specification/specification.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

#include <functional>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

enum class SpecType { safe_set, unsafe_set, custom };

class Specification {
  public:
    /// The reachable sets must lie in the set S (CORA's specification(S, 'safeSet')).
    static Specification safe_set(const Set &S);

    /// The reachable sets must not touch the set S (CORA's specification(S, 'unsafeSet')).
    static Specification unsafe_set(const Set &S);

    /// A predicate on the set (CORA's specification(func, 'custom')): it holds where `holds`
    /// returns true. It has no set.
    static Specification custom(std::function<bool(const ContSet &)> holds);

    SpecType type() const { return type_; }

    /// The set of a safe or unsafe specification; Error::wrong_value for a custom one.
    const Set &set() const;

    /// Whether the set R satisfies the specification.
    bool check(const ContSet &R) const;

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
    Specification(SpecType type, std::optional<Set> set,
                  std::function<bool(const ContSet &)> custom = nullptr)
        : type_(type), set_(std::move(set)), custom_(std::move(custom)) {}

    SpecType type_;
    std::optional<Set> set_;
    std::function<bool(const ContSet &)> custom_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
