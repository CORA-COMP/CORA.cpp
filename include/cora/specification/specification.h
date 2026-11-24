// specification - a specification on reachable sets, as CORA's specification(set, type)
//
// A safe set S must contain the reachable sets (S.contains(R)); an unsafe set S must not touch
// them (!S.is_intersecting(R)); an invariant must intersect them. S is a set of any class,
// typically a Polytope {x | A x <= b}; the check asks the set's own operations (cora::contains and
// cora::is_intersecting), so a pair of classes without one is Error::noops. A custom specification
// is a predicate on the set; a logic one holds a formula of signal temporal logic (Stl). It may be
// active in a time window and in some locations (0-based); a list of specifications (a std::vector)
// holds when each one does. A batch of sets is not supported.
//
// Syntax:     Specification spec = Specification::safe_set(Polytope(A, b));   spec.check(R);
// Operations: check (check.cpp); safe_set, unsafe_set, invariant, custom, with_time,
//             with_locations (specification.cpp); add, is_empty, project, isequal, inverse,
//             robustness, split_logic
// See also:   cora/cont_set/registry.h, cora/cont_set/polytope/polytope.h
//
// CORA: specification/@specification/specification.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"
#include "cora/specification/stl.h"

#include <functional>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

enum class SpecType { safe_set, unsafe_set, invariant, custom, logic };

/// The closed time interval [t0, t1] in which a specification is active.
using TimeSpan = std::pair<double, double>;

class Specification {
  public:
    // Construction --------------------------------------------------------------------------------
    /// The reachable sets must lie in the set S (CORA's specification(S, 'safeSet')).
    static Specification safe_set(const Set &S);

    /// The reachable sets must not touch the set S (CORA's specification(S, 'unsafeSet')).
    static Specification unsafe_set(const Set &S);

    /// The reachable sets must intersect the set S (CORA's specification(S, 'invariant')).
    static Specification invariant(const Set &S);

    /// A predicate on the set (CORA's specification(func, 'custom')): it holds where `holds`
    /// returns true. It has no set.
    static Specification custom(std::function<bool(const ContSet &)> holds);

    /// A temporal logic formula that the traces must satisfy (CORA's specification(eq, 'logic')).
    /// It has no set and no check on sets; the formula's own operations decide it.
    static Specification logic(const Stl &formula);

    // Queries -------------------------------------------------------------------------------------
    SpecType type() const { return type_; }

    /// The formula of a logic specification; Error::wrong_value for another type.
    const Stl &formula() const;

    /// The same specification, active in [t0, t1] (CORA's specification(S, type, time)).
    Specification with_time(double t0, double t1) const;

    /// The same specification, active in these locations of a hybrid automaton (0-based).
    Specification with_locations(std::vector<int64_t> locations) const;

    /// The time window in which it is active; none: the whole horizon.
    const std::optional<TimeSpan> &time() const { return time_; }

    /// The locations in which it is active; empty: everywhere.
    const std::vector<int64_t> &locations() const { return locations_; }

    // Checking ------------------------------------------------------------------------------------
    /// The set of a safe or unsafe specification; Error::wrong_value for a custom one.
    const Set &set() const;

    /// Whether the set R satisfies the specification.
    bool check(const ContSet &R) const;

    /// Whether the set R, which exists over the times `time`, satisfies the specification: one
    /// that is active in another window is satisfied. A timed specification needs the time.
    bool check(const ContSet &R, const TimeSpan &time) const;

    /// Whether the points (N, n), one per row, satisfy it; `times` (empty, one, or N) are their
    /// times, which a timed specification needs. A custom one is Error::not_supported.
    bool check_points(const Tensor &points, const std::vector<double> &times = {}) const;

    /// Whether the reachable sets satisfy it, as CORA's check on a reachSet: each step's
    /// time_interval set (the time_point sets where the window is one instant), against the window;
    /// no step in the window violates it. Tolerance of the time comparisons: tol.
    bool check(const ReachSet &R, Tol tol = std::nullopt) const;

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

    /// The projection of the set onto the dimensions `dims` (0-based); custom: not_supported.
    Specification project(const std::vector<int64_t> &dims) const;

    /// Same type, window, locations and set up to tol (CORA's isequal on one specification); a
    /// custom predicate cannot be compared (Error::not_supported).
    bool isequal(const Specification &other, Tol tol = std::nullopt) const;

  private:
    Specification(SpecType type, std::optional<Set> set,
                  std::function<bool(const ContSet &)> custom = nullptr)
        : type_(type), set_(std::move(set)), custom_(std::move(custom)) {}
    Specification(SpecType type, Stl formula) : type_(type), formula_(std::move(formula)) {}

    /// check(R) on a set over `time` (none: untimed), as the contSet branch of CORA's check.
    bool check_in(const ContSet &R, const std::optional<TimeSpan> &time, double tol) const;

    SpecType type_;
    std::optional<Set> set_;
    std::optional<TimeSpan> time_;
    std::vector<int64_t> locations_;
    std::function<bool(const ContSet &)> custom_;
    std::optional<Stl> formula_;
};

// Lists of specifications -------------------------------------------------------------------------

/// The specifications of both lists in one (CORA's add).
std::vector<Specification> add(const std::vector<Specification> &specs1,
                               const std::vector<Specification> &specs2);

/// Whether the list holds no specification (CORA's isempty: a specification always has a set).
bool is_empty(const std::vector<Specification> &specs);

/// Whether both lists hold the same specifications in any order (CORA's isequal).
bool isequal(const std::vector<Specification> &specs1, const std::vector<Specification> &specs2,
             Tol tol = std::nullopt);

/// The specifications of a list apart: those that are no temporal logic, and those that are (CORA's
/// splitLogic), each in the order of the list.
std::pair<std::vector<Specification>, std::vector<Specification>>
split_logic(const std::vector<Specification> &specs);

/// The specifications with the sets projected onto `dims` (CORA's project).
std::vector<Specification> project(const std::vector<Specification> &specs,
                                   const std::vector<int64_t> &dims);

/// The inverse of untimed specifications of one type (CORA's inverse): safe sets become the
/// unsafe set of their intersection; one unsafe set becomes a safe set; several unsafe sets
/// become the unsafe sets that cover the complement of their union.
std::vector<Specification> inverse(const std::vector<Specification> &specs);

/// The robustness of each point (N, n), one per row, of an untimed specification (CORA's
/// robustness): the distance inside (positive) or outside (negative) a safe set or invariant,
/// reversed for an unsafe set. A timed specification needs the times: the next overload.
Tensor robustness(const Specification &spec, const Tensor &points);

/// The robustness of a trajectory: the least over the points in the specification's window (Inf
/// if none), the trajectory interpolated at the window's center if no point is in it.
double robustness(const Specification &spec, const Tensor &points,
                  const std::vector<double> &times);

/// The least robustness over a list of specifications, per point or over the trajectory.
Tensor robustness(const std::vector<Specification> &specs, const Tensor &points);
double robustness(const std::vector<Specification> &specs, const Tensor &points,
                  const std::vector<double> &times);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
