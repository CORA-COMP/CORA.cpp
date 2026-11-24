// stl_interval - a time interval with open or closed ends, as CORA's stlInterval
//
// The times a signal temporal logic operator looks at: a subset [lb, ub] of the non-negative reals
// with each end open or closed, or empty. An infinite bound is open; lb > ub, or lb == ub with an
// open end, is empty. The bounds are doubles at nearest rounding, as CORALean's float definition
// of STLInterval fixes them, and every operation follows that definition (which keeps the lower
// bound at least zero, as CORA's operations clamp it).
//
// Syntax:     StlInterval I(0, 2);   StlInterval(0, 2, false, true);   (open at 0, closed at 2)
//             StlInterval();   (empty)   StlInterval(1);   (the point [1, 1])
// Operations: and_, plus, minus, minkowski_difference, left_closure, right_closure, to_left,
//             to_right (one file each in src/specification/stl_interval/), infimum, supremum, dim,
//             interval, contains, isequal, representsa
// See also: specification/specification.h
//
// CORA: specification/@stlInterval/stlInterval.m
// CORA: specification/@stlInterval/dim.m
// CORA: specification/@stlInterval/isemptyobject.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/global/tensor/tolerance.h"

#include <iosfwd>
#include <limits>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A bound of an interval and whether it belongs to it.
struct StlEndpoint {
    double value;
    bool closed;
};

class StlInterval {
  public:
    // Construction --------------------------------------------------------------------------------
    /// The empty interval (CORA's stlInterval()).
    StlInterval() = default;

    /// The point [x, x] (CORA's stlInterval(x)).
    explicit StlInterval(double x) : StlInterval(x, x, true, true) {}

    /// [lb, ub], both ends closed unless `closed` is false (CORA's stlInterval(lb, ub, closed)).
    StlInterval(double lb, double ub, bool closed = true) : StlInterval(lb, ub, closed, closed) {}

    /// The interval from lb to ub with the ends closed as lc and rc say (CORA's stlInterval(lb,
    /// ub, lc, rc)); Error::wrong_input_in_constructor for lb < 0 or a NaN bound.
    StlInterval(double lb, double ub, bool lc, bool rc);

    /// The empty interval; only n = 1 (STL intervals are one-dimensional, Error::wrong_value).
    static StlInterval empty(int64_t n = 1);

    // Queries -------------------------------------------------------------------------------------
    /// Whether it is the empty interval (CORA's isemptyobject).
    bool is_empty_object() const noexcept { return empty_; }

    /// The lower bound (0 for the empty interval, as CORALean); the upper one may be infinite.
    double lower() const noexcept { return lower_; }
    double upper() const noexcept { return upper_; }
    bool left_closed() const noexcept { return lc_; }
    bool right_closed() const noexcept { return rc_; }

    /// Always 1.
    int64_t dim() const noexcept { return 1; }

    /// The lower bound and whether it is attained; the empty interval gives (NaN, true).
    StlEndpoint infimum() const;

    /// The upper bound and whether it is attained; the empty interval gives (NaN, true).
    StlEndpoint supremum() const;

    /// The closed one-dimensional Interval [lower, upper] of the bounds (ends ignored).
    Interval interval() const;

    // Operations ----------------------------------------------------------------------------------
    /// The intersection of two intervals.
    StlInterval and_(const StlInterval &other) const;

    /// The sum of the times: {s + t}; empty if either is.
    StlInterval plus(const StlInterval &other) const;

    /// Every time moved by the number c (the lower bound must stay non-negative, else
    /// Error::wrong_input_in_constructor, as CORA).
    StlInterval plus(double c) const;

    /// The bounds moved by the infimum and supremum of the one-dimensional Interval J.
    StlInterval plus(const Interval &J) const;

    /// The differences {s - t} of the times, the lower bound clamped to 0 (closed there).
    StlInterval minus(const StlInterval &other) const;

    /// The times moved down by c, the lower bound clamped to 0.
    StlInterval minus(double c) const;

    /// The times moved down by the one-dimensional Interval J, the lower bound clamped to 0.
    StlInterval minus(const Interval &J) const;

    /// The Minkowski difference {t | t + sub subset of this}; type "exact" or "inner" (the same
    /// result here, as CORA); unbounded this and sub give an unbounded result (CORA: NaN bound).
    StlInterval minkowski_difference(const StlInterval &sub,
                                     const std::string &type = "exact") const;

    /// The interval with its lower end closed (unchanged if empty).
    StlInterval left_closure() const;

    /// The interval with its upper end closed (unchanged if empty or unbounded).
    StlInterval right_closure() const;

    /// [0, lower] with the end open where this is closed; Error::empty_set if empty.
    StlInterval to_left() const;

    /// [upper, inf) with the end open where this is closed; Error::empty_set if empty.
    StlInterval to_right() const;

    // Comparison ----------------------------------------------------------------------------------
    /// Whether the number t lies in it, the bounds compared up to tol.
    bool contains(double t, Tol tol = std::nullopt) const;

    /// Whether the interval S lies in it, the bounds compared up to tol.
    bool contains(const StlInterval &S, Tol tol = std::nullopt) const;

    /// Bounds equal up to tol and the same open and closed ends; two empty ones are equal.
    bool isequal(const StlInterval &other, Tol tol = std::nullopt) const;

    /// Whether it is a set of that type: "emptySet", "stlInterval", "point" (a closed interval of
    /// radius within tol of zero) or "origin" (a closed interval of both bounds within tol of 0);
    /// other types are Error::not_supported.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    bool operator==(const StlInterval &other) const { return isequal(other); }
    bool operator!=(const StlInterval &other) const { return !isequal(other); }

    /// "[0, 2)", or the empty set sign.
    std::string to_string() const;

  private:
    double lower_ = 0, upper_ = 0;
    bool lc_ = false, rc_ = false, empty_ = true;
};

std::ostream &operator<<(std::ostream &os, const StlInterval &I);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
