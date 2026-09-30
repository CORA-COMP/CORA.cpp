// priv_verifyTime - time intervals where a specification is not yet verified, as CORA's verifyTime
//
// A sorted sequence of disjoint [lower, upper] pairs; verified times are removed from it.
//
// Syntax:   VerifyTime T({{0, 2}});   T.setdiff(0.5, 1);   bool hit = T.isIntersecting(0, 0.1);
// Inputs:   bounds - the pairs, in order
// Outputs:  the queries below
// See also: priv_reach_adaptive, priv_verifyRA_zonotope

#pragma once

#include <array>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class VerifyTime {
  public:
    using Bounds = std::vector<std::array<double, 2>>;

    /// The pairs, sorted and disjoint.
    Bounds bounds;

    VerifyTime() = default;
    explicit VerifyTime(Bounds b) : bounds(std::move(b)) {}

    bool empty() const { return bounds.empty(); }
    int numIntervals() const { return static_cast<int>(bounds.size()); }

    /// The smallest lower bound and the largest upper bound.
    double startTime() const { return bounds.front()[0]; }
    double finalTime() const { return bounds.back()[1]; }

    /// Merges intervals whose end meets the next start (tolerance eps * horizon by default).
    VerifyTime compact() const;

    /// Whether some interval contains the time t.
    bool contains(double t) const;

    /// Whether [t0, t1] shares more than a point with the intervals.
    bool isIntersecting(double t0, double t1) const;

    /// The time from t to the end of the interval it is in, or to the start of the next one;
    /// inside says whether the time after that switch is outside (false) or inside (true).
    /// False if there are no intervals or none ends after t.
    bool timeUntilSwitch(double t, double &tSwitch, bool &inside) const;

    /// Removes [t0, t1] from the intervals and drops the remainders shorter than 1e-14.
    void setdiff(double t0, double t1);

    /// The union of the intervals of several objects.
    static VerifyTime unify(const std::vector<VerifyTime> &times);
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
