// priv_verifyTime - time intervals where a specification is not yet verified, as CORA's verifyTime
//
// The methods of VerifyTime: queries on a sorted sequence of disjoint time intervals, and the
// removal of a verified interval.
//
// Syntax:   VerifyTime T({{0, 2}});   T.setdiff(0.5, 1);   bool hit = T.isIntersecting(0, 0.1);
// Inputs:   bounds - the pairs, in order
// Outputs:  the queries of priv_verifyTime.h
// See also: priv_reach_adaptive, priv_verifyRA_zonotope

#include "contDynamics/linearSys/private/priv_verifyTime.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORA's withinTol: equal up to an absolute or a relative tolerance.
bool aux_withinTol(double a, double b, double tol = 1e-8) {
    const double diff = std::abs(a - b);
    return diff <= tol || diff / std::min(std::abs(a), std::abs(b)) <= tol ||
           (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0));
}

} // namespace


// ===========================================  MAIN  =========================================== //

VerifyTime VerifyTime::compact() const {
    if (bounds.size() <= 1) return *this;
    const double tol = 2.220446049250313e-16 * (finalTime() - startTime());
    VerifyTime out;
    out.bounds.push_back(bounds[0]);
    for (std::size_t i = 1; i < bounds.size(); ++i) {
        // An interval starting where the previous one ends extends it.
        if (aux_withinTol(out.bounds.back()[1], bounds[i][0], tol))
            out.bounds.back()[1] = bounds[i][1];
        else
            out.bounds.push_back(bounds[i]);
    }
    return out;
}

bool VerifyTime::contains(double t) const {
    for (const auto &b : bounds)
        if (b[0] <= t && b[1] >= t) return true;
    return false;
}

bool VerifyTime::isIntersecting(double t0, double t1) const {
    if (bounds.empty()) return false;
    if (t1 <= startTime() || t0 > finalTime()) return false;
    // A gap between two intervals that covers [t0, t1] leaves nothing to intersect.
    for (std::size_t i = 0; i + 1 < bounds.size(); ++i)
        if (t0 >= bounds[i][1] && t1 <= bounds[i + 1][0]) return false;
    return true;
}

bool VerifyTime::timeUntilSwitch(double t, double &tSwitch, bool &inside) const {
    if (bounds.empty()) return false;
    const VerifyTime c = compact();
    // The flattened bounds lower_1, upper_1, lower_2, ...: the first one after t is the switch.
    for (std::size_t i = 0; i < 2 * c.bounds.size(); ++i) {
        const double v = c.bounds[i / 2][i % 2];
        if (t < v && !aux_withinTol(t, v)) {
            tSwitch = v - t;
            inside = (i + 1) % 2 == 0;
            return true;
        }
    }
    return false;
}

void VerifyTime::setdiff(double t0, double t1) {
    Bounds kept;
    for (const auto &iv : bounds) {
        if (t1 <= iv[0] || t0 >= iv[1]) {
            kept.push_back(iv);
            continue;
        }
        if (iv[0] < t0) kept.push_back({iv[0], t0});
        if (t1 < iv[1]) kept.push_back({t1, iv[1]});
    }
    // Remainders shorter than 1e-14 are rounding errors of the times.
    kept.erase(std::remove_if(kept.begin(), kept.end(),
                              [](const std::array<double, 2> &iv) {
                                  return std::abs(iv[1] - iv[0]) < 1e-14;
                              }),
               kept.end());
    bounds = std::move(kept);
}

VerifyTime VerifyTime::unify(const std::vector<VerifyTime> &times) {
    Bounds all;
    for (const VerifyTime &t : times) all.insert(all.end(), t.bounds.begin(), t.bounds.end());
    std::sort(all.begin(), all.end());
    VerifyTime out;
    // Overlapping or touching intervals merge into one.
    for (const auto &iv : all) {
        if (!out.bounds.empty() && iv[0] <= out.bounds.back()[1])
            out.bounds.back()[1] = std::max(out.bounds.back()[1], iv[1]);
        else
            out.bounds.push_back(iv);
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
