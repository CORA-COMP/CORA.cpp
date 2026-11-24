// check - whether a set satisfies a specification, as CORA's specification.check
//
// A safe set S holds for R iff S.contains(R); an unsafe set S holds iff !S.is_intersecting(R); an
// invariant S holds iff S.is_intersecting(R).
// Both are the set classes' own operations through the binary registry (cora::contains,
// cora::is_intersecting), so the exactness is theirs: a Polytope decides a zonotope, a box, a
// constrained zonotope or a bundle exactly by support functions or a linear program. A pair of
// classes without an operation is Error::noops. A custom specification asks its predicate, once
// for the whole set. A batch of sets is not supported.
//
// Syntax:   ok = spec.check(R);   ok = spec.check(R, {t0, t1});   ok = spec.check(reach_set);
//           ok = spec.check_points(points, times);
// Inputs:   R - any set, over the times t0..t1 (needed by a timed specification)
//           points - (N, n) points, with times: none, one, or one per point
//           reach_set - a ReachSet
// Outputs:  ok - whether R (every set of the sequence) satisfies the specification
// See also: Specification::first_violation, Polytope::contains, Polytope::is_intersecting
//
// CORA: specification/@specification/check.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "cora/global/within_tol.h"
#include "cora/specification/specification.h"
#include "specification/private/priv_polytope_of.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The refusal of the operands of the operation (they differ in dimension), if any; the entry of
/// check raises it, since a nested entry does not check.
detail::Refusal aux_refusal(BinaryOp op, const ContSet &S, const ContSet &R) {
    return detail::refusal(op, S, R, std::nullopt);
}

/// Whether the windows [a0, a1] and [b0, b1] intersect, up to tol.
bool aux_overlap(const TimeSpan &a, const TimeSpan &b, double tol) {
    return a.first <= b.second + tol && b.first <= a.second + tol;
}

/// Whether the instant t lies in the window, up to tol.
bool aux_contains(const TimeSpan &window, double t, double tol) {
    return window.first - tol <= t && t <= window.second + tol;
}

/// The time of step j of a reach set (CORA's reachSet stores it with every set).
double aux_step_time(const ReachSet &R, std::size_t j) {
    return R.time.empty() ? static_cast<double>(j) * R.time_step : R.time[j];
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Checking one set --------------------------------------------------------------------------------

bool Specification::check_in(const ContSet &R, const std::optional<TimeSpan> &time,
                             double tol) const {
    if (time_ && !time) // not an input check: the window below needs the time
        CORA_FAIL(Error::special_error, "check: a timed specification requires a time interval");
    if (time_ && !aux_overlap(*time_, *time, tol)) return true; // not active here
    if (type_ == SpecType::custom) return custom_(R);
    if (type_ == SpecType::logic)
        CORA_FAIL(Error::not_supported, "check: a logic specification is checked on traces");
    // the set classes decide, through the unchecked forms of the registry
    const ContSet &S = set_->get();
    const BinaryOp op =
        type_ == SpecType::safe_set ? BinaryOp::contains : BinaryOp::is_intersecting;
    const detail::Refusal refusal = aux_refusal(op, S, R);
    if (refusal) CORA_FAIL(refusal->first, "check: " + refusal->second);
    if (type_ == SpecType::safe_set) return detail::contains_impl(S, R, std::nullopt);
    const bool intersecting = detail::is_intersecting_impl(S, R, std::nullopt);
    return type_ == SpecType::invariant ? intersecting : !intersecting;
}

bool Specification::check(const ContSet &R) const {
    detail::Entry entry;
    return check_in(R, std::nullopt, resolve_tol(entry, std::nullopt, "check"));
}

bool Specification::check(const ContSet &R, const TimeSpan &time) const {
    detail::Entry entry;
    CORA_CHECK(time.first <= time.second, Error::wrong_value,
               [] { return std::string("check: the time window [t0, t1] needs t0 <= t1"); });
    return check_in(R, time, resolve_tol(entry, std::nullopt, "check"));
}

// Checking points and reachable sets --------------------------------------------------------------

bool Specification::check_points(const Tensor &points, const std::vector<double> &times) const {
    detail::Entry entry;
    CORA_CHECK(type_ != SpecType::custom && type_ != SpecType::logic, Error::not_supported, [] {
        return std::string("check_points: only a set specification has a points form");
    });
    CORA_CHECK(points.ndim() == 2, Error::wrong_value,
               [] { return std::string("check_points: give the points as (N, n), one per row"); });
    const std::size_t N = static_cast<std::size_t>(points.shape()[0]);
    const int64_t n = points.shape()[1];
    CORA_CHECK(times.size() <= 1 || times.size() == N, Error::special_error, [] {
        return std::string("check_points: the times are empty, one, or one per point");
    });
    CORA_CHECK(!time_ || !times.empty(), Error::special_error,
               [] { return std::string("check: a timed specification requires a time interval"); });
    const double tol = resolve_tol(entry, std::nullopt, "check_points");
    // a point in the window must satisfy the type of the specification
    const Polytope P = priv_polytope_of(*set_);
    const std::vector<double> data = points.data();
    for (std::size_t i = 0; i < N; ++i) {
        if (time_ && !aux_contains(*time_, times.size() == 1 ? times[0] : times[i], tol)) continue;
        // one point against the set
        const std::vector<double> row(data.begin() + i * n, data.begin() + (i + 1) * n);
        const bool inside = P.contains(Tensor::like(points, row, {n}));
        if (inside == (type_ == SpecType::unsafe_set)) return false;
    }
    return true;
}

bool Specification::check(const ReachSet &R, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, "check");
    const bool intervals = !R.time_interval.empty();
    const std::vector<Zonotope> &sets = intervals ? R.time_interval : R.time_point;
    if (!time_) {
        for (const Zonotope &S : sets)
            if (!check_in(S, std::nullopt, tol)) return false;
        return true;
    }
    CORA_CHECK(R.time_step > 0 || !R.time.empty(), Error::special_error, [] {
        return std::string("check: the times of the reach set are unknown (time_step is 0)");
    });
    // the steps that meet the window, each against the set that stands for it
    const std::size_t K = sets.size();
    const auto time_at = [&](std::size_t j) { return aux_step_time(R, j); };
    // a window outside the horizon is a violation (CORA: find(R, 'time') is empty)
    if (K == 0 || !aux_overlap(*time_, {time_at(0), time_at(intervals ? K : K - 1)}, tol))
        return false;
    for (std::size_t j = 0; j < K; ++j) {
        const TimeSpan step{time_at(j), intervals ? time_at(j + 1) : time_at(j)};
        if (!aux_overlap(*time_, step, tol)) continue;
        const ContSet *set = &sets[j];
        TimeSpan at = step;
        if (intervals &&
            time_->second <= time_->first) { // one instant: the time point on the border
            const std::size_t k = within_tol(time_->first, step.first, tol)    ? j
                                  : within_tol(time_->first, step.second, tol) ? j + 1
                                                                               : K + 1;
            if (k < R.time_point.size()) {
                set = &R.time_point[k];
                at = {time_at(k), time_at(k)};
            }
        }
        if (!check_in(*set, at, tol)) return false;
    }
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
