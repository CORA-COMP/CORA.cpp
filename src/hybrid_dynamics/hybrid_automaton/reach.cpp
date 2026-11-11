// reach - the reachable sets of a hybrid automaton, as CORA's hybridAutomaton.reach
//
// A queue holds the sets that enter a location (first the initial set): each is analysed in its
// location until its sets leave the invariant or reach t_final, its jumps join the end of the
// queue with their uncertain times, and the location's sets are stored as one branch. The
// analysis stops when the queue is empty or a specification is violated. CORA's instant
// transitions (a guard that is all of R^n) and output maps are not supported.
//
// Syntax:   HybridReachSet R = HA.reach(problem, alg, specs);
// Inputs:   problem - initial_set, start_location, t_start, t_final, input_set, final_locations
//           alg - alg::HybridStandard (time_step per location, the guard intersection)
//           specs - specifications with the locations where they hold
// Outputs:  R - the branches (location, parent, sets and times) and whether the specs held
// See also: Location, priv_location_reach
//
// CORA: hybridDynamics/@hybridAutomaton/reach.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "global/format.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>
#include <cmath>
#include <deque>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The queue ---------------------------------------------------------------------------------------

/// A set waiting for its location (CORA's list): where, when, and which branch made it.
struct Pending {
    Set set;
    int64_t location;
    TimeRange time;
    int64_t parent;
};

// The checks --------------------------------------------------------------------------------------

/// The checks of the problem, the algorithm and the specifications against the automaton.
void aux_validate(const HybridAutomaton &HA, const HybridReachProblem &p,
                  const alg::HybridStandard &a, const std::vector<HybridSpecification> &specs) {
    detail::Validation entry;
    const int64_t n = HA.dim(), count = int64_t(HA.locations().size());
    // exactly one initial set, of the dimension, and the algorithm that goes with its class
    CORA_CHECK(p.initial_set.has_value() != p.initial_poly_set.has_value(), Error::wrong_value, [] {
        return std::string("reach: the problem needs exactly one of initial_set and "
                           "initial_poly_set");
    });
    CORA_CHECK(
        p.initial_poly_set ? p.initial_poly_set->dim() == n
                           : (p.initial_set->center().shape() == std::vector<int64_t>({n, 1}) &&
                              p.initial_set->generators().ndim() == 2),
        Error::wrong_value, [&] {
            return "reach: the initial set must be one set of the dimension " + std::to_string(n);
        });
    // the algorithm goes with the class of the initial set
    CORA_CHECK(p.initial_poly_set.has_value() == a.polynomialization, Error::wrong_value, [] {
        return std::string("reach: a polynomial zonotope as initial_poly_set goes with "
                           "polynomialization = true, and only with it");
    });
    CORA_CHECK(p.start_location >= 0 && p.start_location < count, Error::wrong_value, [&] {
        return "reach: start_location must be a location from 0 to " + std::to_string(count - 1) +
               ", got " + std::to_string(p.start_location);
    });
    CORA_CHECK(std::isfinite(p.t_start) && std::isfinite(p.t_final) && p.t_final > p.t_start,
               Error::wrong_value, [&] {
                   return "reach: t_final must be finite and after t_start, got " +
                          std::to_string(p.t_start) + " and " + std::to_string(p.t_final);
               });
    // one input set for all locations or one each, of the columns of each B
    const std::size_t inputs = p.input_set.size();
    CORA_CHECK(inputs <= 1 || inputs == std::size_t(count), Error::wrong_value, [&] {
        return "reach: input_set must hold none, one, or one per location (" +
               std::to_string(count) + "), got " + std::to_string(inputs);
    });
    for (std::size_t i = 0; i < std::size_t(count) && inputs > 0; ++i) {
        const Zonotope &U = p.input_set[inputs == 1 ? 0 : i];
        CORA_CHECK(U.center().ndim() == 2 && U.dim() == HA.locations()[i].input_dim(),
                   Error::dimension_mismatch, [&] {
                       return "reach: the input set of location " + std::to_string(i) +
                              " must have the dimension " +
                              std::to_string(HA.locations()[i].input_dim()) +
                              " of the columns of its B, got " + std::to_string(U.dim());
                   });
    }
    // the step sizes, and the locations the specifications name
    CORA_CHECK(a.time_step.size() == 1 || a.time_step.size() == std::size_t(count),
               Error::wrong_value, [&] {
                   return "reach: time_step must hold one step size or one per location (" +
                          std::to_string(count) + "), got " + std::to_string(a.time_step.size());
               });
    for (const HybridSpecification &s : specs)
        for (int64_t loc : s.locations)
            CORA_CHECK(loc >= 0 && loc < count, Error::wrong_value, [&] {
                return "reach: a specification names the location " + std::to_string(loc) +
                       ", but there are " + std::to_string(count);
            });
}

// The specifications per location -----------------------------------------------------------------

/// The specifications that are active in the location (no list: everywhere).
std::vector<Specification> aux_specs_of(const std::vector<HybridSpecification> &specs,
                                        int64_t loc) {
    std::vector<Specification> out;
    for (const HybridSpecification &s : specs)
        if (s.locations.empty() ||
            std::find(s.locations.begin(), s.locations.end(), loc) != s.locations.end())
            out.push_back(s.spec);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

HybridReachSet HybridAutomaton::reach(const HybridReachProblem &problem,
                                      const alg::HybridReach &alg,
                                      const std::vector<HybridSpecification> &specs) const {
    detail::Entry entry;
    const auto &a = std::get<alg::HybridStandard>(alg);
    if (entry.check()) {
        a.validate();
        aux_validate(*this, problem, a, specs);
    }
    HybridReachSet out;
    std::deque<Pending> queue{
        {problem.initial_poly_set ? Set(*problem.initial_poly_set) : Set(*problem.initial_set),
         problem.start_location,
         {problem.t_start, problem.t_start},
         -1}};
    // one location at a time, first come first served, until the queue is empty or a violation
    while (!queue.empty() && out.verified) {
        const Pending next = std::move(queue.front());
        queue.pop_front();
        const auto loc = std::size_t(next.location);
        std::optional<Zonotope> U;
        if (!problem.input_set.empty())
            U = problem.input_set[problem.input_set.size() == 1 ? 0 : loc];
        const double dt = a.time_step[a.time_step.size() == 1 ? 0 : loc];
        LocationResult res = priv_location_reach(
            locations_[loc], next.location, next.set, next.time, problem.t_final, U, dt, a,
            aux_specs_of(specs, next.location), problem.final_locations);
        // the branch, then its jumps at the end of the queue
        res.R.parent = next.parent;
        const auto branch = int64_t(out.branches.size());
        out.branches.push_back(std::move(res.R));
        out.verified = res.verified;
        for (GuardJump &jump : res.jumps)
            queue.push_back({std::move(jump.set), jump.target, jump.time, branch});
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
