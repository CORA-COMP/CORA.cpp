// priv - the private functions of location, as CORA's @location methods that reach calls
//
// Not for callers of the class: HybridAutomaton::reach uses them. A reachable set is a zonotope or
// a zonotope bundle (a Set); a guard intersection that is empty is none.
//
// Syntax:   LocationResult res = priv_location_reach(loc, R0, t_start, t_final, U, dt, alg, ...);
// See also: hybrid_automaton/reach.cpp, location.h
//
// CORA.cpp only: the declarations of the private functions of location.
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#pragma once

#include "cora/hybrid_dynamics/hybrid_automaton/hybrid_automaton.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The reach of a location -------------------------------------------------------------------------

/// A jump out of a location: the set after the reset, the target, and the time of the jump.
struct GuardJump {
    Set set;
    int64_t target;
    TimeRange time;
};

/// What the reach of one location returns (CORA's Rtemp, Rjump and res).
struct LocationResult {
    LocationReachSet R;
    std::vector<GuardJump> jumps;
    bool verified = true;
};

/// Which sets meet which guards (CORA's potInt): per hit, the transition and the set index; the
/// sets are time-point sets if `time_point` comes back true.
struct GuardHits {
    std::vector<int64_t> transitions, sets;
    bool time_point = false;
};

/// The input set in the state space, B U + c (U none: c alone), which the flow then adds as the
/// input of the canonical system x' = A x + u; for a nonlinear flow the input set of f itself.
Zonotope priv_state_input(const Location &loc, const std::optional<Zonotope> &U);

/// The flow at the point x (n, 1) with the input at the center of U_state (priv_state_input):
/// A x + center of B U + c, or f(x, u).
/// A LinParamSys flow fails with Error::not_supported, naming `what`.
Tensor priv_flow_at(const Location &loc, const Tensor &x, const Zonotope &U_state,
                    const char *what);

/// The reach of one location from R0 at the times t_start up to t_final with the input set U (of
/// the flow's B), the specifications `specs` that hold there, and no jumps into final_locations.
LocationResult priv_location_reach(const Location &loc, int64_t index, const Set &R0,
                                   const TimeRange &t_start, double t_final,
                                   const std::optional<Zonotope> &U, double time_step,
                                   const alg::HybridStandard &alg,
                                   const std::vector<Specification> &specs,
                                   const std::vector<int64_t> &final_locations);

// The guards --------------------------------------------------------------------------------------

/// Whether a set (zonotope or bundle) meets a polytope by CORA's 'approx' test.
bool priv_meets(const Polytope &P, const Set &S);

/// The guards that the sets of R meet (CORA's potInt), with CORA's switch to the time-point set
/// when a time-triggered hyperplane is hit exactly between two steps.
GuardHits priv_pot_int(const Location &loc, const LocationReachSet &R,
                       const std::vector<int64_t> &final_locations);

/// One crossing of a guard: the transition and the indices of its sets, without gaps.
struct GuardGroup {
    int64_t transition;
    std::vector<int64_t> sets;
};

/// CORA's aux_groupSets and aux_removeGaps: the hits of each guard (in the order of the guard
/// numbers), split at every gap; the first part of a split goes to the front of the list.
std::vector<GuardGroup> priv_guard_groups(const GuardHits &hits);

/// The enclosures of the hit guards (CORA's guardIntersect), one per group of neighbouring sets
/// that meet the same guard: the transition, the first and the last set index, the set.
struct GuardIntersection {
    int64_t transition, first, last;
    Set set;
};
std::vector<GuardIntersection> priv_guard_intersect(const Location &loc, const GuardHits &hits,
                                                    const LocationReachSet &R,
                                                    const Zonotope &U_state,
                                                    const alg::GuardIntersect &method);

/// The orthogonal bases of the enclosure (CORA's calcBasis), one per method of `enclose`.
std::vector<Tensor> priv_calc_basis(const Location &loc, const std::vector<Set> &R,
                                    const Polytope &guard, const std::vector<alg::Enclose> &enclose,
                                    const Zonotope &U_state);

/// The sets whose flow may point into the hyperplane guard (CORA's checkFlow).
std::vector<Set> priv_check_flow(const Location &loc, const Polytope &guard,
                                 const std::vector<Set> &R, const Zonotope &U_state);

/// The same for the hyperplane a'x = d with the outward unit normal a (the caller's choice).
std::vector<Set> priv_check_flow_outward(const Location &loc, const Tensor &a, double d,
                                         const std::vector<Set> &R, const Zonotope &U_state);

/// The enclosures of the guard intersection of the sets R per basis (CORA's
/// guardIntersect_zonoGirard, _nondetGuard and _conZonotope); none if it is empty.
std::optional<Set> priv_guard_intersect_zono_girard(const std::vector<Set> &R,
                                                    const Polytope &guard,
                                                    const std::vector<Tensor> &B);
std::optional<Set> priv_guard_intersect_nondet_guard(const std::vector<Set> &R,
                                                     const Polytope &guard,
                                                     const std::vector<Tensor> &B,
                                                     std::optional<int> fm_passes);
std::optional<Set> priv_guard_intersect_con_zonotope(const std::vector<Set> &R,
                                                     const Polytope &guard,
                                                     const std::vector<Tensor> &B, int guard_order,
                                                     std::optional<int> fm_passes);

/// The zonotopes of a set: itself, or those of a bundle.
std::vector<Zonotope> priv_zonotopes(const Set &S);

/// One zonotope as itself, several as their bundle.
Set priv_bundle(std::vector<Zonotope> parts);

/// The orthonormal basis (n, n) that CORA's gramSchmidt makes of the column a and the axes: the
/// first column is a / |a| (polytope's priv_gram_schmidt as a matrix like a).
Tensor priv_basis_of(const Tensor &a);

/// The unit normal and offset of a hyperplane guard (its first equality).
std::pair<Tensor, double> priv_hyperplane(const Polytope &guard);

/// The matrix A of a LinearSys flow; a LinParamSys has none to evaluate (CORA's getfcn needs the
/// sampleMatrix of a simulation), nor a NonlinearSys, so `what` fails with Error::not_supported.
const Tensor &priv_flow_matrix(const Location &loc, const char *what);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
