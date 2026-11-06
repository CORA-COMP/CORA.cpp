// lift - a set placed into a space of higher dimension, as CORA's contSet.lift
//
// The classes are Interval, EmptySet, Fullspace and Zonotope::lift; proj has n entries in [0, N).
// Error::noops for a class without it.
//
// Syntax:   R = S.lift(N);   R = S.lift(N, proj);
// Inputs:   S - set of dimension n
//           N - dimension of the space, N >= n
//           proj - the position (0-based) of each coordinate of S in it (default: 0..n-1)
// Outputs:  R - the set in N dimensions, as the class defines it
// See also: project_high_dim, project
//
// CORA: contSet/@contSet/lift.m
// CORA: contSet/@contSet/lift_.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

namespace detail {


LiftPlan lift_plan(int64_t n, int64_t N, const std::vector<int64_t> &dims) {
    LiftPlan plan;
    plan.refusal = project_high_dim_refusal(n, N, dims, "lift");
    if (plan.refusal) return plan;
    if (int64_t(dims.size()) != N) {
        plan.refusal = std::pair{Error::not_defined,
                                 std::string("lift: the new dimensions cannot be unbounded, as the "
                                             "set is always bounded; dims must be a permutation of "
                                             "all N states")};
        return plan;
    }
    // the state at position j of the result is the state i with dims[i] = j
    plan.source.assign(std::size_t(N), -1);
    for (int64_t i = 0; i < N; ++i) {
        if (plan.source[dims[i]] >= 0) {
            const std::string position = std::to_string(dims[i]);
            plan.refusal =
                std::pair{Error::wrong_value, "lift: the position " + position + " is used twice"};
            return plan;
        }
        plan.source[dims[i]] = i;
    }
    return plan;
}

} // namespace detail


Set ContSet::lift(int64_t N, const std::optional<std::vector<int64_t>> &proj) const {
    detail::Entry entry;
    std::vector<int64_t> positions(dim());
    if (proj)
        positions = *proj;
    else
        std::iota(positions.begin(), positions.end(), int64_t{0});
    const detail::Refusal r = detail::project_high_dim_refusal(dim(), N, positions, "lift");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return lift_impl(N, positions);
}

Set ContSet::lift_impl(int64_t, const std::vector<int64_t> &) const { priv_noops("lift", *this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
