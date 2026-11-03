// lift - a set placed into a space of higher dimension, as CORA's contSet.lift
//
// Syntax:   R = S.lift(N);   R = S.lift(N, proj);
// Inputs:   S - set of dimension n;  N - dimension of the space, N >= n;  proj - the position
//           (0-based) of each coordinate of S in it, n entries in [0, N) (default: 0..n-1)
// Outputs:  R - the set in N dimensions, as the class defines it (Interval, EmptySet, Fullspace,
//           Zonotope::lift); Error::noops for a class without it
// See also: project_high_dim, project
//
// CORA: contSet/@contSet/lift.m
// CORA: contSet/@contSet/lift_.m

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

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
