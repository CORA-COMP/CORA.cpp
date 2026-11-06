// project_high_dim - a box placed into a larger space, as CORA's interval.projectHighDim_
//
// Syntax:   Interval R = I.project_high_dim(N, proj);
// Inputs:   N - dimension of the space, N >= dim
//           proj - the N-dimensional coordinates (0-based)
//           of the dim coordinates of I, in the order of I
// Outputs:  R - the box with coordinate proj[j] equal to that of I and [0, 0] elsewhere; for a
//           repeated entry of proj the last one counts; the empty interval stays empty
// See also: lift, project
//
// CORA: contSet/@interval/projectHighDim_.m
// CORA: contSet/@contSet/projectHighDim.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_embed.h"
#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::project_high_dim(int64_t N, const std::vector<int64_t> &proj) const {
    detail::Entry entry;
    const detail::Refusal r = detail::project_high_dim_refusal(dim(), N, proj);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return project_high_dim_impl(N, proj).as<Interval>();
}

Set Interval::project_high_dim_impl(int64_t N, const std::vector<int64_t> &proj) const {
    if (is_empty()) return Interval::empty(N);
    return priv_embed(*this, N, proj, 0.0, 0.0);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
