// lift - a box placed into a higher-dimensional space, as CORA's interval.lift_
//
// Syntax:   Interval R = I.lift(N, dims);
// Inputs:   N - dimension of the space, N >= dim
//           dims - the N-dimensional coordinates (0-based)
//           of the dim coordinates of I, in the order of I
// Outputs:  R - the box with coordinate dims[j] equal to that of I and (-Inf, Inf) elsewhere; for
//           a repeated entry of dims the last one counts, as CORA's assignment
// See also: project
//
// CORA: contSet/@interval/lift_.m
// CORA authors: Tobias Ladner, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_embed.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::Refusal r = detail::project_high_dim_refusal(dim(), N, dims, "lift");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    if (is_empty())
        CORA_FAIL(Error::not_supported, "lift: the operation is not supported for empty intervals");
    const double kInf = std::numeric_limits<double>::infinity();
    return priv_embed(*this, N, dims, -kInf, kInf);
}

Set Interval::lift_impl(int64_t N, const std::vector<int64_t> &proj) const { return lift(N, proj); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
