// project_high_dim - a box placed into a larger space, as CORA's interval.projectHighDim_
//
// Syntax:   Interval R = I.project_high_dim(N, proj);
// Inputs:   N - dimension of the space, N >= dim;  proj - the N-dimensional coordinates (0-based)
//           of the dim coordinates of I, in the order of I
// Outputs:  R - the box with coordinate proj[j] equal to that of I and [0, 0] elsewhere; for a
//           repeated entry of proj the last one counts; the empty interval stays empty
// See also: lift, project
//
// CORA: contSet/@interval/projectHighDim_.m
// CORA: contSet/@contSet/projectHighDim.m

#include "cont_set/interval/private/priv_embed.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::project_high_dim(int64_t N, const std::vector<int64_t> &proj) const {
    detail::Entry entry;
    CORA_CHECK(N >= dim(), Error::wrong_value, [&] {
        return "project_high_dim: the space of " + std::to_string(N) +
               " dimensions must not be smaller than the interval of " + std::to_string(dim());
    });
    CORA_CHECK(static_cast<int64_t>(proj.size()) == dim(), Error::wrong_value, [&] {
        return "project_high_dim: one coordinate per dimension of the interval is needed, got " +
               std::to_string(proj.size()) + " for " + std::to_string(dim());
    });
    for (const int64_t d : proj)
        CORA_CHECK(d >= 0 && d < N, Error::wrong_value, [&] {
            return "project_high_dim: the coordinate " + std::to_string(d) + " is not in [0, " +
                   std::to_string(N) + ")";
        });
    if (is_empty()) return Interval::empty(N);
    return priv_embed(*this, N, proj, 0.0, 0.0);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
