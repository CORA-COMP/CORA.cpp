// priv_embed - a box placed into the coordinates of a higher-dimensional space
//
// Syntax:   Interval R = priv_embed(I, N, dims, lower, upper);
// Inputs:   I - the box (m rows);  N - the dimension of the space;  dims - the coordinate of each
//           of the m rows, in [0, N) (the shapes are not checked);  lower, upper - the bounds of
//           the coordinates that I does not occupy
// Outputs:  R - the box in N dimensions; for a repeated entry of dims the last one counts
// See also: lift, project_high_dim

#pragma once

#include "cora/cont_set/interval/interval.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Interval priv_embed(const Interval &I, int64_t N, const std::vector<int64_t> &dims, double lower,
                    double upper);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
