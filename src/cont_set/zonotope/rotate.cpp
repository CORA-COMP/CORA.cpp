// rotate - the zonotope turned in the plane of two states
//
// The two rows are multiplied by [cos(angle) -sin(angle); sin(angle) cos(angle)]; the other
// states are unchanged.
//
// Syntax:   Zonotope Zr = Z.rotate(dims, angle);
// Inputs:   dims - the two states (0-based, different) that span the plane
//           angle - in radians
// Outputs:  Zr - the zonotope with the center and every generator rotated on those two rows
// See also: linear_map
//
// CORA: contSet/@zonotope/rotate.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows dims of the matrix (..., n, k) turned by R (2, 2), the other rows as they are.
Tensor aux_rotate_rows(const Tensor &M, const Tensor &R, const std::vector<int64_t> &dims) {
    const int64_t n = M.shape()[M.ndim() - 2];
    const Tensor turned = R.matmul(M.take(dims, -2));
    std::vector<int64_t> source(n);
    for (int64_t i = 0; i < n; ++i) source[i] = i;
    source[dims[0]] = n;
    source[dims[1]] = n + 1;
    return Tensor::concat({M, turned}, -2).take(source, -2);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::rotate(const std::vector<int64_t> &dims, double angle) const {
    detail::Entry entry;
    CORA_CHECK(dims.size() == 2 && dims[0] != dims[1], Error::wrong_value,
               [] { return std::string("rotate: dims must be two different states"); });
    for (const int64_t d : dims)
        CORA_CHECK(d >= 0 && d < dim(), Error::wrong_value, [&] {
            return "rotate: the state " + std::to_string(d) + " is not in [0, " +
                   std::to_string(dim()) + ")";
        });
    CORA_CHECK(!std::isnan(angle), Error::wrong_value,
               [] { return std::string("rotate: the angle must not be NaN"); });
    const double cosine = std::cos(angle), sine = std::sin(angle);
    const Tensor R = Tensor::like(c_, {cosine, -sine, sine, cosine}, {2, 2});
    return {aux_rotate_rows(c_, R, dims), aux_rotate_rows(G_, R, dims)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
