// linear_map - the image of the empty set under a matrix, as CORA's emptySet.mtimes
//
// Syntax:   EmptySet R = E.linear_map(M);   EmptySet R = M * E;   EmptySet R = s * E;
// Inputs:   M - a matrix (..., k, n), or an interval matrix of such bounds; s - a number (or a
//           1 x 1 matrix)
// Outputs:  R - the empty set of k dimensions; for a number, E itself
// See also: lift, Zonotope::linear_map
//
// CORA: contSet/@emptySet/mtimes.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet EmptySet::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2, Error::dimension_mismatch, [&] {
        return "linear_map: the matrix must have two dimensions, got the shape " +
               shape_text(M.shape());
    });
    if (M.shape() == std::vector<int64_t>({1, 1})) return *this; // a number, as CORA's isscalar
    CORA_CHECK(M.size(-1) == n_, Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(n_) +
               " columns (the dimension of the set), got the shape " + shape_text(M.shape());
    });
    return EmptySet(M.size(-2));
}

EmptySet EmptySet::linear_map(const Interval &M) const { return linear_map(M.inf()); }

EmptySet EmptySet::linear_map(double) const {
    detail::Entry entry;
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
