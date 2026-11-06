// linear_map - the image of a capsule under a matrix, as CORA's capsule.mtimes
//
// Syntax:   Cres = C.linear_map(M);   Cres = M * C;
// Inputs:   M - matrix (..., m, n)
// Outputs:  Cres - the capsule of M c, M g and the radius r |M|_F: the Frobenius norm bounds the
//           operator norm, so the image is enclosed. CORA scales the radius by max(eig(M M')),
//           the square of the operator norm, which is too small for a map that shrinks
//           (docs/cora-differences.md). The empty capsule stays empty
// See also: lift, Zonotope::linear_map
//
// CORA: contSet/@capsule/mtimes.m

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Capsule Capsule::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.defined() && M.ndim() >= 2 && M.size(-1) == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns (the dimension of C), got the shape " + shape_text(M.shape());
    });
    const auto [c, g] = Tensor::zonotope_linear_map(M, c_, g_);
    const Tensor scaled = r_.multiply(M.matrix_norm(Tensor::Norm::fro));
    // the radius -1 of the empty capsule is a marker, not a length
    return {Unchecked{}, c, g, Tensor::where(priv_empty_mask(r_), r_, scaled)};
}

Set Capsule::linear_map_impl(const Tensor &M) const { return linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
