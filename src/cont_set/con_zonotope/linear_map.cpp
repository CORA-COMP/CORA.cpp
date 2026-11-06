// linear_map - the linear map M * cz of a constrained zonotope, as CORA's mtimes
//
// The constraints stay (the factors are the same).
//
// Syntax:   cZres = cz.linear_map(M);   cZres = M * cz;
// Inputs:   M - a matrix (..., m, n) with the dimension n of cz as its columns
// Outputs:  cZres - exact, of dimension m: the center and the generators map
// See also: plus, Zonotope::linear_map
//
// CORA: contSet/@conZonotope/mtimes.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.shape().back() == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns (the dimension of the set), got the shape " + shape_text(M.shape());
    });
    return priv_affine_image(*this, M, std::nullopt);
}

Set ConZonotope::linear_map_impl(const Tensor &M) const {
    return priv_affine_image(*this, M, std::nullopt);
}

// -cz is (-1) * cz in CORA's uminus: no matrix is needed
Set ConZonotope::uminus_impl() const { return ConZonotope(c_ * -1.0, G_ * -1.0, A_, b_); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
