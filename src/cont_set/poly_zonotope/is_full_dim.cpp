// is_full_dim - whether a polynomial zonotope has a nonempty interior, as CORA's isFullDim
//
// CORA tests rank([G GI]) = n, which is necessary and not sufficient: (a, a^2) has rank 2 and no
// interior (CORALean's counterexample). Here a rank below n is false, and the rank is accepted as
// sufficient where it is: the independent generators alone span the space (the set contains a
// translate of that zonotope), or the dependent part is a zonotope (each factor in its own
// generator). Any other set is reported false, which only says that no proof was found
// (docs/cora-differences.md).
//
// Syntax:   res = pZ.is_full_dim();   res = pZ.is_full_dim(tol);
// Inputs:   tol - rank tolerance; default: the repository tolerance
// Outputs:  res - true if the set is shown to have a nonempty interior
// See also: is_bounded, compact
//
// CORA: contSet/@polyZonotope/isFullDim.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool PolyZonotope::is_full_dim(Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_full_dim");
    const int64_t n = dim();
    if (G_.shape()[0] + GI_.shape()[0] == 0) return n == 0;
    // the independent generators alone span the space: the set holds a translate of their zonotope
    if (GI_.shape()[0] > 0 && GI_.matrix_rank(tol) == n) return true;
    if (Tensor::concat({G_, GI_}, -2).matrix_rank(tol) < n) return false;
    return priv_is_zonotope(*this);
}

bool PolyZonotope::is_full_dim_impl(Tol tol) const { return is_full_dim(tol); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
