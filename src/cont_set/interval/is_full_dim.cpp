// is_full_dim - whether a box has a nonempty interior, as CORA's interval.isFullDim
//
// The comparison is within_tol.
//
// Syntax:   bool b = I.is_full_dim();   bool b = I.is_full_dim(tol);
// Inputs:   tol - how close to zero a radius may be; default: the repository tolerance
// Outputs:  b - false for an empty interval, else true if no radius is within tol of zero (a mask)
// See also: is_bounded, isequal
//
// CORA: contSet/@interval/isFullDim.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_present.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Mask Interval::is_full_dim(Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, inf_.dtype(), "is_full_dim");
    const Tensor r = half_width();
    // no radius within tol of zero, and not empty
    const Tensor wide = r.ones_like() - within_tol(r, r.zeros_like(), tol);
    return Mask(priv_all_entries(wide, *this).minimum(priv_present(*this)));
}

bool Interval::is_full_dim_impl(Tol tol) const { return is_full_dim(tol).all(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
