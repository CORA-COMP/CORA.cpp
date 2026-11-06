// is_full_dim - whether a box has a nonempty interior, as CORA's interval.isFullDim
//
// The comparison is within_tol. For a batch, true if every member is.
//
// Syntax:   bool b = I.is_full_dim();   bool b = I.is_full_dim(tol);
// Inputs:   tol - how close to zero a radius may be; default: the repository tolerance
// Outputs:  b - false for an empty interval, else true if no radius is within tol of zero
// See also: is_bounded, isequal
//
// CORA: contSet/@interval/isFullDim.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Interval::is_full_dim(Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, inf_.dtype(), "is_full_dim");
    if (is_empty()) return false;
    const Tensor r = rad();
    return within_tol(r, r.zeros_like(), tol).max().item() == 0;
}

bool Interval::is_full_dim_impl(Tol tol) const { return is_full_dim(tol); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
