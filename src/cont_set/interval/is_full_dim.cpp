// is_full_dim - whether a box has a nonempty interior, as CORA's interval.isFullDim
//
// Syntax:   bool b = I.is_full_dim();   bool b = I.is_full_dim(tol);
// Inputs:   tol - how close to zero a radius may be; default: the repository tolerance
//           (docs/architecture.md)
// Outputs:  b - false for an empty interval, else true if no radius is within tol of zero (as
//           within_tol); for a batch, true if every member is
// See also: is_bounded, isequal
//
// CORA: contSet/@interval/isFullDim.m

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
