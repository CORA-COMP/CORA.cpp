// is_full_dim - whether a box has a nonempty interior, as CORA's interval.isFullDim
//
// Syntax:   bool b = I.is_full_dim();
// Inputs:   -
// Outputs:  b - false for an empty interval, else true if no radius is within the
//           repository tolerance of zero (as within_tol); for a batch, true if every member is
// See also: is_bounded, isequal
//
// CORA: contSet/@interval/isFullDim.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Interval::is_full_dim() const {
    detail::Entry entry;
    if (is_empty()) return false;
    const Tensor r = rad();
    return within_tol(r, r.zeros_like()).max().item() == 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
