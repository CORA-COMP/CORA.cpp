// box - the smallest box around a polytope, as a polytope, as CORA's polytope.box
//
// Syntax:   Polytope B = P.box();
// Inputs:   P - polytope
// Outputs:  B - the polytope of the rows x <= sup and -x <= -inf of the bounds of interval (a row
//           with an infinite bound is left out); R^n for R^n, the empty set for the empty set.
//           A vertex polytope may be a batch, a halfspace polytope may not
// See also: interval, Polytope::Polytope
//
// CORA: contSet/@polytope/box.m

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::box() const {
    detail::Entry entry;
    const int64_t n = dim();
    if (priv_is_fullspace(*this)) return inf(n);
    const Interval I = interval_impl();
    if (I.is_empty()) return empty(n);
    const Tensor eye = priv_identity(I.inf(), n);
    return Polytope(Tensor::concat({eye, -eye}, -2), Tensor::concat({I.sup(), -I.inf()}, -2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
