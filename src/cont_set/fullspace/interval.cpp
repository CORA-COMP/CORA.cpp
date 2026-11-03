// interval - the interval hull of the whole space, as CORA's fullspace.interval
//
// Syntax:   Interval I = F.interval();
// Inputs:   -
// Outputs:  I - (-Inf, Inf)^n; Error::out_of_domain in R^0, as CORA
// See also: Interval::unbounded, box
//
// CORA: contSet/@fullspace/interval.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Fullspace::interval_impl() const {
    if (n_ == 0) CORA_FAIL(Error::out_of_domain, "interval: the dimension must be positive in R^0");
    return Interval::unbounded(n_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
