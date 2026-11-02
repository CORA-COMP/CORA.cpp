// support_function - the support function of a set in a direction, checked, as CORA's contSet.supportFunc
//
// Syntax:   val = S.support_function(d);
// Inputs:   d - direction, a column (..., n, 1) of the dimension n of S
// Outputs:  val - max_{x in S} d'x, (..., 1, 1)
// See also: interval, Zonotope::support_function_impl
//
// CORA: contSet/@contSet/supportFunc.m

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ContSet::support_function(const Tensor &d) const {
    detail::Entry entry;
    CORA_CHECK(is_column(d, dim()), Error::dimension_mismatch, [&] {
        return "support_function: the direction must be a column of the set's dimension " +
               std::to_string(dim()) + ", got the shape " + shape_text(d.shape());
    });
    return support_function_impl(d);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
