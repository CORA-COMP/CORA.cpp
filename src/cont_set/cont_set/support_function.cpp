// support_function - the support of a set in a direction, checked, as CORA's supportFunc
//
// The class's support_function_impl is CORA's supportFunc_.
//
// Syntax:   val = S.support_function(d);
// Inputs:   d - direction (..., n) of the dimension n of S
// Outputs:  val - max_{x in S} d'x, (...)
// See also: interval, Zonotope::support_function_impl
//
// CORA: contSet/@contSet/supportFunc.m
// CORA: contSet/@contSet/supportFunc_.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ContSet::support_function(const Tensor &d) const {
    detail::Entry entry;
    CORA_CHECK(is_vector(d, dim()), Error::dimension_mismatch, [&] {
        return "support_function: the direction must be a vector of the set's dimension " +

               std::to_string(dim()) + ", got the shape " + shape_text(d.shape());
    });
    entry.release();
    return support_function_impl(d);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
