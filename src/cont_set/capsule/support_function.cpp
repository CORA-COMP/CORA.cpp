// support_function - the support function of a capsule, as CORA's capsule.supportFunc_
//
// The largest value of d'x over the capsule: d'c + |d'g| + r |d|. The empty capsule has -Inf.
//
// Syntax:   val = C.support_function(d);
// Inputs:   d - direction (..., n, 1)
// Outputs:  val - the maximum, (..., 1, 1)
// See also: interval
//
// CORA: contSet/@capsule/supportFunc_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Capsule::support_function_impl(const Tensor &d) const {
    const Tensor dt = d.matrix_transpose();
    // the points of the axis and of the ball add, in the order of CORA
    const Tensor val = dt.matmul(c_).add(dt.matmul(g_).abs()).add(r_.multiply(priv_norm(d)));
    return Tensor::where(priv_empty_mask(r_),
                         val.full_like(val.shape(), -std::numeric_limits<double>::infinity()), val);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
