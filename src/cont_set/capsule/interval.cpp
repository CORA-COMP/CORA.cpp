// interval - the interval hull of a capsule, as CORA's capsule.interval
//
// Syntax:   Interval I = C.interval();
// Inputs:   -
// Outputs:  I - the box c +- (|g| + r): the smallest one containing the capsule; the empty box
//           for the empty capsule. A batch with an empty member has no box and is refused
// See also: center, Zonotope::interval
//
// CORA: contSet/@capsule/interval.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Capsule::interval_impl() const {
    if (priv_empty_mask(r_).detach().max().item() > 0) {
        if (c_.ndim() != 2)
            CORA_FAIL(
                Error::not_supported,
                "interval: a batch with an empty capsule has no box, as boxes cannot be empty "
                "in a batch");
        return Interval::empty(dim());
    }
    // CORA reads the range of each coordinate from the support function: c_i +- (|g_i| + r)
    const Tensor reach = g_.abs().add(r_);
    return {c_.subtract(reach), c_.add(reach)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
