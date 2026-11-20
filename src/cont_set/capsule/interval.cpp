// interval - the interval hull of a capsule, as CORA's capsule.interval
//
// The empty box for the empty capsule. A batch with an empty member is refused.
//
// Syntax:   Interval I = C.interval();
// Inputs:   -
// Outputs:  I - the box c +- (|g| + r): the smallest one containing the capsule
// See also: center, Zonotope::interval
//
// CORA: contSet/@capsule/interval.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   20-November-2026
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Capsule::interval_impl() const {
    if (empty_.any()) {
        if (cc().ndim() != 2)
            CORA_FAIL(Error::not_supported,
                      "interval: the box of a batch with an empty capsule is not supported");
        return Interval::empty(dim());
    }
    // CORA reads the range of each coordinate from the support function: c_i +- (|g_i| + r)
    const Tensor reach = gc().abs().add(rc());
    return {cc().subtract(reach).squeeze(-1), cc().add(reach).squeeze(-1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
