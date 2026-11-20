// priv_empty_mask - which capsules of a batch are empty
//
// Syntax:   Tensor empty = priv_empty_mask(C);
// Inputs:   C - capsule of the batch shape (...)
// Outputs:  empty - 1 for the members flagged in C.empty_mask(), else 0, as (..., 1, 1)
// See also: Capsule::empty, Capsule::is_empty
//
// CORA.cpp only: MATLAB marks the empty capsule by a center without columns; here it is the mask
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   20-November-2026
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_empty_mask(const Capsule &C) {
    const Tensor &r = C.r();
    if (!C.empty_mask().any()) return r.zeros_like().expand_dims(-1).expand_dims(-1);
    const std::vector<uint8_t> flags = C.empty_mask().flags(r.shape());
    return Tensor::like(r, std::vector<double>(flags.begin(), flags.end()), r.shape())
        .expand_dims(-1)
        .expand_dims(-1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
