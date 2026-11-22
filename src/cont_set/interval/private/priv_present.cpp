// priv_present - which members of a batch of intervals are not empty
//
// See also: priv_present.h
//
// CORA.cpp only: the empty mask of a batch is CORA.cpp's.
// Authors:       Claude Opus 5.5, Tobias Ladner
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_present.h"

#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_present(const Interval &I) {
    const Shape batch = I.batch_shape();
    const std::vector<uint8_t> empty = I.empty_mask().flags(batch);
    std::vector<double> present(empty.size());
    for (std::size_t i = 0; i < empty.size(); ++i) present[i] = empty[i] ? 0.0 : 1.0;
    return Tensor::like(I.inf(), present, batch);
}

Tensor priv_all_entries(const Tensor &t, const Interval &I) {
    // a member without entries has every entry
    Shape batch;
    for (int64_t a = 0; a < t.ndim() - I.elem_rank(); ++a) batch.push_back(t.shape()[a]);
    if (t.numel() == 0) return t.full_like(batch, 1.0);
    Tensor flags = t;
    for (int64_t r = 0; r < I.elem_rank(); ++r) flags = flags.all(-1);
    return flags;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
