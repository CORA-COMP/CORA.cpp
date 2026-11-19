// stack - a batch of intervals as one interval
//
// Syntax:   I = Interval::stack({I1, I2, ...});
// Inputs:   Is - intervals of one dimension and one batch shape
// Outputs:  I - one interval whose leading batch dimension indexes Is, in order
// See also: contSet/zonotope/stack
//
// CORA.cpp only: batching, MATLAB CORA has no batch of sets
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/error.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::stack(const std::vector<Interval> &Is) {
    detail::Entry entry;
    CORA_CHECK(!Is.empty(), Error::wrong_value,
               [] { return std::string("stack: give at least one interval"); });
    for (const Interval &I : Is)
        CORA_CHECK(I.inf().shape() == Is[0].inf().shape() && I.elem_rank() == Is[0].elem_rank(),
                   Error::dimension_mismatch, [&] {
                       return "stack: the intervals must have one shape, got the bounds " +
                              shape_text(Is[0].inf().shape()) + " and " +
                              shape_text(I.inf().shape());
                   });
    // the bounds and the masks of the members, stacked along a new first axis
    std::vector<Tensor> infs, sups;
    std::vector<EmptyMask> masks;
    for (const Interval &I : Is) {
        infs.push_back(I.inf());
        sups.push_back(I.sup());
        masks.push_back(I.empty_mask());
    }
    return {Tensor::stack(infs), Tensor::stack(sups), Is[0].elem_rank_,
            EmptyMask::stack(masks, Is[0].batch_shape())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
