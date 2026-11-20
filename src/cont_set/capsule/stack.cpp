// stack - a batch of capsules as one capsule
//
// A member of the batch is empty where its capsule is.
//
// Syntax:   C = Capsule::stack({C1, C2, ...});
// Inputs:   Cs - capsules of one dimension and one batch shape
// Outputs:  C - one capsule whose leading batch dimension indexes Cs, in order
// See also: Interval::stack, Zonotope::stack
//
// CORA.cpp only: batching, MATLAB CORA has no batch of sets
// Authors:       Claude Opus 5.5, Tobias Ladner
// Written:       20-November-2026
// Last update:   20-November-2026
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/error.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Capsule Capsule::stack(const std::vector<Capsule> &Cs) {
    detail::Entry entry;
    CORA_CHECK(!Cs.empty(), Error::wrong_value,
               [] { return std::string("stack: give at least one capsule"); });
    for (const Capsule &C : Cs)
        CORA_CHECK(C.c().shape() == Cs[0].c().shape(), Error::dimension_mismatch, [&] {
            return "stack: the capsules must have one shape, got the centers " +
                   shape_text(Cs[0].c().shape()) + " and " + shape_text(C.c().shape());
        });
    // the parts and the masks of the members, stacked along a new first axis
    std::vector<Tensor> cs, gs, rs;
    std::vector<EmptyMask> masks;
    for (const Capsule &C : Cs) {
        cs.push_back(C.c_.expand_dims(-1));
        gs.push_back(C.g_.expand_dims(-1));
        rs.push_back(C.rc());
        masks.push_back(C.empty_mask());
    }
    return {Unchecked{}, Tensor::stack(cs), Tensor::stack(gs), Tensor::stack(rs),
            EmptyMask::stack(masks, Cs[0].r_.shape())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
