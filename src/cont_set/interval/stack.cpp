// stack - a batch of intervals as one interval
//
// Syntax:   I = Interval::stack({I1, I2, ...});
// Inputs:   Is - intervals of one dimension and one batch shape
// Outputs:  I - one interval whose leading batch dimension indexes Is, in order
// See also: contSet/zonotope/stack
//
// CORA.cpp only: batching, MATLAB CORA has no batch of sets

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
        CORA_CHECK(I.inf().shape() == Is[0].inf().shape(), Error::dimension_mismatch, [&] {
            return "stack: the intervals must have one shape, got the bounds " +
                   shape_text(Is[0].inf().shape()) + " and " + shape_text(I.inf().shape());
        });
    std::vector<Tensor> infs, sups;
    for (const Interval &I : Is) {
        infs.push_back(I.inf());
        sups.push_back(I.sup());
    }
    return {Tensor::stack(infs), Tensor::stack(sups)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
