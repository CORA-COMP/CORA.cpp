// stack - a batch of intervals as one interval
//
// Syntax:   I = Interval::stack({I1, I2, ...});
// Inputs:   Is - intervals of one dimension and one batch shape
// Outputs:  I - one interval whose leading batch dimension indexes Is, in order
// See also: contSet/zonotope/stack

#include "contSet/interval/interval.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ===========================================  MAIN  =========================================== //

Interval Interval::stack(const std::vector<Interval> &Is) {
    if (Is.empty()) throw std::invalid_argument("Interval::stack: no intervals given");
    std::vector<Tensor> infs, sups;
    for (const Interval &I : Is) {
        infs.push_back(I.inf);
        sups.push_back(I.sup);
    }
    return {Tensor::stack(infs), Tensor::stack(sups)};
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
