// lift - the whole space in a space of higher dimension, as CORA's fullspace.lift_
//
// Syntax:   Fullspace R = F.lift(N, dims);
// Inputs:   N - the dimension of the new space, N >= n;  dims - the position (0-based) of each
//           dimension of F in it; CORA does not check it, CORA.cpp as Interval::lift
// Outputs:  R - R^N: the new dimensions are unbounded, as the old ones already are
// See also: Interval::lift
//
// CORA: contSet/@fullspace/lift_.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    CORA_CHECK(lift_refusal(N, dims).empty(), Error::wrong_value,
               [&] { return lift_refusal(N, dims); });
    return Fullspace(N);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
