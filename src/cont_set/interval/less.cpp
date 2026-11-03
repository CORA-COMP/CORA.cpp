// less - whether a box lies strictly in another, as CORA's interval.lt
//
// Syntax:   bool b = I.less(I2);   bool b = I < I2;
// Inputs:   I2 - interval of the dimension of I
// Outputs:  b - true if inf2 < inf and sup < sup2 in every coordinate (I lies strictly inside I2);
//           for a batch, true if all members do
// See also: less_equal, isequal, contains
//
// CORA: contSet/@interval/lt.m

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Interval::less(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "less: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    return inf_.numel() == 0 || I2.inf().less(inf_).minimum(sup_.less(I2.sup())).min().item() != 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
