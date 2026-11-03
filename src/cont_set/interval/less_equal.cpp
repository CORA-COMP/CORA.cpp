// less_equal - whether a box lies in another, as CORA's interval.le
//
// Syntax:   bool b = I.less_equal(I2);   bool b = I <= I2;
// Inputs:   I2 - interval of the dimension of I
// Outputs:  b - true if inf2 <= inf and sup <= sup2 in every coordinate (I is a subset of I2); for
//           a batch, true if all members do
// See also: less, isequal, contains
//
// CORA: contSet/@interval/le.m

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Interval::less_equal(const Interval &I2) const {
    detail::Entry entry;
    CORA_CHECK(I2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "less_equal: the intervals differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I2.dim());
    });
    return inf_.numel() == 0 ||
           I2.inf().less_equal(inf_).minimum(sup_.less_equal(I2.sup())).min().item() != 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
