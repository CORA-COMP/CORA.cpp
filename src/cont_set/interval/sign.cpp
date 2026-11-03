// sign - the sign of a box, as CORA's interval.sign
//
// Syntax:   Interval R = I.sign();
// Inputs:   -
// Outputs:  R - [sign(inf), sign(sup)] elementwise, the hull of the image (sign is monotone). It
//           is tighter than CORALean's float definition, which reads a strictly positive lower
//           bound as 0 because a float comparison certifies only <=
// See also: abs, round
//
// CORA: contSet/@interval/sign.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::sign() const {
    detail::Entry entry;
    return {inf_.sign(), sup_.sign()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
