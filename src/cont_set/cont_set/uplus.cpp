// uplus - the set itself, as CORA's contSet.uplus
//
// Syntax:   R = S.uplus();
// Inputs:   S - a set of dimension n
// Outputs:  R - S, in the class of S, with every number as it was: the translation by -0, which
//           changes no number (x + (-0) = x)
// See also: uminus, minus
//
// CORA: contSet/@contSet/uplus.m

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::uplus() const {
    detail::Entry entry;
    return uplus_impl();
}

Set ContSet::uplus_impl() const {
    return translate_impl(center_impl().full_like({dim_impl(), 1}, -0.0));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
