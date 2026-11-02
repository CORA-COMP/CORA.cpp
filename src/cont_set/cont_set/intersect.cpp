// intersect - the intersection of two sets, as CORA's contSet.and
//
// Syntax:   S = intersect(S1, S2);   S = Set(S1) & Set(S2);
// Inputs:   S1, S2 - sets of one dimension
// Outputs:  S - a set that contains S1 ∩ S2 (exact where the pair allows)
// See also: registry, unite
//
// CORA: contSet/@contSet/and.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set intersect(const ContSet &S1, const ContSet &S2) {
    detail::Entry entry;
    CORA_CHECK(S1.dim() == S2.dim(), Error::dimension_mismatch, [&] {
        return "intersect: the sets differ in dimension, " + S1.class_name() + " of " +
               std::to_string(S1.dim()) + " and " + S2.class_name() + " of " +
               std::to_string(S2.dim());
    });
    return detail::dispatch(BinaryOp::intersect, S1, S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
