// enclose - a set that encloses two sets, as CORA's contSet.enclose
//
// Syntax:   R = S1.enclose(S2);
// Inputs:   S1, S2 - sets of one dimension
// Outputs:  R - a set of the class of S1 that encloses both; Error::noops for a class without
//           the operation or that cannot take the class of S2
// See also: convex_hull, Interval::enclose, Zonotope::enclose
//
// CORA: contSet/@contSet/enclose.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::enclose(const ContSet &S2) const {
    detail::Entry entry;
    CORA_CHECK(S2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "enclose: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S2.dim());
    });
    return enclose_impl(S2);
}

Set ContSet::enclose_impl(const ContSet &S2) const { priv_noops("enclose", *this, &S2); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
