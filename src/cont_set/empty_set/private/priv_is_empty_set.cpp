// priv_is_empty_set - whether a set of any class has no point
//
// See also: priv_is_empty_set.h
//
// CORA.cpp only: a helper of representsa.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/empty_set/private/priv_is_empty_set.h"

#include <algorithm>
#include <cmath>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_is_empty_set(const ContSet &S) {
    // a set with an empty mask says so, as does the empty set
    if (S.empty_mask().all() || S.class_id() == ClassId::empty_set) return true;
    // the classes that encode the empty set in their data have the center NaN then
    const ClassId id = S.class_id();
    if (id != ClassId::capsule && id != ClassId::con_zonotope && id != ClassId::zono_bundle)
        return false;
    const std::vector<double> c = S.center().data();
    return std::all_of(c.begin(), c.end(), [](double x) { return std::isnan(x); });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
