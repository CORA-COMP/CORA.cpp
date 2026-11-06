// priv_represents_empty - whether a set is empty, for the classes that have an empty member
//
// See also: priv_represents_empty.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/fullspace/private/priv_represents_empty.h"
#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_represents_empty(const ContSet &S) {
    return S.class_id() == ClassId::empty_set ||
           (S.class_id() == ClassId::interval && static_cast<const Interval &>(S).is_empty());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
