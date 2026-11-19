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

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_is_empty_set(const ContSet &S) {
    // a set with an empty mask says so; the others still have a center without columns
    return S.empty_mask().all() || S.center().shape().back() == 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
