// priv_empty - the empty interval of the shape of another
//
// See also: priv_empty.h
//
// CORA.cpp only: a helper for the empty interval.
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_empty.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval priv_empty(const Interval &like) {
    // the shape stays, every member is flagged empty
    const Tensor zero = like.inf().zeros_like();
    return {zero, zero, like.elem_rank_, EmptyMask::all_members()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
