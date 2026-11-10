// priv_broadcastable - whether two shapes broadcast to one
//
// See also: priv_broadcastable.h
//
// CORA.cpp only: MATLAB broadcasts implicitly; the shape check is explicit.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_broadcastable.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_broadcastable(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
    const std::size_t common = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t i = 1; i <= common; ++i) {
        const int64_t x = a[a.size() - i], y = b[b.size() - i];
        if (x != y && x != 1 && y != 1) return false;
    }
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
