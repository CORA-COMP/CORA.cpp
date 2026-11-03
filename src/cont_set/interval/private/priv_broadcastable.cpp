// priv_broadcastable - whether two shapes broadcast to one
//
// See also: priv_broadcastable.h

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
