// priv_broadcastable - whether two shapes broadcast to one, and whether numbers fit the elements
//
// See also: priv_broadcastable.h
//
// CORA.cpp only: MATLAB broadcasts implicitly; the shape check is explicit.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
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

// numbers of the shape of the elements of I, with batch dimensions or without
bool priv_fits_elements(const Tensor &x, const Interval &I) {
    const Shape elem = I.elem_shape();
    if (x.ndim() < elem.size()) return false;
    for (int64_t a = 1; a <= elem.size(); ++a)
        if (x.size(static_cast<int>(-a)) != elem[elem.size() - a]) return false;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
