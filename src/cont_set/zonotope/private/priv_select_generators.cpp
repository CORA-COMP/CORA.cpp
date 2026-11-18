// priv_select_generators - some columns of a generator matrix, in order
//
// Syntax:   Tensor Gs = priv_select_generators(G, columns);
// Inputs:   G - generators (..., n, m)
//           columns - indices in [0, m), a host list
// Outputs:  Gs - (..., n, k): those columns, also none (k = 0)
// See also: Zonotope::compact
//
// CORA.cpp only: MATLAB selects with G(:, idx), which needs no function
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_select_generators(const Tensor &G, const std::vector<int64_t> &columns, int axis) {
    const int a = axis < 0 ? int(G.ndim()) + axis : axis;
    if (columns.empty()) {
        Shape shape = G.shape();
        shape[a] = 0;
        return G.zeros_like(shape);
    }
    // all generators in their order: the tensor itself
    if (int64_t(columns.size()) == G.shape()[a]) {
        int64_t next = 0;
        if (std::all_of(columns.begin(), columns.end(), [&](int64_t j) { return j == next++; }))
            return G;
    }
    return G.take(columns, axis);
}


} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
