// priv_select_generators - some columns of a generator matrix, in order
//
// Syntax:   Tensor Gs = priv_select_generators(G, columns);
// Inputs:   G - generators (..., n, m);  columns - indices in [0, m), a host list
// Outputs:  Gs - (..., n, k): those columns, also none (k = 0)
// See also: Zonotope::compact
//
// CORA.cpp only: MATLAB selects with G(:, idx), which needs no function

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_select_generators(const Tensor &G, const std::vector<int64_t> &columns) {
    if (columns.empty()) {
        Shape shape = G.shape();
        shape.back() = 0;
        return G.zeros_like(shape);
    }
    return G.take(columns, -1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
