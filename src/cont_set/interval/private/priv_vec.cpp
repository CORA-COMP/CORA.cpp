// priv_vec - the bounds of an interval matrix stacked column by column into one box
//
// See also: priv_vec.h
//
// CORA.cpp only: a helper of the interval matrix.
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_vec.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval priv_vec(const Interval &I) {
    if (I.elem_rank() == 1) return I; // a box is its own column
    // the transpose lists the columns one after the other
    std::vector<int64_t> shape = I.inf().shape();
    const int64_t entries = shape[shape.size() - 2] * shape.back();
    shape.pop_back();
    shape.back() = entries;
    return {I.inf().matrix_transpose().reshape(shape), I.sup().matrix_transpose().reshape(shape)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
