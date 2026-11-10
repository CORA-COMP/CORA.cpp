// priv_vec - the bounds of an interval matrix stacked into one column
//
// See also: priv_vec.h
//
// CORA.cpp only: a helper of the interval matrix.
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "matrix_set/interval_matrix/private/priv_vec.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval priv_vec(const Interval &I) {
    std::vector<int64_t> shape = I.inf().shape();
    const int64_t entries = shape[shape.size() - 2] * shape.back();
    shape.pop_back();
    shape.back() = entries; // the transpose lists the columns one after the other
    shape.push_back(1);
    return {I.inf().matrix_transpose().reshape(shape), I.sup().matrix_transpose().reshape(shape)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
