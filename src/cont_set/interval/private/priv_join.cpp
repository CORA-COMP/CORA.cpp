// priv_join - boxes joined along a dimension: the work of cat and cart_prod
//
// Syntax:   bool ok = priv_joinable(A, B, 1);   Interval J = priv_join({I1, I2}, 1);
// Inputs:   A, B - bounds
//           Is - intervals;  dim - 1: stacked (rows), 2: side by side (columns)
// Outputs:  priv_joinable - whether the shapes agree in every dimension but dim;
//           priv_join - the interval of the joined bounds (the shapes are not checked)
// See also: cat, cart_prod
//
// CORA.cpp only: a helper of cat and cart_prod.
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_join.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_joinable(const Tensor &A, const Tensor &B, int dim) {
    if (A.ndim() != B.ndim()) return false;
    // the matrix dimensions are the last two: dim 1 is the rows (-2), dim 2 the columns (-1)
    const int64_t joined = A.ndim() - 3 + dim;
    for (int64_t i = 0; i < A.ndim(); ++i)
        if (i != joined && A.shape()[i] != B.shape()[i]) return false;
    return true;
}

Interval priv_join(const std::vector<Interval> &Is, int dim) {
    std::vector<Tensor> infs, sups;
    for (const Interval &I : Is) {
        infs.push_back(I.inf());
        sups.push_back(I.sup());
    }
    const int axis = dim - 3; // -2 for the rows, -1 for the columns
    return {Tensor::concat(infs, axis), Tensor::concat(sups, axis)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
