// priv_join - boxes joined along a dimension: the work of cat and cart_prod
//
// A box joins as the column (n, 1); boxes stacked are a box, anything else an interval matrix,
// and one interval is itself. priv_join does not check the shapes.
//
// Syntax:   bool ok = priv_joinable(I, I2, 1);   Interval J = priv_join({I1, I2}, 1);
// Inputs:   I, I2, Is - intervals
//           dim - 1: stacked (rows), 2: side by side (columns)
// Outputs:  priv_joinable - whether the shapes agree in every dimension but dim
//           priv_join - the interval of the joined bounds
// See also: cat, cart_prod
//
// CORA.cpp only: a helper of cat and cart_prod.
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_join.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// the bounds as matrices: a box is a column
Tensor priv_matrix_bounds(const Interval &I, bool upper) {
    const Tensor &t = upper ? I.sup() : I.inf();
    return I.elem_rank() == 1 ? t.expand_dims(-1) : t;
}

bool priv_joinable(const Interval &I, const Interval &I2, int dim) {
    const Tensor A = priv_matrix_bounds(I, false), B = priv_matrix_bounds(I2, false);
    if (A.ndim() != B.ndim()) return false;
    // the matrix dimensions are the last two: dim 1 is the rows (-2), dim 2 the columns (-1)
    const int64_t joined = A.ndim() - 3 + dim;
    for (int64_t i = 0; i < A.ndim(); ++i)
        if (i != joined && A.shape()[i] != B.shape()[i]) return false;
    return true;
}

Interval priv_join(const std::vector<Interval> &Is, int dim) {
    if (Is.size() == 1) return Is[0]; // one interval is itself, a box stays a box
    std::vector<Tensor> infs, sups;
    bool boxes = true;
    for (const Interval &I : Is) {
        infs.push_back(priv_matrix_bounds(I, false));
        sups.push_back(priv_matrix_bounds(I, true));
        boxes = boxes && I.elem_rank() == 1;
    }
    const int axis = dim - 3; // -2 for the rows, -1 for the columns
    Tensor lo = Tensor::concat(infs, axis), hi = Tensor::concat(sups, axis);
    // boxes stacked are a box; anything else is an interval matrix
    if (boxes && dim == 1) return {lo.squeeze(-1), hi.squeeze(-1)};
    return Interval::matrix(std::move(lo), std::move(hi));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
