// priv_stacking - the helpers of the matrix zonotope that reshape and join matrices
//
// See also: priv_stacking.h

#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Shape priv_replace_tail(const Tensor &t, int k, std::initializer_list<int64_t> tail) {
    Shape shape;
    for (int64_t d = 0; d + k < t.ndim(); ++d) shape.push_back(t.shape()[d]);
    for (const int64_t size : tail) shape.push_back(size);
    return shape;
}

Tensor priv_stack_columns(const Tensor &T) {
    return T.matrix_transpose().reshape(priv_replace_tail(T, 2, {T.size(-2) * T.size(-1), 1}));
}

Tensor priv_stack_generators(const Tensor &Gs) {
    const int64_t h = Gs.size(-3);
    const Tensor rows = Gs.matrix_transpose().reshape(
        priv_replace_tail(Gs, 3, {h, Gs.size(-2) * Gs.size(-1)}));
    return rows.matrix_transpose();
}

// a part without generators is left out: not every backend joins empty tensors
Tensor priv_join(const Tensor &A, const Tensor &B) {
    if (B.size(-3) == 0) return A;
    if (A.size(-3) == 0) return B;
    return Tensor::concat({A, B}, -3);
}

Tensor priv_no_generators(const Tensor &C) {
    return C.zeros_like(priv_replace_tail(C, 2, {0, C.size(-2), C.size(-1)}));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
