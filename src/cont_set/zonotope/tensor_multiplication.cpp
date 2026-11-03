// tensor_multiplication - the image of a zonotope under a third-order tensor, as a zonotope
//
// Syntax:   R = Z.tensor_multiplication(M, list);
// Inputs:   M - tensor (n, n, n, n): output i is sum_jkl M[i, j, k, l] x_j x_k x_l;  list - CORA's
//           options.list: perm (ordered triples of generators), comb (its row count is the number
//           of generators of the result) and map (the column each triple adds to)
// Outputs:  R - zonotope (0, H) that encloses the image of Z, which must be centered at the
//           origin (CORA fails otherwise). The second-order form is not supported: it fails in
//           CORA for every input
// See also: tensor_multiplication_zono, cubic_map
//
// CORA: contSet/@zonotope/tensorMultiplication.m

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::tensor_multiplication(const Tensor &M, const TensorLists &list) const {
    detail::Entry entry;
    const Refusal refusal = priv_tensor_multiplication_refusal(*this, M, list);
    CORA_CHECK(!refusal, refusal->first,
               [&] { return "tensor_multiplication: " + refusal->second; });
    return priv_tensor_multiplication(*this, M, list);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
