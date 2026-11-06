// tensor_multiplication - the image of a zonotope under a third-order tensor, as a zonotope
//
// perm are the ordered triples of generators, comb (its row count is the number of generators of
// the result) and map the column each triple adds to. CORA fails for a center other than the
// origin. The second-order form is not supported: it fails in CORA for every input.
//
// Syntax:   R = Z.tensor_multiplication(M, list);
// Inputs:   M - tensor (n, n, n, n): output i is sum_jkl M[i, j, k, l] x_j x_k x_l
//           list - CORA's options.list: perm, comb and map
// Outputs:  R - zonotope (0, H) that encloses the image of Z, which must be centered at the origin
// See also: tensor_multiplication_zono, cubic_map
//
// CORA: contSet/@zonotope/tensorMultiplication.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

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
