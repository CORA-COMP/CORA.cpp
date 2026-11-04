// tensor_multiplication_zono - the image of a zonotope under a tensor zonotope, as a zonotope
//
// Syntax:   R = Z.tensor_multiplication_zono(M_center, M_generators, list);
// Inputs:   M_center, M_generators - the center and the generators of the tensor zonotope, each
//           a tensor (n, n, n, n);  list - as in tensor_multiplication
// Outputs:  R - the Minkowski sum of tensor_multiplication for the center and for every
//           generator of the tensor zonotope
// See also: tensor_multiplication
//
// CORA: contSet/@zonotope/tensorMultiplication_zono.m

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::tensor_multiplication_zono(const Tensor &M_center,
                                              const std::vector<Tensor> &M_generators,
                                              const TensorLists &list) const {
    detail::Entry entry;
    const Refusal refusal = priv_tensor_multiplication_refusal(*this, M_center, list);
    CORA_CHECK(!refusal, refusal->first,
               [&] { return "tensor_multiplication_zono: " + refusal->second; });
    for (const Tensor &M : M_generators) {
        const Refusal generator = priv_tensor_multiplication_refusal(*this, M, list);
        CORA_CHECK(!generator, generator->first,
                   [&] { return "tensor_multiplication_zono: a generator: " + generator->second; });
    }
    Zonotope result = priv_tensor_multiplication(*this, M_center, list);
    for (const Tensor &M : M_generators)
        result = result.plus(priv_tensor_multiplication(*this, M, list));
    return result;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
