// priv_tensor_multiplication - the third-order tensor image of a zonotope as a zonotope
//
// Each ordered triple p of generators (g_a, g_b, g_c) contributes the vector with the entries
// sum_jkl M[i, j, k, l] g_a[j] g_b[k] g_c[l] to column map[p] of H, as CORA's loop over perm does,
// here for all triples at once: one product of the generator triples and one matrix product.
//
// Syntax:   Zonotope R = priv_tensor_multiplication(Z, M, list);
// Inputs:   Z - zonotope centered at the origin (batches allowed);  M - (n, n, n, n);
//           list - perm (P, 3), comb (C, 3) and map (P), as CORA's options.list
// Outputs:  R - zonotope (0, H) with H (..., n, C)
// See also: tensor_multiplication, tensor_multiplication_zono
//
// CORA.cpp only: CORA's loop over perm as one product, shared by tensor_multiplication and
//                tensor_multiplication_zono

#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The shape of the batch of `like` followed by `tail`.
Shape aux_shape(const Tensor &like, std::initializer_list<int64_t> tail) {
    Shape shape;
    for (int64_t i = 0; i + 2 < like.ndim(); ++i) shape.push_back(like.shape()[i]);
    for (const int64_t size : tail) shape.push_back(size);
    return shape;
}

/// The entries of column `k` of the triples of `perm`, as column indices of G.
std::vector<int64_t> aux_column(const std::vector<std::vector<int64_t>> &perm, std::size_t k) {
    std::vector<int64_t> out;
    for (const auto &triple : perm) out.push_back(triple[k]);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Refusal priv_tensor_multiplication_refusal(const Zonotope &Z, const Tensor &M,
                                           const Zonotope::TensorLists &list) {
    const int64_t n = Z.dim(), m = Z.generators().shape().back(), columns = list.comb.size();
    // the tensor: third order, (n, n, n, n)
    if (M.ndim() == 3)
        return std::pair{Error::not_supported,
                         std::string("the second-order form fails in CORA for every input (it "
                                     "indexes the cell M as an array)")};
    if (M.shape() != Shape{n, n, n, n})
        return std::pair{Error::dimension_mismatch,
                         "M must be a third-order tensor (" + std::to_string(n) + ", " +
                             std::to_string(n) + ", " + std::to_string(n) + ", " +
                             std::to_string(n) + "), got the shape " + shape_text(M.shape())};
    // the lists: a triple of generators per row of perm, a column of H per row of map
    if (list.map.size() != list.perm.size())
        return std::pair{Error::wrong_value, std::string("map needs one entry per row of perm")};
    for (const auto &triple : list.perm) {
        if (triple.size() != 3)
            return std::pair{Error::wrong_value,
                             std::string("every row of perm must have 3 entries, one per factor")};
        for (const int64_t j : triple)
            if (j < 0 || j >= m)
                return std::pair{Error::wrong_value, "perm has the generator " + std::to_string(j) +
                                                         ", outside [0, " + std::to_string(m) +
                                                         ")"};
    }
    // every triple adds to a column of H
    for (const int64_t j : list.map)
        if (j < 0 || j >= columns)
            return std::pair{Error::wrong_value, "map has the column " + std::to_string(j) +
                                                     ", outside [0, " + std::to_string(columns) +
                                                     ")"};
    // CORA leaves H undefined for another center and fails
    if (Z.center().detach().abs().max().item() != 0)
        return std::pair{Error::wrong_value,
                         std::string("the center of the zonotope must be the origin")};
    return std::nullopt;
}

Zonotope priv_tensor_multiplication(const Zonotope &Z, const Tensor &M,
                                    const Zonotope::TensorLists &list) {
    const Tensor &G = Z.generators();
    const int64_t n = Z.dim(), P = list.perm.size(), C = list.comb.size();
    const Tensor zero = Z.center().zeros_like();
    if (P == 0) return {zero, G.zeros_like(aux_shape(G, {n, C}))};
    // the three generators of every triple, side by side in (..., n, P)
    const Tensor g1 = G.take(aux_column(list.perm, 0), -1).reshape(aux_shape(G, {n, 1, 1, P}));
    const Tensor g2 = G.take(aux_column(list.perm, 1), -1).reshape(aux_shape(G, {1, n, 1, P}));
    const Tensor g3 = G.take(aux_column(list.perm, 2), -1).reshape(aux_shape(G, {1, 1, n, P}));
    const Tensor products = g1.multiply(g2).multiply(g3).reshape(aux_shape(G, {n * n * n, P}));
    // tildeH(:, p) for every triple, and the sums into the columns of H
    const Tensor tilde = M.reshape({n, n * n * n}).matmul(products);
    std::vector<double> selection(std::size_t(P * C), 0.0);
    for (int64_t p = 0; p < P; ++p) selection[p * C + list.map[p]] = 1.0;
    return {zero, tilde.matmul(Tensor::like(G, selection, {P, C}))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
