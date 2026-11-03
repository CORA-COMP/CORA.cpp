// cubic_map - an enclosure of a cubic map of zonotopes, as CORA's zonotope.cubMap
//
// Syntax:   Zres = Z.cubic_map(T);   Zres = Z.cubic_map(T, ind);   Zres = Z.cubic_map(Z2, Z3, T);
// Inputs:   T - cubic map: T[i][k] is an (n, n) matrix, output i is z_i = sum_k x_k (x' T[i][k] x)
//           ind - for every output i the k that enter it (CORA's default: all columns of T)
//           Z2, Z3 - the zonotopes of the second and third factor of (x1' T x2) x3
// Outputs:  Zres - zonotope with one dimension per row of T that encloses the image: its
//           generators are the monomials beta_k beta_j beta_c of the generator factors, which a
//           zonotope frees to [-1, 1]; (N+1)(N+2)(N+3)/6 - 1 of them for N generators
// See also: linear_map, plus
//
// CORA: contSet/@zonotope/cubMap.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

#include <algorithm>
#include <array>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Shapes and indices ------------------------------------------------------------------------------

/// The shape of the batch of `like` followed by `tail`.
Shape aux_shape(const Tensor &like, std::initializer_list<int64_t> tail) {
    Shape shape;
    for (int64_t i = 0; i + 2 < like.ndim(); ++i) shape.push_back(like.shape()[i]);
    for (const int64_t size : tail) shape.push_back(size);
    return shape;
}

/// [c G]: the center as the first column.
Tensor aux_full(const Zonotope &Z) { return Tensor::concat({Z.center(), Z.generators()}, -1); }

/// The columns of T that enter output i: ind[i], or all of them.
std::vector<std::vector<int64_t>> aux_indices(const std::vector<std::vector<Tensor>> &T,
                                              const std::vector<std::vector<int64_t>> &ind) {
    if (!ind.empty()) return ind;
    std::vector<int64_t> all(T.front().size());
    for (std::size_t k = 0; k < all.size(); ++k) all[k] = int64_t(k);
    return std::vector<std::vector<int64_t>>(T.size(), all);
}

/// The monomials (k, j, h) of z_k z_j z_h with k <= j <= h, in CORA's order.
std::vector<int64_t> aux_ordered_triples(int64_t N) {
    std::vector<int64_t> flat;
    for (int64_t k = 0; k < N; ++k)
        for (int64_t j = k; j < N; ++j)
            for (int64_t h = j; h < N; ++h) flat.push_back((k * N + j) * N + h);
    return flat;
}

/// A mask (N, N, N) of the entries (j, a, b) for which `holds(j, a, b)`, like `like`.
template <class F>
Tensor aux_mask(const Tensor &like, int64_t N, F holds) {
    std::vector<double> mask;
    for (int64_t j = 0; j < N; ++j)
        for (int64_t a = 0; a < N; ++a)
            for (int64_t b = 0; b < N; ++b) mask.push_back(holds(j, a, b) ? 1.0 : 0.0);
    return Tensor::like(like, mask, {N, N, N});
}

/// The axes of a tensor with `batch` leading axes that exchange its last three as `order` says.
std::vector<int> aux_axes(int64_t batch, std::array<int, 3> order) {
    std::vector<int> axes(batch + 3);
    for (int64_t i = 0; i < batch; ++i) axes[i] = int(i);
    for (int k = 0; k < 3; ++k) axes[batch + k] = int(batch) + order[k];
    return axes;
}

// The Two Maps -----------------------------------------------------------------------------------

/// Output i of the cubic map of one zonotope: the row of its center and generators.
Tensor aux_single_row(const Tensor &Zfull, const std::vector<Tensor> &Ti,
                      const std::vector<int64_t> &ind) {
    const int64_t N = Zfull.shape().back(), batch = Zfull.ndim() - 2;
    const Tensor Zt = Zfull.matrix_transpose();
    // W[j] is CORA's listQuad{j}: sum_k Zfull(ind_k, j) * (the upper triangle of Q_k + Q_k')
    Tensor W = Zfull.zeros_like(aux_shape(Zfull, {N, N, N}));
    for (const int64_t k : ind) {
        const Tensor Q = Zt.matmul(Ti[k]).matmul(Zfull);
        const Tensor folded = Q.triu() + Q.tril(-1).matrix_transpose();
        const Tensor factor = Zfull.slice(-2, k, k + 1).reshape(aux_shape(Zfull, {N, 1, 1}));
        W = W + folded.expand_dims(int(batch)).multiply(factor);
    }
    if (N == 1) return W.reshape(aux_shape(Zfull, {1, 1})); // no generators: the center alone

    // identical factors of different orders add: W[j,a,b] += W[a,j,b] (j<a<b) and += W[b,j,a]
    // (j<=a<=b, b>j), as CORA's loops over k, j, h do
    const Tensor first = aux_mask(Zfull, N, [](int64_t j, int64_t a, int64_t b) {
        return j < a && a < b;
    });
    const Tensor second = aux_mask(Zfull, N, [](int64_t j, int64_t a, int64_t b) {
        return j <= a && a <= b && b > j;
    });
    W = Tensor::where(first, W + W.permute_dims(aux_axes(batch, {1, 0, 2})), W);
    W = Tensor::where(second, W + W.permute_dims(aux_axes(batch, {1, 2, 0})), W);

    // the pure squares: the center takes half of the other diagonal entries, those are halved
    const Tensor head = W.slice(-3, 0, 1);
    const Tensor diagonal = head.diagonal(), halved = diagonal * 0.5;
    const Tensor center_entry = diagonal.slice(-2, 0, 1) + halved.slice(-2, 1, N).sum(-2, true);
    const Tensor corrected = Tensor::concat({center_entry, halved.slice(-2, 1, N)}, -2);
    const Tensor fixed_head = Tensor::where(Zfull.eye_like(N), corrected.diag(), head);
    const Tensor all = Tensor::concat({fixed_head, W.slice(-3, 1, N)}, -3);

    // the entries k <= j <= h in a row
    return all.reshape(aux_shape(Zfull, {1, N * N * N})).take(aux_ordered_triples(N), -1);
}

/// Output i of the mixed cubic map (x1' T x2) x3: the row of center and generators.
Tensor aux_mixed_row(const std::array<Tensor, 3> &Zs, const std::vector<Tensor> &Ti,
                     const std::vector<int64_t> &ind) {
    const Tensor &Z1 = Zs[0], &Z2 = Zs[1], &Z3 = Zs[2];
    const int64_t N3 = Z3.shape().back(), Nq = Z1.shape().back() * Z2.shape().back();
    const Tensor Z1t = Z1.matrix_transpose();
    Tensor blocks = Z1.zeros_like(aux_shape(Z1, {N3, Nq}));
    for (const int64_t k : ind) {
        // the entries of the product matrix column by column, scaled by row k of the third
        const Tensor Q = Z1t.matmul(Ti[k]).matmul(Z2);
        const Tensor vec = Q.matrix_transpose().reshape(aux_shape(Q, {1, Nq}));
        const Tensor factor = Z3.slice(-2, k, k + 1).reshape(aux_shape(Z3, {N3, 1}));
        blocks = blocks + vec.multiply(factor);
    }
    return blocks.reshape(aux_shape(Z1, {1, N3 * Nq}));
}

/// The zonotope of the rows i = 0, 1, ... that `row(T[i], indices[i])` gives.
template <class F>
Zonotope aux_from_rows(const std::vector<std::vector<Tensor>> &T,
                       const std::vector<std::vector<int64_t>> &ind, F row) {
    const auto indices = aux_indices(T, ind);
    std::vector<Tensor> rows;
    for (std::size_t i = 0; i < T.size(); ++i) rows.push_back(row(T[i], indices[i]));
    const Tensor all = Tensor::concat(rows, -2);
    return {all.slice(-1, 0, 1), all.slice(-1, 1, all.shape().back())};
}

// Checks ------------------------------------------------------------------------------------------

/// What is wrong with T and ind for zonotopes of dimension n, or an empty text if nothing.
std::string aux_problem(const std::vector<std::vector<Tensor>> &T,
                        const std::vector<std::vector<int64_t>> &ind, int64_t n) {
    if (T.empty() || T.front().empty())
        return "T needs at least one row and one column of matrices";
    for (const auto &row : T) {
        if (row.size() != T.front().size()) return "the rows of T must have the same length";
        for (const Tensor &M : row)
            if (M.shape() != Shape{n, n})
                return "every matrix of T must be (" + std::to_string(n) + ", " +
                       std::to_string(n) + "), got the shape " + shape_text(M.shape());
    }
    if (ind.empty()) return "";
    if (ind.size() != T.size()) return "ind needs one list per row of T";
    const int64_t columns = std::min<int64_t>(int64_t(T.front().size()), n);
    for (const auto &list : ind)
        for (const int64_t k : list)
            if (k < 0 || k >= columns)
                return "ind has the entry " + std::to_string(k) + ", outside [0, " +
                       std::to_string(columns) + ")";
    return "";
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::cubic_map(const std::vector<std::vector<Tensor>> &T,
                             const std::vector<std::vector<int64_t>> &ind) const {
    detail::Entry entry;
    const std::string problem = aux_problem(T, ind, dim());
    CORA_CHECK(problem.empty(), Error::wrong_value, [&] { return "cubic_map: " + problem; });
    const Tensor full = aux_full(*this);
    // output i of the map for the matrices of row i
    return aux_from_rows(T, ind, [&](const std::vector<Tensor> &Ti, const std::vector<int64_t> &k) {
        return aux_single_row(full, Ti, k);
    });
}

Zonotope Zonotope::cubic_map(const Zonotope &Z2, const Zonotope &Z3,
                             const std::vector<std::vector<Tensor>> &T,
                             const std::vector<std::vector<int64_t>> &ind) const {
    detail::Entry entry;
    const std::string problem = aux_problem(T, ind, dim());
    CORA_CHECK(problem.empty(), Error::wrong_value, [&] { return "cubic_map: " + problem; });
    CORA_CHECK(Z2.dim() == dim() && Z3.dim() == dim(), Error::dimension_mismatch, [&] {
        return "cubic_map: the zonotopes differ in dimension, " + std::to_string(dim()) + ", " +
               std::to_string(Z2.dim()) + " and " + std::to_string(Z3.dim());
    });
    const std::array<Tensor, 3> fulls{aux_full(*this), aux_full(Z2), aux_full(Z3)};
    return aux_from_rows(T, ind, [&](const std::vector<Tensor> &Ti, const std::vector<int64_t> &k) {
        return aux_mixed_row(fulls, Ti, k);
    });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
