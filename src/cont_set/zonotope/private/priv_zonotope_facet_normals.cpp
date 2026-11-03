// priv_zonotope_facet_normals - the unit normals of the facets of a full-dimensional zonotope
//
// The cross products of the (n-1)-subsets of generators, in lexicographic order of the subsets and
// normalized, without those that vanish (dependent generators): the first half of the rows of
// CORA's polytope(Z).A.
//
// Syntax:   Tensor C = priv_zonotope_facet_normals(Z);
// Inputs:   Z - a single zonotope without zero generators whose generators span the space
// Outputs:  C - the normals as rows (k, n)
// See also: Zonotope::minnorm, Zonotope::mink_diff
//
// CORA: contSet/@zonotope/polytope.m

#include "cont_set/zonotope/private/priv.h"

#include <cmath>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The determinant of the k x k row-major matrix, by Gaussian elimination with partial pivoting.
double aux_determinant(std::vector<double> A, int64_t k) {
    double det = 1.0;
    for (int64_t col = 0; col < k; ++col) {
        int64_t pivot = col;
        for (int64_t r = col + 1; r < k; ++r)
            if (std::abs(A[r * k + col]) > std::abs(A[pivot * k + col])) pivot = r;
        if (A[pivot * k + col] == 0.0) return 0.0;
        if (pivot != col) {
            for (int64_t j = 0; j < k; ++j) std::swap(A[pivot * k + j], A[col * k + j]);
            det = -det;
        }
        det *= A[col * k + col];
        for (int64_t r = col + 1; r < k; ++r) {
            const double factor = A[r * k + col] / A[col * k + col];
            for (int64_t j = col; j < k; ++j) A[r * k + j] -= factor * A[col * k + j];
        }
    }
    return det;
}

/// CORA's ndimCross of the n x (n-1) matrix Q (row-major): entry i is (-1)^i times the
/// determinant of Q without row i.
std::vector<double> aux_cross(const std::vector<double> &Q, int64_t n) {
    const int64_t k = n - 1;
    std::vector<double> v(n), minor(k * k);
    for (int64_t i = 0; i < n; ++i) {
        int64_t row = 0;
        for (int64_t r = 0; r < n; ++r) {
            if (r == i) continue;
            for (int64_t j = 0; j < k; ++j) minor[row * k + j] = Q[r * k + j];
            ++row;
        }
        const double det = aux_determinant(minor, k);
        v[i] = (i % 2 == 0) ? det : -det;
    }
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor priv_zonotope_facet_normals(const Zonotope &Z) {
    const HostZonotope H = priv_host_zonotope(Z);
    const Tensor &G = Z.generators();
    if (H.n == 1) return Tensor::like(G, {1.0}, {1, 1});
    const int64_t k = H.n - 1;
    // the subsets of generators whose cross product is not zero, with that product on the host
    std::vector<int64_t> columns;
    std::vector<double> cross;
    std::vector<int64_t> subset(k);
    for (int64_t i = 0; i < k; ++i) subset[i] = i;
    while (H.m >= k) {
        std::vector<double> Q(H.n * k);
        for (int64_t r = 0; r < H.n; ++r)
            for (int64_t j = 0; j < k; ++j) Q[r * k + j] = H.G[r * H.m + subset[j]];
        const std::vector<double> v = aux_cross(Q, H.n);
        double sum = 0.0;
        for (const double x : v) sum += x * x;
        if (sum > 0.0) { // a zero normal is NaN in CORA, and CORA drops the row
            columns.insert(columns.end(), subset.begin(), subset.end());
            cross.insert(cross.end(), v.begin(), v.end());
        }
        // the next subset in lexicographic order
        int64_t i = k - 1;
        while (i >= 0 && subset[i] == H.m - k + i) --i;
        if (i < 0) break;
        ++subset[i];
        for (int64_t j = i + 1; j < k; ++j) subset[j] = subset[j - 1] + 1;
    }
    const int64_t rows = int64_t(columns.size()) / k;
    if (rows == 0) return G.zeros_like({0, H.n});
    // the normal of each subset by tensor operations, so that it carries the gradient: the part
    // of the host cross product w orthogonal to the subset Q, w - Q (Q'Q)^-1 Q' w, normalized
    const Tensor Q = G.take(columns, -1).reshape({H.n, rows, k}).permute_dims({1, 0, 2});
    const Tensor w = Tensor::like(G, cross, {rows, H.n, 1});
    const Tensor Qt = Q.matrix_transpose();
    const Tensor orthogonal = w - Q.matmul(Qt.matmul(Q).solve(Qt.matmul(w)));
    const Tensor unit = orthogonal.divide(orthogonal.vector_norm(Tensor::Norm::two, -2, true));
    return unit.reshape({rows, H.n});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
