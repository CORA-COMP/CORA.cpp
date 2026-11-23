// is_full_dim - whether a polynomial zonotope has a nonempty interior, as CORALean's isFullDim
//
// CORA tests rank([G GI]) = n, which is necessary and not sufficient: (a, a^2) has rank 2 and no
// interior. CORALean certifies instead: the independent generators GI must have a right inverse H
// (candidate GI' (GI GI')^-1, Gauss-Jordan without pivoting) that Zonotope.ballCert accepts, a box
// of positive radius inside the set. Only true is a proof; a set that is full dimensional through
// its dependent generators alone is false (docs/cora-differences.md).
//
// Syntax:   res = pZ.is_full_dim();   res = pZ.is_full_dim(tol);
// Inputs:   tol - checked and unused (the certificate has none); default: the repository tolerance
// Outputs:  res - true if the certificate holds
// See also: is_bounded, compact
//
// CORA: contSet/@polyZonotope/isFullDim.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <functional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Matrix = std::vector<std::vector<double>>;

// The inverse ------------------------------------------------------------------------------------

/// CORALean's basisInverse: Gauss-Jordan elimination of P (n x n) without pivoting.
Matrix aux_basis_inverse(Matrix A) {
    const std::size_t n = A.size();
    Matrix B(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) B[i][i] = 1.0;
    for (std::size_t k = 0; k < n; ++k) {
        // row k to a unit pivot, then cleared from every other row
        const double p = A[k][k];
        std::vector<double> row_a(n), row_b(n), f(n);
        for (std::size_t j = 0; j < n; ++j) {
            row_a[j] = A[k][j] / p;
            row_b[j] = B[k][j] / p;
            f[j] = A[j][k];
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                A[i][j] = i == k ? row_a[j] : A[i][j] - f[i] * row_a[j];
                B[i][j] = i == k ? row_b[j] : B[i][j] - f[i] * row_b[j];
            }
    }
    return B;
}

// The certificate --------------------------------------------------------------------------------

/// The sum of the products of two vectors, from the first entry to the last.
double aux_dot(std::size_t m, const std::function<double(std::size_t)> &a,
               const std::function<double(std::size_t)> &b) {
    double sum = 0.0;
    for (std::size_t l = 0; l < m; ++l) sum += a(l) * b(l);
    return sum;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool PolyZonotope::is_full_dim(Tol tol_arg) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol_arg, c_.dtype(), "is_full_dim"));
    const std::size_t n = std::size_t(dim()), q = std::size_t(GI_.shape()[0]);
    if (q == 0 || n == 0) return n == 0;
    // G = GI' (n x q) of CORALean, as rows of the host
    const std::vector<double> gi = GI_.data();
    const auto G = [&](std::size_t i, std::size_t l) { return gi[l * n + i]; };
    // the candidate right inverse H = G' (G G')^-1 (q x n)
    Matrix gram(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            gram[i][j] = aux_dot(
                q, [&](std::size_t l) { return G(i, l); }, [&](std::size_t l) { return G(j, l); });
    const Matrix inverse = aux_basis_inverse(gram);
    Matrix H(q, std::vector<double>(n));
    for (std::size_t j = 0; j < q; ++j)
        for (std::size_t i = 0; i < n; ++i)
            H[j][i] = aux_dot(
                n, [&](std::size_t l) { return G(l, j); },
                [&](std::size_t l) { return inverse[l][i]; });
    // the radius of the box: a quarter of 1 over the largest absolute row sum of H
    std::vector<double> row_sum(q, 0.0);
    double largest = 0.0;
    for (std::size_t j = 0; j < q; ++j) {
        for (std::size_t i = 0; i < n; ++i) row_sum[j] += std::abs(H[j][i]);
        largest = j == 0 ? row_sum[j] : std::max(largest, row_sum[j]);
    }
    const double B = 1.0 / ((largest + largest) + (largest + largest));
    // ballCert: 0 < B, the residual of G H at most 1/2, and 2 B |H_j| at most 1 for every row
    if (!(0.0 < B)) return false;
    double residual = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double row = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            row += std::abs((i == j ? 1.0 : 0.0) - aux_dot(
                                                       q, [&](std::size_t l) { return G(i, l); },
                                                       [&](std::size_t l) { return H[l][j]; }));
        residual = i == 0 ? row : std::max(residual, row);
    }
    if (!(residual <= 0.5)) return false;
    for (std::size_t j = 0; j < q; ++j)
        if (!((B + B) * row_sum[j] <= 1.0)) return false;
    return true;
}

bool PolyZonotope::is_full_dim_impl(Tol tol) const { return is_full_dim(tol); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
