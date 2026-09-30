// priv_zonotopeMeetsPolytope - whether a zonotope meets a polytope, exactly by a linear program
//
// With beta = u - 1 the question is whether u in [0, 2]^m exists with (a_i'G) u <= r_i, where
// r_i = b_i - a_i'c + a_i'G 1. Phase 1 of the simplex method (Bland's rule, dense tableau) drives
// the artificial variables of the rows with r_i < 0 to zero; the set meets the polytope iff it can.
//
// Syntax:   bool hit = priv_zonotopeMeetsPolytope(c, G, m, a, b);
// Inputs:   c, G (n x m row-major) - the zonotope;  a, b - the halfspaces {x | a[i]'x <= b[i]}
// Outputs:  hit - true if the zonotope and the polytope have a common point (up to 1e-9)
// See also: Specification::check

#include "specification/private/priv.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Pivots the tableau on (row, col): the column becomes a unit vector.
void aux_pivot(std::vector<std::vector<double>> &T, std::vector<double> &cost, int row, int col) {
    const double p = T[row][col];
    for (double &x : T[row]) x /= p;
    for (std::size_t r = 0; r < T.size(); ++r) {
        if (static_cast<int>(r) == row || T[r][col] == 0.0) continue;
        const double f = T[r][col];
        for (std::size_t j = 0; j < T[r].size(); ++j) T[r][j] -= f * T[row][j];
    }
    const double f = cost[col];
    for (std::size_t j = 0; j < cost.size(); ++j) cost[j] -= f * T[row][j];
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool priv_zonotopeMeetsPolytope(const std::vector<double> &c, const std::vector<double> &G, int m,
                                const std::vector<std::vector<double>> &a,
                                const std::vector<double> &b) {
    const int n = static_cast<int>(c.size());
    const int p = static_cast<int>(a.size());

    // Rows ---------------------------------------------------------------------------------

    // rows of the polytope in u: (a_i'G) u <= r_i
    std::vector<std::vector<double>> aG(p, std::vector<double>(m, 0.0));
    std::vector<double> r(p);
    for (int i = 0; i < p; ++i) {
        double ac = 0;
        for (int k = 0; k < n; ++k) ac += a[i][k] * c[k];
        r[i] = b[i] - ac;
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n; ++k) aG[i][j] += a[i][k] * G[k * m + j];
            r[i] += aG[i][j];
        }
    }

    // columns: u (m), slack of the polytope rows (p), slack of the bounds (m), artificials
    int nArt = 0;
    for (double ri : r) nArt += ri < 0;
    const int rows = p + m, cols = m + p + m + nArt, nReal = m + p + m;
    std::vector<std::vector<double>> T(rows, std::vector<double>(cols + 1, 0.0));
    std::vector<int> basis(rows);
    int art = 0;
    for (int i = 0; i < p; ++i) {
        const double sign = r[i] < 0 ? -1.0 : 1.0;  // the right-hand side must be nonnegative
        for (int j = 0; j < m; ++j) T[i][j] = sign * aG[i][j];
        T[i][m + i] = sign;
        T[i][cols] = sign * r[i];
        if (r[i] < 0) {
            T[i][nReal + art] = 1.0;
            basis[i] = nReal + art++;
        } else {
            basis[i] = m + i;
        }
    }
    for (int j = 0; j < m; ++j) {
        T[p + j][j] = 1.0;
        T[p + j][m + p + j] = 1.0;
        T[p + j][cols] = 2.0;
        basis[p + j] = m + p + j;
    }
    if (nArt == 0) return true;  // u = 0 (the corner beta = -1) already satisfies every row

    // Simplex ------------------------------------------------------------------------------

    // phase 1: minimize the artificials; the cost row holds the reduced costs
    std::vector<double> cost(cols + 1, 0.0);
    for (int i = 0; i < p; ++i) {
        if (r[i] >= 0) continue;
        for (int j = 0; j < nReal; ++j) cost[j] -= T[i][j];
        cost[cols] -= T[i][cols];
    }
    const int maxIterations = 50 * (rows + cols);
    for (int it = 0; it < maxIterations; ++it) {
        // Bland's rule: the lowest column with a negative reduced cost enters
        int col = -1;
        for (int j = 0; j < cols && col < 0; ++j)
            if (cost[j] < -1e-12) col = j;
        if (col < 0) break;
        int row = -1;
        double best = 0;
        for (int i = 0; i < rows; ++i) {
            if (T[i][col] <= 1e-12) continue;
            const double ratio = T[i][cols] / T[i][col];
            const bool better = row < 0 || ratio < best - 1e-15 ||
                                (ratio <= best + 1e-15 && basis[i] < basis[row]);
            if (better) {
                row = i;
                best = ratio;
            }
        }
        if (row < 0) break;
        aux_pivot(T, cost, row, col);
        basis[row] = col;
    }

    // the objective is -cost[cols]; a value near zero means a common point exists
    double scale = 1.0;
    for (double ri : r) scale = std::max(scale, std::abs(ri));
    return -cost[cols] <= 1e-9 * scale;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
