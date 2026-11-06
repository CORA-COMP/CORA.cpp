// priv_vertex_candidates - points of a constrained zonotope that contain its vertices
//
// x has |x|_inf <= 1 and A x = b, the fixed factors at +-1; each point appears once and K = 0 for
// an empty set. A vertex of the set is among them, and a convex set contains the set if it
// contains them; they are not all vertices. Error::not_supported beyond
// numerics::kVertexCandidates bases (CORA enumerates the vertices of the polytope of the factors
// instead).
//
// Syntax:   Tensor V = priv_vertex_candidates(cz);
// Inputs:   cz - a single constrained zonotope
// Outputs:  V - the images c + G x (n, K) of the factor vectors x that fix all but rank(A) factors
// See also: priv_blocks, ConZonotope::isequal
//
// CORA: contSet/@conZonotope/private/priv_potVertices.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Elimination -------------------------------------------------------------------------------------

/// The reduced row echelon form of A x = b: the independent rows (rank x m, row-major) with their
/// right-hand sides, whether the zero rows are consistent.
struct Echelon {
    int64_t rank = 0;
    std::vector<double> R, d;
    bool consistent = true;
};

/// Gauss-Jordan elimination of [A b] with the largest pivot of a column; entries up to
/// kSimplexPivot of the largest entry of A count as zero.
Echelon aux_echelon(const HostConZonotope &H) {
    const int64_t p = H.p, m = H.m, width = m + 1;
    std::vector<double> M(std::size_t(p * width));
    double scale = 0.0;
    for (int64_t i = 0; i < p; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            M[i * width + j] = H.A[i * m + j];
            scale = std::max(scale, std::abs(H.A[i * m + j]));
        }
        M[i * width + m] = H.b[i];
    }
    const double tol = numerics::kSimplexPivot * std::max(scale, 1.0);
    // a pivot is the largest entry of the column among the rows not yet used
    int64_t rank = 0;
    for (int64_t j = 0; j < m && rank < p; ++j) {
        int64_t best = rank;
        for (int64_t i = rank; i < p; ++i)
            if (std::abs(M[i * width + j]) > std::abs(M[best * width + j])) best = i;
        if (std::abs(M[best * width + j]) <= tol) continue;
        for (int64_t k = 0; k < width; ++k) std::swap(M[rank * width + k], M[best * width + k]);
        const double pivot = M[rank * width + j];
        for (int64_t k = 0; k < width; ++k) M[rank * width + k] /= pivot;
        for (int64_t i = 0; i < p; ++i) {
            if (i == rank) continue;
            const double factor = M[i * width + j];
            for (int64_t k = 0; k < width; ++k) M[i * width + k] -= factor * M[rank * width + k];
        }
        ++rank;
    }
    // keep the independent rows; the others are 0 = d and decide consistency
    Echelon e;
    e.rank = rank;
    for (int64_t i = 0; i < rank; ++i) {
        e.R.insert(e.R.end(), M.begin() + i * width, M.begin() + i * width + m);
        e.d.push_back(M[i * width + m]);
    }
    for (int64_t i = rank; i < p; ++i)
        e.consistent = e.consistent && std::abs(M[i * width + m]) <= numerics::kSimplexFeasible;
    return e;
}

// Bases -------------------------------------------------------------------------------------------

/// The LU factors (partial pivoting) of the r x r matrix B; false if it is singular.
bool aux_lu(std::vector<double> &B, std::vector<int64_t> &perm, int64_t r) {
    perm.resize(static_cast<std::size_t>(r));
    for (int64_t i = 0; i < r; ++i) perm[i] = i;
    for (int64_t k = 0; k < r; ++k) {
        int64_t best = k;
        for (int64_t i = k; i < r; ++i)
            if (std::abs(B[i * r + k]) > std::abs(B[best * r + k])) best = i;
        if (std::abs(B[best * r + k]) <= numerics::kSimplexPivot) return false;
        if (best != k) {
            for (int64_t j = 0; j < r; ++j) std::swap(B[k * r + j], B[best * r + j]);
            std::swap(perm[k], perm[best]);
        }
        for (int64_t i = k + 1; i < r; ++i) {
            B[i * r + k] /= B[k * r + k];
            for (int64_t j = k + 1; j < r; ++j) B[i * r + j] -= B[i * r + k] * B[k * r + j];
        }
    }
    return true;
}

/// The solution of B x = rhs from the factors of aux_lu.
std::vector<double> aux_lu_solve(const std::vector<double> &B, const std::vector<int64_t> &perm,
                                 const std::vector<double> &rhs, int64_t r) {
    std::vector<double> x(static_cast<std::size_t>(r));
    for (int64_t i = 0; i < r; ++i) {
        x[i] = rhs[perm[i]];
        for (int64_t j = 0; j < i; ++j) x[i] -= B[i * r + j] * x[j];
    }
    for (int64_t i = r - 1; i >= 0; --i) {
        for (int64_t j = i + 1; j < r; ++j) x[i] -= B[i * r + j] * x[j];
        x[i] /= B[i * r + i];
    }
    return x;
}

/// The number of bases and sign vectors: C(m, r) 2^(m - r).
double aux_count(int64_t m, int64_t r) {
    double count = std::ldexp(1.0, int(m - r));
    for (int64_t i = 1; i <= r; ++i) count = count * double(m - r + i) / double(i);
    return count;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Candidates --------------------------------------------------------------------------------------
Tensor priv_vertex_candidates(const ConZonotope &cz) {
    // the constraints in echelon form: rank independent rows, and whether the rest is consistent
    const HostConZonotope H = priv_host_con_zonotope(cz);
    const int64_t m = H.m;
    const Echelon e = aux_echelon(H);
    if (!e.consistent) return cz.c().zeros_like({H.n, 0});
    const int64_t r = e.rank;
    if (aux_count(m, r) > double(numerics::kVertexCandidates))
        CORA_FAIL(Error::not_supported,
                  "vertices: the set has too many bases (more than 2^20) to list its vertices");
    std::set<std::vector<int64_t>> seen;
    std::vector<double> points;
    // every set B of r factors that the constraints determine, every sign vector for the others
    std::vector<int64_t> basic(static_cast<std::size_t>(r)), perm;
    std::iota(basic.begin(), basic.end(), int64_t{0});
    for (bool more = true; more;) {
        // the columns of the basic factors, factored once for all sign vectors
        std::vector<double> B(std::size_t(r * r));
        for (int64_t i = 0; i < r; ++i)
            for (int64_t k = 0; k < r; ++k) B[i * r + k] = e.R[i * m + basic[k]];
        std::vector<int64_t> others;
        for (int64_t j = 0; j < m; ++j)
            if (!std::binary_search(basic.begin(), basic.end(), j)) others.push_back(j);
        // a singular set of columns is no basis
        if (aux_lu(B, perm, r)) {
            for (int64_t mask = 0; mask < (int64_t{1} << others.size()); ++mask) {
                // the sign vector fixes the other factors at +-1; solve for the basic ones
                std::vector<double> x(std::size_t(m), 0.0), rhs = e.d;
                for (std::size_t k = 0; k < others.size(); ++k) {
                    x[others[k]] = ((mask >> k) & 1) ? 1.0 : -1.0;
                    for (int64_t i = 0; i < r; ++i) rhs[i] -= e.R[i * m + others[k]] * x[others[k]];
                }
                const std::vector<double> xb = aux_lu_solve(B, perm, rhs, r);
                bool inside = true;
                for (int64_t k = 0; k < r; ++k) {
                    inside = inside && std::abs(xb[k]) <= 1.0 + numerics::kSimplexFeasible;
                    x[basic[k]] = std::clamp(xb[k], -1.0, 1.0);
                }
                std::vector<int64_t> key(static_cast<std::size_t>(m));
                for (int64_t j = 0; j < m; ++j)
                    key[j] = std::llround(x[j] / numerics::kCornerMerge);
                if (inside && seen.insert(key).second)
                    points.insert(points.end(), x.begin(), x.end());
            }
        }
        // the next r-subset of {0, ..., m-1} in lexicographic order
        int64_t k = r - 1;
        while (k >= 0 && basic[k] == m - r + k) --k;
        more = k >= 0;
        if (more) {
            ++basic[k];
            for (int64_t j = k + 1; j < r; ++j) basic[j] = basic[j - 1] + 1;
        }
    }
    // without factors the set is its center
    if (m == 0) return cz.c();
    const int64_t count = int64_t(points.size()) / m;
    // one factor vector per row of `points`: transpose into columns (m, K)
    std::vector<double> X(points.size());
    for (int64_t k = 0; k < count; ++k)
        for (int64_t j = 0; j < m; ++j) X[j * count + k] = points[k * m + j];
    return cz.c() + cz.generators().matmul(Tensor::like(cz.c(), X, {m, count}));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
