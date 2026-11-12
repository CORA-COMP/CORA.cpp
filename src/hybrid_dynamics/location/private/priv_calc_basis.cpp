// priv_calc_basis - the orthogonal bases that enclose a guard intersection, as CORA's calcBasis
//
// "box" is the identity. "pca" is the left singular vectors of [-G, G], the generators of all
// sets (projected onto a hyperplane guard; at most the 500 longest), found as CORALean's
// Location.calcBasisPcaFloat does: the left singular vectors of G by ten cyclic one-sided Jacobi
// sweeps, by decreasing singular value, in plain double arithmetic (orthonormal up to rounding,
// which the guard intersections account for by the defect term). "flow" is the Gram-Schmidt basis
// of the flow at the mean center (projected onto a hyperplane guard), so its first axis follows the
// flow and the others lie across it.
//
// Syntax:   std::vector<Tensor> B = priv_calc_basis(loc, R, guard, enclose, U);
// Inputs:   loc - the location
//           R - the sets that meet the guard
//           guard - the guard
//           enclose - the methods, one basis each
//           U - the input set in the state space (B U + c), whose center drives "flow"
// Outputs:  B - the bases (n, n), orthonormal columns
// See also: priv_guard_intersect
//
// CORA: hybridDynamics/@location/calcBasis.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORA caps the generators of "pca" at the 500 longest.
constexpr int64_t kMaxPcaGenerators = 500;

// The generators ----------------------------------------------------------------------------------

/// A vector of doubles that is the column of a matrix.
using Column = std::vector<double>;

/// The sum f(0) + (f(1) + (... + (f(n-1) + 0))), nested to the right as CORALean's sumUp.
double aux_sum_right(const std::vector<double> &f) {
    double acc = 0.0;
    for (std::size_t k = f.size(); k-- > 0;) acc = f[k] + acc;
    return acc;
}

/// The dot product of two vectors, summed from the last entry (CORALean's dotUp).
double aux_dot(const Column &a, const Column &b) {
    std::vector<double> f(a.size());
    for (std::size_t k = 0; k < a.size(); ++k) f[k] = a[k] * b[k];
    return aux_sum_right(f);
}

/// The generators of all sets, as columns; projected on the hyperplane with the unit normal c by
/// g - c (c'g), and at most the 500 longest (by the squared length, a stable sort).
std::vector<Column> aux_generators(const std::vector<Set> &R, const Column &nv, bool hyperplane) {
    std::vector<Column> cols;
    for (const Set &S : R) {
        for (const Zonotope &Z : priv_zonotopes(S)) {
            const Tensor G = Z.generators();
            const int64_t n = G.shape()[0], h = G.shape()[1];
            const std::vector<double> d = G.data();
            for (int64_t j = 0; j < h; ++j) {
                Column g(static_cast<std::size_t>(n));
                for (int64_t i = 0; i < n; ++i) g[std::size_t(i)] = d[std::size_t(i * h + j)];
                cols.push_back(std::move(g));
            }
        }
    }
    if (hyperplane) {
        // the unit normal, and the projection of each generator
        const Column &v = nv;
        const double len = std::sqrt(aux_dot(v, v));
        Column c(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) c[i] = v[i] / len;
        // g - c (c'g), each product rounded on its own
        for (Column &g : cols) {
            const double dg = aux_dot(c, g);
            for (std::size_t i = 0; i < g.size(); ++i) {
                const double cd = c[i] * dg;
                g[i] = g[i] - cd;
            }
        }
    }
    // the longest generators by the squared length, a stable sort
    if (static_cast<int64_t>(cols.size()) <= kMaxPcaGenerators) return cols;
    std::vector<double> len(cols.size());
    for (std::size_t j = 0; j < cols.size(); ++j) len[j] = aux_dot(cols[j], cols[j]);
    std::vector<std::size_t> order(cols.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t i, std::size_t j) { return len[i] > len[j]; });
    std::vector<Column> kept;
    for (std::size_t k = 0; k < static_cast<std::size_t>(kMaxPcaGenerators); ++k)
        kept.push_back(cols[order[k]]);
    return kept;
}

/// The map x -> (I - a a') x onto the hyperplane a'x = d with a unit normal (the offset aside).
Tensor aux_onto_hyperplane(const Polytope &guard, const Tensor &M) {
    const Tensor a = priv_hyperplane(guard).first;
    return M - a.matmul(a.matrix_transpose().matmul(M));
}

// The eigenvectors --------------------------------------------------------------------------------

/// The columns of W: the generators by coordinate.
using Columns = std::vector<std::vector<double>>;

/// The dot product of two columns, summed from the first entry (CORALean's step_pcaDot).
double aux_col_dot(const std::vector<double> &a, const std::vector<double> &b) {
    double acc = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        const double prod = a[k] * b[k];
        acc = acc + prod;
    }
    return acc;
}

/// One Hestenes rotation of the columns p and q of W, applied to W and to V, so that the two
/// columns become orthogonal (CORALean's Location.step_pcaRotate); none where they are orthogonal
/// up to 2^-52 of the product of their lengths.
void aux_rotate(std::size_t p, std::size_t q, Columns &W, Columns &V) {
    const double alpha = aux_col_dot(W[p], W[p]);
    const double beta = aux_col_dot(W[q], W[q]);
    const double gamma = aux_col_dot(W[p], W[q]);
    const double gg = gamma * gamma;
    const double ab = alpha * beta;
    if (gg <= std::ldexp(1.0, -104) * ab) return;
    const double zeta = (beta - alpha) / (gamma + gamma);
    const double root = std::sqrt(1.0 + zeta * zeta);
    const double t = (0.0 <= zeta ? 1.0 : -1.0) / (std::fabs(zeta) + root);
    const double c = 1.0 / std::sqrt(1.0 + t * t);
    const double s = t * c;
    // the columns p and q mix
    const auto mix = [&](Columns &M) {
        const std::vector<double> mp = M[p], mq = M[q];
        for (std::size_t k = 0; k < mp.size(); ++k) {
            const double cx = c * mp[k], sy = s * mq[k], sx = s * mp[k], cy = c * mq[k];
            M[p][k] = cx - sy;
            M[q][k] = sx + cy;
        }
    };
    mix(W);
    mix(V);
}

/// The singular vectors of the generators by decreasing length (CORALean's
/// Location.calcBasisPcaFloat): ten cyclic one-sided Jacobi sweeps in plain double arithmetic.
Tensor aux_pca(const std::vector<Column> &G, int64_t n, const Tensor &like) {
    const std::size_t m = static_cast<std::size_t>(n);
    Columns W(m, std::vector<double>(G.size())), V(m, std::vector<double>(m, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        V[i][i] = 1.0;
        for (std::size_t k = 0; k < G.size(); ++k) W[i][k] = G[k][i];
    }
    // ten cyclic sweeps over the pairs p < q
    for (int sweep = 0; sweep < 10; ++sweep)
        for (std::size_t p = 0; p < m; ++p)
            for (std::size_t q = p + 1; q < m; ++q) aux_rotate(p, q, W, V);
    // the columns by decreasing length, ties in order
    std::vector<double> len(m);
    for (std::size_t i = 0; i < m; ++i) len[i] = aux_col_dot(W[i], W[i]);
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t i, std::size_t j) { return len[i] > len[j]; });
    std::vector<double> data(m * m);
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t k = 0; k < m; ++k) data[r * m + k] = V[order[k]][r];
    return Tensor::like(like, data, {n, n});
}

// The flow direction ------------------------------------------------------------------------------

/// The mean of the centers of the sets (each zonotope of a bundle counts).
Tensor aux_mean_center(const std::vector<Set> &R) {
    std::vector<Tensor> c;
    for (const Set &S : R)
        for (const Zonotope &Z : priv_zonotopes(S)) c.push_back(Z.center());
    return Tensor::concat(c, -1).mean(-1, true);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> priv_calc_basis(const Location &loc, const std::vector<Set> &R,
                                    const Polytope &guard, const std::vector<alg::Enclose> &enclose,
                                    const Zonotope &U_state) {
    const bool hyperplane = guard.representsa("conHyperplane", numerics::kGuardHyperplane);
    std::vector<Tensor> B;
    for (alg::Enclose method : enclose) {
        switch (method) {
        case alg::Enclose::box:
            B.push_back(loc.c().eye_like(loc.dim()));
            break;
        case alg::Enclose::pca:
            B.push_back(
                aux_pca(aux_generators(R, hyperplane ? guard.Ae().slice(-2, 0, 1).data() : Column{},
                                       hyperplane),
                        loc.dim(), loc.c()));
            break;
        case alg::Enclose::flow: {
            // the flow at the mean center, on the hyperplane if the guard is one
            Tensor c = aux_mean_center(R);
            if (hyperplane) {
                const auto [a, d] = priv_hyperplane(guard);
                c = aux_onto_hyperplane(guard, c) + a * d;
            }
            const Tensor dir = priv_flow_at(loc, c, U_state, "calcBasis 'flow'");
            B.push_back(priv_basis_of(dir / std::sqrt(dir.square().sum().item())));
            break;
        }
        default:
            CORA_FAIL(Error::wrong_value, "calcBasis: enclose is box, pca or flow");
        }
    }
    return B;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
