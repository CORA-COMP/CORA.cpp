// priv_calc_basis - the orthogonal bases that enclose a guard intersection, as CORA's calcBasis
//
// "box" is the identity. "pca" is the left singular vectors of [-G, G], the generators of all
// sets (projected onto a hyperplane guard; at most the 500 longest). "flow" is the Gram-Schmidt
// basis of the flow at the mean center (projected onto a hyperplane guard), so its first axis
// follows the flow and the others lie across it.
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
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORA caps the generators of "pca" at the 500 longest.
constexpr int64_t kMaxPcaGenerators = 500;

// The generators ----------------------------------------------------------------------------------

/// The map x -> (I - a a') x onto the hyperplane a'x = d with a unit normal (the offset aside).
Tensor aux_onto_hyperplane(const Polytope &guard, const Tensor &M) {
    const Tensor a = priv_hyperplane(guard).first;
    return M - a.matmul(a.matrix_transpose().matmul(M));
}

/// The generators of all sets side by side, projected onto a hyperplane guard.
Tensor aux_generators(const std::vector<Set> &R, const Polytope &guard, bool hyperplane) {
    std::vector<Tensor> G;
    for (const Set &S : R)
        for (const Zonotope &Z : priv_zonotopes(S)) G.push_back(Z.generators());
    Tensor all = Tensor::concat(G, -1);
    if (hyperplane) all = aux_onto_hyperplane(guard, all);
    const int64_t m = all.shape().back();
    if (m <= kMaxPcaGenerators) return all;
    // the longest generators, by the squared length (a stable sort, as MATLAB's)
    const std::vector<double> len = all.square().sum(0).data();
    std::vector<int64_t> order(static_cast<std::size_t>(m));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::stable_sort(order.begin(), order.end(), [&](int64_t i, int64_t j) {
        return len[std::size_t(i)] > len[std::size_t(j)];
    });
    order.resize(std::size_t(kMaxPcaGenerators));
    return all.take(order, -1);
}

/// The left singular vectors of [-G, G], all n of them.
Tensor aux_pca(const Tensor &G) {
    const Tensor M = Tensor::concat({-G, G}, -1);
    const int64_t n = G.shape()[0];
    // a wide matrix gives n vectors; else the n x n Gram matrix has the same left vectors
    if (M.shape().back() >= n) return M.svd()[0];
    return M.matmul(M.matrix_transpose()).svd()[0];
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
            B.push_back(aux_pca(aux_generators(R, guard, hyperplane)));
            break;
        case alg::Enclose::flow: {
            // the flow A c + u at the mean center, on the hyperplane if the guard is one
            Tensor c = aux_mean_center(R);
            if (hyperplane) {
                const auto [a, d] = priv_hyperplane(guard);
                c = aux_onto_hyperplane(guard, c) + a * d;
            }
            const Tensor dir =
                priv_flow_matrix(loc, "calcBasis 'flow'").matmul(c) + U_state.center();
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
