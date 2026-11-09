// priv_check_flow - the sets whose flow may cross a hyperplane guard, as CORA's checkFlow
//
// The normal of the guard is turned to point out of the invariant (judged at the center of the
// first set). A set is kept if the flow at the center of its projection onto the hyperplane points
// along the normal, or else if the flow over a box of the projection (in the Gram-Schmidt basis of
// the normal, reduced by PCA to order 1) can: sup(a'(A x + u)) > 0 for x in it and u in U.
//
// Syntax:   std::vector<Set> P = priv_check_flow(loc, guard, R, U);
// Inputs:   loc - the location
//           guard - a hyperplane guard (one equality, maybe inequalities)
//           R - the sets that meet it
//           U - the input set in the state space (B U + c)
// Outputs:  P - the sets of R that the flow may carry through the guard, in their order
// See also: priv_guard_intersect
//
// CORA: hybridDynamics/@location/checkFlow.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether a'x <= d holds at the point x up to 100 eps (CORA's contains_ of a halfspace).
bool aux_below(const Tensor &a, double d, const Tensor &x) {
    return a.matrix_transpose().matmul(x).item() <= d + numerics::kGuardSide;
}

/// The unit normal a and offset d of the guard, turned so that a points out of the invariant.
std::pair<Tensor, double> aux_outward(const Location &loc, const Polytope &guard,
                                      const Set &first) {
    auto [a, d] = priv_hyperplane(guard);
    const Tensor c = first->center();
    const bool inside = loc.invariant().contains(c, "exact", numerics::kGuardSide);
    // CORA flips when the center is inside the invariant xor below the hyperplane
    if (inside != aux_below(a, d, c)) return {-a, -d};
    return {a, d};
}

/// CORA's reduce(Z, 'pca', 1): more nonzero generators than dimensions become the box of them in
/// their principal axes.
Zonotope aux_reduce_pca(const Zonotope &Z) {
    const Tensor &G0 = Z.generators();
    const std::vector<double> size = G0.abs().sum(0).data();
    std::vector<int64_t> keep;
    for (std::size_t j = 0; j < size.size(); ++j)
        if (size[j] > 0) keep.push_back(int64_t(j));
    const int64_t n = Z.dim();
    if (int64_t(keep.size()) <= n) return Z;
    const Tensor G = G0.take(keep, -1);
    // the covariance of the points [G, -G] has the axes; its scale does not matter for them
    const Tensor U = G.matmul(G.matrix_transpose()).svd()[0];
    const Tensor box = U.matrix_transpose().matmul(G).abs().sum(-1, true);
    return {Z.center(), U.matmul(box.diag())};
}

/// Whether the flow may point along a at some point of the set projected onto the hyperplane.
bool aux_flow_along(const Location &loc, const Tensor &a, double d, const Set &S,
                    const Zonotope &U_state) {
    const Tensor &A = priv_flow_matrix(loc, "checkFlow");
    const Tensor P = A.eye_like() - a.matmul(a.matrix_transpose());
    const Tensor a_t = a.matrix_transpose(), a_A = a_t.matmul(A);
    // the fast test at the projected center
    const Tensor c = P.matmul(S->center()) + a * d;
    if (a_A.matmul(c).item() + a_t.matmul(U_state.center()).item() > 0) return true;
    const Zonotope Z = priv_zonotopes(S).front().linear_map(P) + a * d;
    // a box of the projection in the basis of the normal, and the flow over it and over U
    const Tensor B = priv_basis_of(a);
    const Zonotope R = aux_reduce_pca(Z.linear_map(B.matrix_transpose())).linear_map(B);
    const double flow =
        a_A.matmul(R.center()).item() + a_A.matmul(R.generators()).abs().sum().item() +
        a_t.matmul(U_state.center()).item() + a_t.matmul(U_state.generators()).abs().sum().item();
    return flow > 0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Set> priv_check_flow(const Location &loc, const Polytope &guard,
                                 const std::vector<Set> &R, const Zonotope &U_state) {
    const auto [a, d] = aux_outward(loc, guard, R.front());
    std::vector<Set> out;
    for (const Set &S : R)
        if (aux_flow_along(loc, a, d, S, U_state)) out.push_back(S);
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
