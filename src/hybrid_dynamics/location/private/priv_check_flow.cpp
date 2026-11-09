// priv_check_flow - the sets whose flow may cross a hyperplane guard, as CORA's checkFlow
//
// The normal of the guard is turned to point out of the invariant (judged at the center of the
// first set). A set is kept if the flow at the center of its projection onto the hyperplane points
// along the normal, or else if the flow over a box of the projection (in the Gram-Schmidt basis of
// the normal, reduced by PCA to order 1) can: sup(a'(A x + u)) > 0 for x in it and u in U. For a
// nonlinear flow CORA bounds a' f over that box by a Taylor model; CORA.cpp bounds it by the
// second-order Taylor form with the Hessian over the box, which is as exact for an affine f
// (docs/cora-differences.md).
//
// Syntax:   std::vector<Set> P = priv_check_flow(loc, guard, R, U);
// Inputs:   loc - the location
//           guard - a hyperplane guard (one equality, maybe inequalities)
//           R - the sets that meet it
//           U - the input set in the state space (B U + c), or that of f
// Outputs:  P - the sets of R that the flow may carry through the guard, in their order
// See also: priv_guard_intersect
//
// CORA: hybridDynamics/@location/checkFlow.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>
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

// The flow over a set -----------------------------------------------------------------------------

/// An upper bound of a' f(z) over z in R x box(U): the second-order Taylor form at the center,
/// a' f(z_c) + |a' J(z_c) G| 1 + 1/2 sum_i |a_i| g' max|H_i(Z)| g with g the box radius of Z;
/// exact for an affine f, as CORA's Taylor model of the flow.
double aux_nonlinear_flow_bound(const NonlinearSys &sys, const Tensor &a, const Zonotope &R,
                                const Zonotope &U) {
    const bool inputs = sys.input_dim() > 0;
    const Zonotope Ubox = U.interval().zonotope();
    const Zonotope Z = inputs ? R.cart_prod(Ubox) : R;
    const std::optional<Tensor> uc = inputs ? std::optional<Tensor>(Ubox.center()) : std::nullopt;
    // the value and the linear part at the center
    const Tensor a_t = a.matrix_transpose();
    const Jacobians J = sys.jacobian(R.center(), uc);
    const Tensor AB = inputs ? Tensor::concat({J.A, J.B}, -1) : J.A;
    double bound = a_t.matmul(sys.dynamics(R.center(), uc)).item() +
                   a_t.matmul(AB).matmul(Z.generators()).abs().sum().item();
    // the remainder by the Hessian over the box of Z
    const Interval box = Z.interval();
    const int64_t n = sys.dim(), m = sys.input_dim(), dz = n + m;
    const Range x_box(box.inf().slice(-2, 0, n), box.sup().slice(-2, 0, n));
    const std::optional<Range> u_box =
        inputs ? std::optional<Range>(Range(box.inf().slice(-2, n, dz), box.sup().slice(-2, n, dz)))
               : std::nullopt;
    const std::vector<std::optional<Range>> H = sys.hessian_enclosure(x_box, u_box);
    const std::vector<double> g = Z.generators().abs().sum(-1, true).data(), w = a.data();
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < dz; ++j)
            for (int64_t k = 0; k < dz; ++k)
                if (const auto &h = H[std::size_t((i * dz + j) * dz + k)])
                    bound += 0.5 * std::abs(w[std::size_t(i)]) * g[std::size_t(j)] *
                             g[std::size_t(k)] *
                             std::max(std::abs(h->lo.item()), std::abs(h->hi.item()));
    return bound;
}

// The test of a set -------------------------------------------------------------------------------

/// Whether the flow may point along a at some point of the set projected onto the hyperplane;
/// false if a member of a bundle is certified to have no outward flow.
bool aux_flow_along(const Location &loc, const Tensor &a, double d, const Set &S,
                    const Zonotope &U_state) {
    if (const auto *sys = std::get_if<NonlinearSys>(&loc.flow())) {
        // the fast test at the projected center, then the bound over the reduced projection
        const Tensor P = a.eye_like(a.shape()[0]) - a.matmul(a.matrix_transpose());
        const Tensor c = P.matmul(S->center()) + a * d;
        if (a.matrix_transpose().matmul(priv_flow_at(loc, c, U_state, "checkFlow")).item() > 0)
            return true;
        const Zonotope Z = priv_zonotopes(S).front().linear_map(P) + a * d;
        const Tensor B = priv_basis_of(a);
        const Zonotope R = aux_reduce_pca(Z.linear_map(B.matrix_transpose())).linear_map(B);
        return aux_nonlinear_flow_bound(*sys, a, R, U_state) > 0;
    }
    // a linear flow follows CORALean's checkFlowFloat on each zonotope of the set
    const Tensor &A = priv_flow_matrix(loc, "checkFlow");
    const Tensor P = A.eye_like() - a.matmul(a.matrix_transpose());
    const int64_t n = A.shape()[0];
    const std::vector<double> flow = A.data(), uc = U_state.center().data(), nv = a.data();
    for (const Zonotope &Z : priv_zonotopes(S)) {
        // the set projected onto the hyperplane
        const Zonotope proj = Z.linear_map(P) + a * d;
        // the fast test: the flow at the center, summed from the last term as sumUp
        const std::vector<double> m = proj.center().data();
        double outward = 0;
        for (int64_t i = n - 1; i >= 0; --i) {
            double row = 0;
            for (int64_t j = n - 1; j >= 0; --j)
                row = flow[std::size_t(i * n + j)] * m[std::size_t(j)] + row;
            outward = nv[std::size_t(i)] * (row + uc[std::size_t(i)]) + outward;
        }
        if (!(outward <= 0)) continue;
        // the flow set A proj + V, bounded from above along the normal
        const double up = proj.linear_map(A).plus(U_state).support_function(a).item();
        if (up <= 0) return false;
    }
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Set> priv_check_flow_outward(const Location &loc, const Tensor &a, double d,
                                         const std::vector<Set> &R, const Zonotope &U_state) {
    std::vector<Set> out;
    for (const Set &S : R)
        if (aux_flow_along(loc, a, d, S, U_state)) out.push_back(S);
    return out;
}

std::vector<Set> priv_check_flow(const Location &loc, const Polytope &guard,
                                 const std::vector<Set> &R, const Zonotope &U_state) {
    const auto [a, d] = aux_outward(loc, guard, R.front());
    return priv_check_flow_outward(loc, a, d, R, U_state);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
