// priv_check_flow - the sets whose flow may cross a hyperplane guard, as CORA's checkFlow
//
// The normal of the guard is turned to point out of the invariant (judged at the center of the
// first set; for a linear flow with `certified`, by a row of the invariant, as CORALean's
// Location.guardOrientFloat?, and no filtering if no row certifies a side). A set is kept if the
// flow at the center of its projection onto the hyperplane points along the normal, or else if the
// flow over a box of the projection (in the Gram-Schmidt basis of the normal, reduced by PCA to
// order 1) can: sup(a'(A x + u)) > 0 for x in it and u in U. For a nonlinear flow CORA bounds a' f
// over that box by a Taylor model; CORA.cpp bounds it by the second-order Taylor form with the
// Hessian over the box, which is as exact for an affine f (docs/cora-differences.md). With
// `certified` a linear flow drops a set only where the flow set is certified strictly inward (CORA
// drops at sup <= 0, which loses a run tangential to the guard).
//
// Syntax:   std::vector<Set> P = priv_check_flow(loc, guard, R, U, certified);
// Inputs:   loc - the location
//           guard - a hyperplane guard (one equality, maybe inequalities)
//           R - the sets that meet it
//           U - the input set in the state space (B U + c), or that of f
//           certified - a linear flow: orientation by an invariant row, strict drop
// Outputs:  P - the sets of R that the flow may carry through the guard, in their order
// See also: priv_guard_intersect
//
// CORA: hybridDynamics/@location/checkFlow.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   12-November-2026
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
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
    const bool inside = priv_region_contains(loc.invariant(), c);
    // CORA flips when the center is inside the invariant xor below the hyperplane
    if (inside != aux_below(a, d, c)) return {-a, -d};
    return {a, d};
}

/// CORALean's Location.guardOrientFloat?: the guard normal n and offset g as they are, or negated,
/// whichever an inequality row of the invariant certifies (the row equal to it with a right-hand
/// side at most g); none if no row does, also for an invariant that is not a polytope.
std::optional<std::pair<Tensor, double>> aux_oriented_by_row(const Location &loc,
                                                             const Polytope &guard) {
    const auto *inv = std::get_if<Polytope>(&loc.invariant());
    if (inv == nullptr) return std::nullopt;
    const Tensor n = guard.Ae().slice(-2, 0, 1).matrix_transpose();
    const double g = guard.be().data()[0];
    const std::vector<double> nv = n.data();
    const std::vector<double> A = inv->A().data(), b = inv->b().data();
    const std::size_t dim = nv.size(), rows = b.size();
    // whether some row i has A_i = s n and b_i <= s g
    const auto certifies = [&](double s) {
        for (std::size_t i = 0; i < rows; ++i) {
            bool same = true;
            for (std::size_t j = 0; j < dim && same; ++j) same = A[i * dim + j] == s * nv[j];
            if (same && b[i] <= s * g) return true;
        }
        return false;
    };
    if (certifies(1.0)) return std::make_pair(n, g);
    if (certifies(-1.0)) return std::make_pair(-n, -g);
    return std::nullopt;
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
                    const Zonotope &U_state, bool strict) {
    if (const auto *sys = std::get_if<NonlinearSys>(&loc.flow())) {
        // the fast test at the projected center, then the bound over the reduced projection
        const Tensor P = a.eye_like(a.shape()[0]) - a.matmul(a.matrix_transpose());
        const Tensor c = P.matmul(S->center()) + a * d;
        if (a.matrix_transpose().matmul(priv_flow_at(loc, c, U_state, "checkFlow")).item() > 0)
            return true;
        const Zonotope Z = priv_zonotopes(S).front().linear_map(P) + a * d;
        const Tensor B = priv_basis_of(a);
        const Zonotope R = Z.linear_map(B.matrix_transpose()).reduce_pca().linear_map(B);
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
        if (strict ? up < 0 : up <= 0) return false;
    }
    return true;
}

// The flow along a level set ----------------------------------------------------------------------

/// CORA's aux_getOutside: +1 if g >= 0 is outside the invariant, -1 if g < 0 is, by the first set.
double aux_outside(const Location &loc, const LevelSet &guard, const Set &first) {
    const Tensor c = first->center();
    const double g = guard.fun_han(c).item();
    if (priv_region_contains(loc.invariant(), c)) return g > 0 ? -1.0 : 1.0;
    return g <= 0 ? -1.0 : 1.0;
}

/// outside * grad g(x)' f(x, u) at the point c of the set with the input at the center of U.
double aux_level_flow_at(const NonlinearSys &sys, const LevelSet &guard, double outside,
                         const Tensor &c, const Zonotope &U_state) {
    const std::vector<double> x = c.data();
    const std::vector<double> f =
        (sys.input_dim() == 0 ? sys.dynamics(c) : sys.dynamics(c, U_state.center())).data();
    double flow = 0;
    // sum_j dg/dx_j f_j
    for (int64_t j = 0; j < sys.dim(); ++j)
        flow += guard.gradient(0, j).eval(x) * f[std::size_t(j)];
    return outside * flow;
}

/// Whether the flow may point out along the gradient at some point of the set S: the fast test at
/// the center, then the upper bound of outside * grad g' f over the PCA box of S by Taylor models.
bool aux_level_flow_along(const Location &loc, const LevelSet &guard, double outside, const Set &S,
                          const Zonotope &U_state) {
    const auto *sys = std::get_if<NonlinearSys>(&loc.flow());
    if (sys == nullptr)
        CORA_FAIL(Error::not_supported,
                  "checkFlow: a level set guard is supported for a nonlinear flow only");
    if (aux_level_flow_at(*sys, guard, outside, S->center(), U_state) > 0) return true;
    // the models of the states and, if there are inputs, of the input box
    const Zonotope R = priv_zonotopes(S).front().reduce_pca();
    const std::vector<Taylm> t = sys->input_dim() > 0
                                     ? Taylm::from_zonotope_and_box(R, U_state.interval())
                                     : Taylm::from_zonotope(R);
    // outside * sum_j grad_j(tay) f_j(tay, u)
    Taylm flow = Taylm::constant(0.0, t.front().num_vars());
    for (int64_t j = 0; j < sys->dim(); ++j)
        flow = flow.plus(Taylm::evaluate(guard.gradient(0, j), t)
                             .times(Taylm::evaluate(sys->expressions()[std::size_t(j)], t)));
    // the upper bound of outside * flow, from the bound of flow on the side of its sign
    const std::array<double, 2> range = flow.bounds();
    return (outside > 0 ? range[1] : -range[0]) > 0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Set> priv_check_flow_outward(const Location &loc, const Tensor &a, double d,
                                         const std::vector<Set> &R, const Zonotope &U_state,
                                         bool strict) {
    std::vector<Set> out;
    for (const Set &S : R)
        if (aux_flow_along(loc, a, d, S, U_state, strict)) out.push_back(S);
    return out;
}

std::vector<Set> priv_check_flow(const Location &loc, const LevelSet &guard,
                                 const std::vector<Set> &R, const Zonotope &U_state) {
    if (guard.num_equations() != 1)
        CORA_FAIL(Error::not_supported,
                  "checkFlow: a level set guard needs exactly one constraint");
    // the orientation of the guard by the first set, then each set against it
    const double outside = aux_outside(loc, guard, R.front());
    std::vector<Set> out;
    for (const Set &S : R)
        if (aux_level_flow_along(loc, guard, outside, S, U_state)) out.push_back(S);
    return out;
}

std::vector<Set> priv_check_flow(const Location &loc, const Polytope &guard,
                                 const std::vector<Set> &R, const Zonotope &U_state,
                                 bool certified) {
    if (certified && std::holds_alternative<LinearSys>(loc.flow())) {
        // no row of the invariant certifies a side: no filtering
        const auto oriented = aux_oriented_by_row(loc, guard);
        if (!oriented) return R;
        return priv_check_flow_outward(loc, oriented->first, oriented->second, R, U_state, true);
    }
    const auto [a, d] = aux_outward(loc, guard, R.front());
    return priv_check_flow_outward(loc, a, d, R, U_state, false);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
