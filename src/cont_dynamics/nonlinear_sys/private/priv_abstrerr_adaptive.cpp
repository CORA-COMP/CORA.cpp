// priv_abstrerr_adaptive - the linearization error of an adaptive step
//
// As CORA's priv_abstractionError_adaptive for alg "lin": at tensorOrder 2 the Lagrange remainder
// 1/2 dz' max|H_i| dz from the interval Hessian over the set, which is computed once if the
// Hessian is the same over every box (a test on the first use), as a box; at tensorOrder 3 the
// quadratic term as a zonotope (the set reduced adaptively, the Hessian at x*) plus the cubic
// remainder as a box, reduced adaptively. The generators that reduce boxed in a step are kept for
// the other passes of the step. Interval arithmetic that leaves a domain fails as out_of_domain.
//
// Syntax:   AbstractionError e = priv_abstrerr_adaptive(sys, R, x_star, u_star, run);
// Inputs:   R - the set of the states over the step, relative to x_star (CORA's Rmax)
//           x_star - the expansion point of the states
//           u_star - the expansion point of the inputs
//           run - the tensor order, the options, the caches
// Outputs:  e - the set of the error (VerrorDyn) and its bound per state (trueError)
// See also: priv_abstrerr_lin, priv_lin_reach_adaptive
//
// CORA: contDynamics/@nonlinearSys/private/priv_abstractionError_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/within_tol.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The test of a constant Hessian ------------------------------------------------------------------

/// Whether two ranges are the same box up to CORA's isequal tolerance (eps), entry by entry.
bool aux_same(const std::optional<Range> &a, const std::optional<Range> &b) {
    if (a.has_value() != b.has_value()) return false;
    if (!a) return true;
    const double eps = machine_eps();
    const std::vector<double> alo = a->lo.data(), blo = b->lo.data(), ahi = a->hi.data(),
                              bhi = b->hi.data();
    for (std::size_t k = 0; k < alo.size(); ++k)
        if (!within_tol(alo[k], blo[k], eps) || !within_tol(ahi[k], bhi[k], eps)) return false;
    return true;
}

/// CORA's aux_checkIfHessianConst: the Hessian over the boxes enlarged by 1.1 is the same.
void aux_check_hessian_const(const NonlinearSys &sys, AdaptiveRun &run,
                             const std::vector<std::optional<Range>> &H, const Interval &x_box,
                             const std::optional<Interval> &u_box) {
    run.hessian_const = true;
    const Interval x_big = x_box.enlarge(1.1);
    std::optional<Range> u_range;
    if (u_box) u_range = Range(u_box->enlarge(1.1).inf(), u_box->enlarge(1.1).sup());
    const std::vector<std::optional<Range>> H_test =
        sys.hessian_enclosure(Range(x_big.inf(), x_big.sup()), u_range);
    for (std::size_t k = 0; k < H.size() && run.hessian_const; ++k)
        if (!aux_same(H[k], H_test[k])) run.hessian_const = false;
    if (run.hessian_const) run.hessian_value = H;
    run.hessian_checked = true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The error ---------------------------------------------------------------------------------------

AbstractionError priv_abstrerr_adaptive(const NonlinearSys &sys, const Zonotope &R,
                                        const Tensor &x_star, const std::optional<Tensor> &u_star,
                                        AdaptiveRun &run) {
    const int64_t n = sys.dim();
    const Interval IHx = R.interval();
    std::optional<Interval> IHu;
    if (run.U0) IHu = run.U0->interval();
    // the boxes of the states and of the inputs, absolute
    const Interval x_total(IHx.inf() + x_star, IHx.sup() + x_star);
    std::optional<Interval> u_total;
    if (IHu) u_total = Interval(IHu->inf() + *u_star, IHu->sup() + *u_star);
    // the same as ranges for the derivatives
    const Range x_range(x_total.inf(), x_total.sup());
    std::optional<Range> u_range;
    if (u_total) u_range = Range(u_total->inf(), u_total->sup());

    if (run.tensor_order == 2) {
        // the Hessian over the boxes, unless it is known not to depend on them
        std::vector<std::optional<Range>> H_now;
        if (!run.hessian_const) {
            H_now = sys.hessian_enclosure(x_range, u_range);
            if (!run.hessian_checked) aux_check_hessian_const(sys, run, H_now, x_total, u_total);
        }
        // the largest |z - z*| per coordinate, and the remainder from the Hessian
        std::vector<Tensor> dz = priv_entries(priv_abs_max(IHx.inf(), IHx.sup()));
        if (IHu)
            for (Tensor &e : priv_entries(priv_abs_max(IHu->inf(), IHu->sup())))
                dz.push_back(std::move(e));
        const Tensor err = priv_second_order(run.hessian_const ? run.hessian_value : H_now, dz, n);
        return {err, Zonotope(err.zeros_like(), err.diag())};
    }

    // tensorOrder 3: the quadratic term as a set, the cubic remainder as a box, both reduced
    const Zonotope R_red = priv_reduce_cached(R, std::sqrt(run.alg.zeta_z), run.idx_red, run.i);
    const Zonotope Z = run.U0 ? R_red.cart_prod(*run.U0) : R_red;
    const Zonotope second = Z.quad_map(sys.hessian(x_star, u_star)) * 0.5;
    std::optional<Interval> third;
    if (!run.third_order_empty)
        third = priv_third_order(sys.third_order_enclosure(x_range, u_range), priv_dz(IHx, IHu), n,
                                 true);
    if (!third) run.third_order_empty = true;
    const Zonotope V = priv_reduce_cached(third ? second + third->zonotope() : second,
                                          10 * run.alg.zeta_z, run.idx_err, run.i);
    // the bound of the error: |center| + the sum of the lengths of the generators
    const Tensor bound = V.center().abs() + V.generators().abs().sum(-1, true);
    return {bound, V};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
