// priv_poly_window - the linear step from a polynomial zonotope of the poly-adaptive algorithm
//
// The system x' = A x + u with the input u in V, from the polynomial zonotope X0: the window over
// the step and the differences to its start, and the set at the end for a constant input with a
// polynomial zonotope over the factors of X0. e^{A dt} maps X0 exactly, so the dependency on its
// factors stays; the correction F X0 and the solutions of the inputs are enclosures that join the
// independent part. The input V is time-varying with its center held through hull(0, u), so the
// window covers every time of the step. The end set adds the dependent part of the constant input
// through the integral of e^{As} by exact_plus, the independent part is part of the input
// (CORALean's polyWindowFloat and polyPointFloat; CORA's initReach_adaptive). As CORA does, the
// homogeneous solution and the solutions of the inputs are reduced adaptively at zeta_z before they
// are added, which keeps the independent part small (CORALean has no such reduction).
//
// Syntax:   [Rti, Rdiff] = priv_poly_window(lin, taylor, dt, eta, V, X0, zeta_z);
//           Rtp = priv_poly_point(lin, taylor, dt, eta, V, X0, Vs, zeta_z);
// Inputs:   lin - the linear system, taylor - the powers of A
//           eta - an admissible Taylor order
// Outputs:  the window and the differences; the set at the end of the step
// See also: priv_lin_reach_poly_adaptive_pass.cpp, priv_poly_reach.cpp
//
// CORA: contDynamics/@linearSys/initReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The solution at the end of the step of the input u(t) in Z: the series of A as one matrix
/// applied to Z, and the remainder. The terms are summed first, which is tighter than CORALean's
/// term by term sum and keeps the generators few.
Zonotope aux_solution(const LinearSys &lin, TaylorLinSys &taylor, const Zonotope &Z, double dt,
                      int eta) {
    return Z.linear_map(priv_series(lin, taylor, dt, eta))
        .plus(Z.linear_map(priv_remainder(lin, dt, eta)));
}

} // namespace


// ===========================================  MAIN  =========================================== //

/// The integral of e^{As} over the step as an interval matrix (CORA's e_dttInt of
/// errorSolution_adaptive): the series sum_{i<=eta} A^i dt^(i+1)/(i+1)! and the remainder dt [-W,
/// W], W = |expm(|A| dt) - sum_{i<=eta} |A|^i dt^i / i!|.
Interval priv_adaptive_integral(const LinearSys &lin, TaylorLinSys &taylor, double dt, int eta) {
    const Tensor &A = lin.A();
    const Tensor Aabs = A.abs();
    Tensor M = A.eye_like(), Asum = A.eye_like() * dt, power = A.eye_like();
    for (int i = 1; i <= eta; ++i) {
        power = power.matmul(Aabs);
        M = M + power * taylor.dtoverfac(dt, i);
        Asum = Asum + taylor.Apower(i) * taylor.dtoverfac(dt, i + 1);
    }
    const Tensor W = ((Aabs * dt).expm() - M).abs() * dt;
    return {Asum - W, Asum + W};
}


std::pair<PolyZonotope, PolyZonotope> priv_poly_window(const LinearSys &lin, TaylorLinSys &taylor,
                                                       double dt, int eta, const Zonotope &V,
                                                       const PolyZonotope &X0, double zeta_z) {
    const int64_t n = X0.dim();
    // the input: the center u held through the segment [0, u], and the rest, centered
    const Tensor &u = V.center();
    const Zonotope segment(u * 0.5, u * 0.5);
    const Zonotope PuR = aux_solution(lin, taylor, segment, dt, eta).reduce_adaptive(zeta_z);
    const Zonotope PU = aux_solution(lin, taylor, V - u, dt, eta).reduce_adaptive(zeta_z);
    // e^{A dt} X0 and the curvature F X0
    const Tensor e_dt = taylor.e_adt(dt);
    const PolyZonotope FX = X0.linear_map(lin.correction_matrix_state(dt, eta));
    const PolyZonotope hom =
        priv_reduce_adaptive_poly(X0.enclose(X0.linear_map(e_dt)).plus(FX), zeta_z);
    const PolyZonotope window = hom.plus(PuR).plus(PU);
    // the differences: from the origin (with the factors of X0) to (e^{A dt} - I) X0
    const Tensor zero = X0.center().zeros_like();
    const PolyZonotope origin(zero, zero.zeros_like({n, 0}), zero.zeros_like({n, 0}), {}, X0.ids());
    const PolyZonotope diff_hom = priv_reduce_adaptive_poly(
        origin.enclose(X0.linear_map(e_dt - e_dt.eye_like())).plus(FX), zeta_z);
    const PolyZonotope diff = diff_hom.plus(PuR + PU);
    return {window, diff};
}

PolyZonotope priv_poly_point(const LinearSys &lin, TaylorLinSys &taylor, double dt, int eta,
                             const Zonotope &V, const PolyZonotope &X0, const PolyZonotope &Vs,
                             double zeta_z) {
    const int64_t n = X0.dim();
    // the input: V with the time-varying (independent) part of Vs, its center held
    const Tensor zero = X0.center().zeros_like();
    const Zonotope Vt = V + Zonotope(zero, Vs.independent_generators());
    const Tensor &u = Vt.center();
    const Zonotope Pu = aux_solution(lin, taylor, Zonotope(u, zero.zeros_like({n, 0})), dt, eta);
    const Zonotope PU = aux_solution(lin, taylor, Vt - u, dt, eta).reduce_adaptive(zeta_z);
    // e^{A dt} X0 plus the integral of the dependent part of Vs, the dependency on X0 kept
    const PolyZonotope dep(Vs.center(), Vs.generators(), zero.zeros_like({n, 0}), Vs.exponents(),
                           Vs.ids());
    const PolyZonotope Ps = dep.linear_map(priv_adaptive_integral(lin, taylor, dt, eta));
    // the homogeneous solution is reduced before the inputs join (CORA's initReach_adaptive)
    const PolyZonotope hom = priv_reduce_adaptive_poly(X0.linear_map(taylor.e_adt(dt)), zeta_z);
    return hom.exact_plus(Ps).plus(Pu).plus(PU);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
