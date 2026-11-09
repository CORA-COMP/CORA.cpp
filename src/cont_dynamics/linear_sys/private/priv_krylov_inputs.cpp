// priv_krylov_inputs - the constant input and the input solution of reach with alg::Krylov
//
// Both run Arnoldi on the integrator system [A u; 0 0] from e_{n+1}, whose state rows of
// e^{[A u; 0 0] t} e_{n+1} are the solution of x' = A x + u from 0. The constant input keeps the
// subspace for every step and the curvature C V G(H) e_1 (G the input correction matrix). The input
// solution sums, per generator u of V, the Taylor terms V H^i e_1 dt^i / i! (i = 1..eta+1) as
// generators and the supremum of V E(H) e_1 (E the remainder) as a box, as CORA does: entries
// below eps of the box are 0, and krylov_error times the number of generators is added to it
// when krylov_error > 2 eps.
//
// Syntax:   KrylovConstantInput u = priv_krylov_constant_input(A, u_trans, C, alg, t_final);
//           Zonotope RV = priv_krylov_input_solution(A, V, alg, t_final);
// Inputs:   A - the products with the state matrix
//           u_trans - the constant input B u (host, n)
//           C - the output matrix (p, n)
//           V - the centered input set B U of the state space
//           alg - the options
//           t_final - the time of the error bound
// Outputs:  u - the subspace, mapped by C, and the input curvature
//           RV - the input solution of the first step (n)
// See also: priv_reach_krylov, priv_krylov_subspace
//
// CORA: contDynamics/@linearSys/private/priv_krylov_R_uTrans.m
// CORA: contDynamics/@linearSys/private/priv_inputSolution_Krylov.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Helpers -----------------------------------------------------------------------------------------

/// The first column of an interval matrix.
Interval aux_first_column(const Interval &M) {
    return {M.inf().slice(-1, 0, 1), M.sup().slice(-1, 0, 1)};
}

/// e_{n+1}, the equivalent initial state of the integrator system.
std::vector<double> aux_last_unit(int64_t n) {
    std::vector<double> e(size_t(n) + 1, 0.0);
    e[size_t(n)] = 1.0;
    return e;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The constant input ------------------------------------------------------------------------------

KrylovConstantInput priv_krylov_constant_input(const HostMatVec &A, const std::vector<double> &u,
                                               const Tensor &C, const alg::Krylov &alg,
                                               double t_final) {
    const int64_t n = A.n;
    int64_t order = 1;
    const KrylovSubspace S =
        priv_krylov_subspace(priv_integrator_matvec(A, u), aux_last_unit(n), order,
                             alg.krylov_error, alg.krylov_step, t_final, C);
    const Tensor V_state = S.V.slice(0, 0, n);
    // the curvature of the constant input, |e_{n+1}| = 1
    const Interval G = LinearSys(S.H).correction_matrix_input(alg.time_step, alg.taylor_terms);
    const Zonotope correction = aux_first_column(G).linear_map(V_state).zonotope().linear_map(C);
    return {C.matmul(V_state), S.H, correction};
}

// The input solution ------------------------------------------------------------------------------

Zonotope priv_krylov_input_solution(const HostMatVec &A, const Zonotope &V, const alg::Krylov &alg,
                                    double t_final) {
    const int64_t n = A.n, m = V.generators().shape().back();
    const double dt = alg.time_step;
    const Tensor &like = V.center();
    const std::vector<double> G = V.generators().data();
    std::vector<Tensor> terms;
    Tensor error = like.zeros_like({n, 1});
    int64_t order = 1;
    // each nonzero generator u of V in the subspace of [A u; 0 0]
    for (int64_t g = 0; g < m; ++g) {
        std::vector<double> u(static_cast<size_t>(n));
        bool zero = true;
        for (int64_t r = 0; r < n; ++r) {
            u[size_t(r)] = G[size_t(r * m + g)];
            zero = zero && within_tol(u[size_t(r)], 0.0);
        }
        if (zero) continue;
        const HostMatVec integrator = priv_integrator_matvec(A, u);
        const KrylovSubspace S = priv_krylov_subspace(integrator, aux_last_unit(n), order,
                                                      alg.krylov_error, alg.krylov_step, t_final,
                                                      like);
        // the Taylor terms V H^i e_1 dt^i / i!, i = 1..eta+1, of the state rows
        Tensor power = S.H;
        for (int i = 1; i <= alg.taylor_terms + 1; ++i) {
            if (i > 1) power = power.matmul(S.H);
            const Tensor term =
                S.V.matmul(power.slice(-1, 0, 1)) * std::pow(dt, i) / priv_factorial(i);
            terms.push_back(term.slice(0, 0, n));
        }
        // the remainder of the series, its supremum over the state rows
        const Interval E = LinearSys(S.H).expm_remainder(dt, alg.taylor_terms);
        error = error + aux_first_column(E).linear_map(S.V).supremum().slice(0, 0, n);
    }
    // entries below eps are 0, as CORA rounds them
    std::vector<double> radius = error.data();
    for (double &r : radius)
        if (std::abs(r) < machine_eps()) r = 0.0;
    const Tensor r = Tensor::like(like, radius, {n, 1});
    Interval box(r.negative(), r);
    if (alg.krylov_error > 2 * machine_eps()) {
        const Tensor red = like.full_like({n, 1}, alg.krylov_error * double(m));
        box = box + Interval(red.negative(), red);
    }
    const Tensor generators = terms.empty() ? like.zeros_like({n, 0}) : Tensor::concat(terms, -1);
    return Zonotope(like.zeros_like({n, 1}), generators) + box.zonotope();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
