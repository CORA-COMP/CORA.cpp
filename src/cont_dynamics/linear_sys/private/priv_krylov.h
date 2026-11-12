// priv_krylov - the Krylov subspaces of linear_sys.reach with alg::Krylov
//
// A subspace of the vector v under A is the Arnoldi basis V and the Hessenberg matrix H with
// e^{A t} v ~ |v| V e^{H t} e_1; its order grows by krylov_step until the error bound of CORALean's
// Krylov.errorBoundFloat at t_final is at most krylov_error (it holds for every A, where CORA's
// Jawecki test is valid for a nonexpansive A only). Arnoldi runs on the host, through the products
// A x only.
//
// Syntax:   KrylovSubspace S = priv_krylov_subspace(A, v, order, error, step, t_final, like);
// See also: priv_reach_krylov, LinearSys::reach
//
// CORA: contDynamics/@linearSys/private/priv_subspace_Krylov_individual_Jawecki.m
// CORA: contDynamics/@linearSys/private/priv_krylov_R_uTrans.m
// CORA: contDynamics/@linearSys/private/priv_inputSolution_Krylov.m
// CORA: contDynamics/@linearSys/private/priv_reach_krylov.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/sparse_matrix.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The products with the matrix --------------------------------------------------------------------

/// The products y = M x of a square matrix with host vectors, summed in CORALean's order.
struct HostMatVec {
    int64_t n = 0;
    std::shared_ptr<const SparseMatrix> M;
    mutable std::optional<double> quad_;

    void apply(const double *x, double *y) const { M->matvec_down(x, y); }
    /// CORALean's Krylov.quadBoundUp of the matrix, formed on the first call.
    double quad_bound() const {
        if (!quad_) quad_ = M->quad_bound_up();
        return *quad_;
    }
};

/// The products of the state matrix of the system: the sparse A, or the dense one on the host.
HostMatVec priv_state_matvec(const LinearSys &sys);

/// The products of the integrator system [A u; 0 0] of dimension n + 1 (CORA's A_int).
HostMatVec priv_integrator_matvec(const HostMatVec &A, const std::vector<double> &u);

/// The squarings of the exponential in the bound of the order test (CORALean's default).
inline constexpr int kKrylovSquarings = 10;

/// The most passes of Arnoldi that one restart window of a trajectory may take.
inline constexpr int64_t kKrylovMaxOrder = 400;

// The float Arnoldi iteration and its error bound -------------------------------------------------

/// The Arnoldi data of the start vector c: v(1), ..., v(k+1) in V, column j of H, the residuals.
struct ArnoldiFloat {
    std::vector<double> c;
    double nu = 0;
    std::vector<std::vector<double>> V, H;
    std::vector<double> residual;
    int64_t passes() const { return int64_t(H.size()); }
};

/// The state before the first pass: nu = |c| and v(1) = c / nu (Krylov.step_arnoldiStart).
ArnoldiFloat priv_arnoldi_start(const std::vector<double> &c);

/// Runs the passes until `passes` are made (Krylov.step_arnoldiPass), the residual of each column
/// on the way; a pass at a zero vector is exact and skips the product.
void priv_arnoldi_extend(ArnoldiFloat &state, const HostMatVec &A, int64_t passes,
                         double breakdown);

/// The Euclidean norm from above: the machine's root of the sum of squares (Float.normUp).
double priv_krylov_norm_up(const std::vector<double> &x);

/// Krylov.quadBoundUp of the leading k x k block of the Hessenberg matrix of the state.
double priv_hessenberg_quad_bound(const ArnoldiFloat &state, int64_t k);

/// Krylov.errorBoundFloat of the first k passes as a function of the time: mu, the residuals' share
/// and the start error are formed once, the decay product per time.
class KrylovBoundFn {
  public:
    KrylovBoundFn(const ArnoldiFloat &state, double mu_A, int64_t k);
    /// The growth rate max(0, mu_A, mu_T).
    double mu() const { return mu_; }
    /// The bound of ||e^{tA}c - nu V_k e^{tT_k} e_1|| over [0, t].
    double at(double t) const;
    /// Krylov.orderOkFloat: 1 <= k, t mu within the range of the exponential, the bound <= tol.
    bool ok(double t, double tol, int q) const;

  private:
    const ArnoldiFloat &s_;
    int64_t k_;
    double mu_ = 0, wn_ = 0, share_ = 0, rho_ = 0;
};

/// Krylov.errorBoundFloat: the bound over [0, t] for the first k passes, q squarings of exp.
double priv_krylov_error_bound(const ArnoldiFloat &state, double mu_A, int64_t k, double t, int q);

/// Krylov.orderOkFloat: 1 <= k, t mu within the range of the exponential, the bound <= tol.
bool priv_krylov_order_ok(const ArnoldiFloat &state, double mu_A, int64_t k, double t, double tol,
                          int q);

// The order of a subspace -------------------------------------------------------------------------

/// What the order search of Krylov.subspaceFloat found: the Arnoldi data, the bound, whether ok.
struct KrylovSearch {
    ArnoldiFloat state;
    int64_t order = 0;
    bool ok = false;
    double bound = 0;
};

/// The order search of Krylov.subspaceFloat on c: from k0, up by `step` (to n) while the bound at
/// tf exceeds tol, at most `fuel` times; q squarings of exp, `breakdown` the happy-breakdown limit.
KrylovSearch priv_krylov_search(const HostMatVec &A, const std::vector<double> &c, int64_t k0,
                                int64_t step, double tf, double tol, int64_t fuel, int q,
                                double breakdown);

/// A Krylov subspace of v: V (n, k), H (k, k) like `like`, |v|, the order and the bound of its
/// error over [0, t_final]; `ok` is whether the bound meets the tolerance (no certificate if not).
struct KrylovSubspace {
    Tensor V, H;
    double norm = 0;
    int64_t order = 0;
    bool ok = true;
    double bound = 0;
};

/// The subspace of v (Krylov.subspaceFloat): the order starts at `order`, grows by krylov_step
/// until the bound at t_final is at most krylov_error; `order` returns the one found.
KrylovSubspace priv_krylov_subspace(const HostMatVec &A, const std::vector<double> &v,
                                    int64_t &order, double krylov_error, int krylov_step,
                                    double t_final, const Tensor &like);

/// Fails with not_converged if the subspace does not meet the tolerance.
void priv_krylov_require_ok(const KrylovSubspace &S, double krylov_error, const char *caller);

// Restart windows ---------------------------------------------------------------------------------

/// One restart window of e^{tA} v: its subspace, the steps it is valid for, and the error bound
/// of |s| V e^{tT} e_1 against the true trajectory at every time of it (earlier windows included).
struct KrylovWindow {
    Tensor V, H;
    double norm = 0;
    int64_t order = 0, steps = 0;
    double error = 0;
};

/// The windows of e^{tA} v over `steps` steps of `time_step`, one after the other (see
/// priv_krylov_windows.cpp): each restarts Arnoldi from the approximate state at its start.
class KrylovWindows {
  public:
    KrylovWindows(HostMatVec A, std::vector<double> v, double time_step, int64_t steps, double tol,
                  int krylov_step, int64_t first_order, const Tensor &like);
    bool done() const { return done_ >= steps_; }
    /// The next window; fails with not_converged if not even one step meets the tolerance.
    KrylovWindow next();

  private:
    HostMatVec A_;
    std::vector<double> start_;
    double dt_, tol_, mu_A_, error_ = 0;
    int64_t steps_, done_ = 0, first_order_;
    int step_;
    Tensor like_;
};

// The inputs and the sets -------------------------------------------------------------------------

/// The constant input u in its subspace (CORA's priv_krylov_R_uTrans): the state rows of V mapped
/// by C, H, the input curvature C V G e_1 and the bound of the error of the subspace.
struct KrylovConstantInput {
    Tensor proj, H;
    Zonotope correction;
    double bound = 0;
};
KrylovConstantInput priv_krylov_constant_input(const HostMatVec &A, const std::vector<double> &u,
                                               const Tensor &C, const alg::Krylov &alg,
                                               double t_final);

/// The input solution of the first step of V = B U (CORA's priv_inputSolution_Krylov): the Taylor
/// terms of each generator in its subspace plus the box of their remainders and subspace errors.
Zonotope priv_krylov_input_solution(const HostMatVec &A, const Zonotope &V, const alg::Krylov &alg,
                                    double t_final);

/// The sets of reach with alg::Krylov from R0 with the input set (none: no input); with
/// on_time_interval, the time-interval sets go to it and none are stored.
ReachSet priv_reach_krylov(const LinearSys &sys, const Zonotope &R0,
                           const std::optional<Zonotope> &input_set, const alg::Krylov &alg,
                           double t_final,
                           const std::function<void(const Zonotope &)> &on_time_interval = nullptr);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
