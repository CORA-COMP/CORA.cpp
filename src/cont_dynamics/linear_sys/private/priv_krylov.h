// priv_krylov - the Krylov subspaces of linear_sys.reach with alg::Krylov
//
// A subspace of the vector v under A is the Arnoldi basis V and the Hessenberg matrix H with
// e^{A t} v ~ |v| V e^{H t} e_1; its order grows by krylov_step until Jawecki's error bound
// at t_final is at most krylov_error. Arnoldi runs on the host, through the products A x only.
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

#include <cstdint>
#include <functional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The product y = M x of a square matrix with host vectors of its dimension.
struct HostMatVec {
    int64_t n = 0;
    std::function<void(const double *x, double *y)> apply;
};

/// The products of the state matrix of the system: the sparse A, or the dense one on the host.
HostMatVec priv_state_matvec(const LinearSys &sys);

/// The products of the integrator system [A u; 0 0] of dimension n + 1 (CORA's A_int).
HostMatVec priv_integrator_matvec(const HostMatVec &A, const std::vector<double> &u);

/// A Krylov subspace of v: V (n, k), H (k, k) like `like`, and |v|; k = 0 for v = 0.
struct KrylovSubspace {
    Tensor V, H;
    double norm = 0;
    int64_t order = 0;
};

/// The subspace of v (CORA's priv_subspace_Krylov_individual_Jawecki): the order starts at
/// `order` - krylov_step and grows by krylov_step; `order` returns the one found.
KrylovSubspace priv_krylov_subspace(const HostMatVec &A, const std::vector<double> &v,
                                    int64_t &order, double krylov_error, int krylov_step,
                                    double t_final, const Tensor &like);

/// The constant input u in its subspace (CORA's priv_krylov_R_uTrans): of [A u; 0 0] from
/// e_{n+1}, the state rows of V mapped by C, H, and the input curvature C V G e_1.
struct KrylovConstantInput {
    Tensor proj, H;
    Zonotope correction;
};
KrylovConstantInput priv_krylov_constant_input(const HostMatVec &A, const std::vector<double> &u,
                                               const Tensor &C, const alg::Krylov &alg,
                                               double t_final);

/// The input solution of the first step of V = B U in the state space (CORA's
/// priv_inputSolution_Krylov): the Taylor terms of each generator in its subspace plus the box
/// of their remainders (and of krylov_error, if above 2 eps).
Zonotope priv_krylov_input_solution(const HostMatVec &A, const Zonotope &V, const alg::Krylov &alg,
                                    double t_final);

/// The sets of reach with alg::Krylov from R0 with the input set (none: no input).
ReachSet priv_reach_krylov(const LinearSys &sys, const Zonotope &R0,
                           const std::optional<Zonotope> &input_set, const alg::Krylov &alg,
                           double t_final);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
