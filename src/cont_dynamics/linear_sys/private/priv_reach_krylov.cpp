// priv_reach_krylov - the algorithm "krylov" of linear_sys.reach, as CORA's priv_reach_krylov
//
// Every vector of R0 (center c, generators g) and of the first input solution RV gets its own
// subspace (V, H) under A (priv_initReach_Krylov); only C V is kept, so no set of the state space
// is formed. Step k (t = k dt) of the homogeneous solution maps each vector v to |v| C V E_k e_1,
// E_k = e^{H dt} E_{k-1}, and adds the curvature |v| C V E_k F(H) e_1 (an interval); the input
// solution adds the box of |v| C V e^{H t} e_1 over the vectors of RV, the constant input its
// subspace at t. Then, with U_0 the sum of the input boxes so far,
//    time_point = reduce(H_tp + R_trans + U_0),
//    time_interval = reduce(enclose(H_tp,prev, H_tp + R_trans) + U_0 + curvature + correction).
// krylov_error > 2 eps adds error I to each mapped set (error = krylov_error times its vectors).
// A zero generator of R0 stays a zero column, where CORA fails on it.
//
// Syntax:   ReachSet R = priv_reach_krylov(sys, R0, U, alg, t_final);
// Inputs:   sys - the system
//           R0 - the initial set
//           U - the input set (in the input space of B) or none
//           alg - the options
//           t_final - the final time
// Outputs:  R - the output sets of each step
// See also: priv_krylov_subspace, priv_krylov_inputs, LinearSys::reach
//
// CORA: contDynamics/@linearSys/private/priv_reach_krylov.m
// CORA: contDynamics/@linearSys/private/priv_initReach_Krylov.m
// CORA: contDynamics/@linearSys/private/priv_subspace_Krylov_jaweckiBound.m
// CORA: contDynamics/@linearSys/private/priv_exponential_Krylov_projected_linSysInput.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cont_dynamics/linear_sys/private/priv_krylov.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The subspaces -----------------------------------------------------------------------------------

/// The vectors of a zonotope in their subspaces, mapped by C: |v| C V and H, or none for v = 0;
/// the center first (CORA's priv_subspace_Krylov_jaweckiBound, the order passed on from 15).
struct Projected {
    Tensor proj, H;
};
std::vector<std::optional<Projected>> aux_subspaces(const HostMatVec &A, const Zonotope &Z,
                                                    const Tensor &C, const alg::Krylov &alg,
                                                    double t_final) {
    const int64_t n = A.n, m = Z.generators().shape().back();
    const std::vector<double> c = Z.center().data(), G = Z.generators().data();
    std::vector<std::optional<Projected>> out;
    int64_t order = 15;
    // the center (j = -1), then the generators; a zero vector has no subspace
    for (int64_t j = -1; j < m; ++j) {
        std::vector<double> v(static_cast<size_t>(n));
        double sum = 0.0;
        for (int64_t r = 0; r < n; ++r) {
            v[size_t(r)] = j < 0 ? c[size_t(r)] : G[size_t(r * m + j)];
            sum += v[size_t(r)] * v[size_t(r)];
        }
        if (sum == 0) {
            out.emplace_back();
            continue;
        }
        const KrylovSubspace S =
            priv_krylov_subspace(A, v, order, alg.krylov_error, alg.krylov_step, t_final, C);
        out.emplace_back(Projected{C.matmul(S.V) * S.norm, S.H});
    }
    return out;
}

/// The error box error I (p) that CORA adds to a mapped set, error = krylov_error times the
/// number of its vectors; none at krylov_error <= 2 eps.
Zonotope aux_with_error(const Tensor &points, const alg::Krylov &alg, int64_t vectors) {
    const int64_t p = points.shape()[0];
    const Tensor c = points.slice(-1, 0, 1), G = points.slice(-1, 1, points.shape().back());
    if (!(alg.krylov_error > 2 * machine_eps())) return {c, G};
    const double error = alg.krylov_error * double(vectors);
    return {c, Tensor::concat({G, c.eye_like(p) * error}, -1)};
}

// The homogeneous solution ------------------------------------------------------------------------

/// The vectors of R0 step by step: E_k = e^{H dt} E_{k-1} and the curvature F(H) e_1 of each.
class Homogeneous {
  public:
    Homogeneous(std::vector<std::optional<Projected>> spaces, const alg::Krylov &alg)
        : spaces_(std::move(spaces)) {
        for (const auto &s : spaces_) {
            if (!s) {
                steps_.emplace_back();
                continue;
            }
            const Interval F =
                LinearSys(s->H).correction_matrix_state(alg.time_step, alg.taylor_terms);
            steps_.emplace_back(Step{(s->H * alg.time_step).expm(), s->H.eye_like(),
                                     Interval(F.inf().slice(-1, 0, 1), F.sup().slice(-1, 0, 1))});
        }
    }

    /// The next time point [c, G] (p, 1 + m) and the curvature over the step (CORA's
    /// aux_propagate_HomSol).
    std::pair<Tensor, Interval> next(int64_t p, const Tensor &like) {
        std::vector<Tensor> columns;
        Interval curvature(like.zeros_like({p, 1}), like.zeros_like({p, 1}));
        for (size_t j = 0; j < spaces_.size(); ++j) {
            if (!spaces_[j]) { // a zero vector stays zero
                columns.push_back(like.zeros_like({p, 1}));
                continue;
            }
            Step &s = *steps_[j];
            s.E = s.e_hdt.matmul(s.E);
            const Tensor M = spaces_[j]->proj.matmul(s.E);
            columns.push_back(M.slice(-1, 0, 1));
            curvature = curvature + s.F1.linear_map(M);
        }
        return {Tensor::concat(columns, -1), curvature};
    }

  private:
    struct Step {
        Tensor e_hdt, E;
        Interval F1;
    };
    std::vector<std::optional<Projected>> spaces_;
    std::vector<std::optional<Step>> steps_;
};

// The input solution ------------------------------------------------------------------------------

/// |v| C V e^{H t} e_1 of each vector (zero for none), as columns (CORA's
/// priv_exponential_Krylov_projected_linSysInput).
Tensor aux_at_time(const std::vector<std::optional<Projected>> &spaces, double t, int64_t p,
                   const Tensor &like) {
    std::vector<Tensor> columns;
    columns.reserve(spaces.size());
    for (const auto &s : spaces)
        columns.push_back(s ? s->proj.matmul((s->H * t).expm().slice(-1, 0, 1))
                            : like.zeros_like({p, 1}));
    return Tensor::concat(columns, -1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachSet priv_reach_krylov(const LinearSys &sys, const Zonotope &R0,
                           const std::optional<Zonotope> &input_set, const alg::Krylov &alg,
                           double t_final) {
    const int64_t n = sys.dim();
    const Tensor &like = R0.center();
    const Tensor C = sys.C() ? *sys.C() : like.eye_like(n);
    const int64_t p = C.shape()[0];
    const HostMatVec A = priv_state_matvec(sys);
    // the input set split into its center u_trans and the rest, both through B
    Zonotope V(like.zeros_like({n, 1}), like.zeros_like({n, 0}));
    std::vector<double> u_trans(size_t(n), 0.0);
    if (input_set) {
        const Tensor &B = *sys.B();
        V = Zonotope(input_set->center().zeros_like(), input_set->generators()).linear_map(B);
        u_trans = B.matmul(input_set->center()).data();
    }

    // the subspaces of the constant input, the input solution and the initial set
    const KrylovConstantInput constant = priv_krylov_constant_input(A, u_trans, C, alg, t_final);
    const Zonotope RV = priv_krylov_input_solution(A, V, alg, t_final);
    const auto input_spaces = aux_subspaces(A, RV, C, alg, t_final);
    const int64_t rv_vectors = RV.generators().shape().back() + 1;
    Homogeneous homogeneous(aux_subspaces(A, R0, C, alg, t_final), alg);
    const int64_t r0_vectors = R0.generators().shape().back() + 1;
    const Projected u_space{constant.proj, constant.H};

    ReachSet R;
    R.time_point.push_back(R0.linear_map(C));
    Zonotope previous = R.time_point.back();
    std::optional<Interval> U0;
    const int steps = priv_num_steps(t_final, alg.time_step);
    for (int k = 1; k <= steps; ++k) {
        const double t = k * alg.time_step;
        // the homogeneous solution, the input solution of this step and the constant input
        const auto [points, curvature] = homogeneous.next(p, like);
        const Zonotope Htp = aux_with_error(points, alg, r0_vectors);
        const Zonotope RV_proj =
            U0 ? aux_with_error(aux_at_time(input_spaces, t, p, like), alg, rv_vectors)
               : RV.linear_map(C);
        const Zonotope R_trans = aux_with_error(aux_at_time({u_space}, t, p, like), alg, 1);
        U0 = U0 ? *U0 + RV_proj.interval() : RV_proj.interval();
        const Zonotope U0_set = U0->zonotope();
        // the sets of the step
        const Zonotope Rhom_tp = Htp + R_trans;
        const Zonotope R_tp = Rhom_tp + U0_set;
        const Zonotope R_ti =
            previous.enclose(Rhom_tp) + U0_set + curvature.zonotope() + constant.correction;
        R.time_interval.push_back(R_ti.reduce(alg.zonotope_order));
        R.time_point.push_back(R_tp.reduce(alg.zonotope_order));
        previous = Rhom_tp;
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
