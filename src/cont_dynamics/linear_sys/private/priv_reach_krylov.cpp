// priv_reach_krylov - the algorithm "krylov" of linear_sys.reach, as CORA's priv_reach_krylov
//
// Every vector of R0 (center c, generators g) and of the first input solution RV gets its own
// subspace (V, H) under A (priv_initReach_Krylov), whose order CORALean's error bound accepts
// (priv_krylov_subspace; the vectors of R0 in restart windows, KrylovWindows); only C V is kept.
// Step k of the homogeneous solution maps each vector v to |v| C V y_k, y_k = e^{H t_k} e_1, and
// adds the curvature |v| C V F(H) y_k (an interval at the start of the step: the standard reach
// of y' = H y; CORA applies it at the end, CORALean's Krylov.alg2Corrected); the input solution
// adds the box of |v| C V e^{H t} e_1 over the vectors of RV, the constant input its subspace at
// t. With U_0 the input boxes so far and E the box of the subspace errors (the bound of each
// vector times the length of the rows of C): time_point = reduce(H_tp + R_trans + U_0 + E),
// time_interval = reduce(enclose(H_tp,prev, H_tp + R_trans) + U_0 + curvature + E + correction).
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

/// The center and the generators of the zonotope as host vectors; none for a zero vector.
std::vector<std::optional<std::vector<double>>> aux_vectors(const Zonotope &Z) {
    const int64_t n = Z.center().shape()[0], m = Z.generators().shape().back();
    const std::vector<double> c = Z.center().data(), G = Z.generators().data();
    std::vector<std::optional<std::vector<double>>> out;
    // the center (j = -1), then the generators
    for (int64_t j = -1; j < m; ++j) {
        std::vector<double> v(static_cast<size_t>(n));
        double sum = 0.0;
        for (int64_t r = 0; r < n; ++r) {
            v[size_t(r)] = j < 0 ? c[size_t(r)] : G[size_t(r * m + j)];
            sum += v[size_t(r)] * v[size_t(r)];
        }
        if (sum == 0)
            out.emplace_back();
        else
            out.emplace_back(std::move(v));
    }
    return out;
}

/// The vectors of a zonotope in their subspaces, mapped by C: |v| C V and H, or none for v = 0.
struct Projected {
    Tensor proj, H;
};
std::vector<std::optional<Projected>> aux_subspaces(const HostMatVec &A, const Zonotope &Z,
                                                    const Tensor &C, const alg::Krylov &alg,
                                                    double t_final, double &bound) {
    std::vector<std::optional<Projected>> out;
    int64_t order = 15;
    // each nonzero vector in the first subspace that CORALean's error bound accepts
    for (const auto &v : aux_vectors(Z)) {
        if (!v) {
            out.emplace_back();
            continue;
        }
        const KrylovSubspace S =
            priv_krylov_subspace(A, *v, order, alg.krylov_error, alg.krylov_step, t_final, C);
        priv_krylov_require_ok(S, alg.krylov_error, "reach");
        bound += S.bound;
        out.emplace_back(Projected{C.matmul(S.V) * S.norm, S.H});
    }
    return out;
}

/// The lengths of the rows of C: the error of a state of 2-norm e is at most e times it per output.
Tensor aux_row_norms(const Tensor &C) {
    // in blocks of rows, so that a large C is not squared as a whole
    std::vector<Tensor> blocks;
    const int64_t p = C.shape()[0];
    for (int64_t first = 0; first < p; first += 64)
        blocks.push_back(C.slice(0, first, std::min<int64_t>(first + 64, p)).square().sum(1, true));
    return Tensor::concat(blocks, 0).sqrt();
}

/// The set [c G] of the columns `points` with the box of the radius (p, 1) of an error added; the
/// box is left out where the radius is zero.
Zonotope aux_with_error(const Tensor &points, const Tensor &radius) {
    const Tensor c = points.slice(-1, 0, 1), G = points.slice(-1, 1, points.shape().back());
    if (!(radius.max().item() > 0)) return {c, G};
    return {c, Tensor::concat({G, radius.diag()}, -1)};
}

// The homogeneous solution ------------------------------------------------------------------------

/// The vectors of R0 step by step, each in its restart windows (KrylovWindows): y_k = e^{H dt}
/// y_{k-1} and the curvature F(H) y_k of each, from the start of the step, as the standard reach of
/// y' = H y from e_1; the error bound of each vector is that of its window.
class Homogeneous {
  public:
    Homogeneous(const HostMatVec &A, const Zonotope &Z, const Tensor &C, const alg::Krylov &alg,
                double t_final)
        : C_(C), alg_(alg) {
        const int64_t steps = priv_num_steps(t_final, alg.time_step);
        // a window sequence per nonzero vector
        for (auto &v : aux_vectors(Z)) {
            if (!v) {
                vectors_.emplace_back();
                continue;
            }
            vectors_.emplace_back(
                Vector{KrylovWindows(A, std::move(*v), alg.time_step, steps, alg.krylov_error,
                                     alg.krylov_step, 15, Z.center()),
                       {},
                       {},
                       {},
                       std::nullopt,
                       0,
                       0});
        }
    }

    /// The next time point [c, G] (p, 1 + m), the curvature over the step and the bound of the
    /// error of the vectors: the point of the start of the step gets F(H) y_k, a generator's
    /// trajectory is scaled by [-1, 1].
    std::tuple<Tensor, Interval, double> next(int64_t p, const Tensor &like) {
        std::vector<Tensor> columns;
        Interval curvature(like.zeros_like({p, 1}), like.zeros_like({p, 1}));
        double error = 0;
        for (size_t j = 0; j < vectors_.size(); ++j) {
            if (!vectors_[j]) { // a zero vector stays zero
                columns.push_back(like.zeros_like({p, 1}));
                continue;
            }
            Vector &s = *vectors_[j];
            if (s.left == 0) aux_restart(s);
            // the curvature acts on the state at the start of the step
            const int64_t k = s.y.shape()[0];
            const Zonotope point(s.y, s.y.zeros_like({k, 0}));
            const Interval small = point.linear_map(*s.F).interval();
            Interval lifted = small.linear_map(s.proj);
            if (j > 0) { // a generator: alpha in [-1, 1] times the interval
                const Tensor radius = lifted.inf().abs().maximum(lifted.sup().abs());
                lifted = Interval(radius.negative(), radius);
            }
            curvature = curvature + lifted;
            s.y = s.e_hdt.matmul(s.y);
            --s.left;
            error += s.error;
            columns.push_back(s.proj.matmul(s.y));
        }
        return {Tensor::concat(columns, -1), curvature, error};
    }

  private:
    /// A vector with its windows and the current one: C V |s|, e^{H dt}, F(H), y, steps left.
    struct Vector {
        KrylovWindows windows;
        Tensor proj, e_hdt, y;
        std::optional<Interval> F;
        double error;
        int64_t left;
    };

    /// The next window of the vector.
    void aux_restart(Vector &s) const {
        const KrylovWindow w = s.windows.next();
        s.proj = C_.matmul(w.V) * w.norm;
        s.e_hdt = (w.H * alg_.time_step).expm();
        s.F = LinearSys(w.H).correction_matrix_state(alg_.time_step, alg_.taylor_terms);
        s.y = w.H.eye_like(w.order).slice(-1, 0, 1);
        s.error = w.error;
        s.left = w.steps;
    }

    Tensor C_;
    alg::Krylov alg_;
    std::vector<std::optional<Vector>> vectors_;
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
                           double t_final,
                           const std::function<void(const Zonotope &)> &on_time_interval) {
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

    // the subspaces of the constant input and the input solution, and the length of the rows of C
    const Tensor rows = aux_row_norms(C);
    const KrylovConstantInput constant = priv_krylov_constant_input(A, u_trans, C, alg, t_final);
    double bound_rv = 0;
    const Zonotope RV = priv_krylov_input_solution(A, V, alg, t_final);
    const auto input_spaces = aux_subspaces(A, RV, C, alg, t_final, bound_rv);
    Homogeneous homogeneous(A, R0, C, alg, t_final);
    const Projected u_space{constant.proj, constant.H};
    const Tensor error_rv = rows * bound_rv, error_trans = rows * constant.bound;

    ReachSet R;
    Zonotope previous = R0.linear_map(C);
    if (!on_time_interval) R.time_point.push_back(previous);
    std::optional<Interval> U0;
    const int steps = priv_num_steps(t_final, alg.time_step);
    for (int k = 1; k <= steps; ++k) {
        const double t = k * alg.time_step;
        // the homogeneous solution, the input solution of this step and the constant input
        const auto [points, curvature, error] = homogeneous.next(p, like);
        const Tensor error_r0 = rows * error;
        const Interval error_box(error_r0.negative(), error_r0);
        const Zonotope Htp(points.slice(-1, 0, 1), points.slice(-1, 1, points.shape().back()));
        const Zonotope RV_proj =
            U0 ? aux_with_error(aux_at_time(input_spaces, t, p, like), error_rv) : RV.linear_map(C);
        const Zonotope R_trans = aux_with_error(aux_at_time({u_space}, t, p, like), error_trans);
        U0 = U0 ? *U0 + RV_proj.interval() : RV_proj.interval();
        const Zonotope U0_set = U0->zonotope();
        // the sets of the step, the error of the homogeneous part added to both of its ends
        const Zonotope Rhom_tp = Htp + R_trans;
        const Zonotope R_ti = previous.enclose(Rhom_tp) + U0_set +
                              (curvature + error_box).zonotope() + constant.correction;
        if (on_time_interval) {
            on_time_interval(R_ti.reduce(alg.zonotope_order));
        } else {
            const Zonotope R_tp = Rhom_tp + U0_set + error_box.zonotope();
            R.time_interval.push_back(R_ti.reduce(alg.zonotope_order));
            R.time_point.push_back(R_tp.reduce(alg.zonotope_order));
        }
        previous = Rhom_tp;
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
