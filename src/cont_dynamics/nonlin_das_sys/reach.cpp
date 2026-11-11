// reach - the reachable sets of a differential-algebraic system, as CORA's reach and initReach
//
// The first step makes the algebraic variables consistent with the center of the initial set and
// linearizes from there with the error 0; every later step starts from the set at the end of the
// one before, its error bounds and the algebraic variables over it, and reduces both new sets to
// zonotope_order (post). The constant input of step k is column k of u_trans.
//
// Syntax:   DaeReachSet R = sys.reach(problem, alg::DaeLinearization{.time_step = 0.005});
//           DaeStep s = sys.init_reach(problem, alg);   s = sys.post(s, problem, u_trans, alg);
// Inputs:   problem - initial_set, input_set, u_trans, y0_guess, t_start, t_final, decomposition
//           alg - alg::DaeLinearization (time_step, taylor_terms, zonotope_order, error_order, ...)
// Outputs:  R - time_interval[k] encloses the states over [t_start + k dt, t_start + (k+1) dt]
// See also: linearize, simulate, NonlinearSys::reach
//
// CORA: contDynamics/@contDynamics/reach.m
// CORA: contDynamics/@nonlinDASys/initReach.m
// CORA: contDynamics/@nonlinDASys/post.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cont_dynamics/nonlin_das_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The checks --------------------------------------------------------------------------------------

/// Whether M is a matrix of the shape (rows, columns).
bool aux_has_shape(const Tensor &M, int64_t rows, int64_t columns) {
    return M.defined() && M.shape() == std::vector<int64_t>({rows, columns});
}

/// The checks of a decomposition against the whole system: the shapes of the selection matrices,
/// and that every state and every constraint is owned by some subsystem (an unowned row would have
/// no remainder).
void aux_validate_decomposition(const NonlinDASys &sys, const DaeDecomposition &dec) {
    detail::Validation entry;
    const int64_t n = sys.dim(), p = sys.constraint_dim(), m = sys.input_dim();
    // the subsystems and the voltages
    CORA_CHECK(!dec.subsystems.empty() && dec.v_gen.defined() && dec.v_gen.shape().size() == 2 &&
                   dec.v_gen.shape()[1] == 1,
               Error::wrong_value, [] {
                   return std::string("reach: a decomposition needs subsystems and the voltages "
                                      "v_gen as a column (nv, 1)");
               });
    const int64_t nv = dec.v_gen.shape()[0];
    std::vector<double> owned_x(std::size_t(n), 0.0), owned_y(std::size_t(p), 0.0);
    for (const DaeSubsystem &s : dec.subsystems) {
        const int64_t ni = s.sys.dim(), pi = s.sys.constraint_dim(), mi = s.sys.input_dim();
        CORA_CHECK(aux_has_shape(s.X, ni, n) && aux_has_shape(s.Y, pi, p) &&
                       aux_has_shape(s.Uv, mi, nv) && aux_has_shape(s.Uy, mi, p) &&
                       aux_has_shape(s.Uu, mi, m),
                   Error::dimension_mismatch, [&] {
                       return std::string("reach: the selection matrices of a subsystem must be "
                                          "X (n_i, n), Y (p_i, p), Uv (m_i, nv), Uy (m_i, p) and "
                                          "Uu (m_i, m) for its n_i, p_i and m_i");
                   });
        // every column of a selection matrix with an entry is owned
        const std::vector<double> X = s.X.data(), Y = s.Y.data();
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < ni; ++i)
                owned_x[std::size_t(j)] += std::abs(X[std::size_t(i * n + j)]);
        for (int64_t j = 0; j < p; ++j)
            for (int64_t i = 0; i < pi; ++i)
                owned_y[std::size_t(j)] += std::abs(Y[std::size_t(i * p + j)]);
    }
    for (const double v : owned_x)
        CORA_CHECK(v > 0, Error::wrong_value, [] {
            return std::string("reach: a state of the system is owned by no subsystem");
        });
    for (const double v : owned_y)
        CORA_CHECK(v > 0, Error::wrong_value, [] {
            return std::string("reach: a constraint of the system is owned by no subsystem");
        });
}

/// The checks of a reach problem against the system.
void aux_validate_problem(const NonlinDASys &sys, const DaeReachProblem &problem) {
    detail::Validation entry;
    const int64_t n = sys.dim(), p = sys.constraint_dim(), m = sys.input_dim();
    // what the problem needs, then the shapes of its sets
    CORA_CHECK(
        problem.initial_set.has_value() && problem.y0_guess.has_value(), Error::wrong_value,
        [] { return std::string("reach: the problem needs an initial_set and a y0_guess"); });
    // the horizon
    CORA_CHECK(problem.t_final > problem.t_start && problem.t_final < 1e300 && problem.t_start >= 0,
               Error::wrong_value, [&] {
                   return "reach: t_final must be above t_start >= 0, got " +
                          std::to_string(problem.t_start) + " and " +
                          std::to_string(problem.t_final);
               });
    CORA_CHECK(aux_has_shape(problem.initial_set->center(), n, 1) &&
                   problem.initial_set->generators().shape().size() == 2,
               Error::not_supported, [&] {
                   return "reach: the initial set must be one zonotope of the dimension " +
                          std::to_string(n) + "; batches are not supported";
               });
    // the guess of y, the input set and the constant inputs
    CORA_CHECK(aux_has_shape(*problem.y0_guess, p, 1), Error::dimension_mismatch,
               [&] { return "reach: y0_guess must be a column (" + std::to_string(p) + ", 1)"; });
    CORA_CHECK(!problem.input_set || (m > 0 && aux_has_shape(problem.input_set->center(), m, 1)),
               Error::dimension_mismatch, [&] {
                   return "reach: the input set must be one zonotope of the " + std::to_string(m) +
                          " inputs (the system has " + (m == 0 ? "none" : "them") + ")";
               });
    CORA_CHECK(!problem.u_trans || (m > 0 && problem.u_trans->shape().size() == 2 &&
                                    problem.u_trans->shape()[0] == m),
               Error::dimension_mismatch, [&] {
                   return "reach: u_trans must have the m = " + std::to_string(m) +
                          " inputs as rows, one column per step";
               });
    // the decomposition against the whole system
    if (problem.decomposition) aux_validate_decomposition(sys, *problem.decomposition);
}

/// The constant input of step k (0-based): column k of u_trans, its only column, or 0.
Tensor aux_u_trans(const DaeReachProblem &problem, const Tensor &like, int64_t m, int k) {
    if (!problem.u_trans) return like.zeros_like({m, 1});
    const int64_t columns = problem.u_trans->shape()[1];
    if (k >= columns && columns != 1)
        CORA_FAIL(Error::dimension_mismatch, "reach: u_trans has " + std::to_string(columns) +
                                                 " columns, one per step, and step " +
                                                 std::to_string(k + 1) + " has none");
    return columns == 1 ? *problem.u_trans : problem.u_trans->slice(-1, k, k + 1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The steps ---------------------------------------------------------------------------------------

// The first step: y consistent with the center of the initial set, the errors 0.
DaeStep NonlinDASys::init_reach(const DaeReachProblem &problem, const alg::DaeReach &alg) const {
    detail::Entry entry;
    if (entry.check()) {
        aux_validate_problem(*this, problem);
        std::visit([](const auto &a) { a.validate(); }, alg);
    }
    const alg::DaeLinearization &options = std::get<alg::DaeLinearization>(alg);
    const Zonotope &R0 = *problem.initial_set;
    const Tensor u_trans = aux_u_trans(problem, R0.center(), m_, 0);
    const Tensor u_star = problem.input_set ? problem.input_set->center() + u_trans : u_trans;
    const Tensor y0 = consistent_initial_state(R0.center(), *problem.y0_guess, u_star);
    return priv_dae_lin_reach(*this, R0, y0, R0.center().zeros_like(), y0.zeros_like(),
                              problem.input_set, u_trans, problem.decomposition, options);
}

// CORA's post reduces the sets that initReach returns, and keeps the errors.
DaeStep NonlinDASys::post(const DaeStep &step, const DaeReachProblem &problem,
                          const Tensor &u_trans, const alg::DaeReach &alg) const {
    detail::Entry entry;
    if (entry.check()) {
        aux_validate_problem(*this, problem);
        std::visit([](const auto &a) { a.validate(); }, alg);
        CORA_CHECK(
            step.error_x.defined() && step.error_x.shape() == std::vector<int64_t>({n_, 1}) &&
                step.error_y.defined() && step.error_y.shape() == std::vector<int64_t>({p_, 1}),
            Error::dimension_mismatch, [&] {
                return "post: the errors of the step must be columns (" + std::to_string(n_) +
                       ", 1) and (" + std::to_string(p_) + ", 1)";
            });
    }
    // the step from the end of the one before, then both sets are reduced
    const alg::DaeLinearization &options = std::get<alg::DaeLinearization>(alg);
    DaeStep next = priv_dae_lin_reach(*this, step.time_point, step.algebraic.center(), step.error_x,
                                      step.error_y, problem.input_set, u_trans,
                                      problem.decomposition, options);
    next.time_point = next.time_point.reduce(options.zonotope_order);
    next.time_interval = next.time_interval.reduce(options.zonotope_order);
    return next;
}

// The reachable sets ------------------------------------------------------------------------------

// The start set, the first step, then the step after each one.
DaeReachSet NonlinDASys::reach(const DaeReachProblem &problem, const alg::DaeReach &alg) const {
    detail::Entry entry;
    // the problem and the options
    if (entry.check()) {
        aux_validate_problem(*this, problem);
        std::visit([](const auto &a) { a.validate(); }, alg);
    }
    const double time_step = std::get<alg::DaeLinearization>(alg).time_step;
    const int steps = priv_num_steps(problem.t_final - problem.t_start, time_step);
    // the start set, then the sets of each step
    DaeReachSet R;
    R.time_step = time_step;
    R.time_point.push_back(*problem.initial_set);
    DaeStep step = init_reach(problem, alg);
    for (int k = 1; k <= steps; ++k) {
        R.time_interval.push_back(step.time_interval);
        R.time_point.push_back(step.time_point);
        R.algebraic.push_back(step.algebraic);
        if (k < steps)
            step = post(step, problem, aux_u_trans(problem, step.time_point.center(), m_, k), alg);
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
