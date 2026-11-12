// reach - the reachable sets of x' = f(x, u), as CORA's nonlinearSys.reach with algorithm "lin"
//
// The first step is init_reach from the initial set, every further one post from the end of the
// step before: each linearizes f at the center of the set moved half a step along the flow, solves
// the affine system with the input set U, and adds the solution of the linearization error
// (CORA's linReach, private/priv_lin_reach.cpp). post reduces both sets to zonotope_order; the
// first step is not reduced, as in CORA. The error of a step is the first guess of the next.
//
// A PolyZonotope as the initial set with alg::Polynomialization runs the same steps on polynomial
// zonotopes (priv_poly_reach, CORA's linReach "poly"); CORA's restructuring (polyZono, default
// Inf: off) is not ported.
//
// With alg::LinearizationAdaptive the step sizes and orders are tuned by the algorithm
// (priv_reach_adaptive: CORALean's adaptStepFloat) and R.time holds the step times; init_reach and
// post are of alg::Linearization only.
//
// Syntax:   R = sys.reach(problem, alg);
//           Rp = sys.reach({.initial_set = pZ, .t_final = 1}, alg::Polynomialization{...});
//           R = sys.reach({.initial_set = X0, .t_final = 1}, alg::Linearization{.time_step = .01});
// Inputs:   problem - initial_set (one zonotope), input_set (of the m inputs, optional), t_final
//           alg - alg::Linearization (time_step, taylor_terms, zonotope_order, the orders)
// Outputs:  R - time_interval[k] encloses the states over [k*time_step, (k+1)*time_step]
// See also: linearize, simulate, LinearSys::reach
//
// CORA: contDynamics/@contDynamics/reach.m
// CORA: contDynamics/@nonlinearSys/initReach.m
// CORA: contDynamics/@nonlinearSys/post.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   10-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The fixed-step algorithm of init_reach and post: the adaptive one has no such steps.
const alg::Linearization &aux_fixed(const alg::NonlinearReach &alg, const char *op) {
    const auto *fixed = std::get_if<alg::Linearization>(&alg);
    if (fixed == nullptr)
        CORA_FAIL(Error::not_supported, std::string(op) + ": the algorithm " + name_of(alg) +
                                            " chooses its step sizes itself; use reach");
    return *fixed;
}

/// The sets of reach from the first step on, the same for the sets of both algorithms: init for
/// the first step, post for each one after it.
template <class Reach, class Step, class Start, class Init, class Post>
Reach aux_steps(const Start &X0, int steps, double time_step, Init init, Post post) {
    Reach R;
    R.time_point.push_back(X0);
    Step step = init();
    for (int k = 1; k < steps; ++k) {
        R.time_interval.push_back(step.time_interval);
        R.time_point.push_back(step.time_point);
        step = post(step);
    }
    R.time_interval.push_back(step.time_interval);
    R.time_point.push_back(step.time_point);
    R.time_step = time_step;
    return R;
}


/// One step on polynomial zonotopes by the algorithm: "poly" or "lin".
PolyNonlinearStep aux_poly_step(const NonlinearSys &sys, const PolyZonotope &R, const Tensor &error,
                                const std::optional<Zonotope> &U,
                                const alg::Polynomialization &alg) {
    return priv_poly_reach(sys, R, error, U, alg);
}

PolyNonlinearStep aux_poly_step(const NonlinearSys &sys, const PolyZonotope &R, const Tensor &error,
                                const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    return priv_lin_reach_poly(sys, R, error, U, alg);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The checks --------------------------------------------------------------------------------------

void NonlinearSys::validate_inputs(const char *op, const std::optional<Zonotope> &U) const {
    detail::Validation entry;
    CORA_CHECK(!U || m_ > 0, Error::not_supported, [&] {
        return std::string(op) + ": the system has no inputs; leave the input set empty";
    });
    // and it is one zonotope of the m inputs
    CORA_CHECK(!U || (U->center().shape() == std::vector<int64_t>({m_, 1}) &&
                      U->generators().shape().size() == 2),
               Error::dimension_mismatch, [&] {
                   return std::string(op) + ": the input set must be one zonotope of the " +
                          std::to_string(m_) + " inputs";
               });
}

void NonlinearSys::validate_sets(const char *op, const Zonotope &R,
                                 const std::optional<Zonotope> &U) const {
    detail::Validation entry;
    CORA_CHECK(R.center().shape() == std::vector<int64_t>({n_, 1}) &&
                   R.generators().shape().size() == 2,
               Error::not_supported, [&] {
                   return std::string(op) + ": the set of a nonlinear system is one zonotope of " +
                          "the dimension " + std::to_string(n_) + "; batches are not supported";
               });
    validate_inputs(op, U);
}

// The steps ---------------------------------------------------------------------------------------

NonlinearStep NonlinearSys::init_reach(const Zonotope &R0, const std::optional<Zonotope> &U,
                                       const alg::NonlinearReach &alg) const {
    detail::Entry entry;
    if (entry.check()) {
        validate_sets("init_reach", R0, U);
        std::visit([](const auto &a) { a.validate(); }, alg);
    }
    // CORA starts from the error 0
    const alg::Linearization &options = aux_fixed(alg, "init_reach");
    return priv_lin_reach(*this, R0, R0.center().zeros_like(), U, options);
}

// CORA's post reduces the sets that initReach returns, and keeps the error.
NonlinearStep NonlinearSys::post(const NonlinearStep &step, const std::optional<Zonotope> &U,
                                 const alg::NonlinearReach &alg) const {
    detail::Entry entry;
    if (entry.check()) {
        validate_sets("post", step.time_point, U);
        std::visit([](const auto &a) { a.validate(); }, alg);
        CORA_CHECK(step.error.defined() && step.error.shape() == std::vector<int64_t>({n_, 1}),
                   Error::dimension_mismatch, [&] {
                       return "post: the error of the step must be a column (" +
                              std::to_string(n_) + ", 1)";
                   });
    }
    const alg::Linearization &options = aux_fixed(alg, "post");
    NonlinearStep next = priv_lin_reach(*this, step.time_point, step.error, U, options);
    next.time_point = next.time_point.reduce(options.zonotope_order);
    next.time_interval = next.time_interval.reduce(options.zonotope_order);
    return next;
}

// The reachable sets ------------------------------------------------------------------------------

ReachSet NonlinearSys::reach(const ReachProblem &problem, const alg::NonlinearReach &alg) const {
    detail::Entry entry;
    // one initial zonotope of the dimension, an input set of the inputs, a positive horizon
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("reach: the problem needs an initial_set"); });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "reach: t_final must be positive and finite, got " + std::to_string(problem.t_final);
    });
    if (entry.check()) {
        validate_sets("reach", *problem.initial_set, problem.input_set);
        std::visit([](const auto &a) { a.validate(); }, alg);
    }
    // the adaptive algorithm tunes its steps
    if (const auto *adaptive = std::get_if<alg::LinearizationAdaptive>(&alg))
        return priv_reach_adaptive(*this, problem, *adaptive);
    // the start set, the first step, then the step after each one
    const double time_step = std::get<alg::Linearization>(alg).time_step;
    return aux_steps<ReachSet, NonlinearStep>(
        *problem.initial_set, priv_num_steps(problem.t_final, time_step), time_step,
        [&] { return init_reach(*problem.initial_set, problem.input_set, alg); },
        [&](const NonlinearStep &step) { return post(step, problem.input_set, alg); });
}

// The reachable sets of polynomial zonotopes ------------------------------------------------------


void NonlinearSys::validate_poly_sets(const char *op, const PolyZonotope &R,
                                      const std::optional<Zonotope> &U) const {
    detail::Validation entry;
    // the dimension of the set and the input set, as for a zonotope
    CORA_CHECK(R.dim() == n_, Error::dimension_mismatch, [&] {
        return std::string(op) + ": the polynomial zonotope has the dimension " +
               std::to_string(R.dim()) + ", the system " + std::to_string(n_);
    });
    validate_inputs(op, U);
}

// The checks of a step on polynomial zonotopes: the set, the input, the algorithm, the error.
template <class Alg>
void NonlinearSys::validate_poly_step(const char *op, const PolyZonotope &R,
                                      const std::optional<Tensor> &error,
                                      const std::optional<Zonotope> &U, const Alg &alg) const {
    detail::Validation entry;
    validate_poly_sets(op, R, U);
    alg.validate();
    CORA_CHECK(!error || (error->defined() && error->shape() == std::vector<int64_t>({n_, 1})),
               Error::dimension_mismatch, [&] {
                   return std::string(op) + ": the error of the step must be a column (" +
                          std::to_string(n_) + ", 1)";
               });
}

// The first step from the initial set, with the error 0, for either algorithm on polynomial sets.
PolyNonlinearStep NonlinearSys::init_reach(const PolyZonotope &R0, const std::optional<Zonotope> &U,
                                           const alg::Polynomialization &alg) const {
    detail::Entry entry;
    if (entry.check()) validate_poly_step("init_reach", R0, std::nullopt, U, alg);
    // CORA starts from the error 0
    return aux_poly_step(*this, R0, R0.center().zeros_like(), U, alg);
}

// The first step of "lin" from a polynomial zonotope, also with the error 0.
PolyNonlinearStep NonlinearSys::init_reach(const PolyZonotope &R0, const std::optional<Zonotope> &U,
                                           const alg::Linearization &alg) const {
    detail::Entry entry;
    if (entry.check()) validate_poly_step("init_reach", R0, std::nullopt, U, alg);
    return aux_poly_step(*this, R0, R0.center().zeros_like(), U, alg);
}

// CORA's post reduces the sets that initReach returns, and keeps the error.
PolyNonlinearStep NonlinearSys::post(const PolyNonlinearStep &step,
                                     const std::optional<Zonotope> &U,
                                     const alg::Polynomialization &alg) const {
    detail::Entry entry;
    if (entry.check()) validate_poly_step("post", step.time_point, step.error, U, alg);
    // the step from the end of the one before, then both sets are reduced
    const PolyNonlinearStep next = aux_poly_step(*this, step.time_point, step.error, U, alg);
    return {next.time_interval.reduce(alg.zonotope_order),
            next.time_point.reduce(alg.zonotope_order), next.error};
}

PolyNonlinearStep NonlinearSys::post(const PolyNonlinearStep &step,
                                     const std::optional<Zonotope> &U,
                                     const alg::Linearization &alg) const {
    detail::Entry entry;
    if (entry.check()) validate_poly_step("post", step.time_point, step.error, U, alg);
    // the same step and reduction for "lin"
    const PolyNonlinearStep next = aux_poly_step(*this, step.time_point, step.error, U, alg);
    return {next.time_interval.reduce(alg.zonotope_order),
            next.time_point.reduce(alg.zonotope_order), next.error};
}

PolyReachSet NonlinearSys::reach(const PolyReachProblem &problem,
                                 const alg::PolyNonlinearReach &alg) const {
    detail::Entry entry;
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("reach: the problem needs an initial_set"); });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "reach: t_final must be positive and finite, got " + std::to_string(problem.t_final);
    });
    if (entry.check()) validate_poly_sets("reach", *problem.initial_set, problem.input_set);
    // the start set, the first step, then the step after each one, by the algorithm's own steps
    return std::visit(
        [&](const auto &options) {
            // init_reach from the start set, then post from each step
            return aux_steps<PolyReachSet, PolyNonlinearStep>(
                *problem.initial_set, priv_num_steps(problem.t_final, options.time_step),
                options.time_step,
                [&] { return init_reach(*problem.initial_set, problem.input_set, options); },
                [&](const PolyNonlinearStep &step) {
                    return post(step, problem.input_set, options);
                });
        },
        alg);
}

PolyReachSet NonlinearSys::reach(const PolyReachProblem &problem,
                                 const alg::PolynomializationAdaptive &alg) const {
    detail::Entry entry;
    // as for the other algorithms: a start set and a positive finite horizon
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("reach: the problem needs an initial_set"); });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "reach: t_final must be positive and finite, got " + std::to_string(problem.t_final);
    });
    if (entry.check()) {
        validate_poly_sets("reach", *problem.initial_set, problem.input_set);
        alg.validate();
    }
    // the steps are tuned by the algorithm
    return priv_reach_poly_adaptive(*this, problem, alg);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
