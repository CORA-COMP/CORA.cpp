// algorithms - the algorithms of reach, simulate and verify, as types (CORA's options)
//
// Each algorithm is a struct that holds exactly its own options, with CORA's defaults, a name
// (its Python spelling) and a field list; validate() checks them once per call. A system takes a
// std::variant of the ones it implements, so a wrong combination does not compile.
//
// Syntax:   sys.reach(problem, alg::Standard{.time_step = 0.01, .taylor_terms = 4});
//           sys.verify(problem, specs, alg::ReachAvoidZonotope{});
// See also: cont_dynamics/cont_dynamics.h, cont_dynamics/linear_sys/linear_sys.h
//
// CORA: contDynamics/@linearSys/reach.m
// CORA: contDynamics/@linearSys/verify.m
// CORA: contDynamics/@nonlinearSys/reach.m

#pragma once

#include "cora/global/options.h"
#include "cora/global/settings.h"

#include <string>
#include <variant>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::alg {

/// A positive step size, checked with the name of the algorithm in the message.
inline void validate_time_step(double time_step, const char *name) {
    detail::Validation entry;
    CORA_CHECK(time_step > 0 && time_step < 1e300, Error::wrong_value, [&] {
        return std::string(name) + ": time_step must be positive and finite, got " +
               std::to_string(time_step) + "; e.g. alg::" + name + "{.time_step = 0.01}";
    });
}

/// An order of at least `least`, checked with its name in the message.
inline void validate_order(int value, int least, const char *what, const char *name) {
    detail::Validation entry;
    CORA_CHECK(value >= least, Error::wrong_value, [&] {
        return std::string(name) + ": " + what + " must be at least " + std::to_string(least) +
               ", got " + std::to_string(value);
    });
}

// Reachability of linear systems ----------------------------------------------------------------

/// CORA's linAlg "standard": the sets of every step are enclosed and propagated.
struct Standard {
    double time_step = 0;    ///< the step size, required
    int taylor_terms = 10;   ///< the order of the Taylor series of the curvature enlargement
    int zonotope_order = 50; ///< with an input set, the order the sets are reduced to per step

    static constexpr const char *name = "standard";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &Standard::time_step),
                               field("taylor_terms", &Standard::taylor_terms),
                               field("zonotope_order", &Standard::zonotope_order));
    }
    void validate() const {
        validate_time_step(time_step, "Standard");
        validate_order(taylor_terms, 1, "taylor_terms", "Standard");
        validate_order(zonotope_order, 1, "zonotope_order", "Standard");
    }
};

/// CORA's linAlg "wrapping-free": the first step is enclosed once and mapped forward.
struct WrappingFree {
    double time_step = 0;  ///< the step size, required
    int taylor_terms = 10; ///< the order of the Taylor series of the curvature enlargement

    static constexpr const char *name = "wrapping_free";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &WrappingFree::time_step),
                               field("taylor_terms", &WrappingFree::taylor_terms));
    }
    void validate() const {
        validate_time_step(time_step, "WrappingFree");
        validate_order(taylor_terms, 1, "taylor_terms", "WrappingFree");
    }
};

// Verification of linear systems ----------------------------------------------------------------

/// CORA's verifyAlg "reachavoid:supportFunc": the support function of the affine solution along
/// the specification directions, with an adaptive step size; tunes itself.
struct ReachAvoidSupportFunction {
    static constexpr const char *name = "reach_avoid_support_function";
    static constexpr auto fields() { return std::tuple<>(); }
    void validate() const {}
};

/// CORA's verifyAlg "reachavoid:zonotope": adaptive zonotope reachable sets with a bound on their
/// error, refined until the outer and inner approximations decide; tunes itself.
struct ReachAvoidZonotope {
    static constexpr const char *name = "reach_avoid_zonotope";
    static constexpr auto fields() { return std::tuple<>(); }
    void validate() const {}
};

// Simulation ---------------------------------------------------------------------------------

/// The exact solution of a linear system, sampled every time_step.
struct Exact {
    double time_step = 0; ///< the sampling step, required

    static constexpr const char *name = "exact";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &Exact::time_step));
    }
    void validate() const { validate_time_step(time_step, "Exact"); }
};

/// The classical Runge-Kutta method, sampled every time_step with steps of at most max_step.
struct RungeKutta4 {
    double time_step = 0;   ///< the sampling step, required
    double max_step = 1e-3; ///< the largest integration step

    static constexpr const char *name = "runge_kutta4";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &RungeKutta4::time_step),
                               field("max_step", &RungeKutta4::max_step));
    }
    void validate() const {
        validate_time_step(time_step, "RungeKutta4");
        validate_time_step(max_step, "RungeKutta4");
    }
};

// Reachability of nonlinear systems -------------------------------------------------------------

/// CORA's alg "lin": each step linearizes at the center of the set and bounds the linearization
/// error by the Lagrange remainder over an enclosure of the step.
struct Linearization {
    double time_step = 0;    ///< the step size, required
    int taylor_terms = 4;    ///< the order of the Taylor series of the linearized system
    int zonotope_order = 50; ///< the sets are reduced to zonotope_order * n generators

    static constexpr const char *name = "linearization";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &Linearization::time_step),
                               field("taylor_terms", &Linearization::taylor_terms),
                               field("zonotope_order", &Linearization::zonotope_order));
    }
    void validate() const {
        validate_time_step(time_step, "Linearization");
        validate_order(taylor_terms, 1, "taylor_terms", "Linearization");
        validate_order(zonotope_order, 1, "zonotope_order", "Linearization");
    }
};

/// The algorithms of each system and operation; the first is the default of Python.
using LinearReach = std::variant<Standard, WrappingFree>;
using LinearVerify = std::variant<ReachAvoidSupportFunction, ReachAvoidZonotope>;
using LinearSimulate = std::variant<Exact>;
using NonlinearReach = std::variant<Linearization>;
using NonlinearSimulate = std::variant<RungeKutta4>;

/// The name of the algorithm a variant holds.
template <class Variant>
const char *name_of(const Variant &alg) {
    return std::visit([](const auto &a) { return std::decay_t<decltype(a)>::name; }, alg);
}

} // namespace cora::alg

// ---------------------------------------  END OF CODE  ---------------------------------------- //
