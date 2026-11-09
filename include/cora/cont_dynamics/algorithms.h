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
// CORA: contDynamics/@linearSys/private/priv_reach_fromStart.m
// CORA: contDynamics/@linearSys/private/priv_reach_decomp.m
// CORA: contDynamics/@contDynamics/reach.m
// CORA: hybridDynamics/@hybridAutomaton/reach.m
// CORA authors: Mark Wetzlinger, Tobias Ladner, Matthias Althoff, Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/options.h"
#include "cora/global/settings.h"

#include <array>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

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

/// A partition of the state into blocks [first, last] (0-based), non-empty.
inline void validate_partition(const std::vector<std::array<int64_t, 2>> &partition,
                               const char *name) {
    detail::Validation entry;
    CORA_CHECK(!partition.empty(), Error::wrong_value, [&] {
        return std::string(name) +
               ": partition must list the blocks of the state, e.g. {{0, 1}, {2, 2}}";
    });
    int64_t next = 0;
    for (const std::array<int64_t, 2> &block : partition) {
        CORA_CHECK(block[0] == next && block[1] >= block[0], Error::wrong_value, [&] {
            return std::string(name) + ": the blocks of partition must follow each other from 0 "
                                       "without a gap, each [first, last] with first <= last";
        });
        next = block[1] + 1;
    }
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

/// CORA's linAlg "fromStart": every step is the first step's solution mapped forward by e^{A k dt},
/// so the error of the sets does not accumulate; the input solutions are summed and reduced.
struct FromStart {
    double time_step = 0;    ///< the step size, required
    int taylor_terms = 10;   ///< the order of the Taylor series of the curvature enlargement
    int zonotope_order = 50; ///< with an input set, the order the input solution is reduced to

    static constexpr const char *name = "from_start";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &FromStart::time_step),
                               field("taylor_terms", &FromStart::taylor_terms),
                               field("zonotope_order", &FromStart::zonotope_order));
    }
    void validate() const {
        validate_time_step(time_step, "FromStart");
        validate_order(taylor_terms, 1, "taylor_terms", "FromStart");
        validate_order(zonotope_order, 1, "zonotope_order", "FromStart");
    }
};

/// CORA's linAlg "decomp": as fromStart, but the state is split into the blocks of `partition`
/// (0-based [first, last] pairs that cover 0..n-1 in order), and each block is propagated on its
/// own.
struct Decomposition {
    double time_step = 0;    ///< the step size, required
    int taylor_terms = 10;   ///< the order of the Taylor series of the curvature enlargement
    int zonotope_order = 50; ///< with an input set, the order the input solution is reduced to
    std::vector<std::array<int64_t, 2>> partition; ///< the blocks of the state, required

    static constexpr const char *name = "decomposition";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &Decomposition::time_step),
                               field("taylor_terms", &Decomposition::taylor_terms),
                               field("zonotope_order", &Decomposition::zonotope_order),
                               field("partition", &Decomposition::partition));
    }
    void validate() const {
        validate_time_step(time_step, "Decomposition");
        validate_order(taylor_terms, 1, "taylor_terms", "Decomposition");
        validate_order(zonotope_order, 1, "zonotope_order", "Decomposition");
        validate_partition(partition, "Decomposition");
    }
};

// Reachability of linear parametric systems ----------------------------------------------------

/// CORA's reach of linParamSys: the exponential of the matrix set to intermediate_terms as a
/// matrix zonotope and to taylor_terms as interval matrices, then step by step.
struct LinParamStandard {
    double time_step = 0;        ///< the step size, required
    int taylor_terms = 4;        ///< the order of the Taylor series of the exponential
    int zonotope_order = 50;     ///< the order the sets are reduced to per step
    int intermediate_terms = 2;  ///< the terms of the series kept as a matrix zonotope
    bool comp_time_point = true; ///< whether the sets at the step times are computed

    static constexpr const char *name = "standard";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &LinParamStandard::time_step),
                               field("taylor_terms", &LinParamStandard::taylor_terms),
                               field("zonotope_order", &LinParamStandard::zonotope_order),
                               field("intermediate_terms", &LinParamStandard::intermediate_terms),
                               field("comp_time_point", &LinParamStandard::comp_time_point));
    }
    void validate() const {
        validate_time_step(time_step, "LinParamStandard");
        validate_order(zonotope_order, 1, "zonotope_order", "LinParamStandard");
        validate_order(intermediate_terms, 2, "intermediate_terms", "LinParamStandard");
        validate_order(taylor_terms, intermediate_terms, "taylor_terms (>= intermediate_terms)",
                       "LinParamStandard");
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

// Reachability of hybrid automata --------------------------------------------------------------

/// How a guard intersection is enclosed (CORA's options.enclose): in the axes ("box"), in the
/// principal axes of the generators ("pca"), or orthogonal to the flow ("flow"). Several give a
/// zonotope bundle, the intersection of the enclosures.
enum class Enclose { box, pca, flow };

/// The enclosure methods of a guard intersection, one at least.
inline void validate_enclose(const std::vector<Enclose> &enclose, const char *name) {
    detail::Validation entry;
    CORA_CHECK(!enclose.empty(), Error::wrong_value, [&] {
        return std::string(name) + ": enclose must name a method, e.g. {Enclose::box}";
    });
}

/// CORA's guardIntersect "zonoGirard": the intersection of each set with the hyperplane of the
/// guard, bounded per axis of the basis by Girard's 2-D zonogon search.
struct ZonoGirard {
    std::vector<Enclose> enclose{Enclose::box}; ///< the bases of the enclosure

    static constexpr const char *name = "zono_girard";
    static constexpr auto fields() {
        return std::make_tuple(field("enclose", &ZonoGirard::enclose));
    }
    void validate() const { validate_enclose(enclose, "ZonoGirard"); }
};

/// CORA's guardIntersect "nondetGuard": the sets are enclosed by one box per basis first, then
/// intersected with the guard as constrained zonotopes; for large uncertain guards.
struct NondetGuard {
    std::vector<Enclose> enclose{Enclose::box}; ///< the bases of the enclosure

    static constexpr const char *name = "nondet_guard";
    static constexpr auto fields() {
        return std::make_tuple(field("enclose", &NondetGuard::enclose));
    }
    void validate() const { validate_enclose(enclose, "NondetGuard"); }
};

/// CORA's guardIntersect "conZonotope": each set, reduced to guard_order, is intersected with the
/// guard as a constrained zonotope, and the union is enclosed per basis.
struct ConZonotopeGuard {
    std::vector<Enclose> enclose{Enclose::box}; ///< the bases of the enclosure
    int guard_order = 0; ///< the order the sets are reduced to first, required (CORA's guardOrder)

    static constexpr const char *name = "con_zonotope";
    static constexpr auto fields() {
        return std::make_tuple(field("enclose", &ConZonotopeGuard::enclose),
                               field("guard_order", &ConZonotopeGuard::guard_order));
    }
    void validate() const {
        validate_enclose(enclose, "ConZonotopeGuard");
        validate_order(guard_order, 1, "guard_order", "ConZonotopeGuard");
    }
};

/// The guard intersection methods (CORA's options.guardIntersect).
using GuardIntersect = std::variant<ZonoGirard, NondetGuard, ConZonotopeGuard>;

/// CORA's reach of a hybridAutomaton with linAlg "standard": in each location the algorithm
/// Standard until the sets leave the invariant, then one jump per hit guard.
struct HybridStandard {
    std::vector<double> time_step; ///< one step size, or one per location, required
    int taylor_terms = 10;         ///< the order of the Taylor series of the curvature enlargement
    int zonotope_order = 50;       ///< the order the sets are reduced to per step
    int intermediate_terms = 2;    ///< a LinParamSys flow: its LinParamStandard::intermediate_terms
    GuardIntersect guard_intersect = ZonoGirard{}; ///< how the guards are intersected

    // the fields as Python sees them, and their checks
    static constexpr const char *name = "standard";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &HybridStandard::time_step),
                               field("taylor_terms", &HybridStandard::taylor_terms),
                               field("zonotope_order", &HybridStandard::zonotope_order),
                               field("intermediate_terms", &HybridStandard::intermediate_terms),
                               field("guard_intersect", &HybridStandard::guard_intersect));
    }
    void validate() const {
        detail::Validation entry;
        CORA_CHECK(!time_step.empty(), Error::wrong_value, [] {
            return std::string("HybridStandard: time_step must give one step size, or one per "
                               "location, e.g. {.time_step = {0.01}}");
        });
        // every step size and order, then the guard method
        for (double dt : time_step) validate_time_step(dt, "HybridStandard");
        validate_order(taylor_terms, 1, "taylor_terms", "HybridStandard");
        validate_order(zonotope_order, 1, "zonotope_order", "HybridStandard");
        validate_order(intermediate_terms, 2, "intermediate_terms", "HybridStandard");
        std::visit([](const auto &g) { g.validate(); }, guard_intersect);
    }
};

/// The algorithms of each system and operation; the first is the default of Python.
using LinearReach = std::variant<Standard, WrappingFree, FromStart, Decomposition>;
using LinearVerify = std::variant<ReachAvoidSupportFunction, ReachAvoidZonotope>;
using LinearSimulate = std::variant<Exact>;
using NonlinearReach = std::variant<Linearization>;
using LinParamReach = std::variant<LinParamStandard>;
using NonlinearSimulate = std::variant<RungeKutta4>;
using HybridReach = std::variant<HybridStandard>;

/// The name of the algorithm a variant holds.
template <class Variant>
const char *name_of(const Variant &alg) {
    return std::visit([](const auto &a) { return std::decay_t<decltype(a)>::name; }, alg);
}

} // namespace cora::alg

// ---------------------------------------  END OF CODE  ---------------------------------------- //
