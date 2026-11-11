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
// CORA: contDynamics/@linearSys/private/priv_reach_krylov.m
// CORA: contDynamics/@contDynamics/reach.m
// CORA: hybridDynamics/@hybridAutomaton/reach.m
// CORA authors: Mark Wetzlinger, Tobias Ladner, Matthias Althoff, Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Opus 5.5, Claude Sonnet 5.5
// Written:       02-November-2026
// Last update:   11-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/options.h"
#include "cora/global/settings.h"
#include "cora/global/tensor/tolerance.h"

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

/// CORA's linAlg "krylov": the center and each generator of the initial set (and of the input
/// solution) are propagated in their own Krylov subspace, whose order grows by krylov_step until
/// the error bound of Jawecki et al. is below krylov_error; the sets are those of the output.
struct Krylov {
    double time_step = 0;    ///< the step size, required
    int taylor_terms = 10;   ///< the order of the Taylor series of the curvature enlargement
    int zonotope_order = 50; ///< the order the sets are reduced to per step
    double krylov_error = machine_eps(); ///< the error per vector (CORA has no default)
    int krylov_step = 20;                ///< how much the order of a subspace grows per try

    static constexpr const char *name = "krylov";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &Krylov::time_step),
                               field("taylor_terms", &Krylov::taylor_terms),
                               field("zonotope_order", &Krylov::zonotope_order),
                               field("krylov_error", &Krylov::krylov_error),
                               field("krylov_step", &Krylov::krylov_step));
    }
    void validate() const {
        validate_time_step(time_step, "Krylov");
        validate_order(taylor_terms, 1, "taylor_terms", "Krylov");
        validate_order(zonotope_order, 1, "zonotope_order", "Krylov");
        validate_order(krylov_step, 1, "krylov_step", "Krylov");
        detail::Validation entry;
        CORA_CHECK(krylov_error >= 0 && krylov_error < 1e300, Error::wrong_value, [&] {
            return "Krylov: krylov_error must be nonnegative and finite, got " +
                   std::to_string(krylov_error);
        });
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

/// The Krylov support function algorithm for high-dimensional systems (ARCH-COMP26 AFF RND04-06):
/// the support function of the reachable output set along the specification directions, from the
/// Krylov subspace of each direction under A', with the step halved until every halfspace holds.
struct ReachAvoidKrylov {
    double time_step = 0;         ///< the first step size; 0 is t_final / 1000
    double min_time_step = 1e-10; ///< the step size below which it gives up
    int taylor_terms = 20;        ///< the order of the Taylor series of the curvature enlargement
    double krylov_error = 2 * machine_eps(); ///< the error per vector
    int krylov_step = 50;                    ///< how much the order of a subspace grows per try

    static constexpr const char *name = "reach_avoid_krylov";
    static constexpr auto fields() {
        // the options of the Python keywords
        return std::make_tuple(field("time_step", &ReachAvoidKrylov::time_step),
                               field("min_time_step", &ReachAvoidKrylov::min_time_step),
                               field("taylor_terms", &ReachAvoidKrylov::taylor_terms),
                               field("krylov_error", &ReachAvoidKrylov::krylov_error),
                               field("krylov_step", &ReachAvoidKrylov::krylov_step));
    }
    void validate() const {
        detail::Validation entry;
        CORA_CHECK(time_step >= 0 && time_step < 1e300, Error::wrong_value, [&] {
            return "ReachAvoidKrylov: time_step must be nonnegative, got " +
                   std::to_string(time_step);
        });
        // the step, the orders and the error
        validate_time_step(min_time_step, "ReachAvoidKrylov");
        validate_order(taylor_terms, 1, "taylor_terms", "ReachAvoidKrylov");
        validate_order(krylov_step, 1, "krylov_step", "ReachAvoidKrylov");
        CORA_CHECK(krylov_error >= 0 && krylov_error < 1e300, Error::wrong_value, [&] {
            return "ReachAvoidKrylov: krylov_error must be nonnegative and finite, got " +
                   std::to_string(krylov_error);
        });
    }
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

/// The orders of CORA's alg "lin" that NonlinearSys reach and a nonlinear location share: the
/// Taylor order of the linearization error, and the orders the sets of that error are reduced to.
inline void validate_tensor_orders(int tensor_order, int error_order, int intermediate_order,
                                   const char *name) {
    detail::Validation entry;
    CORA_CHECK(tensor_order == 2 || tensor_order == 3, Error::wrong_value, [&] {
        return std::string(name) + ": tensor_order must be 2 or 3, got " +
               std::to_string(tensor_order);
    });
    // CORA asks for errorOrder and intermediateOrder only at tensorOrder 3
    if (tensor_order == 3) {
        validate_order(error_order, 1, "error_order (needed at tensor_order 3)", name);
        validate_order(intermediate_order, 1, "intermediate_order (needed at tensor_order 3)",
                       name);
    }
}

/// CORA's alg "lin": each step linearizes at the center of the set (moved half a step along the
/// flow) and bounds the linearization error by the Lagrange remainder of the Taylor expansion of
/// order tensor_order over the reachable set of the step, enlarged until it holds.
struct Linearization {
    double time_step = 0;       ///< the step size, required
    int taylor_terms = 4;       ///< the order of the Taylor series of the linearized system
    int zonotope_order = 50;    ///< the sets are reduced to zonotope_order * n generators
    int tensor_order = 2;       ///< 2: the remainder by the Hessian; 3: by the third derivatives
    int error_order = 0;        ///< tensor_order 3: the order of the set the Hessian maps
    int intermediate_order = 0; ///< tensor_order 3: the order of the error set

    static constexpr const char *name = "linearization";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &Linearization::time_step),
                               field("taylor_terms", &Linearization::taylor_terms),
                               field("zonotope_order", &Linearization::zonotope_order),
                               field("tensor_order", &Linearization::tensor_order),
                               field("error_order", &Linearization::error_order),
                               field("intermediate_order", &Linearization::intermediate_order));
    }
    void validate() const {
        validate_time_step(time_step, "Linearization");
        validate_order(taylor_terms, 1, "taylor_terms", "Linearization");
        validate_order(zonotope_order, 1, "zonotope_order", "Linearization");
        validate_tensor_orders(tensor_order, error_order, intermediate_order, "Linearization");
    }
};

/// CORA's alg "poly" [Althoff, HSCC 2013] on polynomial zonotopes: as Linearization at tensor_order
/// 3, but the static linearization error (the Hessian at the expansion point) is a polynomial
/// zonotope that keeps its dependency on the initial set; only the rest is enclosed.
struct Polynomialization {
    double time_step = 0;        ///< the step size, required
    int taylor_terms = 4;        ///< the order of the Taylor series of the linearized system
    int zonotope_order = 50;     ///< the sets are reduced to zonotope_order * n generators
    int tensor_order = 3;        ///< 3 only: the Hessian as a set, the third derivatives as a box
    int error_order = 10;        ///< the order of the sets the Hessian maps
    int intermediate_order = 50; ///< the order of the error sets

    // the name and the fields, for the algorithm tables of Python and the checks
    static constexpr const char *name = "polynomialization";
    static constexpr auto fields() {
        // every field by its name
        return std::make_tuple(field("time_step", &Polynomialization::time_step),
                               field("taylor_terms", &Polynomialization::taylor_terms),
                               field("zonotope_order", &Polynomialization::zonotope_order),
                               field("tensor_order", &Polynomialization::tensor_order),
                               field("error_order", &Polynomialization::error_order),
                               field("intermediate_order", &Polynomialization::intermediate_order));
    }
    void validate() const {
        validate_time_step(time_step, "Polynomialization");
        validate_order(taylor_terms, 1, "taylor_terms", "Polynomialization");
        validate_order(zonotope_order, 1, "zonotope_order", "Polynomialization");
        validate_tensor_orders(tensor_order, error_order, intermediate_order, "Polynomialization");
        detail::Validation entry;
        CORA_CHECK(tensor_order == 3, Error::not_supported, [] {
            return std::string("Polynomialization: only tensor_order 3 is ported (the Hessian as a "
                               "set); order 2 has no static error and CORA's poly needs order > 2");
        });
    }
};

/// CORA's alg "lin-adaptive" [Wetzlinger et al., NAHS 2022] as CORALean repairs it
/// (reachAdaptiveFloat): step size, tensor order (2 or 3) and horizon are tuned from the
/// abstraction error. The constants are CORA's: zeta_z is its redFactor.
struct LinearizationAdaptive {
    double zeta_z = 0.0005;   ///< the share of the width a reduction may add (redFactor)
    double decr_factor = 0.9; ///< the factor by which a step size is scaled down while tuning
    std::vector<double> zeta_phi{0.85, 0.76, 0.68}; ///< the step size gain per order (0, 1, 2)
    double zeta_k = 0.9;       ///< below this gain of the tensor order, the lower order is kept
    int taylor_terms = 10;     ///< the least order of the Taylor series of the linear step
    int zonotope_order = 1000; ///< the order of Girard's reduction inside the linear step
    int fuel = 10000;          ///< the bound of every loop CORA writes as while true

    // the name and the fields, for the algorithm tables of Python and the checks
    static constexpr const char *name = "linearization_adaptive";
    static constexpr auto fields() {
        return std::make_tuple(field("zeta_z", &LinearizationAdaptive::zeta_z),
                               field("decr_factor", &LinearizationAdaptive::decr_factor),
                               field("zeta_phi", &LinearizationAdaptive::zeta_phi),
                               field("zeta_k", &LinearizationAdaptive::zeta_k),
                               field("taylor_terms", &LinearizationAdaptive::taylor_terms),
                               field("zonotope_order", &LinearizationAdaptive::zonotope_order),
                               field("fuel", &LinearizationAdaptive::fuel));
    }
    // every constant lies in the range the algorithm needs it in
    void validate() const {
        detail::Validation entry;
        CORA_CHECK(zeta_z > 0 && zeta_z < 1 && zeta_k > 0 && zeta_k < 1 && decr_factor > 0 &&
                       decr_factor < 1,
                   Error::wrong_value, [] {
                       return std::string("LinearizationAdaptive: zeta_z, zeta_k and decr_factor "
                                          "must lie in (0, 1)");
                   });
        CORA_CHECK(zeta_phi.size() == 3, Error::wrong_value, [] {
            return std::string("LinearizationAdaptive: zeta_phi needs three entries");
        });
        validate_order(taylor_terms, 1, "taylor_terms", "LinearizationAdaptive");
        validate_order(zonotope_order, 1, "zonotope_order", "LinearizationAdaptive");
        CORA_CHECK(fuel >= 1, Error::wrong_value,
                   [] { return std::string("LinearizationAdaptive: fuel must be at least 1"); });
    }
};

// Reachability of nonlinear differential-algebraic systems --------------------------------------

/// CORA's alg "lin" of nonlinDASys [Althoff, Krogh, TAC 2014]: each step linearizes f and g at the
/// center of the set, eliminates y, and bounds the linearization error by the Lagrange remainder
/// of the second order, enlarged from the step before's until it holds.
struct DaeLinearization {
    double time_step = 0;    ///< the step size, required
    int taylor_terms = 4;    ///< the order of the Taylor series of the linearized system
    int zonotope_order = 50; ///< the sets are reduced to zonotope_order * n generators
    int tensor_order = 2;    ///< 2 only: the remainder by the Hessian over the reachable set
    int error_order = 3;     ///< the order of the set the Hessians map (Girard)
    std::vector<double> max_error_x{}; ///< the largest error per state (maxError_x); empty: Inf
    std::vector<double> max_error_y{}; ///< the largest error per constraint (maxError_y)

    static constexpr const char *name = "dae_linearization";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &DaeLinearization::time_step),
                               field("taylor_terms", &DaeLinearization::taylor_terms),
                               field("zonotope_order", &DaeLinearization::zonotope_order),
                               field("tensor_order", &DaeLinearization::tensor_order),
                               field("error_order", &DaeLinearization::error_order),
                               field("max_error_x", &DaeLinearization::max_error_x),
                               field("max_error_y", &DaeLinearization::max_error_y));
    }
    void validate() const {
        // the step, the orders and the tensor order
        validate_time_step(time_step, "DaeLinearization");
        validate_order(taylor_terms, 1, "taylor_terms", "DaeLinearization");
        validate_order(zonotope_order, 1, "zonotope_order", "DaeLinearization");
        validate_order(error_order, 1, "error_order", "DaeLinearization");
        detail::Validation entry;
        CORA_CHECK(tensor_order == 2, Error::not_supported, [] {
            return std::string("DaeLinearization: only tensor_order 2 is ported (the Hessian over "
                               "the reachable set); order 3 is CORA's priv_linError_thirdOrder");
        });
    }
};

// Reachability of hybrid automata --------------------------------------------------------------

/// How a guard intersection is enclosed (CORA's options.enclose): in the axes ("box"), in the
/// principal axes of the generators ("pca"), or orthogonal to the flow ("flow"). Several give a
/// zonotope bundle, the intersection of the enclosures.
enum class Enclose { box, pca, flow };

/// How a cut by a polytope guard is boxed (CORALean's `BoxMethod`): by Fourier-Motzkin passes
/// ("fm", looser than CORA), by checked dual solutions of CORA's programs ("cert", as tight as
/// CORA), or by the cut of both ("both").
enum class CutBox { fm, cert, both };

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
    CutBox box = CutBox::fm;                    ///< how a cut is boxed
    int fm_passes = 10;                         ///< the Fourier-Motzkin passes of box fm and both

    static constexpr const char *name = "nondet_guard";
    static constexpr auto fields() {
        return std::make_tuple(field("enclose", &NondetGuard::enclose),
                               field("box", &NondetGuard::box),
                               field("fm_passes", &NondetGuard::fm_passes));
    }
    void validate() const {
        validate_enclose(enclose, "NondetGuard");
        validate_order(fm_passes, 0, "fm_passes", "NondetGuard");
    }
};

/// CORA's guardIntersect "conZonotope": each set, reduced to guard_order, is intersected with the
/// guard as a constrained zonotope, and the union is enclosed per basis.
struct ConZonotopeGuard {
    std::vector<Enclose> enclose{Enclose::box}; ///< the bases of the enclosure
    int guard_order = 0; ///< the order the sets are reduced to first, required (CORA's guardOrder)
    CutBox box = CutBox::fm; ///< how a cut is boxed
    int fm_passes = 10;      ///< the Fourier-Motzkin passes of box fm and both

    static constexpr const char *name = "con_zonotope";
    static constexpr auto fields() {
        return std::make_tuple(field("enclose", &ConZonotopeGuard::enclose),
                               field("guard_order", &ConZonotopeGuard::guard_order),
                               field("box", &ConZonotopeGuard::box),
                               field("fm_passes", &ConZonotopeGuard::fm_passes));
    }
    void validate() const {
        validate_enclose(enclose, "ConZonotopeGuard");
        validate_order(guard_order, 1, "guard_order", "ConZonotopeGuard");
        validate_order(fm_passes, 0, "fm_passes", "ConZonotopeGuard");
    }
};

/// The guard intersection methods (CORA's options.guardIntersect).
using GuardIntersect = std::variant<ZonoGirard, NondetGuard, ConZonotopeGuard>;

/// CORA's reach of a hybridAutomaton with linAlg "standard": in each location the algorithm
/// Standard (a NonlinearSys flow: Linearization) until the sets leave the invariant, then one
/// jump per hit guard.
struct HybridStandard {
    std::vector<double> time_step; ///< one step size, or one per location, required
    int taylor_terms = 10;         ///< the order of the Taylor series of the curvature enlargement
    int zonotope_order = 50;       ///< the order the sets are reduced to per step
    int intermediate_terms = 2;    ///< a LinParamSys flow: its LinParamStandard::intermediate_terms
    int tensor_order = 2;          ///< a NonlinearSys flow: its Linearization::tensor_order
    int error_order = 0;           ///< a NonlinearSys flow: its Linearization::error_order
    int intermediate_order = 0;    ///< a NonlinearSys flow: its Linearization::intermediate_order
    bool polynomialization = false; ///< a NonlinearSys flow: alg::Polynomialization (poly sets)
    GuardIntersect guard_intersect = ZonoGirard{}; ///< how the guards are intersected

    // the fields as Python sees them, and their checks
    static constexpr const char *name = "standard";
    static constexpr auto fields() {
        return std::make_tuple(field("time_step", &HybridStandard::time_step),
                               field("taylor_terms", &HybridStandard::taylor_terms),
                               field("zonotope_order", &HybridStandard::zonotope_order),
                               field("intermediate_terms", &HybridStandard::intermediate_terms),
                               field("tensor_order", &HybridStandard::tensor_order),
                               field("error_order", &HybridStandard::error_order),
                               field("intermediate_order", &HybridStandard::intermediate_order),
                               field("polynomialization", &HybridStandard::polynomialization),
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
        validate_tensor_orders(tensor_order, error_order, intermediate_order, "HybridStandard");
        std::visit([](const auto &g) { g.validate(); }, guard_intersect);
    }
};

/// The algorithms of each system and operation; the first is the default of Python.
using LinearReach = std::variant<Standard, WrappingFree, FromStart, Decomposition>;
using LinearVerify = std::variant<ReachAvoidSupportFunction, ReachAvoidZonotope, ReachAvoidKrylov>;
using LinearSimulate = std::variant<Exact>;
using NonlinearReach = std::variant<Linearization, LinearizationAdaptive>;
using PolyNonlinearReach = std::variant<Polynomialization, Linearization>;
using LinParamReach = std::variant<LinParamStandard>;
using NonlinearSimulate = std::variant<RungeKutta4>;
using DaeReach = std::variant<DaeLinearization>;
using HybridReach = std::variant<HybridStandard>;

/// The name of the algorithm a variant holds.
template <class Variant>
const char *name_of(const Variant &alg) {
    return std::visit([](const auto &a) { return std::decay_t<decltype(a)>::name; }, alg);
}

} // namespace cora::alg

// ---------------------------------------  END OF CODE  ---------------------------------------- //
