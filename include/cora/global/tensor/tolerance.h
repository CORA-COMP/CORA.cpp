// tolerance - the one tolerance policy: what a `tol` argument means when it is left out
//
// Every `tol` of the public API defaults to unset. An unset tol is resolved at the entry point,
// in this order: the explicit argument, the innermost ToleranceScope of the thread, the process
// default (set_default_tolerance), default_tol(dtype). No function has a number of its own.
//
// Syntax:   double t = resolve_tol(tol, dtype);   ToleranceScope loose(1e-10);
//           set_default_tolerance(1e-10);   set_default_tolerance(std::nullopt);
// Inputs:   tol - an explicit tolerance or std::nullopt;  dtype - the dtype of the data compared
// Outputs:  t - the tolerance to compare with
// See also: global/within_tol.h, global/settings.h (the same scoping), docs/architecture.md
//
// CORA.cpp only: CORA has a default per function; CORA.cpp has one policy

#pragma once

#include <limits>
#include <optional>

#include "cora/global/settings.h"
#include "cora/global/tensor/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A tolerance argument: a nonnegative number, or unset (resolve_tol picks it).
using Tol = std::optional<double>;

/// The smallest x with 1 + x != 1 in dtype (MATLAB's eps).
inline double machine_eps(DType dtype) noexcept {
    return dtype == DType::float32 ? double(std::numeric_limits<float>::epsilon())
                                   : std::numeric_limits<double>::epsilon();
}

/// The machine epsilon of double, for scalar code.
inline double machine_eps() noexcept { return machine_eps(DType::float64); }

/// 100 machine epsilons of dtype: CORA's contains default, which absorbs the rounding of the
/// few operations between a set and its scaled copy (contains(Z, 1.0*Z) holds); float32 gets
/// the same multiple of its own, larger epsilon.
inline double default_tol(DType dtype) noexcept { return 100 * machine_eps(dtype); }

/// The tolerance on this thread without an argument: its innermost ToleranceScope, else the
/// process default; empty if both are unset.
Tol current_tolerance() noexcept;

/// The process default for threads without a ToleranceScope; std::nullopt restores default_tol.
void set_default_tolerance(Tol tol) noexcept;

/// `tol` if set, else current_tolerance(), else default_tol(dtype).
double resolve_tol(Tol tol, DType dtype) noexcept;

/// resolve_tol at a public entry point: if the entry checks, a negative or NaN tol is
/// Error::wrong_value, "<operation>: tol must be nonnegative, got <tol>".
double resolve_tol(const detail::Entry &entry, Tol tol, DType dtype, const char *operation);

/// resolve_tol for numbers, which are doubles.
inline double resolve_tol(Tol tol) noexcept { return resolve_tol(tol, DType::float64); }

/// Sets the tolerance of every operation on this thread that is given none, until the scope ends.
class ToleranceScope {
  public:
    explicit ToleranceScope(double tol) noexcept;
    ~ToleranceScope();
    ToleranceScope(const ToleranceScope &) = delete;
    ToleranceScope &operator=(const ToleranceScope &) = delete;

  private:
    Tol previous_;
};

/// The thresholds of the numerical methods: fixed numbers of an algorithm (a rounding slack, a
/// pivot size, a picture resolution), not tolerances that a user relaxes. Numeric literals of
/// this kind live here, each with its meaning; library code has none of its own.
namespace numerics {

// Time and steps ----------------------------------------------------------------------------------

inline constexpr double kStepCount = 1e-9; ///< slack of ceil(T / dt - slack): 0.3 / 0.1 is 3
inline constexpr double kTimeStepMatch = 1e-9; ///< adaptive reach: step sizes this close are one
inline constexpr double kTimeRound = 1e-14; ///< times this close are the same instant
inline constexpr double kAtMaxStep = 1e-12; ///< a step this close to the maximum is the maximum
inline constexpr double kMinTimeStep = 1e-12; ///< below this step size an adaptive pass gives up
inline constexpr double kTaylorStepMatch = 1e-10; ///< step sizes this close share a cache entry
inline constexpr double kBoxMargin = 1e-10; ///< the absolute margin of an inflated enclosure
inline constexpr double kInternalConsistency = 1e-9; ///< recomputed bounds agree up to this

// Series and iterations ---------------------------------------------------------------------------

inline constexpr double kExpmTermRatio = 1e-18; ///< a term below this part of the sum ends a series
inline constexpr double kJacobiOffDiagonal = 1e-30; ///< squared off-diagonal part that ends Jacobi
inline constexpr double kPowerIteration = 1e-15; ///< relative change that ends a power iteration
inline constexpr double kTiny = 1e-300; ///< keeps a denominator or a range off zero
inline constexpr double kOdeRelTol = 1e-3; ///< relative tolerance of the step control (ode45)
inline constexpr double kOdeAbsTol = 1e-6; ///< absolute tolerance of the step control (ode45)

// Linear programs ---------------------------------------------------------------------------------

inline constexpr double kSimplexPivot = 1e-9; ///< smallest usable tableau entry in a ratio test
inline constexpr double kSimplexOptimal = 1e-9; ///< reduced costs above -this are optimal
inline constexpr double kSimplexFeasible = 1e-8; ///< relative phase-1 residual that is feasible
inline constexpr double kSimplexTie = 1e-12; ///< relative gap under which two ratios tie
inline constexpr double kSimplexDrive = 1e-7; ///< entry that lets an artificial leave the basis
inline constexpr double kTableauCost = 1e-12; ///< Bland's rule: a reduced cost below -this enters
inline constexpr double kTableauPivot = 1e-12; ///< Bland's rule: a column entry above this leaves
inline constexpr double kTableauTie = 1e-15; ///< Bland's rule: ratios this close tie
inline constexpr double kTableauFeasible = 1e-9; ///< a common point exists if the cost is below
inline constexpr double kLpDistanceZero = 1e-9; ///< a program distance below this counts as 0
inline constexpr double kPolytopeContains = 1e-12; ///< a point beyond a facet by less is inside

// Geometry of zonotopes and pictures --------------------------------------------------------------

inline constexpr double kGeneratorZero = 1e-14; ///< a generator shorter than this is dropped
inline constexpr double kParallel2d = 1e-12; ///< sine below which 2d generators are parallel
inline constexpr double kParallel3d = 1e-10; ///< the same in 3d, and the face plane distance
inline constexpr double kAngleMerge = 1e-9; ///< angles this close are one
inline constexpr double kCornerMerge = 1e-9; ///< corners this close (of the extent) are one
inline constexpr double kRootFloor = 1e-9; ///< keeps floor(root) from missing an integer root
inline constexpr double kHullExtent = 1e-10; ///< hull points this close (of the extent) are flat
inline constexpr double kCoplanar = 1e-9; ///< 1 - cos of the angle below which faces are flat
inline constexpr double kUnionExtent = 1e-9; ///< distance (of the extent) at which polygons merge
inline constexpr double kConvexCorner = 1e-12; ///< a corner turning back by less is convex
inline constexpr double kPlotResolution = 1e-4; ///< part of the extent that union_polygons thins
inline constexpr double kThinTolerance = 1e-3; ///< part of the extent that tikz thinning drops
inline constexpr double kColorEqual = 5e-5; ///< colors this close in every channel are the same
inline constexpr double kPlotEqual = 1e-9; ///< a style number this close to its default is it
inline constexpr double kAxisRange = 1e-12; ///< an axis range below this is a point
inline constexpr double kPrintZero = 1e-12; ///< a coordinate below this prints as 0
inline constexpr double kTickSlack = 1e-9; ///< part of a step that the last tick may overshoot

} // namespace numerics

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
