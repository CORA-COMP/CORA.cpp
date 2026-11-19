// priv_verify - the private functions behind linear_sys verify, as CORA's @linearSys/private
//
// Not for callers of the class: verify uses them.
//
// Syntax:   priv_verify_ra_support_function(sys, params, specs);   priv_verify_ra_zonotope(sys,
// ...); See also: linear_sys.h, verify.cpp
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

#include <chrono>
#include <optional>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

struct VerifyZonotopeTrace; // the grids and witnesses of a run, priv_verify_zonotope_pass.h
struct VerifyKrylovTrace;   // the directions and windows of a run, priv_verify_krylov.h

/// The input in canonical form x' = A x + u, u in u_trans + GU [-1, 1]^k: u_trans = B center(U)
/// and GU = B generators(U), without zero generators (GU (n, 0) if there are none); n states
/// and p outputs.
struct CanonicalInput {
    Tensor u_trans, GU;
    int64_t n = 0, p = 0;
};
CanonicalInput priv_canonical_input(const LinearSys &sys, const Zonotope &U);

/// A halfspace {y | a'y <= b} of a specification.
struct SpecRow {
    std::vector<double> a;
    double b = 0;
};

/// The halfspaces of the polytope of a safe or unsafe specification, read from its own constraints
/// (an equality as two rows). The only place where verify uses the representation of the set; a
/// set that is not a Polytope, or a custom specification, is refused.
std::vector<SpecRow> priv_spec_rows(const Specification &spec);

/// The algorithm "reachavoid:supportFunc": the support function of the affine solution along
/// the specification directions, with an adaptive step size. The inputs are checked by verify.
VerifyResult priv_verify_ra_support_function(const LinearSys &sys, const Zonotope &R0,
                                             const Zonotope &U, double t_final,
                                             const std::vector<Specification> &specs);

/// The algorithm "reachavoid:zonotope": adaptive zonotope reachable sets with a bound on their
/// error to the exact sets, refined until the outer and inner approximations decide.
VerifyResult priv_verify_ra_zonotope(const LinearSys &sys, const Zonotope &R0, const Zonotope &U,
                                     double t_final, const std::vector<Specification> &specs,
                                     VerifyZonotopeTrace *trace = nullptr);

/// The algorithm "reach_avoid_krylov": Krylov subspaces of the specification directions under A',
/// the support function of the output set bounded per step, the step halved until it decides.
VerifyResult priv_verify_ra_krylov(const LinearSys &sys, const Zonotope &R0, const Zonotope &U,
                                   double t_final, const std::vector<Specification> &specs,
                                   const alg::ReachAvoidKrylov &alg,
                                   VerifyKrylovTrace *trace = nullptr);

/// The seconds since `start`: the computation time of verify (CORA's tComp).
inline double priv_seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// The inverse of a square matrix, empty if it is (numerically) singular.
std::optional<Tensor> priv_inverse(const Tensor &A);

/// Flushes denormal numbers to zero while it lives: the powers and squarings of a stiff matrix
/// underflow through the denormal range, where every product is slow, and nothing there counts.
struct VerifyFlushDenormals {
#if defined(__SSE2__) || defined(_M_X64)
    unsigned saved = _mm_getcsr();
    VerifyFlushDenormals() { _mm_setcsr(saved | 0x8040); }
    ~VerifyFlushDenormals() { _mm_setcsr(saved); }
#endif
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
