// verify - verifies a linear system against reach-avoid specifications, as CORA's linearSys.verify
//
// Syntax:   res = sys.verify(problem, specs, alg);
// Inputs:   problem - initial_set (one zonotope of the state space), input_set (one zonotope of
//                     the input space of B; without B it must be zero), t_final
//           specs - safe sets and unsafe sets over the outputs y = C x
//           alg - alg::ReachAvoidSupportFunction (unsafe sets of one halfspace) or
//                 alg::ReachAvoidZonotope (unsafe polytopes too)
// Outputs:  res - verified, the iterations of the adaptive loop, the last step size and step
//           count, and a falsifying initial state when a specification is hit
// See also: reach, private/priv_verify_ra_support_function
//
// CORA: contDynamics/@linearSys/verify.m
// CORA: contDynamics/@linearSys/private/priv_verifyRA_supportFunc.m
// CORA: contDynamics/@linearSys/private/priv_verifyRA_zonotope.m
// CORA authors: Tobias Ladner, Mark Wetzlinger, Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether every number of t is zero (a NaN is not): the sum of |t| is exactly zero then only.
bool aux_all_zero(const Tensor &t) { return t.abs().sum().item() == 0; }

/// Whether a halfspace normal is a nonzero column of p rows.
bool aux_normal_ok(const Halfspace &h, int64_t p) {
    if (h.a.shape() != std::vector<int64_t>{p, 1}) return false;
    return !aux_all_zero(h.a);
}

} // namespace


// ===========================================  MAIN  =========================================== //

VerifyResult LinearSys::verify(const ReachProblem &problem, const std::vector<Specification> &specs,
                               const alg::LinearVerify &alg) const {
    detail::Entry entry;
    // one system, one initial set and one input set of the right dimensions
    const int64_t n = dim();
    CORA_CHECK(A_.shape().size() == 2, Error::not_supported,
               [] { return std::string("verify: takes a single system, not a batch"); });
    CORA_CHECK(
        problem.initial_set.has_value() && problem.input_set.has_value(), Error::wrong_value,
        [] { return std::string("verify: the problem needs an initial_set and an input_set"); });
    const Zonotope &R0 = *problem.initial_set, &U = *problem.input_set;
    CORA_CHECK(R0.center().shape() == std::vector<int64_t>({n, 1}) &&
                   R0.generators().shape().size() == 2,
               Error::dimension_mismatch, [&] {
                   return "verify: the initial set must be one zonotope of the dimension " +
                          std::to_string(n) + ", got the center " + shape_text(R0.center().shape());
               });
    CORA_CHECK(problem.t_final > 0 && std::isfinite(problem.t_final), Error::wrong_value, [&] {
        return "verify: t_final must be positive and finite, got " +
               std::to_string(problem.t_final);
    });
    // the input set: of the columns of B, and zero without B
    const int64_t m = B_ ? B_->shape()[1] : U.dim();
    CORA_CHECK(U.center().shape() == std::vector<int64_t>({m, 1}) &&
                   U.generators().shape().size() == 2,
               Error::dimension_mismatch, [&] {
                   return "verify: the input set must be one zonotope of the dimension " +
                          std::to_string(m) + " of the columns of B, got the center " +
                          shape_text(U.center().shape());
               });
    CORA_CHECK(
        B_ || (aux_all_zero(U.center()) && aux_all_zero(U.generators())), Error::wrong_value,
        [] { return std::string("verify: a nonzero input set needs B; use LinearSys(A, B)"); });
    CORA_CHECK(!specs.empty(), Error::wrong_value,
               [] { return std::string("verify: give at least one specification"); });
    const int64_t p = C_ ? C_->shape()[0] : n;
    for (const Specification &spec : specs)
        for (const Halfspace &h : spec.halfspaces())
            CORA_CHECK(aux_normal_ok(h, p), Error::wrong_value, [&] {
                return "verify: a halfspace normal must be a nonzero column of the output "
                       "dimension " +
                       std::to_string(p) + ", got the shape " + shape_text(h.a.shape());
            });
    // the support function algorithm decides unsafe sets of one halfspace only
    if (std::holds_alternative<alg::ReachAvoidSupportFunction>(alg))
        for (const Specification &spec : specs)
            CORA_CHECK(spec.type() != SpecType::unsafe_set || spec.halfspaces().size() == 1,
                       Error::not_supported, [] {
                           return std::string("verify: reach_avoid_support_function takes unsafe "
                                              "sets of one halfspace; use reach_avoid_zonotope");
                       });

    if (std::holds_alternative<alg::ReachAvoidSupportFunction>(alg))
        return priv_verify_ra_support_function(*this, R0, U, problem.t_final, specs);
    return priv_verify_ra_zonotope(*this, R0, U, problem.t_final, specs);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
