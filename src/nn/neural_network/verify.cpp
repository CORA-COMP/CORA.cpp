// verify - the verdict of a network on a VNN-COMP property: falsification and branch and bound
//
// verify runs the falsification method and the bounding method's branch and bound in the order of
// falsification_order (CORA's verify.m is the bounding "conzonotope"). attack-first: the attack
// for attack_time_share of the timeout (at most attack_first_rounds rounds), then the branch and
// bound; bound-first skips that. Either way an open verdict gets the attack again until the
// deadline. A network the bounding cannot take gets the attack alone. "sat" needs the confirm
// check of the counterexample (the scoring's, from Python; else the float64 network's output).
//
// Syntax:   NnVerifyResult r = nn.verify(property, timeout);
//           NnVerifyResult r = nn.verify(property, timeout, vnncomp_options("acasxu"), confirm);
// Inputs:   property - the input boxes and the unsafe polytopes (NnProperty)
//           timeout - seconds of wall time from the call
//           options - the strategies and their options (NnVerifyOptions)
//           confirm - the check of a counterexample (unset: the network's float64 output)
// Outputs:  r - the verdict, the counterexample for "sat", and the counts
// See also: cora/nn/neural_network/verify.h, docs/nn_verification.md
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Lukas Koller, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_layers.h"
#include "nn/neural_network/private/priv_verify.h"

#include <cmath>
#include <exception>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The order of the strategies ---------------------------------------------------------------------

/// The verdict of the problem within the deadline, for checked inputs.
NnVerifyResult aux_verify(NnProblem &problem, double start, double timeout) {
    const NnVerifyOptions &o = problem.options();
    const double deadline = start + timeout;
    NnVerifyResult result;
    std::unique_ptr<Attack> attack = make_attack(o.falsification_method, o);
    std::unique_ptr<Bounding> bounding = priv_make_bounding(o.bounding_method, o);
    bool can_bound = false;
    try {
        can_bound = bounding->prepare(problem);
        if (!can_bound) result.note = bounding->why;
    } catch (const Error &error) {
        // a bound beyond the float range (an exp): the attack alone
        if (!priv_is_out_of_range(error)) std::rethrow_exception(std::current_exception());
        result.note = error.what();
    }
    // the result of a confirmed counterexample, and an attack with its time
    const auto sat = [&](const AttackResult &found) {
        result.status = "sat";
        result.x = found.x;
        result.y = found.y;
        result.stats["checks"] = double(problem.checks);
        return result;
    };
    const auto run_attack = [&](double until, int64_t rounds) {
        const double t = now_seconds();
        // a step the deadline cuts short finds nothing
        AttackResult found;
        problem.deadline = until;
        try {
            found = attack->run(problem, until, rounds);
        } catch (const Error &error) {
            if (!is_deadline(error)) std::rethrow_exception(std::current_exception());
            result.stats["attack_cut"] += 1;
        }
        // its time and steps
        result.stats["attack_seconds"] += now_seconds() - t;
        result.stats["attack_iterations"] += double(found.iterations);
        return found;
    };
    // the attack first, within its share of the time
    if (o.falsification_order == "attack-first") {
        const AttackResult found =
            can_bound ? run_attack(start + o.attack_time_share * timeout, o.attack_first_rounds)
                      : run_attack(deadline, -1);
        if (!found.x.empty()) return sat(found);
    }
    // the branch and bound, then the attack in the time that is left
    if (can_bound) {
        // a deadline part-way through leaves the verdict open
        const double t = now_seconds();
        AttackResult found;
        problem.deadline = deadline;
        std::string status = "unknown";
        try {
            status = bounding->branch_and_bound(problem, deadline, result.stats, found);
        } catch (const Error &error) {
            // the deadline, or a bound beyond the float range: the verdict stays open
            if (!is_deadline(error) && !priv_is_out_of_range(error))
                std::rethrow_exception(std::current_exception());
            result.stats[is_deadline(error) ? "bab_cut" : "bab_out_of_range"] += 1;
        }
        result.stats["bab_seconds"] = now_seconds() - t;
        if (status == "sat") return sat(found);
        if (status == "unsat") {
            result.status = "unsat";
            result.stats["checks"] = double(problem.checks);
            return result;
        }
    }
    // the attack in the time that is left
    const AttackResult found = run_attack(deadline, -1);
    if (!found.x.empty()) return sat(found);
    result.status = "unknown";
    result.stats["checks"] = double(problem.checks);
    return result;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The entry ---------------------------------------------------------------------------------------

NnVerifyResult NeuralNetwork::verify(const NnProperty &property, double timeout,
                                     const NnVerifyOptions &options,
                                     const ConfirmFn &confirm) const {
    detail::Entry entry;
    const double start = now_seconds();
    // the boxes have the input dimension, the polytopes the output dimension
    CORA_CHECK(property.lo.ndim() == 2 && property.lo.shape() == property.hi.shape() &&
                   property.lo.shape()[1] == input_dim() && property.lo.shape()[0] > 0,
               Error::dimension_mismatch, [&] {
                   return "verify: the boxes must be (C, " + std::to_string(input_dim()) +
                          "), got " + shape_text(property.lo.shape()) + " and " +
                          shape_text(property.hi.shape());
               });
    CORA_CHECK(property.A.ndim() == 3 && property.A.shape()[2] == output_dim() &&
                   property.b.ndim() == 2 && property.b.shape()[0] == property.A.shape()[0] &&
                   property.b.shape()[1] == property.A.shape()[1] &&
                   int64_t(property.poly_case.size()) == property.A.shape()[0],
               Error::dimension_mismatch, [&] {
                   return "verify: the polytopes must be A (P, K, " + std::to_string(output_dim()) +
                          ") and b (P, K) with a case each, got " + shape_text(property.A.shape()) +
                          " and " + shape_text(property.b.shape());
               });
    // every polytope belongs to a case, and the timeout is a duration
    const int64_t C = property.lo.shape()[0];
    for (const int64_t k : property.poly_case)
        CORA_CHECK(k >= 0 && k < C, Error::wrong_value, [&] {
            return "verify: a polytope of the case " + std::to_string(k) + " of " +
                   std::to_string(C) + " cases";
        });
    CORA_CHECK(timeout >= 0 && !std::isnan(timeout), Error::wrong_value,
               [] { return std::string("verify: the timeout must not be negative"); });
    options.validate();
    NnProblem problem(*this, property, options, confirm);
    return aux_verify(problem, start, timeout);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
