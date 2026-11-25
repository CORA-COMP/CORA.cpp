// verify - the verdict of a network on a VNN-COMP property: falsification and branch and bound
//
// verify runs the falsification method and the bounding method's branch and bound in the order of
// falsification_order (CORA's verify.m is the bounding "conzonotope"). attack-first: the attack
// for attack_time_share of the timeout (at most attack_first_rounds rounds), then the branch and
// bound; bound-first skips that. Either way an open verdict gets the attack again until the
// deadline. A network the bounding cannot take gets the attack alone. "sat" needs the confirm
// check of the counterexample (the scoring's, from Python; else the float64 network's output).
// Out of memory (std's or libtorch's) where the strategies' own back-offs end: the cache freed and
// that step without a verdict (no proof, no counterexample); the verdict then stays open.
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
// Last update:   25-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "global/tensor/device_memory.h"
#include "nn/neural_network/private/priv_layers.h"
#include "nn/neural_network/private/priv_verify.h"

#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The chunks of cases -----------------------------------------------------------------------------

/// The rows of polytopes per chunk of cases at most (NnProblem::case_chunks).
constexpr int64_t kCaseChunkRows = 256;

/// The branch and bound of the cases chunk by chunk: unsat if every chunk is, sat if one is.
std::string aux_chunks(NnProblem &problem, const std::vector<std::vector<int64_t>> &chunks,
                       double deadline, std::map<std::string, double> &stats, AttackResult &found) {
    const NnVerifyOptions &o = problem.options();
    for (const std::vector<int64_t> &chunk : chunks) {
        check_deadline(deadline);
        NnProblem sub = problem.subproblem(chunk);
        sub.deadline = deadline;
        const std::unique_ptr<Bounding> bounding = priv_make_bounding(o.bounding_method, o);
        if (!bounding->prepare(sub)) return "unknown";
        const std::string status = bounding->branch_and_bound(sub, deadline, stats, found);
        problem.checks += sub.checks;
        stats["case_chunks"] += 1;
        if (status != "unsat") return status;
    }
    return "unsat";
}

// Out of memory -----------------------------------------------------------------------------------

/// Whether error is out of memory: then the device's cache freed and the count of `what` in stats.
bool aux_out_of_memory(const NnProblem &problem, const std::exception &error,
                       std::map<std::string, double> &stats, const char *what) {
    if (!priv_is_oom(error)) return false;
    free_device_cache(problem.A.device_id());
    stats[what] += 1;
    return true;
}

// The order of the strategies ---------------------------------------------------------------------

/// The verdict of the problem within the deadline, for checked inputs.
NnVerifyResult aux_verify(NnProblem &problem, double start, double timeout) {
    const NnVerifyOptions &o = problem.options();
    const double deadline = start + timeout;
    NnVerifyResult result;
    std::unique_ptr<Attack> attack = make_attack(o.falsification_method, o);
    std::unique_ptr<Bounding> bounding = priv_make_bounding(o.bounding_method, o);
    bool can_bound = false;
    // many cases (lindex): chunk by chunk, the first one tells whether the bounding takes the net
    const std::vector<std::vector<int64_t>> chunks = problem.case_chunks(kCaseChunkRows);
    try {
        if (chunks.empty()) {
            can_bound = bounding->prepare(problem);
        } else {
            NnProblem first = problem.subproblem(chunks.front());
            can_bound = bounding->prepare(first);
        }
        if (!can_bound) result.note = bounding->why;
    } catch (const std::exception &error) {
        // a bound beyond the float range (an exp), or out of memory: the attack alone
        if (!priv_is_out_of_range(error) && !priv_is_oom(error))
            std::rethrow_exception(std::current_exception());
        aux_out_of_memory(problem, error, result.stats, "prepare_oom");
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
        } catch (const std::exception &error) {
            // the deadline, or out of memory: no counterexample from this step
            if (!is_deadline(error) && !priv_is_oom(error))
                std::rethrow_exception(std::current_exception());
            if (!aux_out_of_memory(problem, error, result.stats, "attack_oom"))
                result.stats["attack_cut"] += 1;
            found = AttackResult();
        }
        // its time and steps
        result.stats["attack_seconds"] += now_seconds() - t;
        result.stats["attack_iterations"] += double(found.iterations);
        return found;
    };
    // a long attack first: the bounding's first batch over the root boxes may decide it already
    if (o.root_check_first && o.falsification_order == "attack-first" && can_bound &&
        chunks.empty()) {
        NnVerifyOptions root = o;
        root.max_verif_iter = 1;
        const std::unique_ptr<Bounding> first = priv_make_bounding(root.bounding_method, root);
        AttackResult found;
        problem.deadline = deadline;
        std::string status = "unknown";
        try {
            if (first->prepare(problem)) {
                status = first->branch_and_bound(problem, deadline, result.stats, found);
            }
        } catch (const std::exception &error) {
            // the deadline, a bound beyond the float range or out of memory: the attack goes on
            if (!is_deadline(error) && !priv_is_out_of_range(error) && !priv_is_oom(error))
                std::rethrow_exception(std::current_exception());
            aux_out_of_memory(problem, error, result.stats, "root_check_oom");
            status = "unknown";
        }
        if (status == "sat") return sat(found);
        if (status == "unsat") {
            result.status = "unsat";
            result.stats["checks"] = double(problem.checks);
            return result;
        }
    }
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
            if (chunks.empty()) {
                status = bounding->branch_and_bound(problem, deadline, result.stats, found);
            } else {
                status = aux_chunks(problem, chunks, deadline, result.stats, found);
            }
        } catch (const std::exception &error) {
            // the deadline, a bound beyond the float range or out of memory: the verdict stays open
            if (!is_deadline(error) && !priv_is_out_of_range(error) && !priv_is_oom(error))
                std::rethrow_exception(std::current_exception());
            if (!aux_out_of_memory(problem, error, result.stats, "bab_oom"))
                result.stats[is_deadline(error) ? "bab_cut" : "bab_out_of_range"] += 1;
            status = "unknown";
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
    // the activations of this port's layers beyond VNN-COMP's: sets and points only
    for (const Activation a : activations_)
        CORA_CHECK(a != Activation::softplus && a != Activation::root &&
                       a != Activation::inv_sqrt_root && a != Activation::group_sort,
                   Error::not_supported, [] {
                       return std::string("verify: softplus, root, inv_sqrt_root and group_sort "
                                          "layers are not in the verifier's activations");
                   });
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
