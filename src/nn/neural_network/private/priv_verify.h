// priv_verify - the shared steps of verify's strategies: candidates, heuristics, the registries
//
// The built-in strategies register themselves when a registry is first read (attacks: center,
// pgd, fgsm, zonotack; boundings: zonotope, conzonotope; CORA's ten neuron heuristics; the input
// split heuristics of the zonotope bounding). A candidate point is checked in the search dtype,
// clamped into its case's float32 box: those in the closed unsafe set, the least loss first and at
// most max_candidates of them, go to the problem's confirm check.
//
// Syntax:   priv_register_builtins();   bool found = priv_confirm_rows(problem, x, loss, result);
// See also: cora/nn/neural_network/verify.h
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Lukas Koller, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/nn/neural_network/verify.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Registers the built-in strategies of each file (called once by the registries).
void priv_register_attacks();
void priv_register_heuristics();
void priv_register_zonotack();
void priv_register_naive();

/// The bounding of a name and the input split heuristic of a name (wrong_field_value if unknown).
std::unique_ptr<Bounding> priv_make_bounding(const std::string &name,
                                             const NnVerifyOptions &options);
const InputSplitHeuristic &priv_input_split_heuristic(const std::string &name);

/// Whether the error is an allocation that failed (CORA's catch of out-of-memory errors).
bool priv_is_oom(const std::exception &error);

/// A point where the verifier may run out of memory: Error::out_of_memory where priv_inject_oom
/// armed `site`, else nothing (one relaxed load while none is armed).
void priv_oom_site(const char *site);

/// The tests' out of memory: the calls of priv_oom_site(site) after the next `skip` throw, `count`
/// of them (0: disarmed).
void priv_inject_oom(const std::string &site, int64_t skip, int64_t count);

/// The rows of x with loss <= 0, the least first and at most max_candidates, through the confirm
/// check; the first that passes goes into result (true).
bool priv_confirm_rows(NnProblem &problem, const Tensor &x, const Tensor &loss,
                       AttackResult &result);

/// One sign step of size alpha (broadcast to x) against the gradient g: x - alpha sign(g). PGD
/// and FGSM of verify take it, and so does NeuralNetwork::compute_fgsm_attack (alpha < 0 ascends).
Tensor priv_sign_step(const Tensor &x, const Tensor &g, const Tensor &alpha);

/// The candidates x (S, n) of the cases (S), clamped into their search boxes, checked.
bool priv_check_points(NnProblem &problem, const Tensor &x, const Tensor &cases,
                       AttackResult &result);

/// CORA's computeHeuristic: the scores; -inf for a stable neuron (only_unstable) and for a neuron
/// split before (its id in splits (B, s), the layer's ids from first_id; undefined: none).
Tensor priv_compute_heuristic(const std::string &name, const HeuristicInputs &in,
                              const Tensor &splits, int64_t first_id, bool only_unstable);

/// The rows (..., S) of the indices of the flat positions in `flat` of a (rows, cols) grid.
std::vector<int64_t> priv_rows_of(const std::vector<int64_t> &flat, int64_t cols);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
