// priv_verify_options - the check of verify's options, and CORA's VNN-COMP preset of each benchmark
//
// validate refuses an unknown strategy name (the registries' names) or option value, as CORA's
// validateNNoptions. vnncomp_options is CORA's getDefaultVNNCOMPoptions (examples/nn/vnncomp):
// the bounding "conzonotope" with zonotack falsification, input splits by zono-norm-gradient and
// the refinement of Prop. 2, then the benchmark's own changes (generator slots, neuron splits,
// ReLU rows, batch sizes). conzonotope_bounding_method "dual-iter" tightens the box of the inputs
// only, which the refinement does not read, so the factor box of Fourier-Motzkin decides alone.
//
// Syntax:   NnVerifyOptions o = vnncomp_options("cora_2024");   o.validate();
// See also: cora/nn/neural_network/verify.h, verify.cpp
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "nn/neural_network/private/priv_verify.h"

#include <algorithm>
#include <limits>
#include <regex>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Checks ------------------------------------------------------------------------------------------

/// Fails unless value is one of the known ones.
void aux_one_of(const std::string &field, const std::string &value,
                const std::vector<std::string> &known) {
    if (std::find(known.begin(), known.end(), value) != known.end()) return;
    std::string list;
    for (const std::string &k : known) list += (list.empty() ? "" : ", ") + k;
    CORA_FAIL(Error::wrong_field_value,
              "NnVerifyOptions: the " + field + " '" + value + "' is unknown; known: " + list);
}

/// Fails unless the count is at least `least`.
void aux_at_least(const std::string &field, double value, double least) {
    if (!(value >= least))
        CORA_FAIL(Error::wrong_field_value,
                  "NnVerifyOptions: " + field + " must be at least " + std::to_string(least));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The check of the options ------------------------------------------------------------------------

void NnVerifyOptions::validate() const {
    detail::Entry entry;
    // the strategies by the registries' names
    aux_one_of("falsification_method", falsification_method, attack_names());
    aux_one_of("bounding_method", bounding_method, bounding_names());
    const bool latent = bounding_method != "zonotope";
    aux_one_of("input_split_heuristic", input_split_heuristic,
               latent ? heuristic_names() : input_split_heuristic_names());
    std::vector<std::string> neuron = heuristic_names();
    neuron.emplace_back("none");
    if (latent) {
        aux_one_of("neuron_split_heuristic", neuron_split_heuristic, neuron);
        aux_one_of("relu_constraint_heuristic", relu_constraint_heuristic, neuron);
        aux_one_of("input_generator_heuristic", input_generator_heuristic, neuron);
    } else {
        if (neuron_split_heuristic != "none")
            CORA_FAIL(Error::wrong_field_value,
                      "NnVerifyOptions: neuron splits need the latent-space verifier "
                      "(bounding_method 'conzonotope')");
    }
    // the values of the other switches
    aux_one_of("falsification_order", falsification_order, {"attack-first", "bound-first"});
    aux_one_of("verify_dequeue_type", verify_dequeue_type, {"front", "back", "half-half"});
    aux_one_of("verify_enqueue_type", verify_enqueue_type, {"prepend", "append"});
    aux_one_of("refinement_method", refinement_method, {"zonotack", "naive", "relu-bab"});
    aux_one_of("relu_bab_order", relu_bab_order, {"output", "error"});
    aux_one_of("bab_falsification_method", bab_falsification_method,
               {"zonotack", "center", "fgsm"});
    aux_one_of("approx_error_order", approx_error_order,
               {"sensitivity*length", "length", "sequential"});
    aux_one_of("poly_method", poly_method, {"bounds", "singh"});
    aux_one_of("conzonotope_bounding_method", conzonotope_bounding_method,
               {"fourier-motzkin", "dual-iter"});
    // the switches of the splits, the rows and Phase 2's bounds
    aux_one_of("neuron_split_position", neuron_split_position, {"zero", "middle"});
    aux_one_of("fourier_motzkin_order", fourier_motzkin_order, {"cora", "coralean"});
    aux_one_of("alpha_method", alpha_method, {"zonotope", "crown"});
    aux_one_of("relu_constraint_faces", relu_constraint_faces, {"cora", "triangle"});
    // the counts
    aux_at_least("num_splits", double(num_splits), 1);
    aux_at_least("num_dimensions", double(num_dimensions), 0);
    aux_at_least("num_neuron_splits", double(num_neuron_splits), 0);
    aux_at_least("num_relu_constraints", num_relu_constraints, 0);
    aux_at_least("num_init_gens", num_init_gens, 1);
    aux_at_least("num_approx_err", num_approx_err, 0);
    aux_at_least("mini_batch_size", double(mini_batch_size), 1);
    aux_at_least("batch_memory_budget", double(batch_memory_budget), 0);
    // relu-bab's
    aux_at_least("relu_bab_dual_steps", double(relu_bab_dual_steps), 0);
    aux_at_least("relu_bab_witnesses", double(relu_bab_witnesses), 0);
    aux_at_least("relu_bab_frontier_bytes", double(relu_bab_frontier_bytes), 0);
    aux_at_least("relu_bab_memory_budget", double(relu_bab_memory_budget), 0);
    aux_at_least("relu_bab_init_gens", relu_bab_init_gens, 1);
    aux_at_least("relu_bab_faces", double(relu_bab_faces), 0);
    aux_at_least("relu_bab_root_steps", double(relu_bab_root_steps), 0);
    aux_at_least("relu_bab_farkas_steps", double(relu_bab_farkas_steps), 0);
    aux_at_least("relu_bab_alpha_iterations", double(relu_bab_alpha_iterations), 0);
    // CROWN's bounds of the layer inputs: relu-bab's root and the check
    aux_at_least("relu_bab_crown_rounds", double(relu_bab_crown_rounds), 0);
    aux_at_least("relu_bab_crown_iterations", double(relu_bab_crown_iterations), 0);
    aux_at_least("relu_bab_crown_step", relu_bab_crown_step, 0);
    aux_at_least("check_crown_rounds", double(check_crown_rounds), 0);
    aux_at_least("check_crown_iterations", double(check_crown_iterations), 0);
    aux_at_least("refinement_max_iter", double(refinement_max_iter), 1);
    aux_at_least("refinement_min_move", refinement_min_move, 0);
    aux_at_least("polytope_bound_approx_max_iter", double(polytope_bound_approx_max_iter), 0);
    aux_at_least("max_candidates", double(max_candidates), 1);
    aux_at_least("pgd_min_steps", double(pgd_min_steps), 1);
    aux_at_least("pgd_steps", double(pgd_steps), 1);
    aux_at_least("pgd_restarts", double(pgd_restarts), 1);
    // the counts of Phase 2's steps
    aux_at_least("alpha_iterations", double(alpha_iterations), 0);
    aux_at_least("look_ahead_candidates", double(look_ahead_candidates), 1);
    aux_at_least("alpha_patience", double(alpha_patience), 1);
    aux_at_least("optimize_gap", optimize_gap, 0);
    aux_at_least("dual_iterations", double(dual_iterations), 0);
    aux_at_least("dual_relu_rows", double(dual_relu_rows), 0);
    for (const double w : pgd_sign_windows)
        if (!(w > 0))
            CORA_FAIL(Error::wrong_field_value,
                      "NnVerifyOptions: pgd_sign_windows must be positive");
    if (!(dual_step_size > 0))
        CORA_FAIL(Error::wrong_field_value, "NnVerifyOptions: dual_step_size must be positive");
    if (!(alpha_step_size > 0))
        CORA_FAIL(Error::wrong_field_value, "NnVerifyOptions: alpha_step_size must be positive");
    if (!(attack_time_share >= 0 && attack_time_share <= 1))
        CORA_FAIL(Error::wrong_field_value, "NnVerifyOptions: attack_time_share must be in [0, 1]");
}

// CORA's presets ----------------------------------------------------------------------------------

NnVerifyOptions cora_vnncomp_options(const std::string &category) {
    detail::Entry entry;
    const std::string name = std::regex_replace(category, std::regex("[_-](20[0-9][0-9])$"), "");
    const double inf = std::numeric_limits<double>::infinity();
    NnVerifyOptions o;
    // CORA's defaults of getDefaultVNNCOMPoptions
    o.bounding_method = "conzonotope";
    o.refinement_method = "zonotack";
    o.bab_falsification_method = "zonotack";
    o.approx_error_order = "sensitivity*length";
    o.conzonotope_bounding_method = "dual-iter";
    o.polytope_bound_approx_max_iter = 3;
    o.input_generator_heuristic = o.input_split_heuristic = o.neuron_split_heuristic =
        o.relu_constraint_heuristic = "zono-norm-gradient";
    o.verify_dequeue_type = "half-half";
    // the benchmark's own changes
    if (name == "cgan") {
        o.num_neuron_splits = 1;
        o.mini_batch_size = 4;
        o.max_verif_iter = 100;
        o.neuron_split_heuristic = "least-unstable";
    } else if (name == "cifar100") {
        o.mini_batch_size = 8;
        o.neuron_split_heuristic = "least-unstable";
        o.bab_falsification_method = "fgsm";
        o.num_neuron_splits = 1;
        o.num_relu_constraints = 100;
        o.num_init_gens = 250;
        o.num_approx_err = 25;
        o.max_verif_iter = 1000;
        o.verify_enqueue_type = "append";
        // the benchmarks with fewer error slots or more splits
    } else if (name == "collins_rul_cnn") {
        o.num_approx_err = 100;
    } else if (name == "cora") {
        o.num_relu_constraints = inf;
        o.mini_batch_size = 32;
        o.num_neuron_splits = 1;
    } else if (name == "metaroom") {
        o.num_init_gens = 500;
        o.num_approx_err = 100;
        o.mini_batch_size = 4;
        o.max_verif_iter = 10;
        // the image benchmarks: fewer input generators
    } else if (name == "mnist_fc") {
        o.num_init_gens = 500;
        o.num_approx_err = 100;
    } else if (name == "oval21") {
        o.num_init_gens = 100;
        o.num_approx_err = 10;
    } else if (name == "safenlp") {
        o.num_neuron_splits = 1;
        o.num_relu_constraints = 100;
    } else if (name == "tinyimagenet") {
        o.num_init_gens = 500;
        o.num_approx_err = 0;
        o.num_neuron_splits = 1;
        o.mini_batch_size = 32;
    }
    return o;
}

// The competition preset --------------------------------------------------------------------------

NnVerifyOptions vnncomp_options(const std::string &category) {
    detail::Entry entry;
    const std::string name = std::regex_replace(category, std::regex("[_-](20[0-9][0-9])$"), "");
    // CORA's preset, then Phase 2's bounds as measured on the dev sets (docs/nn_verification.md)
    NnVerifyOptions o = cora_vnncomp_options(category);
    o.bounding_method = "zonotope-alpha";
    o.backward_bound = true;
    // acasxu's few inputs: the zonotope's slopes, which move every later bound; else alpha-CROWN
    o.alpha_method = name == "acasxu" ? "zonotope" : "crown";
    // sat_relu: CORA's bounding alone (docs/nn_verification.md, the dev seeds' scores)
    if (name == "sat_relu") {
        o.bounding_method = "conzonotope";
        o.backward_bound = false;
    }
    // the wide conv nets whose alpha-CROWN steps cost more than they decide (the dev sets)
    if (name == "metaroom" || name == "soundnessbench" || name == "tinyimagenet")
        o.alpha_iterations = 0;
    // lsnc_relu: the refinement decides its boxes, the slopes none; children leave it at 0.03
    if (name == "lsnc_relu") {
        o.alpha_iterations = 0;
        o.refinement_min_move = 0.03;
    }
    // traffic_signs' binarized nets: more attack, alpha-beta-CROWN's sign windows
    if (name == "traffic_signs_recognition") {
        o.attack_time_share = 0.9;
        o.attack_first_rounds = 1000000; // the share, not three rounds
        o.pgd_sign_windows = {0.1, 5.0};
    }
    // relusplitter: the ReLU-split branch and bound (seed 213: 7 unsat of 12 against 5)
    if (name == "relusplitter") o.refinement_method = "relu-bab";
    // nn4sys: CORA's bounding of lindex' many small cases, the attack only briefly (dev seed 1001)
    if (name == "nn4sys") {
        o.bounding_method = "conzonotope";
        o.backward_bound = false;
        o.attack_time_share = 0.05;
    }
    // vit: all 3072 inputs a generator (32 GiB), no slope steps (docs/nn_verification.md)
    if (name == "vit") {
        o.memory_budget = int64_t(32) << 30;
        o.alpha_iterations = 0;
    }
    return o;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
