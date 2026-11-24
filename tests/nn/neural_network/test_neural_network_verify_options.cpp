// test_neural_network_verify_options - the options of verify and its registries of strategies
//
// The options refuse unknown strategy names and option values, cora_vnncomp_options gives CORA's
// presets (the year of a benchmark dropped) and vnncomp_options adds Phase 2's bounds, the built-in
// strategies are registered, and a program's own attack is used by its name. Every heuristic scores
// every entry, and an out-of-memory error is recognized.
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/nn/neural_network/verify.h"
#include "nn/neural_network/private/priv_verify.h"
#include "testing.h"

#include <cmath>
#include <new>
#include <stdexcept>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A random network of the sizes with the activation after each hidden layer.
NeuralNetwork aux_network(Rng &rng, const std::vector<int64_t> &sizes, Activation hidden) {
    std::vector<NeuralNetwork::Layer> layers;
    std::vector<Activation> acts;
    for (std::size_t k = 0; k + 1 < sizes.size(); ++k) {
        std::vector<double> W(static_cast<std::size_t>(sizes[k + 1] * sizes[k])),
            b(static_cast<std::size_t>(sizes[k + 1]));
        rng.normal(W.data(), W.size(), 1.0 / std::sqrt(double(sizes[k])));
        rng.normal(b.data(), b.size(), 0.3);
        layers.push_back(
            {Tensor::from_data(W, {sizes[k + 1], sizes[k]}), Tensor::from_data(b, {sizes[k + 1]})});
        acts.push_back(k + 2 < sizes.size() ? hidden : Activation::identity);
    }
    return {layers, acts};
}

/// The box [lo, hi]^n with the unsafe set y_0 <= t.
NnProperty aux_below(int64_t n, int64_t m, double lo, double hi, double t) {
    NnProperty p;
    p.lo = Tensor::from_data(std::vector<double>(std::size_t(n), lo), {1, n});
    p.hi = Tensor::from_data(std::vector<double>(std::size_t(n), hi), {1, n});
    std::vector<double> A(std::size_t(m), 0.0);
    A[0] = 1.0;
    p.A = Tensor::from_data(A, {1, 1, m});
    p.b = Tensor::from_data(std::vector<double>{t}, {1, 1});
    p.poly_case = {0};
    return p;
}

/// The options refuse unknown names, the presets are CORA's, a registered attack is used.
void options_and_strategies() {
    NnVerifyOptions o;
    o.falsification_method = "nope";
    check(test::raises([&] { o.validate(); }, Error::wrong_field_value), "an unknown attack");
    o = NnVerifyOptions{};
    o.bounding_method = "nope";
    check(test::raises([&] { o.validate(); }, Error::wrong_field_value), "an unknown bounding");
    o = NnVerifyOptions{};
    o.neuron_split_heuristic = "least-unstable";
    check(test::raises([&] { o.validate(); }, Error::wrong_field_value),
          "neuron splits need the conzonotope bounding");
    // every other switch refuses an unknown value, each field set wrong on its own
    for (const std::string field :
         {"order", "dequeue", "enqueue", "refinement", "fm", "position"}) {
        NnVerifyOptions w = vnncomp_options("x");
        if (field == "order") w.falsification_order = "x";
        if (field == "dequeue") w.verify_dequeue_type = "x";
        if (field == "enqueue") w.verify_enqueue_type = "x";
        if (field == "refinement") w.refinement_method = "x";
        if (field == "fm") w.fourier_motzkin_order = "x";
        if (field == "position") w.neuron_split_position = "x";
        check(test::raises([&] { w.validate(); }, Error::wrong_field_value), "an unknown " + field);
    }
    // the counts of Phase 2, PGD's steps, the batch's budget and the refinement's least move
    for (const std::string field : {"alpha_iterations", "alpha_step_size", "candidates",
                                    "pgd_min_steps", "batch_memory_budget", "min_move"}) {
        NnVerifyOptions w = vnncomp_options("x");
        w.bounding_method = "zonotope-alpha";
        // one count out of range
        if (field == "alpha_iterations") w.alpha_iterations = -1;
        if (field == "alpha_step_size") w.alpha_step_size = 0;
        if (field == "candidates") w.look_ahead_candidates = 0;
        if (field == "pgd_min_steps") w.pgd_min_steps = 0;
        if (field == "batch_memory_budget") w.batch_memory_budget = -1;
        if (field == "min_move") w.refinement_min_move = -0.1;
        check(test::raises([&] { w.validate(); }, Error::wrong_field_value), "a wrong " + field);
    }
    vnncomp_options("cora_2024").validate();
    const NnVerifyOptions cora = cora_vnncomp_options("cora_2024"),
                          safe = cora_vnncomp_options("safenlp");
    check(cora.bounding_method == "conzonotope" && !cora.backward_bound,
          "CORA's presets stay CORA's");
    // the competition's preset adds Phase 2's bounds to CORA's
    const NnVerifyOptions ours = vnncomp_options("cora_2024"),
                          acas = vnncomp_options("acasxu_2023");
    check(ours.bounding_method == "zonotope-alpha" && ours.backward_bound &&
              ours.alpha_method == "crown" && acas.alpha_method == "zonotope" &&
              ours.num_neuron_splits == cora.num_neuron_splits,
          "the VNN-COMP preset: CORA's with Phase 2's bounds");
    // the benchmarks' own changes of it (docs/nn_verification.md)
    const NnVerifyOptions sat = vnncomp_options("sat_relu");
    check(sat.bounding_method == "conzonotope" && !sat.backward_bound,
          "sat_relu's preset: CORA's bounding without CROWN");
    check(vnncomp_options("traffic_signs_recognition_2023").attack_time_share == 0.9 &&
              ours.attack_time_share == cora.attack_time_share,
          "traffic_signs' preset: nine tenths of the time for the attack");
    check(vnncomp_options("traffic_signs_recognition_2023").pgd_sign_windows.size() == 2 &&
              ours.pgd_sign_windows == std::vector<double>{1.0},
          "traffic_signs' preset: the tight and the loose sign window");
    // nn4sys: lindex's many cases
    const NnVerifyOptions sys = vnncomp_options("nn4sys");
    check(sys.bounding_method == "conzonotope" && !sys.backward_bound &&
              sys.attack_time_share == 0.05,
          "nn4sys' preset: CORA's bounding without CROWN, a brief attack");
    check(vnncomp_options("tinyimagenet_2024").alpha_iterations == 0 &&
              vnncomp_options("metaroom_2023").alpha_iterations == 0 &&
              vnncomp_options("soundnessbench_2026").alpha_iterations == 0 &&
              vnncomp_options("cifar100_2024").alpha_iterations == ours.alpha_iterations,
          "the wide conv nets' preset: no alpha-CROWN steps, except cifar100");
    // lsnc_relu's preset
    check(vnncomp_options("lsnc_relu").alpha_iterations == 0 &&
              vnncomp_options("lsnc_relu").refinement_min_move == 0.03 &&
              ours.refinement_min_move == 0,
          "lsnc_relu's preset: no alpha steps, a refinement tolerance");
    check(cora.num_neuron_splits == 1 && std::isinf(cora.num_relu_constraints) &&
              cora.mini_batch_size == 32 && safe.num_relu_constraints == 100,
          "CORA's presets, the year dropped");
    check(heuristic_names().size() == 11 && attack_names().size() >= 4 &&
              bounding_names().size() >= 3 && input_split_heuristic_names().size() >= 2,
          "the built-ins are registered");
    // a registered attack replaces the search
    struct Fixed : Attack {
        AttackResult run(NnProblem &problem, double, int64_t) override {
            AttackResult r;
            r.x = problem.lo.slice(0, 0, 1).squeeze(0).data();
            r.y = problem.search_net().evaluate(problem.lo.slice(0, 0, 1)).data();
            return r;
        }
    };
    register_attack("fixed", [](const NnVerifyOptions &) { return std::make_unique<Fixed>(); });
    Rng rng(1);
    const NeuralNetwork nn = aux_network(rng, {2, 3, 2}, Activation::relu);
    NnVerifyOptions mine;
    mine.falsification_method = "fixed";
    check(nn.verify(aux_below(2, 2, 0, 1, 0), 5, mine).status == "sat", "the registered attack");
}

/// Every benchmark's preset is valid and has its own changes.
void presets() {
    const NnVerifyOptions cgan = vnncomp_options("cgan_2023"), c100 = vnncomp_options("cifar100");
    const NnVerifyOptions rul = vnncomp_options("collins_rul_cnn_2022");
    const NnVerifyOptions room = vnncomp_options("metaroom_2023"), fc = vnncomp_options("mnist_fc");
    const NnVerifyOptions oval = vnncomp_options("oval21"), tiny = vnncomp_options("tinyimagenet");
    for (const NnVerifyOptions *o : {&cgan, &c100, &rul, &room, &fc, &oval, &tiny}) o->validate();
    check(cgan.max_verif_iter == 100 && c100.verify_enqueue_type == "append" &&
              rul.num_approx_err == 100 && room.max_verif_iter == 10 && fc.num_init_gens == 500 &&
              oval.num_approx_err == 10 && tiny.num_approx_err == 0,
          "each benchmark's preset");
    // a share of the timeout beyond the whole
    NnVerifyOptions share;
    share.attack_time_share = 2;
    check(test::raises([&] { share.validate(); }, Error::wrong_field_value),
          "attack_time_share beyond 1");
}

/// Every heuristic scores every entry; unknown names and missing inputs fail.
void heuristics_and_registries() {
    // the bounds l < 0 < u, positive errors, sensitivities and gradients
    Rng rng(4);
    std::vector<double> v(12);
    rng.uniform(v.data(), v.size(), 0.1, 1.0);
    const Tensor pos = Tensor::from_data(v, {2, 6});
    const HeuristicInputs in{pos.negative(), pos * 2.0, pos * 0.5, pos + 1.0, pos - 0.5};
    for (const std::string &name : heuristic_names()) {
        const Tensor h = priv_compute_heuristic(name, in, Tensor(), 0, false);
        check(h.shape() == pos.shape() && h.isfinite().all(1).min().item() > 0,
              "the heuristic " + name + " scores every entry");
    }
    // a heuristic needs what it reads, and unknown names fail
    const HeuristicInputs bare{pos.negative(), pos, pos, Tensor(), Tensor()};
    check(test::raises(
              [&] { priv_compute_heuristic("zono-norm-gradient", bare, Tensor(), 0, false); },
              Error::wrong_value),
          "a heuristic without its gradient");
    const NnVerifyOptions o;
    check(test::raises([&] { make_attack("nope", o); }, Error::wrong_field_value) &&
              test::raises([&] { heuristic("nope"); }, Error::wrong_field_value) &&
              test::raises([&] { priv_make_bounding("nope", o); }, Error::wrong_field_value) &&
              test::raises([&] { priv_input_split_heuristic("nope"); }, Error::wrong_field_value),
          "unknown strategy names");
    // the errors that mean out of memory
    check(priv_is_oom(std::bad_alloc()) && priv_is_oom(std::runtime_error("CUDA out of memory")) &&
              priv_is_oom(std::runtime_error("CORA:outOfMemory")) &&
              !priv_is_oom(std::runtime_error("a shape mismatch")),
          "out of memory is recognized");
}

/// The options are the same with the checks off.
void checks_and_their_switch() {
    NnVerifyOptions o = vnncomp_options("cora");
    o.poly_method = "x";
    check(test::raises([&] { o.validate(); }, Error::wrong_field_value), "an unknown poly_method");
    o.num_splits = 0;
    ChecksScope off(false);
    check(vnncomp_options("cora").num_neuron_splits == 1, "the presets with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    options_and_strategies();
    presets();
    heuristics_and_registries();
    checks_and_their_switch();
    return test::finish("neuralNetwork verify options");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
