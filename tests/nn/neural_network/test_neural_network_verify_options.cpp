// test_neural_network_verify_options - the options of verify and its registries of strategies
//
// The options refuse unknown strategy names and option values, vnncomp_options gives CORA's
// presets (the year of a benchmark dropped), the built-in strategies are registered, and a
// program's own attack is used by its name.
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/nn/neural_network/verify.h"
#include "testing.h"

#include <cmath>

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
    // every other switch refuses an unknown value
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
    vnncomp_options("cora_2024").validate();
    const NnVerifyOptions cora = vnncomp_options("cora_2024"), safe = vnncomp_options("safenlp");
    check(cora.num_neuron_splits == 1 && std::isinf(cora.num_relu_constraints) &&
              cora.mini_batch_size == 32 && safe.num_relu_constraints == 100,
          "CORA's presets, the year dropped");
    check(heuristic_names().size() == 10 && attack_names().size() >= 4 &&
              bounding_names().size() >= 2 && input_split_heuristic_names().size() >= 2,
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
    checks_and_their_switch();
    return test::finish("neuralNetwork verify options");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
