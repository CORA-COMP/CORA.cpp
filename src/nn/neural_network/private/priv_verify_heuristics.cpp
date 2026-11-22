// priv_verify_heuristics - CORA's ten heuristics by name, and the registries of verify's strategies
//
// A heuristic scores every neuron (or input dimension) from its bounds l, u, its approximation
// error er, its sensitivity and the gradient of the output's F-radius (CORA's computeHeuristic);
// verify splits or constrains the best ones. The registries map an option name to a strategy;
// the built-ins are registered when a registry is first read, so a program's own registration
// (register_attack, ...) may replace one.
//
// Syntax:   Tensor h = priv_compute_heuristic("least-unstable", in, splits, first_id, true);
// See also: cora/nn/neural_network/verify.h
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"
#include "nn/neural_network/private/priv_verify.h"

#include <chrono>
#include <limits>
#include <mutex>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The registries, filled with the built-ins on first use.
struct Registries {
    std::map<std::string, AttackFactory> attacks;
    std::map<std::string, BoundingFactory> boundings;
    std::map<std::string, Heuristic> heuristics;
    std::map<std::string, InputSplitHeuristic> input_splits;
};

// The maps ----------------------------------------------------------------------------------------

std::mutex &aux_mutex() {
    static std::mutex m;
    return m;
}

/// The maps, and whether the built-ins are being registered (they never replace an entry).
Registries &aux_raw() {
    static Registries r;
    return r;
}
bool &aux_filling() {
    static bool filling = false;
    return filling;
}

/// The maps with the built-ins in them.
Registries &aux_registries() {
    static std::once_flag filled;
    std::call_once(filled, [] {
        aux_filling() = true;
        priv_register_heuristics();
        priv_register_attacks();
        priv_register_zonotack();
        priv_register_naive();
        aux_filling() = false;
    });
    return aux_raw();
}

/// Puts a strategy into its map: a program's replaces, a built-in only fills a free name.
template <class M, class V>
void aux_put(M &map, const std::string &name, V value) {
    const std::lock_guard<std::mutex> lock(aux_mutex());
    if (aux_filling())
        map.emplace(name, std::move(value));
    else
        map[name] = std::move(value);
}

/// The maps to register into: the built-ins first, unless they are being registered.
Registries &aux_target() { return aux_filling() ? aux_raw() : aux_registries(); }

/// The names of a map, sorted (a map is).
template <class M>
std::vector<std::string> aux_names(const M &map) {
    std::vector<std::string> names;
    names.reserve(map.size());
    for (const auto &[name, value] : map) names.push_back(name);
    return names;
}

// Heuristic parts ---------------------------------------------------------------------------------

/// The smaller distance of the bounds to zero, min(-l, u).
Tensor aux_unstable(const HeuristicInputs &in) { return in.l.negative().minimum(in.u); }

/// The magnitude of the gradient, kept off zero.
Tensor aux_grad(const HeuristicInputs &in) { return in.grad.abs() + numerics::kHeuristicGradient; }

} // namespace


// ===========================================  MAIN  =========================================== //

// The registries ----------------------------------------------------------------------------------

void register_attack(const std::string &name, AttackFactory factory) {
    aux_put(aux_target().attacks, name, std::move(factory));
}

// a program's strategy replaces a built-in of its name
void register_bounding(const std::string &name, BoundingFactory factory) {
    aux_put(aux_target().boundings, name, std::move(factory));
}

void register_heuristic(const std::string &name, Heuristic h) {
    aux_put(aux_target().heuristics, name, std::move(h));
}

void register_input_split_heuristic(const std::string &name, InputSplitHeuristic h) {
    aux_put(aux_target().input_splits, name, std::move(h));
}

// the names of each kind
std::vector<std::string> attack_names() { return aux_names(aux_registries().attacks); }
std::vector<std::string> bounding_names() { return aux_names(aux_registries().boundings); }
std::vector<std::string> heuristic_names() { return aux_names(aux_registries().heuristics); }
std::vector<std::string> input_split_heuristic_names() {
    return aux_names(aux_registries().input_splits);
}

std::unique_ptr<Attack> make_attack(const std::string &name, const NnVerifyOptions &options) {
    const Registries &r = aux_registries();
    // the attack of the name, or an error
    const auto it = r.attacks.find(name);
    if (it == r.attacks.end())
        CORA_FAIL(Error::wrong_field_value, "verify: the attack '" + name + "' is unknown");
    return it->second(options);
}

std::unique_ptr<Bounding> priv_make_bounding(const std::string &name,
                                             const NnVerifyOptions &options) {
    const Registries &r = aux_registries();
    const auto it = r.boundings.find(name);
    if (it == r.boundings.end())
        CORA_FAIL(Error::wrong_field_value,
                  "verify: the bounding_method '" + name + "' is unknown");
    return it->second(options);
}

const Heuristic &heuristic(const std::string &name) {
    const Registries &r = aux_registries();
    // the heuristic of the name, or an error
    const auto it = r.heuristics.find(name);
    if (it == r.heuristics.end())
        CORA_FAIL(Error::wrong_field_value, "verify: the heuristic '" + name + "' is unknown");
    return it->second;
}

const InputSplitHeuristic &priv_input_split_heuristic(const std::string &name) {
    const Registries &r = aux_registries();
    const auto it = r.input_splits.find(name);
    if (it == r.input_splits.end())
        CORA_FAIL(Error::wrong_field_value,
                  "verify: the input_split_heuristic '" + name + "' is unknown");
    return it->second;
}

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// The heuristics ----------------------------------------------------------------------------------

void priv_register_heuristics() {
    const auto add = [](const std::string &name, bool sens, bool grad,
                        std::function<Tensor(const HeuristicInputs &)> f) {
        register_heuristic(name, {std::move(f), sens, grad});
    };
    add("input-radius", false, false,
        [](const HeuristicInputs &in) { return (in.u - in.l) * 0.5; });
    add("least-unstable", true, false,
        [](const HeuristicInputs &in) { return in.sens.divide(aux_unstable(in)); });
    add("least-unstable-gradient", false, true,
        [](const HeuristicInputs &in) { return aux_grad(in).divide(aux_unstable(in)); });
    add("most-unstable", true, false,
        [](const HeuristicInputs &in) { return aux_unstable(in).multiply(in.sens); });
    // the gap of the ReLU's triangle, zero for a point range
    add("gap-sensitivity", true, false, [](const HeuristicInputs &in) {
        const Tensor gap = in.l.negative().multiply(in.u).divide(in.u - in.l);
        return Tensor::where(gap.isnan(), gap.zeros_like(), gap).multiply(in.sens);
    });
    // the products and quotients of the bounds, times the sensitivity
    add("product-sensitivity", true, false,
        [](const HeuristicInputs &in) { return in.l.negative().multiply(in.u).multiply(in.sens); });
    add("centered-sensitivity", true, false, [](const HeuristicInputs &in) {
        return (1.0 - (in.l + in.u).abs().divide(in.u - in.l)).multiply(in.sens);
    });
    add("most-sensitive-approx-error", true, false,
        [](const HeuristicInputs &in) { return in.er.multiply(in.sens); });
    add("most-sensitive-input-radius", true, false,
        [](const HeuristicInputs &in) { return ((in.u - in.l) * 0.5).multiply(in.sens); });
    // the gradient of the F-radius times the error
    add("zono-norm-gradient", false, true,
        [](const HeuristicInputs &in) { return aux_grad(in).multiply(in.er); });
}

Tensor priv_compute_heuristic(const std::string &name, const HeuristicInputs &in,
                              const Tensor &splits, int64_t first_id, bool only_unstable) {
    const Heuristic &h = heuristic(name);
    if ((h.sens && !in.sens.defined()) || (h.grad && !in.grad.defined()))
        CORA_FAIL(Error::wrong_value,
                  "verify: the heuristic '" + name + "' needs what was not computed");
    Tensor score = h.score(in);
    const Tensor none = score.full_like(score.shape(), -kInf);
    // a stable neuron and a neuron split before are not candidates
    if (only_unstable) {
        const Tensor zero = in.l.zeros_like();
        score = Tensor::where(in.l.less(zero).multiply(zero.less(in.u)), score, none);
    }
    if (splits.defined() && splits.shape()[1] > 0) {
        const int64_t n = score.shape()[1];
        const Tensor ids = (score.arange_like(n) + double(first_id)).reshape({1, 1, n});
        const Tensor was = splits.abs().expand_dims(-1).equal(ids).any(1);
        score = Tensor::where(was, none, score);
    }
    return score;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
