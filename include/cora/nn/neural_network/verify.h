// verify - the options, the property, the result and the strategies of NeuralNetwork::verify
//
// A VNN-COMP instance is a network and a property: per case an input box and an unsafe union of
// polytopes A y <= b. verify answers "sat" (a counterexample the confirm check accepts), "unsat"
// (every box verified) or "unknown" (the deadline). It runs a falsification method and a bounding
// method in the order of falsification_order; every strategy is chosen by name from a registry
// (CORA's options.nn, validateNNoptions.m), and a program may register its own: an Attack, a
// Bounding (its own branch and bound), a neuron heuristic (CORA's computeHeuristic) or an input
// split heuristic. The search runs in NnVerifyOptions::dtype (float32) inside the float32 boxes,
// the bounds in float64 over the outer boxes; a box is verified when, for every polytope of its
// case, some row's lower bound exceeds verified_margin. One set fits memory_budget (bytes: fewer
// error slots and input generators otherwise), and a batch of sets batch_memory_budget.
//
// Syntax:   NnVerifyResult r = nn.verify(property, timeout, vnncomp_options("acasxu_2023"));
//           register_attack("mine", factory);
// See also: cora/nn/neural_network/neural_network.h, docs/nn_verification.md
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

#pragma once

#include "cora/global/tensor/tensor.h"
#include "cora/global/tensor/tolerance.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class NeuralNetwork;

// The property ------------------------------------------------------------------------------------

/// The property of an instance, its polytopes stacked: case k has the input box [lo_k, hi_k], and
/// polytope p (of case poly_case[p]) is the unsafe set A_p y <= b_p; a padding row is 0 <= +inf.
struct NnProperty {
    Tensor lo, hi;                  ///< (C, n), float64
    std::vector<int64_t> poly_case; ///< (P)
    Tensor A, b;                    ///< (P, K, m) and (P, K), float64
};

/// The check of a counterexample: its output if the point x (a float32 input as float64 values)
/// passes it, else nothing. Unset: the network's float64 output lies in a polytope of a case.
using ConfirmFn = std::function<std::optional<std::vector<double>>(const std::vector<double> &x)>;

// The options -------------------------------------------------------------------------------------

/// The options of verify, named after CORA's options.nn where one exists, with CORA's defaults.
struct NnVerifyOptions {
    // the order and the falsification around the branch and bound
    std::string falsification_method = "pgd";         ///< or "fgsm", "center", "zonotack"
    std::string falsification_order = "attack-first"; ///< or "bound-first"
    double attack_time_share = 0.3;                   ///< of the timeout at most, for attack-first
    int64_t attack_first_rounds = 3; ///< rounds of restarts at most, for attack-first
    int64_t pgd_steps = 100;
    int64_t pgd_restarts = 64;
    bool pgd_targeted = false;   ///< restarts per polytope of a box's case, each toward its own one
    double pgd_step_size = 0.25; ///< times the box radius
    double pgd_decay = 0.5;
    std::vector<double> pgd_decay_at = {0.5, 0.75}; ///< fractions of the steps
    /// The step of each restart as a share of the box radius, cycled over a box's restarts (a
    /// basin only a certain step reaches, alpha-beta-CROWN's multi-alpha PGD); empty:
    /// pgd_step_size.
    std::vector<double> pgd_step_sizes = {};
    /// The factor of the step at every step (alpha-beta-CROWN's pgd_lr_decay), besides pgd_decay.
    double pgd_step_decay = 1.0;
    /// The sign step takes the sign of the running mean of the gradients with this momentum (the
    /// sign of Adam's step, alpha-beta-CROWN's attack): 0 is the sign of the gradient itself.
    double pgd_momentum = 0.0;
    /// Hops after each round (alpha-beta-CROWN's iterated local search): the pgd_hop_keep best
    /// points jump by a random share of the radius (pgd_hop_jumps, in turn) and descend again for
    /// pgd_hop_steps steps, a decoy basin's way out; 0: none.
    int64_t pgd_hops = 0;
    int64_t pgd_hop_steps = 800;
    int64_t pgd_hop_keep = 128;
    std::vector<double> pgd_hop_jumps = {0.1, 0.2, 0.4, 0.6};
    /// The polish after each round and hop that comes within 1e-4 of the unsafe set: Adam's steps
    /// of pgd_polish_step times the radius (decaying 0.999 a step) from the pgd_polish_keep best
    /// points, which settle a point into the basin that the sign steps oscillate around; 0: none.
    int64_t pgd_polish_steps = 0;
    double pgd_polish_step = 0.0006;
    int64_t pgd_polish_keep = 8;
    int64_t pgd_polish_every = 1; ///< the hops between polishes (after each round, always)
    /// The half-widths of the sign activation's straight-through gradient (1 on |z| <= w), one per
    /// round of restarts in turn (alpha-beta-CROWN's tight 0.1 and loose 5 SignMerge on
    /// traffic_signs); the candidates are always checked on the true network.
    std::vector<double> pgd_sign_windows = {1.0};
    /// attack-first: the bounding's first batch over the root boxes comes before the attack (an
    /// unsat the bounds decide at once is not left to a long attack).
    bool root_check_first = false;
    int64_t max_candidates = 8; ///< confirm checks per attack step
    /// PGD's restarts shrink until this many steps fit in the attack's time (a slow, wide network).
    int64_t pgd_min_steps = 10;
    uint64_t seed = 0;
    // the bounding and its splits
    std::string bounding_method = "zonotope"; ///< "conzonotope" (CORA's verify), "zonotope-alpha"
    std::string input_split_heuristic = "most-sensitive-input-radius";
    std::string neuron_split_heuristic = "none";
    int64_t num_splits = 2;                      ///< pieces per split
    int64_t num_dimensions = 1;                  ///< input dimensions split per set
    std::string verify_dequeue_type = "front";   ///< "front", "back" (zonotope) or "half-half"
    std::string verify_enqueue_type = "prepend"; ///< or "append"
    int64_t max_verif_iter = std::numeric_limits<int64_t>::max();
    double verified_margin = numerics::kVerifiedMargin; ///< float32 net vs float64 bounds
    int64_t memory_budget = 0;                          ///< bytes per set; 0: 1 GiB
    int64_t batch_memory_budget = 0; ///< bytes per batch of sets; 0: memory_budget
    std::string device;              ///< where the network goes ("cpu", "cuda"); empty: its own
    DType dtype = DType::float32;    // NOLINT(backend): the search dtype, an option of verify
    // CORA's verify (bounding_method "conzonotope"), CORA's names
    std::string refinement_method = "zonotack";        ///< or "naive"
    std::string bab_falsification_method = "zonotack"; ///< or "center", "fgsm"
    double num_init_gens = std::numeric_limits<double>::infinity();
    double num_approx_err = std::numeric_limits<double>::infinity();
    std::string approx_error_order = "sensitivity*length";       ///< or "length", "sequential"
    std::string poly_method = "bounds";                          ///< or "singh" (CORALean's)
    std::string conzonotope_bounding_method = "fourier-motzkin"; ///< or "dual-iter" (the same)
    int64_t polytope_bound_approx_max_iter = 8;                  ///< Fourier-Motzkin passes
    int64_t refinement_max_iter = 8; ///< propagations of Prop. 2 per set, plus one
    /// A child leaves Prop. 2's passes once its input box moves by at most this much in a pass, in
    /// the factors' [-1, 1] (CORA.cpp only; 0: only a box that did not move, as CORA's results).
    double refinement_min_move = 0;
    int64_t num_neuron_splits = 0;
    std::string neuron_split_position = "zero"; ///< or "middle"
    double num_relu_constraints = 0;
    std::string relu_constraint_heuristic = "most-sensitive-approx-error";
    /// The faces of a ReLU row: "cora" (ReLU(x) >= 0, >= x) or "triangle" (also <= its chord).
    std::string relu_constraint_faces = "cora";
    std::string input_generator_heuristic = "most-sensitive-input-radius";
    int64_t mini_batch_size = 1024;
    /// The heuristics' F-radius gradient through the slopes, offsets and errors of the lines
    /// (CORA's options.nn.train.exact_backprop); false holds the slopes.
    bool exact_backprop = true;
    /// "cora": a row's bound of a factor is the row's sum minus its term (fast); "coralean": the
    /// sum of the other terms in CORALean's order (contractHalfspace, bit-exact on lean).
    std::string fourier_motzkin_order = "cora";
    /// The check also takes CROWN's backward bound over the same relaxation, the better of the two.
    bool backward_bound = false;
    /// The check also takes the dual certificate: per row a Lagrangian bound under the splits'
    /// rows and the three faces of dual_relu_rows crossing ReLUs (beta-CROWN's multipliers).
    bool dual_certificate = false;
    int64_t dual_iterations = 20; ///< projected Adam steps on the multipliers
    double dual_step_size = 0.1;  ///< Adam's step, times the row's largest coefficient
    int64_t dual_relu_rows = 32;  ///< crossing ReLUs whose faces are rows, by their error
    // refinement_method "relu-bab": ReLU splits on the frozen root zonotope (nn_verification.md)
    int64_t relu_bab_dual_steps = 20;    ///< dual steps per node, from its parent's multipliers
    int64_t relu_bab_witnesses = 32;     ///< nodes per depth whose factors are confirm candidates
    int64_t relu_bab_frontier_bytes = 0; ///< the open nodes' bytes at most; 0: 1/4 batch budget
    int64_t relu_bab_memory_budget = 0;  ///< the root set's bytes; 0: 3/4 of the batch budget
    double relu_bab_init_gens = std::numeric_limits<double>::infinity(); ///< the root's at most
    std::string relu_bab_order = "output"; ///< the splits: "output" (|w| at the slot), "error"
    int64_t relu_bab_faces = 64; ///< the next unsplit neurons' triangle faces as rows of a node
    int64_t relu_bab_root_steps = 100;     ///< dual steps of the root's faces
    int64_t relu_bab_farkas_steps = 0;     ///< dual steps of Farkas' certificate per node (0: none)
    int64_t relu_bab_alpha_iterations = 0; ///< the root's slope steps (no patience)
    /// Rounds of CROWN's bounds of the root's layer inputs, intersected with the zonotope's before
    /// its lines (0: the zonotope's bounds alone).
    int64_t relu_bab_crown_rounds = 1;
    int64_t relu_bab_crown_iterations =
        20;                            ///< alpha-CROWN steps on their ReLU lower slopes per row
    double relu_bab_crown_step = 0.25; ///< Adam's step on those slopes in [0, 1]
    /// The check's rounds of CROWN's bounds of the layer inputs (0: none), as relu-bab's root.
    int64_t check_crown_rounds = 0;
    int64_t check_crown_iterations = 0; ///< their alpha-CROWN steps per row
    /// CROWN's passes take a softmax's upper bound by its exps' log-sum-exp (nn_verification.md),
    /// not by its exp and reciprocal lines (the check, CROWN's layer bounds, relu-bab's root).
    bool crown_softmax_lse = false;
    /// The candidates of a look-ahead neuron split heuristic ("fsb") per set.
    int64_t look_ahead_candidates = 8;
    // the bounding "zonotope-alpha": CORA's verify whose check optimizes the ReLU slopes
    int64_t alpha_iterations = 10; ///< projected Adam steps on the open sets' margins
    double alpha_step_size = 0.3;  ///< Adam's step on the slopes in [0, 1]
    int64_t alpha_patience = 3;    ///< steps without a newly excluded polytope before it stops
    /// "zonotope": the slopes of the zonotope's lines, every later bound moves; "crown": CROWN's
    /// lower slopes per row in the backward pass (alpha-CROWN), the zonotope's bounds held.
    std::string alpha_method = "zonotope";
    /// The open sets whose slopes or multipliers are optimized: their worst polytope's best row is
    /// below 0 by at most this share of the row's width (inf: all).
    double optimize_gap = std::numeric_limits<double>::infinity();
    // the repairs of CORA's unsound steps (docs/nn_verification.md), on by default
    bool project_stale_factors = true; ///< the split rows keep the input factors after one pass
    bool drop_dummy_duplicates = true; ///< a dummy split drops its duplicate child only
    bool remember_point_splits_only = true; ///< a split bound is remembered at a point center only

    /// Fails with wrong_field_value for an unknown name of a strategy or an option.
    void validate() const;
};

/// CORA's getDefaultVNNCOMPoptions of a benchmark (a year suffix _2024 is dropped): the
/// conzonotope bounding with zonotack, the PGD attack around it.
NnVerifyOptions cora_vnncomp_options(const std::string &category);

/// The VNN-COMP preset: CORA's, with Phase 2's bounds chosen on the dev sets (zonotope-alpha,
/// backward_bound, alpha_method by benchmark).
NnVerifyOptions vnncomp_options(const std::string &category);

// The result --------------------------------------------------------------------------------------

/// The verdict ("sat", "unsat", "unknown"), for sat the counterexample (x, y), and the counts.
struct NnVerifyResult {
    std::string status;
    std::vector<double> x, y;
    std::map<std::string, double> stats;
    std::string note; ///< why the bounding was not used, if it was not
};

// The problem -------------------------------------------------------------------------------------

/// An instance prepared once for the strategies: the network in the search dtype and in float64,
/// the boxes (float32 inside the property's, float64 around them) and the polytopes as tensors.
class NnProblem {
  public:
    NnProblem(const NeuralNetwork &net, const NnProperty &property, const NnVerifyOptions &options,
              ConfirmFn confirm);

    const NeuralNetwork &net() const { return *net_; }          ///< float64
    const NeuralNetwork &search_net() const { return *net32_; } ///< the search dtype
    const NnVerifyOptions &options() const { return options_; }
    const NnProperty &property() const { return property_; }
    int64_t num_boxes() const { return lo.shape()[0]; }

    /// Per row of y (S, m): the least over the polytopes of its case (case (S)) of the largest row
    /// A_k y - b_k; at most 0 inside the unsafe set.
    Tensor loss(const Tensor &y, const Tensor &cases) const;

    /// The gradient at x (S, n) of the attack's objective: per polytope the violated rows' excess
    /// plus a tenth of the worst row, of the case's best polytope or the sample's target (S, 1);
    /// loss gets loss() at x; a sign's gradient is 1 on |z| <= sign_window.
    Tensor objective_gradient(const Tensor &x, const Tensor &cases, Tensor *loss = nullptr,
                              const Tensor *targets = nullptr, double sign_window = 1.0) const;

    /// The confirm check of one point; the output if it is a counterexample.
    std::optional<std::vector<double>> confirm(const Tensor &x);

    /// How many points of the search fit into the memory budget.
    int64_t attack_batch() const;

    /// Many cases of few polytopes each (nn4sys' lindex): the cases in chunks of at most max_rows
    /// rows of polytopes, in order; none when all the rows are at most max_rows.
    std::vector<std::vector<int64_t>> case_chunks(int64_t max_rows) const;

    /// The problem of some cases only: their boxes and polytopes, the same networks and check.
    NnProblem subproblem(const std::vector<int64_t> &cases) const;

    /// The search boxes (float32 inside the property's) and the outer float64 ones (C, n).
    Tensor lo, hi, outer_lo, outer_hi;
    Tensor box_case;    ///< (C): 0, 1, ..., the case of each box, float64
    Tensor search_case; ///< the same in the search dtype
    Tensor A, b;        ///< the polytopes in float64
    Tensor poly_case;   ///< (P), float64
    int64_t memory_budget = 0;
    int64_t batch_memory_budget = 0; ///< the boxes of one batch (options, else memory_budget)
    int64_t checks = 0;              ///< confirm calls
    /// The deadline of the step that runs (verify sets it): objective_gradient and the bounding's
    /// passes check it once per layer (check_deadline).
    double deadline = std::numeric_limits<double>::infinity();

  private:
    std::shared_ptr<const NeuralNetwork> net_, net32_;
    NnProperty property_;
    NnVerifyOptions options_;
    ConfirmFn confirm_;
    Tensor A32_, b32_, poly_case32_;
    /// Many cases of few polytopes each (nn4sys' lindex: thousands of boxes): each row's margins
    /// over its own case's polytopes only, not over all of them (memory S P K otherwise).
    bool per_case_ = false;
    std::vector<std::vector<int64_t>> case_polys_; ///< the polytopes of each case
    int64_t max_polys_ = 0;                        ///< the most polytopes of a case
    struct CaseMargins {
        Tensor margin;             ///< (S, Q, K): the rows of the case's polytopes, Q = max_polys_
        std::vector<int64_t> poly; ///< (S Q): the polytope of each slot (0 for a padding slot)
        Tensor valid;              ///< (S, Q): 1 for a polytope of the case, 0 for a padding slot
    };
    CaseMargins aux_case_margins(const Tensor &y, const Tensor &cases) const;
    /// The property's values on the host, for subproblem (filled once).
    struct HostProperty {
        std::vector<double> lo, hi, A, b;
    };
    mutable std::shared_ptr<const HostProperty> host_;
    /// The problem of the property with the parent's networks, options and check (subproblem).
    NnProblem(const NnProblem &parent, const NnProperty &property);
    /// The boxes and the polytopes as tensors (the constructor and subproblem).
    void aux_set_property(const NnProperty &property);
};

// The strategies ----------------------------------------------------------------------------------

/// What an attack found: x and y (empty if nothing) and the effort.
struct AttackResult {
    std::vector<double> x, y;
    int64_t iterations = 0, evaluations = 0, candidates = 0;
};

/// A falsification method: run searches until now_seconds() reaches the deadline, or for `rounds`
/// rounds of restarts (negative: no limit) where the method has rounds.
class Attack {
  public:
    virtual ~Attack() = default;
    virtual AttackResult run(NnProblem &problem, double deadline, int64_t rounds) = 0;
};

/// A bounding method with its own branch and bound: prepare once (false: it cannot bound this
/// network, with the reason in why), then the verdict "unsat", "sat" (found) or "unknown".
class Bounding {
  public:
    virtual ~Bounding() = default;
    virtual bool prepare(NnProblem &problem) = 0;
    virtual std::string branch_and_bound(NnProblem &problem, double deadline,
                                         std::map<std::string, double> &stats,
                                         AttackResult &found) = 0;
    std::string why;
};

/// The inputs of a neuron heuristic per entry (B, n): the bounds, the approximation errors, the
/// sensitivity and the gradient of the output's F-radius (undefined when not needed).
struct HeuristicInputs {
    Tensor l, u, er, sens, grad;
};

/// A neuron heuristic: the score of every entry (CORA's computeHeuristic); `sens` and `grad` say
/// what it reads.
struct Heuristic {
    std::function<Tensor(const HeuristicInputs &)> score;
    bool sens = false, grad = false;
    /// As a neuron split heuristic: the score filters look_ahead_candidates neurons, and the one
    /// whose two children have the best backward bound is split (FSB, filtered smart branching).
    bool look_ahead = false;
};

/// An input split heuristic of the zonotope bounding: the dimension to split per box (B, n).
using InputSplitHeuristic = std::function<std::vector<int64_t>(
    NnProblem &, const Tensor &lo, const Tensor &hi, const Tensor &cases)>;

using AttackFactory = std::function<std::unique_ptr<Attack>(const NnVerifyOptions &)>;
using BoundingFactory = std::function<std::unique_ptr<Bounding>(const NnVerifyOptions &)>;

/// Registers a strategy under its option name (a name registered again is replaced).
void register_attack(const std::string &name, AttackFactory factory);
void register_bounding(const std::string &name, BoundingFactory factory);
void register_heuristic(const std::string &name, Heuristic heuristic);
void register_input_split_heuristic(const std::string &name, InputSplitHeuristic heuristic);

/// The registered names of each kind, sorted.
std::vector<std::string> attack_names();
std::vector<std::string> bounding_names();
std::vector<std::string> heuristic_names();
std::vector<std::string> input_split_heuristic_names();

/// The attack of a name, the heuristic of a name; unknown names fail with wrong_field_value.
std::unique_ptr<Attack> make_attack(const std::string &name, const NnVerifyOptions &options);
const Heuristic &heuristic(const std::string &name);

/// The seconds of a steady clock, the time base of the deadlines.
double now_seconds();

/// Raises Error::special_error ("verify: the deadline passed") once now_seconds() reaches the
/// deadline: one clock read, cheap enough for once per layer of a pass. verify then answers with
/// what it has, "unknown" if nothing is decided.
void check_deadline(double deadline);

/// Whether the error is check_deadline's.
bool is_deadline(const std::exception &error);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
