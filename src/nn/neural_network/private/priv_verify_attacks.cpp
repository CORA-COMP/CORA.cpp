// priv_verify_attacks - the falsification methods center, pgd and fgsm of verify
//
// "center" checks the center of every box (CORA's falsification_method 'center'). "pgd" is
// projected gradient descent on the property's objective: sign steps of pgd_step_size times the
// box radius, multiplied by pgd_decay at the pgd_decay_at fractions of the steps, from the center
// and from uniform restarts (pgd_restarts per box, as many as the memory budget holds, fewer once
// pgd_min_steps steps would not fit in the time), in rounds until the deadline (CORA's
// computePGDAttack). With pgd_targeted, the restarts are per polytope of a box's case, each
// toward its own polytope (a targeted attack per output class of a disjunction, as
// alpha-beta-CROWN's per-OR restarts). "fgsm" is one sign step of the full radius from the center
// (CORA's computeFGSMAttack). Every point of a step whose loss is at most 0 is a candidate for the
// confirm check.
//
// Syntax:   auto attack = make_attack("pgd", options);   attack->run(problem, deadline, 3);
// See also: priv_verify.h, priv_verify_zonotack.cpp (the attack "zonotack")
// References:
//    [1] A. Kurakin, I. J. Goodfellow, S. Bengio. "Adversarial machine learning at scale", ICLR
//        2017.
//    [2] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. "Towards deep learning models
//        resistant to adversarial attacks", ICLR 2018.
//    [3] I. J. Goodfellow, J. Shlens, C. Szegedy. "Explaining and harnessing adversarial
//        examples", ICLR 2015.
//
// CORA: nn/@neuralNetwork/computePGDAttack.m
// CORA: nn/@neuralNetwork/computeFGSMAttack.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_verify.h"

#include <exception>
#include <set>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The center --------------------------------------------------------------------------------------

/// CORA's 'center': the center of every box.
class CenterAttack : public Attack {
  public:
    AttackResult run(NnProblem &problem, double, int64_t) override {
        AttackResult result;
        const Tensor x = (problem.lo + problem.hi) * 0.5;
        const Tensor loss = problem.loss(problem.search_net().evaluate(x), problem.search_case);
        result.evaluations = x.shape()[0];
        result.iterations = 1;
        priv_confirm_rows(problem, x, loss, result);
        return result;
    }
};

// PGD and FGSM ------------------------------------------------------------------------------------

/// The sign steps of PGD (FGSM: one step of the full radius, one round).
class PgdAttack : public Attack {
  public:
    PgdAttack(const NnVerifyOptions &o, bool fgsm)
        : steps_(fgsm ? 1 : o.pgd_steps), restarts_(fgsm ? 1 : o.pgd_restarts),
          rounds_(fgsm ? 1 : -1), min_steps_(fgsm ? 1 : o.pgd_min_steps),
          step_size_(fgsm ? 1.0 : o.pgd_step_size), decay_(o.pgd_decay), seed_(o.seed),
          targeted_(!fgsm && o.pgd_targeted) {
        if (!fgsm)
            for (const double f : o.pgd_decay_at) decay_steps_.insert(int64_t(f * double(steps_)));
    }

    AttackResult run(NnProblem &problem, double deadline, int64_t rounds) override {
        // a step that the deadline cuts short ends the search, its counts kept
        AttackResult result;
        try {
            aux_search(problem, deadline, rounds, result);
        } catch (const Error &error) {
            if (!is_deadline(error)) std::rethrow_exception(std::current_exception());
        }
        return result;
    }

  private:
    /// The rounds of restarts until the deadline; a counterexample goes into result.
    void aux_search(NnProblem &problem, double deadline, int64_t rounds, AttackResult &result) {
        Rng rng(seed_);
        // restarts samples per group (a box, or a box and a polytope) per round, as memory allows
        const std::vector<Group> groups = aux_groups(problem);
        const int64_t boxes = int64_t(groups.size());
        int64_t per_box = std::max<int64_t>(
            1, std::min(restarts_, problem.attack_batch() / std::max<int64_t>(1, boxes)));
        Rows r = aux_rows(problem, groups, per_box);
        const int64_t limit = rounds >= 0 ? rounds : rounds_;
        for (int64_t round = 0; now_seconds() < deadline && (limit < 0 || round < limit); ++round) {
            // uniform restarts, the first one of each box at its center in the first round
            std::vector<double> u(std::size_t(r.lo.numel()));
            rng.uniform(u.data(), u.size(), 0.0, 1.0);
            Tensor x = r.lo + Tensor::like(r.lo, u, r.lo.shape()).multiply(r.hi - r.lo);
            if (round == 0) x = Tensor::where(r.is_first, r.center, x);
            Tensor alpha = r.radius * step_size_;
            for (int64_t step = 0; step < steps_; ++step) {
                if (decay_steps_.count(step)) alpha = alpha * decay_;
                // one sign step on the objective, projected back into the box
                const double started = now_seconds();
                Tensor loss;
                const Tensor g =
                    problem.objective_gradient(x, r.cases, &loss, targeted_ ? &r.targets : nullptr);
                const Tensor before = x;
                x = (x - alpha.multiply(g.sign())).maximum(r.lo).minimum(r.hi);
                ++result.iterations;
                result.evaluations += x.shape()[0];
                if (priv_confirm_rows(problem, before, loss, result)) return;
                if (now_seconds() >= deadline) break;
                // too slow for pgd_min_steps steps in the time left: fewer restarts per box
                const double took = now_seconds() - started, left = deadline - now_seconds();
                if (step == 0 && per_box > 1 && took * double(min_steps_) > left) {
                    const double share = left / (took * double(min_steps_));
                    const int64_t keep = std::max<int64_t>(1, int64_t(double(per_box) * share));
                    std::vector<int64_t> kept;
                    for (int64_t k = 0; k < boxes; ++k)
                        for (int64_t j = 0; j < keep; ++j) kept.push_back(k * per_box + j);
                    per_box = keep;
                    r = aux_rows(problem, groups, per_box);
                    x = x.take(kept, 0);
                    alpha = alpha.take(kept, 0);
                }
            }
            // the last iterate too
            const Tensor loss = problem.loss(problem.search_net().evaluate(x), r.cases);
            if (priv_confirm_rows(problem, x, loss, result)) return;
        }
    }

    /// A box of the problem and the polytope its samples aim at (-1: the best one of the case).
    struct Group {
        int64_t box, target;
    };
    /// The boxes, or with targeted every box with each polytope of its case.
    std::vector<Group> aux_groups(const NnProblem &problem) const {
        std::vector<Group> groups;
        const int64_t boxes = problem.num_boxes();
        if (!targeted_) {
            for (int64_t k = 0; k < boxes; ++k) groups.push_back({k, -1});
            return groups;
        }
        const std::vector<double> box_case = problem.box_case.data();
        const std::vector<double> poly_case = problem.poly_case.data();
        for (int64_t k = 0; k < boxes; ++k)
            for (std::size_t p = 0; p < poly_case.size(); ++p)
                if (poly_case[p] == box_case[std::size_t(k)]) groups.push_back({k, int64_t(p)});
        return groups;
    }

    /// The boxes of the samples, per_box of each group in a row, the first of each box marked.
    struct Rows {
        Tensor lo, hi, cases, radius, center, is_first, targets;
    };
    static Rows aux_rows(const NnProblem &problem, const std::vector<Group> &groups,
                         int64_t per_box) {
        std::vector<int64_t> rows;
        std::vector<double> first, target;
        for (std::size_t g = 0; g < groups.size(); ++g)
            for (int64_t j = 0; j < per_box; ++j) {
                rows.push_back(groups[g].box);
                // the center once per box: its first group's first sample
                const bool new_box = g == 0 || groups[g - 1].box != groups[g].box;
                first.push_back(j == 0 && new_box ? 1.0 : 0.0);
                target.push_back(double(std::max<int64_t>(0, groups[g].target)));
            }
        // their boxes, centers and radii
        Rows r;
        r.lo = problem.lo.take(rows, 0);
        r.hi = problem.hi.take(rows, 0);
        r.cases = problem.search_case.take(rows, 0);
        r.radius = (r.hi - r.lo) * 0.5;
        r.center = (r.lo + r.hi) * 0.5;
        r.is_first = Tensor::like(r.lo, first, {int64_t(first.size()), 1});
        r.targets = Tensor::like(r.lo, target, {int64_t(target.size()), 1});
        return r;
    }

    int64_t steps_, restarts_, rounds_, min_steps_;
    double step_size_, decay_;
    uint64_t seed_;
    bool targeted_ = false;
    std::set<int64_t> decay_steps_;
};

} // namespace


// ===========================================  MAIN  =========================================== //

// The registry ------------------------------------------------------------------------------------

void priv_register_attacks() {
    register_attack("center",
                    [](const NnVerifyOptions &) { return std::make_unique<CenterAttack>(); });
    register_attack("pgd",
                    [](const NnVerifyOptions &o) { return std::make_unique<PgdAttack>(o, false); });
    register_attack("fgsm",
                    [](const NnVerifyOptions &o) { return std::make_unique<PgdAttack>(o, true); });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
