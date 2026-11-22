// priv_verify_attacks - the falsification methods center, pgd and fgsm of verify
//
// "center" checks the center of every box (CORA's falsification_method 'center'). "pgd" is
// projected gradient descent on the property's objective: sign steps of pgd_step_size times the
// box radius, multiplied by pgd_decay at the pgd_decay_at fractions of the steps, from the center
// and from uniform restarts (pgd_restarts per box, as many as the memory budget holds), in rounds
// until the deadline (CORA's computePGDAttack). "fgsm" is one sign step of the full radius from
// the center (CORA's computeFGSMAttack). Every point of a step whose loss is at most 0 is a
// candidate for the confirm check.
//
// Syntax:   auto attack = make_attack("pgd", options);   attack->run(problem, deadline, 3);
// See also: priv_verify.h, priv_verify_zonotack.cpp (the attack "zonotack")
//
// CORA: nn/@neuralNetwork/computePGDAttack.m
// CORA: nn/@neuralNetwork/computeFGSMAttack.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_verify.h"

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
          rounds_(fgsm ? 1 : -1), step_size_(fgsm ? 1.0 : o.pgd_step_size), decay_(o.pgd_decay),
          seed_(o.seed) {
        if (!fgsm)
            for (const double f : o.pgd_decay_at) decay_steps_.insert(int64_t(f * double(steps_)));
    }

    AttackResult run(NnProblem &problem, double deadline, int64_t rounds) override {
        AttackResult result;
        Rng rng(seed_);
        // every box gets restarts samples per round, as many as the memory budget allows
        const int64_t boxes = problem.num_boxes();
        const int64_t per_box = std::max<int64_t>(
            1, std::min(restarts_, problem.attack_batch() / std::max<int64_t>(1, boxes)));
        std::vector<int64_t> rows;
        std::vector<double> first;
        for (int64_t k = 0; k < boxes; ++k)
            for (int64_t j = 0; j < per_box; ++j) {
                rows.push_back(k);
                first.push_back(j == 0 ? 1.0 : 0.0);
            }
        const Tensor lo = problem.lo.take(rows, 0), hi = problem.hi.take(rows, 0);
        const Tensor cases = problem.search_case.take(rows, 0), radius = (hi - lo) * 0.5;
        const Tensor center = (lo + hi) * 0.5;
        const Tensor is_first = Tensor::like(lo, first, {int64_t(first.size()), 1});
        const int64_t limit = rounds >= 0 ? rounds : rounds_;
        std::vector<double> u(std::size_t(lo.numel()));
        for (int64_t round = 0; now_seconds() < deadline && (limit < 0 || round < limit); ++round) {
            // uniform restarts, the first one of each box at its center in the first round
            rng.uniform(u.data(), u.size(), 0.0, 1.0);
            Tensor x = lo + Tensor::like(lo, u, lo.shape()).multiply(hi - lo);
            if (round == 0) x = Tensor::where(is_first, center, x);
            Tensor alpha = radius * step_size_;
            for (int64_t step = 0; step < steps_; ++step) {
                if (decay_steps_.count(step)) alpha = alpha * decay_;
                // one sign step on the objective, projected back into the box
                Tensor loss;
                const Tensor g = problem.objective_gradient(x, cases, &loss);
                const Tensor before = x;
                x = (x - alpha.multiply(g.sign())).maximum(lo).minimum(hi);
                ++result.iterations;
                result.evaluations += x.shape()[0];
                if (priv_confirm_rows(problem, before, loss, result)) return result;
                if (now_seconds() >= deadline) break;
            }
            // the last iterate too
            const Tensor loss = problem.loss(problem.search_net().evaluate(x), cases);
            if (priv_confirm_rows(problem, x, loss, result)) return result;
        }
        return result;
    }

  private:
    int64_t steps_, restarts_, rounds_;
    double step_size_, decay_;
    uint64_t seed_;
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
