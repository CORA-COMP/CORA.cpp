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
#include <limits>
#include <set>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
/// The polish: Adam's floor of the second moment's root, and the loss within which it starts.
const double kAdamFloor = 5.0e7 * machine_eps(), kPolishGate = 5.0e11 * machine_eps();

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
          targeted_(!fgsm && o.pgd_targeted),
          windows_(o.pgd_sign_windows.empty() ? std::vector<double>{1.0} : o.pgd_sign_windows),
          sizes_(o.pgd_step_sizes.empty() || fgsm ? std::vector<double>{step_size_}
                                                  : o.pgd_step_sizes),
          step_decay_(fgsm ? 1.0 : o.pgd_step_decay), momentum_(fgsm ? 0.0 : o.pgd_momentum),
          hops_(fgsm ? 0 : o.pgd_hops), hop_steps_(o.pgd_hop_steps), hop_keep_(o.pgd_hop_keep),
          hop_jumps_(o.pgd_hop_jumps.empty() ? std::vector<double>{0.4} : o.pgd_hop_jumps),
          polish_steps_(fgsm ? 0 : o.pgd_polish_steps), polish_step_(o.pgd_polish_step),
          polish_keep_(o.pgd_polish_keep), polish_every_(o.pgd_polish_every) {
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
    /// A box of the problem and the polytope its samples aim at (-1: the best one of the case).
    struct Group {
        int64_t box, target;
    };
    /// The samples' boxes, centers, radii and steps, a row each; the first of each box marked.
    struct Rows {
        Tensor lo, hi, cases, radius, center, is_first, targets, fraction;
    };

    /// The rounds of restarts until the deadline, each followed by hops; a counterexample goes
    /// into result.
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
            const double window = windows_[std::size_t(round) % windows_.size()];
            Tensor best_x = x, best_loss = x.full_like({x.shape()[0]}, kInfinity);
            const std::set<int64_t> *decay = &decay_steps_;
            if (aux_descend(problem, deadline, r, x, best_x, best_loss, steps_, decay, window,
                            {&groups, &per_box}, result))
                return;
            // the last iterate too
            const Tensor loss = problem.loss(problem.search_net().evaluate(x), r.cases);
            if (priv_confirm_rows(problem, x, loss, result)) return;
            if (aux_polish(problem, deadline, r, best_x, best_loss, window, result)) return;
            // hops: the best points jump out of their basin and descend again, the best kept
            for (int64_t hop = 0; hop < hops_ && now_seconds() < deadline; ++hop) {
                const int64_t keep = std::min<int64_t>(hop_keep_, r.lo.shape()[0]);
                const Tensor order = best_loss.argsort(0, false).slice(0, 0, keep);
                Rows h = aux_take(r, order);
                std::vector<double> w(std::size_t(h.lo.numel()));
                rng.uniform(w.data(), w.size(), -1.0, 1.0);
                const double jump = hop_jumps_[std::size_t(hop) % hop_jumps_.size()];
                Tensor hx = (best_x.take(order, 0) +
                             Tensor::like(h.lo, w, h.lo.shape()).multiply(h.radius) * jump)
                                .maximum(h.lo)
                                .minimum(h.hi);
                Tensor hbest_x = hx, hbest_loss = best_loss.full_like({keep}, kInfinity);
                const std::set<int64_t> none;
                if (aux_descend(problem, deadline, h, hx, hbest_x, hbest_loss, hop_steps_, &none,
                                window, {nullptr, nullptr}, result))
                    return;
                if ((hop + 1) % polish_every_ == 0 &&
                    aux_polish(problem, deadline, h, hbest_x, hbest_loss, window, result))
                    return;
                // the elite: the hop's points replace the worst ones of the round
                const Tensor all_loss = Tensor::concat({best_loss, hbest_loss}, 0);
                const Tensor all_x = Tensor::concat({best_x, hbest_x}, 0);
                const Tensor top = all_loss.argsort(0, false).slice(0, 0, best_loss.shape()[0]);
                best_loss = all_loss.take(top, 0);
                best_x = all_x.take(top, 0);
            }
        }
    }

    /// The polish: Adam's small steps from the polish_keep best points of r, once one is within
    /// 1e-4 of the unsafe set; true once a candidate passes the confirm check.
    bool aux_polish(NnProblem &problem, double deadline, const Rows &r, const Tensor &best_x,
                    const Tensor &best_loss, double window, AttackResult &result) {
        if (polish_steps_ <= 0 || best_loss.min(0).data()[0] > kPolishGate) return false;
        const Tensor order = best_loss.argsort(0, false).slice(0, 0, polish_keep_);
        Rows p = aux_take(r, order);
        Tensor px = best_x.take(order, 0), pbest_x = px, pbest_loss = best_loss.take(order, 0);
        const std::set<int64_t> none;
        return aux_descend(problem, deadline, p, px, pbest_x, pbest_loss, polish_steps_, &none,
                           window, {nullptr, nullptr}, result, true);
    }

    /// The groups and the restarts per box of the round's rows, which may shrink (nullptr: not).
    struct Shrink {
        const std::vector<Group> *groups;
        int64_t *per_box;
    };

    /// steps sign steps of x (the rows r) on the objective, projected into the box, the best point
    /// and loss of every row kept; true once a candidate passes the confirm check (in result).
    bool aux_descend(NnProblem &problem, double deadline, Rows &r, Tensor &x, Tensor &best_x,
                     Tensor &best_loss, int64_t steps, const std::set<int64_t> *decay,
                     double window, const Shrink &shrink, AttackResult &result, bool adam = false) {
        // the polish: Adam's steps of its own size, decaying at every step
        Tensor alpha = r.radius.multiply(adam ? r.radius.ones_like() * polish_step_ : r.fraction);
        Tensor mean = x.zeros_like(), square = x.zeros_like();
        for (int64_t step = 0; step < steps; ++step) {
            if (decay->count(step)) alpha = alpha * decay_;
            // one sign step on the objective, projected back into the box
            const double started = now_seconds();
            Tensor loss;
            const Tensor g = problem.objective_gradient(x, r.cases, &loss,
                                                        targeted_ ? &r.targets : nullptr, window);
            const Tensor before = x;
            // the best point of each row
            if (hops_ > 0 || polish_steps_ > 0) {
                const Tensor better = loss.less(best_loss);
                best_x = Tensor::where(better.expand_dims(-1), before, best_x);
                best_loss = Tensor::where(better, loss, best_loss);
            }
            // Adam's step (the polish), else the sign of the running mean of the gradients
            if (adam) {
                mean = mean * 0.9 + g * 0.1;
                square = square * 0.999 + g.square() * 0.001;
                const Tensor direction =
                    (mean * (1.0 / (1.0 - std::pow(0.9, double(step + 1)))))
                        .divide(
                            (square * (1.0 / (1.0 - std::pow(0.999, double(step + 1))))).sqrt() +
                            kAdamFloor);
                x = (x - alpha.multiply(direction)).maximum(r.lo).minimum(r.hi);
                alpha = alpha * 0.999;
            } else if (momentum_ > 0) {
                mean = mean * momentum_ + g * (1.0 - momentum_);
                x = priv_sign_step(x, mean, alpha).maximum(r.lo).minimum(r.hi);
            } else {
                x = priv_sign_step(x, g, alpha).maximum(r.lo).minimum(r.hi);
            }
            if (!adam && step_decay_ < 1.0) alpha = alpha * step_decay_;
            // the counts, and the candidates of this step's points
            ++result.iterations;
            result.evaluations += x.shape()[0];
            if (priv_confirm_rows(problem, before, loss, result)) return true;
            if (now_seconds() >= deadline) break;
            // too slow for pgd_min_steps steps in the time left: fewer restarts per box
            const double took = now_seconds() - started, left = deadline - now_seconds();
            // the shrunk rows, each state with them
            if (step == 0 && shrink.per_box && *shrink.per_box > 1 &&
                took * double(min_steps_) > left) {
                const double share = left / (took * double(min_steps_));
                int64_t &per_box = *shrink.per_box;
                const int64_t keep = std::max<int64_t>(1, int64_t(double(per_box) * share));
                std::vector<int64_t> kept;
                const int64_t boxes = int64_t(shrink.groups->size());
                for (int64_t k = 0; k < boxes; ++k)
                    for (int64_t j = 0; j < keep; ++j) kept.push_back(k * per_box + j);
                per_box = keep;
                r = aux_rows(problem, *shrink.groups, per_box);
                x = x.take(kept, 0);
                alpha = alpha.take(kept, 0);
                mean = mean.take(kept, 0);
                best_x = best_x.take(kept, 0);
                best_loss = best_loss.take(kept, 0);
            }
        }
        return false;
    }

    /// The rows of r at the indices.
    static Rows aux_take(const Rows &r, const Tensor &rows) {
        Rows t;
        t.lo = r.lo.take(rows, 0);
        t.hi = r.hi.take(rows, 0);
        t.cases = r.cases.take(rows, 0);
        t.radius = r.radius.take(rows, 0);
        t.center = r.center.take(rows, 0);
        t.is_first = r.is_first.take(rows, 0);
        t.targets = r.targets.take(rows, 0);
        t.fraction = r.fraction.take(rows, 0);
        return t;
    }

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
    Rows aux_rows(const NnProblem &problem, const std::vector<Group> &groups,
                  int64_t per_box) const {
        std::vector<int64_t> rows;
        std::vector<double> first, target, fraction;
        for (std::size_t g = 0; g < groups.size(); ++g)
            for (int64_t j = 0; j < per_box; ++j) {
                rows.push_back(groups[g].box);
                // the center once per box: its first group's first sample
                const bool new_box = g == 0 || groups[g - 1].box != groups[g].box;
                first.push_back(j == 0 && new_box ? 1.0 : 0.0);
                target.push_back(double(std::max<int64_t>(0, groups[g].target)));
                // the step of this restart, cycled over the sizes
                fraction.push_back(sizes_[std::size_t(j) % sizes_.size()]);
            }
        // their boxes, centers and radii
        Rows r;
        r.lo = problem.lo.take(rows, 0);
        r.hi = problem.hi.take(rows, 0);
        r.cases = problem.search_case.take(rows, 0);
        r.radius = (r.hi - r.lo) * 0.5;
        r.center = (r.lo + r.hi) * 0.5;
        // the marks and the steps
        r.is_first = Tensor::like(r.lo, first, {int64_t(first.size()), 1});
        r.targets = Tensor::like(r.lo, target, {int64_t(target.size()), 1});
        r.fraction = Tensor::like(r.lo, fraction, {int64_t(fraction.size()), 1});
        return r;
    }

    int64_t steps_, restarts_, rounds_, min_steps_;
    double step_size_, decay_;
    uint64_t seed_;
    bool targeted_ = false;
    std::vector<double> windows_, sizes_;
    double step_decay_, momentum_;
    int64_t hops_, hop_steps_, hop_keep_;
    std::vector<double> hop_jumps_;
    int64_t polish_steps_;
    double polish_step_;
    int64_t polish_keep_, polish_every_;
    std::set<int64_t> decay_steps_;
};

} // namespace


// ===========================================  MAIN  =========================================== //

// The sign step -----------------------------------------------------------------------------------

Tensor priv_sign_step(const Tensor &x, const Tensor &g, const Tensor &alpha) {
    return x - alpha.multiply(g.sign());
}

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
