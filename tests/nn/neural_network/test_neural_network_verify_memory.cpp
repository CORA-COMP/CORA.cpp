// test_neural_network_verify_memory - verify out of memory: a sound verdict, never a crash
//
// crown-bab on small ReLU networks whose outputs are sampled, with an out of memory injected at
// each of its allocations (priv_inject_oom: the root net's crossing pass, the root's forward and
// CROWN bounds, CROWN's row chunks, the roots of several polytopes, a node batch, the look-ahead):
// verify answers, every back-off is counted, and an "unsat" holds every sample. Budgets of one
// byte answer soundly too, and a bounding, an attack or a prepare that runs out of memory leaves
// the verdict open (verify's last resort). The same on a GPU, where the budgets come from it.
//
// CORA.cpp only: CORA's verify catches out of memory around a batch alone
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/nn/neural_network/verify.h"
#include "nn/neural_network/private/priv_verify.h"
#include "testing.h"

#include <cmath>
#include <limits>
#include <new>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kTimeout = 2; // seconds per verify: an open verdict takes all of them

NeuralNetwork aux_network(Rng &rng, const std::vector<int64_t> &sizes) {
    std::vector<NeuralNetwork::Layer> layers;
    std::vector<Activation> acts;
    for (std::size_t k = 0; k + 1 < sizes.size(); ++k) {
        std::vector<double> W(static_cast<std::size_t>(sizes[k + 1] * sizes[k])),
            b(static_cast<std::size_t>(sizes[k + 1]));
        rng.normal(W.data(), W.size(), 1.0 / std::sqrt(double(sizes[k])));
        rng.normal(b.data(), b.size(), 0.3);
        layers.push_back(
            {Tensor::from_data(W, {sizes[k + 1], sizes[k]}), Tensor::from_data(b, {sizes[k + 1]})});
        acts.push_back(k + 2 < sizes.size() ? Activation::relu : Activation::identity);
    }
    return {layers, acts};
}

/// The box [lo, hi]^n and the unsafe set y_0 <= t, or y_0 <= t or y_0 >= far (two polytopes).
NnProperty aux_below(int64_t n, double lo, double hi, double t, double far = 0) {
    NnProperty p;
    p.lo = Tensor::from_data(std::vector<double>(std::size_t(n), lo), {1, n});
    p.hi = Tensor::from_data(std::vector<double>(std::size_t(n), hi), {1, n});
    if (far == 0) {
        p.A = Tensor::from_data(std::vector<double>{1.0, 0.0}, {1, 1, 2});
        p.b = Tensor::from_data(std::vector<double>{t}, {1, 1});
        p.poly_case = {0};
    } else {
        p.A = Tensor::from_data(std::vector<double>{1.0, 0.0, -1.0, 0.0}, {2, 1, 2});
        p.b = Tensor::from_data(std::vector<double>{t, -far}, {2, 1});
        p.poly_case = {0, 0};
    }
    return p;
}

/// crown-bab, bound first, a few nodes at once.
NnVerifyOptions aux_options(const std::string &b) {
    NnVerifyOptions o = cora_vnncomp_options("test");
    o.bounding_method = "zonotope-alpha";
    o.refinement_method = "crown-bab";
    o.falsification_order = "bound-first";
    o.alpha_iterations = 5;
    o.relu_bab_crown_iterations = 3;
    o.crown_bab_batch = 8;
    o.crown_bab_fsb = 3;
    o.pgd_restarts = 8;
    o.pgd_steps = 20;
    if (b == "lean") o.dtype = DType::float64;
    return o;
}

/// A strategy that runs out of memory where `where` says.
class OomAttack : public Attack {
  public:
    AttackResult run(NnProblem &, double, int64_t) override { throw std::bad_alloc(); }
};
class OomBounding : public Bounding {
  public:
    explicit OomBounding(bool in_prepare) : in_prepare_(in_prepare) {}
    bool prepare(NnProblem &) override {
        if (in_prepare_) throw std::bad_alloc();
        return true;
    }
    std::string branch_and_bound(NnProblem &, double, std::map<std::string, double> &,
                                 AttackResult &) override {
        throw std::bad_alloc();
    }

  private:
    bool in_prepare_;
};

/// Each allocation of crown-bab out of memory once (or always): an answer, sound, and counted.
void injected(const std::string &b) {
    if (b == "lean") return; // many small verifications: slow on lean
    struct Site {
        const char *name;
        int64_t skip, count;
        const char *stat; ///< the back-off's count
    };
    const std::vector<Site> sites = {
        {"relu_bab_crossing", 1, 1, "relu_bab_root_oom"},   // the root net's pass: fewer gens
        {"crown_bab_root", 0, 1, "crown_bab_root_oom"},     // the root: a smaller net
        {"crown_bab_root", 0, 1000, "crown_bab_root_oom"},  // always: open
        {"crown_bounds", 0, 1, "relu_bab_crown_oom"},       // CROWN's rows: half the chunk
        {"crown_bounds", 0, 1000, "relu_bab_crown_oom"},    // always: the zonotope's bounds
        {"crown_bab_roots", 0, 1000, "crown_bab_oom"},      // the roots: none verified at once
        {"crown_bab_bound", 0, 1, "crown_bab_oom"},         // a node's bound (the root's first)
        {"crown_bab_bound", 2, 1000, "crown_bab_oom"},      // the batches: half, then open
        {"crown_bab_look_ahead", 0, 1000, "crown_bab_oom"}, // the look-ahead: BaBSR's pick
        {"affine_bab_slopes", 0, 1000, "affine_bab_oom"},   // the forms' slopes: the search alone
        {"affine_bab_table", 0, 1000, "affine_bab_oom"},    // the forms' rows: the search alone
        {"affine_bab", 0, 1000, "affine_bab_oom"},          // the forms' nodes: the search alone
    };
    Rng rng(41);
    int wrong = 0, answered = 0, proved = 0;
    std::vector<int> counted(sites.size(), 0); // per site: the runs that backed off there
    for (int net_i = 0; net_i < 1; ++net_i) {
        const NeuralNetwork nn = aux_network(rng, {2, 12, 12, 2});
        std::vector<double> xs(std::size_t(20000 * 2));
        rng.uniform(xs.data(), xs.size(), -1.0, 1.0);
        const std::vector<double> y = nn.evaluate(Tensor::from_data(xs, {20000, 2})).data();
        double least = kInf, most = -kInf;
        for (std::size_t i = 0; i < 20000; ++i) {
            least = std::min(least, y[i * 2]);
            most = std::max(most, y[i * 2]);
        }
        NnVerifyOptions o = aux_options(b);
        for (std::size_t si = 0; si < sites.size(); ++si) {
            const Site &site = sites[si];
            o.affine_bab = std::string(site.name).rfind("affine_bab", 0) == 0;
            for (const double t : {least - 0.3, least - 0.02}) {
                // two polytopes for the roots, one else
                const bool two = std::string(site.name) == "crown_bab_roots";
                const double far = two ? most + 0.05 : 0.0;
                // armed for this run only: an answer, the back-off's count, a sound unsat
                priv_inject_oom(site.name, site.skip, site.count);
                NnVerifyResult r;
                bool threw = false;
                try {
                    r = nn.verify(aux_below(2, -1.0, 1.0, t, far), kTimeout, o);
                } catch (const std::exception &) {
                    threw = true;
                }
                priv_inject_oom(site.name, 0, 0);
                answered += !threw && (r.status == "unsat" || r.status == "unknown") ? 1 : 0;
                counted[si] += !threw && r.stats.count(site.stat) ? 1 : 0;
                proved += !threw && r.status == "unsat" ? 1 : 0;
                if (!threw && r.status == "unsat")
                    for (std::size_t i = 0; i < 20000; ++i)
                        wrong += y[i * 2] <= t || (two && y[i * 2] >= far) ? 1 : 0;
            }
        }
    }
    // every run answered, every site backed off, every unsat sound
    const int runs = 2 * int(sites.size());
    check(answered == runs, b + ": out of memory anywhere in crown-bab, verify still answers");
    int sites_hit = 0;
    for (const int c : counted) sites_hit += c > 0 ? 1 : 0;
    check(sites_hit == int(sites.size()), b + ": each site's out of memory backs off, counted");
    check(wrong == 0, b + ": no unsat out of memory where a sample violates the property");
    check(proved > 0, b + ": a back-off that completes still proves");
}

/// Budgets of one byte: the smallest root net, row chunks and batches of one, a sound answer.
void tiny_budgets(const std::string &b) {
    if (b == "lean") return;
    Rng rng(43);
    const NeuralNetwork nn = aux_network(rng, {2, 12, 12, 2});
    std::vector<double> xs(std::size_t(20000 * 2));
    rng.uniform(xs.data(), xs.size(), -1.0, 1.0);
    const std::vector<double> y = nn.evaluate(Tensor::from_data(xs, {20000, 2})).data();
    double least = kInf;
    for (std::size_t i = 0; i < 20000; ++i) least = std::min(least, y[i * 2]);
    // every budget one byte: unsat, unknown or a counterexample, never an unsound unsat
    NnVerifyOptions o = aux_options(b);
    o.memory_budget = o.batch_memory_budget = o.relu_bab_memory_budget = 1;
    o.crown_bab_memory_budget = o.relu_bab_frontier_bytes = 1;
    int wrong = 0, answered = 0;
    for (const double t : {least - 0.3, least - 0.02, least + 0.05}) {
        const NnVerifyResult r = nn.verify(aux_below(2, -1.0, 1.0, t), kTimeout, o);
        answered += r.status == "unsat" || r.status == "unknown" || r.status == "sat" ? 1 : 0;
        if (r.status == "unsat")
            for (std::size_t i = 0; i < 20000; ++i) wrong += y[i * 2] <= t ? 1 : 0;
    }
    check(answered == 3 && wrong == 0, b + ": budgets of one byte give a sound answer");
}

/// A bounding or an attack out of memory: verify's last resort leaves the verdict open.
void last_resort(const std::string &b) {
    Rng rng(47);
    const NeuralNetwork nn = aux_network(rng, {2, 6, 2});
    register_attack("oom", [](const NnVerifyOptions &) { return std::make_unique<OomAttack>(); });
    register_bounding("oom",
                      [](const NnVerifyOptions &) { return std::make_unique<OomBounding>(false); });
    register_bounding("oom-prepare",
                      [](const NnVerifyOptions &) { return std::make_unique<OomBounding>(true); });
    NnVerifyOptions o = aux_options(b);
    o.falsification_method = "oom";
    o.bounding_method = "oom";
    const NnVerifyResult bab = nn.verify(aux_below(2, -1.0, 1.0, -1000.0), 1, o);
    check(bab.status == "unknown" && bab.stats.count("bab_oom") && bab.stats.count("attack_oom"),
          b + ": a bounding and an attack out of memory leave the verdict open");
    o.bounding_method = "oom-prepare";
    const NnVerifyResult prep = nn.verify(aux_below(2, -1.0, 1.0, -1000.0), 1, o);
    check(prep.status == "unknown" && prep.stats.count("prepare_oom"),
          b + ": a prepare out of memory leaves the attack alone, the verdict open");
    // the root check first, out of memory: the attack goes on
    o.bounding_method = "oom";
    o.root_check_first = true;
    o.falsification_order = "attack-first";
    const NnVerifyResult root = nn.verify(aux_below(2, -1.0, 1.0, -1000.0), 1, o);
    check(root.status == "unknown" && root.stats.count("root_check_oom"),
          b + ": the root check out of memory leaves the verdict open");
    // a site armed for one call throws once, then not
    priv_inject_oom("test_site", 1, 1);
    bool first = false, second = false, third = false;
    try {
        priv_oom_site("test_site");
    } catch (const Error &e) {
        first = e.id() == Error::out_of_memory;
    }
    try {
        priv_oom_site("test_site");
    } catch (const Error &e) {
        second = e.id() == Error::out_of_memory;
    }
    try {
        priv_oom_site("test_site");
    } catch (const Error &e) {
        third = e.id() == Error::out_of_memory;
    }
    check(!first && second && !third, b + ": an injected out of memory skips, throws, disarms");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        injected(b);
        tiny_budgets(b);
        last_resort(b);
    });
    return test::finish("neuralNetwork verify out of memory");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
