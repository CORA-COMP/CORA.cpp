// benchmark_neur_net_contr_reach_vcas - the VCAS instances of ARCH-COMP 2026
//
// The vertical collision avoidance system has discrete time steps and nine controllers, which
// CORA's neurNetContrSys does not run; CORA's example_neuralNet_reach_07_VCAS.m is its own loop
// over the nine ReLU networks, the advisory of the previous step choosing the network. This is that
// loop, composed of NeuralNetwork::evaluate on a polynomial zonotope and the affine step of the
// plant. Random runs first (10; a violation is FALSIFIED), then the reach: per step the logits are
// enclosed, the advisory is the one that beats all others on the whole set (else UNKNOWN), the
// acceleration is the middle or worst one of the advisory, or 0 by the report's compliance rule
// (the advisory is the previous one and hdot complies; a set that complies in part takes the hull),
// CORACPP_VCAS_RULE=script takes the script's (0 whenever hdot complies). The unsafe set |h| <= 100
// is checked after every step k = 1..10, as the report states (the script checks the last only).
//
// Syntax:   benchmark_neur_net_contr_reach_vcas [instance ...]
// Outputs:  one line per instance on stdout
// See also: benchmark_neur_net_contr_reach_acc.cpp, benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_07_VCAS.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_neur_net.h"
#include "benchmark_run.h"
#include "cora/cora.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::AinncsInstance;
using cora::bench::box_poly;
using cora::bench::kInf;
using cora::bench::load_network;

constexpr double kG = 32.2;
constexpr int kAdvisories = 9, kSteps = 10;

/// The accelerations of an advisory (worst, middle, best) and the climb rates that comply with it.
struct Advisory {
    std::array<double, 3> acc;
    double comply_lo, comply_hi;
};

const std::array<Advisory, kAdvisories> &aux_advisories() {
    static const std::array<Advisory, kAdvisories> table{{
        {{-kG / 8, 0, kG / 8}, -kInf, -kInf},             // COC (nothing complies)
        {{-kG / 3, -7 * kG / 24, -kG / 4}, -kInf, 0},     // DNC
        {{kG / 4, 7 * kG / 24, kG / 3}, 0, kInf},         // DND
        {{-kG / 3, -7 * kG / 24, -kG / 4}, -kInf, -1500}, // DES1500
        {{kG / 4, 7 * kG / 24, kG / 3}, 1500, kInf},      // CL1500
        {{-kG / 3, -kG / 3, -kG / 3}, -kInf, -1500},      // SDES1500
        {{kG / 3, kG / 3, kG / 3}, 1500, kInf},           // SCL1500
        {{-kG / 3, -kG / 3, -kG / 3}, -kInf, -2500},      // SDES2500
        {{kG / 3, kG / 3, kG / 3}, 2500, kInf},           // SCL2500
    }};
    return table;
}

/// The nine networks, one per previous advisory.
std::vector<NeuralNetwork> aux_networks() {
    std::vector<NeuralNetwork> nets;
    for (int i = 1; i <= kAdvisories; ++i)
        nets.push_back(load_network("vcas_" + std::to_string(i)));
    return nets;
}

/// The state after one step with the acceleration a: h - hdot - a/2, hdot + a, tau - 1.
Tensor aux_step_matrix() { return Tensor::from_data({1, -1, 0, 0, 1, 0, 0, 0, 1}, {3, 3}); }

/// Whether a state (h, hdot, tau) is in the unsafe set of the script.
bool aux_unsafe(double h, double hdot, double tau) {
    return -100 <= h && h <= 100 && -1000 <= hdot && hdot <= 1000 && 25 - kSteps <= tau &&
           tau <= 25;
}

/// The runs from random points of R0: the advisory of the first maximal logit, the acceleration of
/// the report's rule on a point; whether a run reaches the unsafe set (the first one is in out).
bool aux_simulate(bool middle, double hdot0, const std::vector<NeuralNetwork> &nets, Rng &rng,
                  ClosedLoopVerdict &out) {
    const auto &advisories = aux_advisories();
    std::vector<double> u(10);
    rng.uniform(u.data(), u.size(), 0.0, 1.0);
    for (const double ui : u) {
        // the state, the start of the run, and the advisory of the step before
        double x[3] = {-133 + 4 * ui, hdot0, 25};
        const std::array<double, 3> x0{x[0], x[1], x[2]};
        int adv = 1;
        bool violated = false;
        int k = 0;
        while (k < kSteps && !violated) {
            ++k;
            const std::vector<double> logits =
                nets[std::size_t(adv - 1)]
                    .evaluate(Tensor::from_data({x[0], x[1], x[2]}, {3, 1}))
                    .data();
            const int next =
                int(std::max_element(logits.begin(), logits.end()) - logits.begin()) + 1;
            const Advisory &a = advisories[std::size_t(next - 1)];
            const double choice = a.acc[middle ? 1 : 0];
            // the report's rule: 0 only if the advisory is the previous one and hdot complies
            const bool complies = a.comply_lo <= x[1] && x[1] <= a.comply_hi;
            const double acc =
                (next == adv || std::getenv("CORACPP_VCAS_RULE")) && complies ? 0.0 : choice;
            x[0] += -x[1] - 0.5 * acc;
            x[1] += acc;
            x[2] -= 1;
            adv = next;
            violated = aux_unsafe(x[0], x[1], x[2]);
        }
        if (violated) {
            out.violating_start = Tensor::from_data({x0[0], x0[1], x0[2]}, {3, 1});
            out.violating_state = Tensor::from_data({x[0], x[1], x[2]}, {3, 1});
            out.violation_time = k;
            return true;
        }
    }
    return false;
}

/// The reach of the script's loop; the verdict is VERIFIED when all ten steps are done and no set
/// meets the unsafe set, UNKNOWN when the advisory is not determined or a set meets it.
ClosedLoopVerdict aux_reach(bool middle, double hdot0, const std::vector<NeuralNetwork> &nets,
                            bool script_rule) {
    const auto &advisories = aux_advisories();
    ClosedLoopVerdict out;
    PolyZonotope R = box_poly({-133, hdot0, 25}, {-129, hdot0, 25});
    int adv = 1;
    NnOptions options;
    for (int k = 1; k <= kSteps; ++k) {
        // the advisory that beats every other one for the whole set
        const Interval logits =
            nets[std::size_t(adv - 1)].evaluate(R, options).zonotope().interval();
        const std::vector<double> lo = logits.inf().data(), hi = logits.sup().data();
        int next = -1;
        for (int j = 0; j < kAdvisories && next < 0; ++j) {
            bool wins = true;
            for (int i = 0; i < kAdvisories; ++i)
                wins = wins && (i == j || hi[std::size_t(i)] <= lo[std::size_t(j)]);
            if (wins) next = j + 1;
        }
        if (next < 0) {
            std::cerr << "  unable to determine the maximum at step " << k << '\n';
            return out;
        }
        // the acceleration: 0 where the climb rate complies and the advisory stays (the report)
        const Advisory &a = advisories[std::size_t(next - 1)];
        const double choice = a.acc[middle ? 1 : 0];
        const Interval box = R.zonotope().interval();
        const double hl = box.inf().data()[1], hu = box.sup().data()[1];
        const bool contained = a.comply_lo <= hl && hu <= a.comply_hi;
        const bool meets = !(hu < a.comply_lo || hl > a.comply_hi);
        const bool same = script_rule || next == adv;
        R = R.linear_map(aux_step_matrix());
        if (same && contained) {
            R = R.plus(Tensor::from_data({0.0, 0.0, -1.0}, {3, 1}));
        } else if (same && meets) {
            // the hull of the accelerations 0 and the choice: a segment along (-1/2, 1, 0)
            const double mid = choice / 2, half = std::abs(choice) / 2;
            R = R.plus(Zonotope(Tensor::from_data({-0.5 * mid, mid, -1.0}, {3, 1}),
                                Tensor::from_data({-0.5 * half, half, 0.0}, {3, 1})));
        } else {
            R = R.plus(Tensor::from_data({-0.5 * choice, choice, -1.0}, {3, 1}));
        }
        adv = next;
        // the unsafe set after step k (the script: the last step only)
        const Interval now = R.zonotope().interval();
        const double h_lo = now.inf().data()[0], h_hi = now.sup().data()[0];
        out.time_reached = k;
        if ((!script_rule || k == kSteps) && !(h_hi < -100 || h_lo > 100)) {
            std::cerr << "  the set at step " << k << " has h in [" << h_lo << ", " << h_hi
                      << "]\n";
            return out;
        }
    }
    out.verdict = Verdict::verified;
    return out;
}

/// One instance: the random runs, then the reach if none violates.
ClosedLoopVerdict aux_vcas(bool middle, double hdot0, Rng &rng) {
    const std::vector<NeuralNetwork> nets = aux_networks();
    ClosedLoopVerdict out;
    if (aux_simulate(middle, hdot0, nets, rng, out)) {
        out.verdict = Verdict::falsified;
        return out;
    }
    return aux_reach(middle, hdot0, nets, std::getenv("CORACPP_VCAS_RULE") != nullptr);
}

std::vector<AinncsInstance> aux_instances() {
    // CORA's verdicts and times of the report (Table 1)
    struct Row {
        bool middle;
        double hdot0;
        Verdict expected;
        double seconds;
    };
    const Verdict V = Verdict::verified, F = Verdict::falsified;
    const std::vector<Row> rows{{true, -19.5, V, 0.215},  {true, -22.5, V, 0.129},
                                {true, -25.5, F, 0.065},  {true, -28.5, F, 0.064},
                                {false, -19.5, V, 0.118}, {false, -22.5, F, 0.061},
                                {false, -25.5, F, 0.042}, {false, -28.5, F, 0.032}};
    std::vector<AinncsInstance> v;
    for (const Row &r : rows) {
        char label[32];
        std::snprintf(label, sizeof label, "%s-%.1f", r.middle ? "middle" : "worst", -r.hdot0);
        v.push_back({"VCAS", label, r.expected, r.seconds,
                     [=](Rng &rng) { return aux_vcas(r.middle, r.hdot0, rng); }});
    }
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
