// benchmark_sound_verdict_arch23 - a verdict of CORA.cpp, proved by the CORALean oracle
//
// For one small ARCH-COMP instance per family that CORA.cpp verifies, asks the oracle for the same
// verdict at the sound dtype binary64 (outward rounding), through CORALean's reach and its
// specification check: `true` there is a verdict backed by a proof of the float definitions. The
// oracle's reach has a fixed step, more conservative than CORA's adaptive verify. The program
// fails if CORA.cpp or the oracle does not verify; a family the oracle cannot serve exits 77
// (ctest: skipped) and says why. linear: RAND01. hybrid: skipped, the oracle has the steps of an
// automaton, not hybridAutomaton.reach. nonlinear: skipped, CORA.cpp has no client yet.
//
// Syntax:   build/core/benchmarks/benchmark_sound_verdict_arch23 linear|hybrid|nonlinear
// Outputs:  one line per family: what CORA.cpp says, what the oracle proves, the oracle's time
// Needs:    CORACPP_ORACLE (with CORACPP_REQUIRE_ORACLE its absence is a failure)
// See also: benchmark_data.h, include/cora/global/oracle/linear_sys.h, specification.h
//
// CORA.cpp only: the oracle's sound verdicts have no CORA counterpart.
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_data.h"
#include "cora/global/oracle/linear_sys.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/specification.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;

constexpr int kSkip = 77; // ctest's SKIP_RETURN_CODE of these tests

/// Whether the oracle is configured; without it the test skips, or fails under the requirement.
bool aux_oracle() {
    const char *command = std::getenv("CORACPP_ORACLE");
    if (command && *command) return true;
    const char *required = std::getenv("CORACPP_REQUIRE_ORACLE");
    if (required && *required && std::string(required) != "0")
        throw std::runtime_error("CORACPP_REQUIRE_ORACLE is set, but CORACPP_ORACLE is not");
    return false;
}

/// The constraint rows A, b of the safe or unsafe sets of an instance over the states: A_s C.
struct StateSpec {
    std::string kind;
    Tensor A, b;
};

std::vector<StateSpec> aux_state_specs(const bench::Json &meta, const Tensor &C) {
    std::vector<StateSpec> out;
    for (const bench::Json &spec : meta["specs"].array) {
        std::vector<double> rows, offsets;
        const int64_t m = static_cast<int64_t>(spec["A"].array.size());
        const int64_t p = static_cast<int64_t>(spec["A"][0].array.size());
        for (std::size_t k = 0; k < spec["A"].array.size(); ++k) {
            for (const bench::Json &x : spec["A"][k].array) rows.push_back(x.number);
            offsets.push_back(spec["b"][k].number);
        }
        out.push_back({spec["type"].string, Tensor::from_data(rows, {m, p}).matmul(C),
                       Tensor::from_data(offsets, {m, 1})});
    }
    return out;
}

/// The linear family: RAND01, verified by CORA.cpp and proved by the oracle.
int aux_linear() {
    const std::string name = "RAND01";
    const bench::Instance inst = bench::load_instance(name);
    const VerifyResult mine = inst.sys->verify(*inst.params, inst.specs, inst.alg);
    std::cout << name << ": CORA.cpp verifies " << mine.verified << " in " << mine.num_steps
              << " steps\n";
    if (!mine.verified) return 1;
    if (!aux_oracle()) return kSkip;

    // the same system, in the oracle's dtype: rounded outward in every operation
    const auto [meta, bin] = bench::read_instance_files(name);
    const Tensor C = bench::read_tensor(meta["matrices"], bin, "C");
    const std::vector<StateSpec> specs = aux_state_specs(meta, C);
    const lean::DTypeScope scope("binary64");
    const lean::LinearSys sys(bench::read_tensor(meta["matrices"], bin, "A"),
                              bench::read_tensor(meta["matrices"], bin, "B"));
    const Zonotope &X0 = *inst.params->initial_set, &U = *inst.params->input_set;
    // the oracle's step is fixed: the horizon is a multiple of it, in exact binary arithmetic
    const double t_final = inst.params->t_final, time_step = t_final / 512; // a power of two: exact
    const auto start = std::chrono::steady_clock::now();
    const lean::ReachSet R = sys.reach(
        lean::Zonotope(X0.center(), X0.generators()), lean::Zonotope(U.center(), U.generators()),
        t_final, alg::Standard{.time_step = time_step, .taylor_terms = 8, .zonotope_order = 20});
    bool proved = true;
    for (const lean::Zonotope &Z : R.time_interval)
        for (const StateSpec &s : specs)
            proved = proved && lean::check_specification(s.kind, s.A, s.b, Z);
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
    std::cout << name << ": the oracle proves " << proved << " over " << R.time_interval.size()
              << " time-interval sets, step " << time_step << ", " << wall.count() << " s\n";
    return proved ? 0 : 1;
}

/// The hybrid family has no sound verdict: the oracle serves location and transition steps only.
int aux_hybrid() {
    std::cout
        << "hybrid: skipped. CORALean serves the steps of a hybrid automaton (location.reach, "
           "transition.jump, the guard intersections), not hybridAutomaton.reach, so the "
           "verdict of an instance with a jump has no sound counterpart yet\n";
    return kSkip;
}

/// The nonlinear family has no sound verdict yet.
int aux_nonlinear() {
    std::cout
        << "nonlinear: skipped. CORA.cpp has no client for nonlinearSys.reach of the oracle\n";
    return kSkip;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    const std::string family = argc > 1 ? argv[1] : "";
    try {
        if (family == "linear") return aux_linear();
        if (family == "hybrid") return aux_hybrid();
        if (family == "nonlinear") return aux_nonlinear();
        std::cerr << "usage: benchmark_sound_verdict_arch23 linear|hybrid|nonlinear\n";
    } catch (const std::exception &e) {
        std::cerr << family << ": " << e.what() << '\n';
    }
    return 1;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
