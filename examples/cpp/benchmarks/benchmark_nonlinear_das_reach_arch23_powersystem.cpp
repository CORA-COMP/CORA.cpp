// benchmark_nonlinear_das_reach_arch23_powersystem - the IEEE 14 bus power system (TSPS25)
//
// MATLAB CORA's benchmark_nonlinearDAS_reach_ARCH23_powerSystem.m: the nonlinear differential-
// algebraic system of 15 states, 27 algebraic variables and 19 inputs, with the Lagrange remainder
// evaluated on two subsystems (the compositional analysis of Althoff, TPS 2014). Four reach runs
// in sequence, each from the end set of the one before: normal operation to 0.1 s (time step
// 0.005), the fault of bus 1 to 0.13 s (0.001), normal operation to 2 s (0.005) and to 5 s (0.02),
// zonotope order 400, from the steady state with the box 0.01, 0.1, 0.001 around it. The verdict
// is the script's: the set at 5 s lies in the initial box (the report: R(t) returns to R(0)).
// The sets of a step are dropped after the next, so the memory does not grow with the horizon.
//
// Syntax:   build/core/benchmarks/benchmark_nonlinear_das_reach_arch23_powersystem [TSPS25]
// Outputs:  the line 'benchmark,instance,result,time'; the runs and the final set on stderr
// See also: benchmark_power_system.h, benchmark_nonlinear_reach_arch23_laubloomis.cpp
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_power_system.h"

#include <chrono>
#include <iomanip>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;

/// The options the script gives every run, with the time step of the run.
alg::DaeLinearization aux_options(double time_step) {
    alg::DaeLinearization a;
    a.time_step = time_step;
    a.taylor_terms = 6;
    a.zonotope_order = 400;
    a.error_order = 3;
    a.max_error_x.assign(15, 0.0);
    for (int i = 5; i < 10; ++i) a.max_error_x[std::size_t(i)] = 0.6;
    a.max_error_y.assign(27, 0.01);
    return a;
}

/// A column of a vector made of the parts.
Tensor aux_column(const std::vector<double> &gen, std::size_t loads) {
    std::vector<double> u = gen;
    u.resize(gen.size() + loads, 0.0);
    return column(u);
}

/// The subsystems of a case: the selection matrices into the whole system and the voltages of its
/// generator buses (Vgen = VM(bus.generator)).
DaeDecomposition aux_decomposition(const bench::PowerSystemData &data, const std::string &full,
                                   const std::string &sub1, const std::string &sub2) {
    const bench::PowerSystemCase &c = data.at(full);
    std::vector<double> vgen;
    vgen.reserve(c.bus.generator.size());
    for (const int g : c.bus.generator) vgen.push_back(c.VM[std::size_t(g - 1)]);
    return {{bench::subsystem_index(data, c, sub1), bench::subsystem_index(data, c, sub2)},
            column(vgen)};
}

/// One run of the script: reach from `start`, keeping only the last time-point set; the steps and
/// the time of reach are returned with it.
struct Run {
    Zonotope last;
    int steps = 0;
    double seconds = 0;
};

Run aux_reach(const NonlinDASys &sys, const Zonotope &start, const Tensor &y0_guess, double t_start,
              double t_final, double time_step, const Tensor &u, const DaeDecomposition &dec) {
    // the problem of the run, and its steps one after the other
    const alg::DaeReach alg = aux_options(time_step);
    const DaeReachProblem problem{.initial_set = start,
                                  .input_set = std::nullopt,
                                  .u_trans = u,
                                  .y0_guess = y0_guess,
                                  .t_start = t_start,
                                  .t_final = t_final,
                                  .decomposition = dec};
    const int steps = static_cast<int>(std::ceil((t_final - t_start) / time_step - 1e-9));
    const auto begin = std::chrono::steady_clock::now();
    DaeStep step = sys.init_reach(problem, alg);
    for (int k = 1; k < steps; ++k) step = sys.post(step, problem, u, alg);
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - begin;
    return {step.time_point, steps, wall.count()};
}

/// Runs the four reach runs and decides the instance; prints the MATLAB line.
bool aux_run(const std::string &name) {
    if (name != "TSPS25") CORA_FAIL(Error::wrong_value, "powerSystem: unknown instance " + name);
    const bench::PowerSystemData data = bench::load_power_system();
    // the models and their subsystems, in normal operation and under the fault
    const NonlinDASys model = data.model("IEEE14"), fault = data.model("IEEE14_fault");
    const DaeDecomposition normal = aux_decomposition(data, "IEEE14", "IEEE14_sub1", "IEEE14_sub2");
    const DaeDecomposition faulty =
        aux_decomposition(data, "IEEE14_fault", "IEEE14_fault_sub1", "IEEE14_sub2");
    // the steady state of the normal operation, from the script's guesses
    const Tensor u_normal = aux_column({2, 0.4, 0, 0, 0}, 14),
                 u_fault = aux_column({0, 0.4, 0, 0, 0}, 14);
    std::vector<double> x_guess(15, 0.0), y_guess(27, 0.0);
    for (int i = 5; i < 10; ++i) x_guess[std::size_t(i)] = 377;
    for (int i = 10; i < 15; ++i) x_guess[std::size_t(i)] = 1;
    for (int i = 0; i < 14; ++i) y_guess[std::size_t(i)] = 1;
    auto [x0, y0_guess] = model.steady_state(column(x_guess), column(y_guess), u_normal);
    // the box around it: 0.01 on the angles, 0.1 on the speeds, 0.001 on the voltages
    std::vector<double> radius(15, 0.01);
    for (int i = 5; i < 15; ++i) radius[std::size_t(i)] = i < 10 ? 0.1 : 0.001;
    const Tensor G0 = column(radius).diag();
    const Zonotope R0(x0, G0);
    // the four runs
    double total = 0;
    Zonotope R = R0;
    int steps = 0;
    const auto run = [&](const NonlinDASys &sys, double t_start, double t_final, double dt,
                         const Tensor &u, const DaeDecomposition &dec) {
        const Run r = aux_reach(sys, R, y0_guess, t_start, t_final, dt, u, dec);
        R = r.last;
        total += r.seconds;
        steps += r.steps;
        std::cerr << "  [" << t_start << ", " << t_final << "], dt " << dt << ": " << r.steps
                  << " steps, " << r.seconds << " s\n";
        // CORACPP_BENCHMARK_VERBOSE=1: the end set of the run (center and widths), as MATLAB prints
        if (std::getenv("CORACPP_BENCHMARK_VERBOSE")) {
            std::cerr << std::setprecision(12) << "    center:";
            for (const double v : R.center().data()) std::cerr << " " << v;
            std::cerr << "\n    widths:";
            const Interval box = R.interval();
            const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
            for (std::size_t i = 0; i < lo.size(); ++i) std::cerr << " " << hi[i] - lo[i];
            std::cerr << "\n";
        }
    };
    run(model, 0, 0.1, 0.005, u_normal, normal);
    run(fault, 0.1, 0.13, 0.001, u_fault, faulty);
    run(model, 0.13, 2, 0.005, u_normal, normal);
    run(model, 2, 5, 0.02, u_normal, normal);
    // the script's verdict: the interval hull of the end set lies in the initial box
    const auto check = std::chrono::steady_clock::now();
    const bool verified = R0.contains(R.interval());
    total += std::chrono::duration<double>(std::chrono::steady_clock::now() - check).count();
    cora::bench::print_line("PowerSystem", name, verified, total);
    const Interval box = R.interval();
    std::cerr << "  " << steps << " steps; x1 in [" << box.inf().data()[0] << ", "
              << box.sup().data()[0] << "], x6 in [" << box.inf().data()[5] << ", "
              << box.sup().data()[5] << "] at t = 5\n";
    return verified;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_family(argc, argv, {"TSPS25"}, aux_run); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
