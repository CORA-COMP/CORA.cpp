// benchmark_neur_net_contr_reach_arch26_ainncs - the AINNCS instances of ARCH-COMP 2026
//
// MATLAB CORA's examples/contDynamics/neurNetContrSys/example_neuralNet_reach_*.m, run through
// verify: Unicycle, ACC, TORA (remain, reach-tanh, reach-sigmoid), Single Pendulum, Double
// Pendulum (less-robust, more-robust), Airplane and Attitude Control. The networks are the
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the example scripts; the plants are written
// out as the model files write them, the numbers and the settings (time step, orders, sampling
// time) are those of the scripts, which the comments name where they differ. Each instance prints
// the line of the scripts, 'benchmark,instance,verdict,time' (the time of verify, simulation
// included), and the expected verdict of the ARCH-COMP 2026 report (CORA, Table 1) on stderr. The
// exit code counts the verdicts that differ from the report's.
//
// Syntax:   build/core/benchmarks/benchmark_neur_net_contr_reach_arch26_ainncs [Unicycle ...]
// Outputs:  one line per instance on stdout
// See also: benchmark_nonlinear_reach_arch23_laubloomis.cpp, benchmark_data.h
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_data.h"
#include "benchmark_run.h"
#include "cora/cora.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::Json;
using cora::bench::JsonParser;
using cora::bench::read_file;

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The network <name>.nn.json / .nn.bin of examples/benchmarks/data/networks.
NeuralNetwork aux_network(const std::string &name) {
    // the folder of the data, and the numbers of the binary file
    const std::string folder = std::getenv("CORACPP_BENCHMARK_DATA")
                                   ? std::getenv("CORACPP_BENCHMARK_DATA")
                                   : CORACPP_BENCHMARK_DATA;
    const Json meta = JsonParser(read_file(folder + "/networks/" + name + ".nn.json")).parse();
    const std::string bin = read_file(folder + "/networks/" + name + ".nn.bin");
    const auto *numbers = reinterpret_cast<const double *>(bin.data());
    // a layer: W row-major and b at their offsets, and the activation by name
    std::vector<NeuralNetwork::Layer> layers;
    std::vector<Activation> activations;
    for (const Json &layer : meta["layers"].array) {
        const int64_t rows = layer["out"].integer(), cols = layer["in"].integer();
        const double *W = numbers + layer["W"].integer(), *b = numbers + layer["b"].integer();
        layers.push_back({Tensor::from_data(std::vector<double>(W, W + rows * cols), {rows, cols}),
                          Tensor::from_data(std::vector<double>(b, b + rows), {rows, 1})});
        const std::string &act = layer["activation"].string;
        activations.push_back(act == "relu"      ? Activation::relu
                              : act == "tanh"    ? Activation::tanh
                              : act == "sigmoid" ? Activation::sigmoid
                                                 : Activation::identity);
    }
    return {layers, activations};
}

/// The box [lo, hi] (bounds may be infinite) as the halfspaces of a polytope.
Polytope aux_box(const std::vector<double> &lo, const std::vector<double> &hi) {
    // one halfspace per finite bound
    const auto n = int64_t(lo.size());
    std::vector<double> A, b;
    int64_t rows = 0;
    // the upper bound as x_i <= hi, the lower one as -x_i <= -lo
    for (int64_t i = 0; i < n; ++i) {
        for (const double sign : {1.0, -1.0}) {
            const double bound = sign > 0 ? hi[std::size_t(i)] : -lo[std::size_t(i)];
            if (std::isinf(bound)) continue;
            // the row of the halfspace
            for (int64_t j = 0; j < n; ++j) A.push_back(j == i ? sign : 0.0);
            b.push_back(bound);
            ++rows;
        }
    }
    return {Tensor::from_data(A, {rows, n}), Tensor::from_data(b, {rows, 1})};
}

Zonotope aux_zonotope(const std::vector<double> &lo, const std::vector<double> &hi) {
    const auto n = int64_t(lo.size());
    std::vector<double> c, r;
    for (std::size_t i = 0; i < lo.size(); ++i) {
        c.push_back((lo[i] + hi[i]) / 2);
        r.push_back((hi[i] - lo[i]) / 2);
    }
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(r, {n, 1}).diag()};
}

PolyZonotope aux_poly(const std::vector<double> &lo, const std::vector<double> &hi) {
    const auto n = int64_t(lo.size());
    return PolyZonotope(Interval(Tensor::from_data(lo, {n, 1}), Tensor::from_data(hi, {n, 1})));
}

/// One instance: how to verify it, and the verdict of the ARCH-COMP 2026 report (CORA).
struct Instance {
    std::string benchmark, label;
    Verdict expected;
    double expected_time; ///< CORA's time in the report (Table 1)
    std::function<ClosedLoopVerdict(Rng &)> run;
    std::string tuning = ""; ///< what differs from the script (settings, input), if anything
};

// The plants -------------------------------------------------------------------------------------

using Vars = std::vector<Expr>;

/// example_neuralNet_reach_01_unicycle.m
NonlinearSys aux_unicycle() {
    return {[](const Vars &x, const Vars &u) {
                return Vars{x[3] * cos(x[2]), x[3] * sin(x[2]), u[1] - 20.0, u[0] - 20.0 + u[2]};
            },
            4, 3};
}

/// example_neuralNet_reach_02_ACC.m: both cars with the air drag u_f x^2 on the velocity.
NonlinearSys aux_acc() {
    constexpr double u_f = 0.0001, a_lead = -2;
    return {[=](const Vars &x, const Vars &u) {
                return Vars{x[1], x[2], -2.0 * x[2] + 2.0 * a_lead - u_f * pow(x[1], 2),
                            x[4], x[5], -2.0 * x[5] + 2.0 * u[0] - u_f * pow(x[3], 2)};
            },
            6, 1};
}

/// example_neuralNet_reach_03_TORA.m (the output is u = f(x) - 10) and the heterogeneous ones
/// (u = 11 f(x)).
NonlinearSys aux_tora(double gain, double offset) {
    return {[=](const Vars &x, const Vars &u) {
                return Vars{x[1], -1.0 * x[0] + 0.1 * sin(x[2]), x[3], gain * u[0] + offset};
            },
            4, 1};
}

/// example_neuralNet_reach_04_singlePendulum.m: m = 0.5, L = 0.5, c = 0, g = 1.
NonlinearSys aux_single_pendulum() {
    constexpr double m = 0.5, L = 0.5, c = 0, g = 1;
    return {[=](const Vars &x, const Vars &u) {
                return Vars{x[1], g / L * sin(x[0]) + 1 / (m * L * L) * (u[0] - c * x[1])};
            },
            2, 1};
}

/// models/Cora/doublePendulum.m
NonlinearSys aux_double_pendulum() {
    constexpr double m = 0.5, L = 0.5, c = 0, g = 1;
    return {[=](const Vars &x, const Vars &u) {
                const Expr d = x[0] - x[1], s = cos(d);
                // the force terms of the first link, shared by both accelerations
                const Expr B = g / L * sin(x[0]) - 0.5 * pow(x[3], 2) * sin(d) +
                               (u[0] - c * x[2]) / (2 * L * L * m);
                const Expr T = pow(x[2], 2) * sin(d) - s * B + g / L * sin(x[1]) +
                               (u[1] - c * x[3]) / (L * L * m);
                const Expr den = 0.5 * pow(cos(d), 2) - 1.0;
                return Vars{x[2], x[3], s * T / (2.0 * den) + B, -1.0 * T / den};
            },
            4, 2};
}

/// A 3x3 matrix of expressions times a vector.
Vars aux_matvec(const std::vector<Vars> &M, const Vars &v) {
    Vars out;
    for (const Vars &row : M) out.push_back(row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
    return out;
}

/// models/Cora/airplane.m: m = 1, g = 1, Ix = Iy = Iz = 1, Ixz = 0, so M = [0 1 0; 1 0 0; 0 0 1].
NonlinearSys aux_airplane() {
    constexpr double m = 1, g = 1, Ix = 1, Iy = 1, Iz = 1, Ixz = 0;
    return {
        [=](const Vars &x, const Vars &u) {
            const Expr c9 = cos(x[8]), s9 = sin(x[8]), c8 = cos(x[7]), s8 = sin(x[7]),
                       c7 = cos(x[6]), s7 = sin(x[6]);
            const Expr zero(0.0), one(1.0);
            // the positions: the rotation T1 T2 T3 of the body velocities
            const Vars f1 = aux_matvec(
                {{c9, -1.0 * s9, zero}, {s9, c9, zero}, {zero, zero, one}},
                aux_matvec({{c8, zero, s8}, {zero, one, zero}, {-1.0 * s8, zero, c8}},
                           aux_matvec({{one, zero, zero}, {zero, c7, -1.0 * s7}, {zero, s7, c7}},
                                      {x[3], x[4], x[5]})));
            // the velocities
            const Vars f2{-g * sin(x[7]) + u[0] / m - x[11] * x[5] + x[9] * x[4],
                          g * cos(x[7]) * sin(x[6]) + u[1] / m - x[9] * x[3] + x[10] * x[5],
                          g * cos(x[7]) * cos(x[6]) + u[2] / m - x[10] * x[4] + x[11] * x[3]};
            // the angles
            const Vars f3 = aux_matvec({{one, tan(x[7]) * sin(x[6]), tan(x[7]) * cos(x[6])},
                                        {zero, c7, -1.0 * s7},
                                        {zero, s7 / c8, c7 / c8}},
                                       {x[10], x[11], x[9]});
            // the rotation rates, with M = [Ixz/a -Ix/a 0; -Iz/a Ixz/a 0; 0 0 Iy], a = -1
            constexpr double a = 1 / (Ixz * Ixz - Ix * Iz);
            const Vars v{u[3] - (Iz - Iy) * x[11] * x[9] - Ixz * x[10] * x[11],
                         u[5] - (Iy - Ix) * x[11] * x[10] - Ixz * x[9] * x[11],
                         u[4] - Ixz * (pow(x[9], 2) - pow(x[10], 2)) - (Ix - Iz) * x[10] * x[9]};
            const Vars f4 = aux_matvec({{Expr(Ixz / a) * one, Expr(-Ix / a) * one, zero},
                                        {Expr(-Iz / a) * one, Expr(Ixz / a) * one, zero},
                                        {zero, zero, Expr(Iy) * one}},
                                       v);
            Vars f;
            for (const Vars *part : {&f1, &f2, &f3, &f4})
                f.insert(f.end(), part->begin(), part->end());
            return f;
        },
        12, 6};
}

/// examples/contDynamics/neurNetContrSys/dynamics/dynamics_attitudeControl.m
NonlinearSys aux_attitude() {
    return {[](const Vars &x, const Vars &u) {
                const Expr &w1 = x[0], &w2 = x[1], &w3 = x[2], &p1 = x[3], &p2 = x[4], &p3 = x[5];
                const Expr p = pow(p1, 2) + pow(p2, 2) + pow(p3, 2);
                return Vars{0.25 * (u[0] + w2 * w3),
                            0.5 * (u[1] - 3.0 * w1 * w3),
                            u[2] + 2.0 * w1 * w2,
                            0.5 * (w2 * (p - p3) + w3 * (p + p2) + w1 * (p + 1.0)),
                            0.5 * (w1 * (p + p3) + w3 * (p - p1) + w2 * (p + 1.0)),
                            0.5 * (w1 * (p - p2) + w2 * (p + p1) + w3 * (p + 1.0))};
            },
            6, 3};
}

// The instances ----------------------------------------------------------------------------------

/// The options of the scripts: alg lin with the step, orders and the sampling of the network.
ClosedLoopOptions aux_options(double time_step, int tensor_order, int taylor_terms,
                              int zonotope_order, int error_order = 0, int intermediate_order = 0) {
    ClosedLoopOptions o;
    // CORACPP_AINNCS_OPTIONS = time_step,tensor_order,taylor,zonotope,error,intermediate overrides
    if (const char *tune = std::getenv("CORACPP_AINNCS_OPTIONS")) {
        std::sscanf(tune, "%lf,%d,%d,%d,%d,%d", &time_step, &tensor_order, &taylor_terms,
                    &zonotope_order, &error_order, &intermediate_order);
    }
    o.plant = {.time_step = time_step,
               .taylor_terms = taylor_terms,
               .zonotope_order = zonotope_order,
               .tensor_order = tensor_order,
               .error_order = error_order,
               .intermediate_order = intermediate_order};
    return o;
}

template <class S>
std::function<ClosedLoopVerdict(Rng &)>
aux_verify(const NonlinearSys &plant, const std::string &net, double dt, const S &X0,
           const std::optional<Zonotope> &U, double t_final, const ClosedLoopOptions &options,
           const TimedSpecification &spec) {
    // the closure owns copies, the instance outlives its arguments
    return [=](Rng &rng) {
        const NeurNetContrSys sys(plant, aux_network(net), dt);
        // CORACPP_BENCHMARK_VERBOSE prints the box of every set of a reach without the spec
        if (std::getenv("CORACPP_BENCHMARK_VERBOSE")) {
            const ClosedLoopReach<S> R = sys.reach(X0, U, t_final, options);
            for (std::size_t k = 0; k < R.time_point.size(); ++k) {
                const Interval box = R.time_point[k].interval();
                std::cerr << "  t = " << R.time[k] << ": inf " << box.inf().matrix_transpose()
                          << " sup " << box.sup().matrix_transpose() << '\n';
            }
        }
        return sys.verify(X0, U, t_final, options, spec, 20, rng);
    };
}

std::vector<Instance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    const std::vector<double> inf4(4, kInf), box4(4, 0.0);
    std::vector<Instance> v;
    // Unicycle: the disturbance input is the script's default zonotope(0)
    const Zonotope no_disturbance(Tensor::from_data({0.0}, {1, 1}), Tensor::zeros({1, 0}));
    v.push_back({"Unicycle", "reach", Verdict::verified, 8.015,
                 aux_verify(aux_unicycle(), "unicycle", 0.2,
                            aux_zonotope({9.5, -4.5, 2.1, 1.5}, {9.55, -4.45, 2.11, 1.51}),
                            no_disturbance, 10.0, aux_options(0.1, 2, 1, 10),
                            {Specification::safe_set(
                                 aux_box({-0.6, -0.2, -0.06, -0.3}, {0.6, 0.2, 0.06, 0.3})),
                             std::make_pair(10.0, 10.0)})});
    // the same with the report's disturbance w in 1e-4 [-1, 1] on the speed, which the script drops
    const Zonotope disturbance(Tensor::from_data({0.0}, {1, 1}), Tensor::from_data({1e-4}, {1, 1}));
    v.push_back({"Unicycle", "reach-disturbance", Verdict::verified, 8.015,
                 aux_verify(aux_unicycle(), "unicycle", 0.2,
                            aux_zonotope({9.5, -4.5, 2.1, 1.5}, {9.55, -4.45, 2.11, 1.51}),
                            disturbance, 10.0, aux_options(0.1, 2, 1, 10),
                            {Specification::safe_set(
                                 aux_box({-0.6, -0.2, -0.06, -0.3}, {0.6, 0.2, 0.06, 0.3})),
                             std::make_pair(10.0, 10.0)}),
                 "the disturbance w of the report (the script sets w = 0)"});
    // ACC: the safe distance D_rel >= 10 + 1.4 v_ego is one halfspace
    v.push_back(
        {"ACC", "safe-distance", Verdict::verified, 3.176,
         aux_verify(
             aux_acc(), "acc", 0.1, aux_poly({90, 32, 0, 10, 30, 0}, {110, 32.2, 0, 11, 30.2, 0}),
             none, 5.0, aux_options(0.1, 2, 1, 10),
             {Specification::safe_set(Polytope(Tensor::from_data({-1, 0, 0, 1, 1.4, 0}, {1, 6}),
                                               Tensor::from_data({-10.0}, {1, 1})))})});
    // TORA, remain
    v.push_back({"TORA", "remain", Verdict::verified, 8.998,
                 aux_verify(aux_tora(1.0, -10.0), "tora_relu", 1.0,
                            aux_poly({0.6, -0.7, -0.4, 0.5}, {0.7, -0.6, -0.3, 0.6}), none, 20.0,
                            aux_options(0.1, 3, 4, 200, 10, 20),
                            {Specification::safe_set(aux_box({-2, -2, -2, -2}, {2, 2, 2, 2}))}),
                 "time_step 0.1 (script 0.25)"});
    // TORA, reach (tanh and sigmoid controllers): the goal set at the end
    for (const bool tanh : {true, false}) {
        v.push_back(
            {"TORA", tanh ? "reach-tanh" : "reach-sigmoid", Verdict::verified, tanh ? 2.356 : 4.730,
             aux_verify(aux_tora(11.0, 0.0), tanh ? "tora_tanh" : "tora_sigmoid", 0.5,
                        aux_poly({-0.77, -0.45, 0.51, -0.3}, {-0.75, -0.43, 0.54, -0.28}), none,
                        5.0,
                        tanh ? aux_options(0.1, 2, 1, 10) : aux_options(0.025, 3, 4, 20, 10, 50),
                        {Specification::safe_set(
                             aux_box({-0.1, -0.9, -kInf, -kInf}, {0.2, -0.6, kInf, kInf})),
                         std::make_pair(5.0, 5.0)}),
             tanh ? "" : "time_step 0.025 (script 0.1)"});
    }
    // Single Pendulum: the angle in [0, 1] from 0.5 to 1
    v.push_back(
        {"Single_Pendulum", "reach", Verdict::verified, 4.083,
         aux_verify(aux_single_pendulum(), "single_pendulum", 0.05,
                    aux_poly({1.0, 0.0}, {1.175, 0.2}), none, 1.0, aux_options(0.02, 2, 1, 10),
                    {Specification::safe_set(aux_box({0.0, -kInf}, {1.0, kInf})),
                     std::make_pair(0.5, 1.0)})});
    // Double Pendulum
    v.push_back(
        {"Double_Pendulum", "less-robust", Verdict::unknown, 5.604,
         aux_verify(aux_double_pendulum(), "double_pendulum_less_robust", 0.05,
                    aux_zonotope({1, 1, 1, 1}, {1.3, 1.3, 1.3, 1.3}), none, 1.0,
                    aux_options(0.01, 2, 4, 20),
                    {Specification::safe_set(aux_box({-1.7, -1.7, -1.7, -1.7}, {2, 2, 2, 2}))})});
    v.push_back(
        {"Double_Pendulum", "more-robust", Verdict::falsified, 5.589,
         aux_verify(
             aux_double_pendulum(), "double_pendulum_more_robust", 0.02,
             aux_poly({1, 1, 1, 1}, {1.3, 1.3, 1.3, 1.3}), none, 0.4, aux_options(0.02, 2, 1, 10),
             {Specification::safe_set(aux_box({-1.5, -1.5, -1.5, -1.5}, {1.5, 1.5, 1.5, 1.5}))})});
    // Airplane: y, phi, theta, psi in [-1, 1]; falsified by a simulation
    std::vector<double> lo(12, -kInf), hi(12, kInf);
    for (const int i : {1, 6, 7, 8}) lo[std::size_t(i)] = -1, hi[std::size_t(i)] = 1;
    std::vector<double> a_lo(12, 0.0), a_hi(12, 0.0);
    for (const int i : {3, 4, 5, 6, 7, 8}) a_hi[std::size_t(i)] = 1;
    v.push_back(
        {"Airplane", "continuous", Verdict::falsified, 6.809,
         aux_verify(aux_airplane(), "airplane", 0.1, aux_poly(a_lo, a_hi), none, 2.0,
                    aux_options(0.1, 2, 4, 10), {Specification::safe_set(aux_box(lo, hi))})});
    // Attitude Control: the unsafe set at t = 3 must not be reached
    v.push_back({"Attitude_Control", "avoid", Verdict::verified, 5.226,
                 aux_verify(aux_attitude(), "attitude_control", 0.1,
                            aux_poly({-0.45, -0.55, 0.65, -0.75, 0.85, -0.65},
                                     {-0.44, -0.54, 0.66, -0.74, 0.86, -0.64}),
                            none, 3.0, aux_options(0.1, 2, 1, 10),
                            {Specification::unsafe_set(aux_box({-0.2, -0.5, 0, -0.7, 0.7, -0.4},
                                                               {0, -0.4, 0.2, -0.6, 0.8, -0.2})),
                             std::make_pair(3.0, 3.0)})});
    return v;
}

const char *aux_name(Verdict v) {
    return v == Verdict::verified ? "VERIFIED" : v == Verdict::falsified ? "FALSIFIED" : "UNKNOWN";
}

/// Runs one instance, prints the line of the scripts and CORA's verdict and time of the report.
bool aux_run(const std::string &name) {
    // the instance of the name
    for (const Instance &inst : aux_instances()) {
        if (inst.benchmark + "-" + inst.label != name) continue;
        // verify with a seeded generator, timed with the simulations as the scripts' tTotal
        Rng rng(1);
        const auto start = std::chrono::steady_clock::now();
        const ClosedLoopVerdict out = inst.run(rng);
        const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
        // the line of the scripts, then what the reach and the report add on stderr
        std::cout << inst.benchmark << "," << inst.label << "," << aux_name(out.verdict) << ","
                  << wall.count() << '\n'
                  << std::flush;
        if (out.verdict == Verdict::unknown)
            std::cerr << "  reach stopped at t = " << out.time_reached
                      << (out.exploded   ? " (sets exploded)"
                          : out.violated ? " (violation)"
                                         : "")
                      << '\n';
        if (out.violating_start)
            std::cerr << "  a simulated run from " << out.violating_start->matrix_transpose()
                      << " violates at t = " << out.violation_time << " in "
                      << out.violating_state->matrix_transpose() << '\n';
        if (!inst.tuning.empty()) std::cerr << "  differs from the script: " << inst.tuning << '\n';
        std::cerr << "  " << name << ": ARCH26 report (CORA): " << aux_name(inst.expected) << ", "
                  << inst.expected_time << " s\n";
        return out.verdict == inst.expected;
    }
    CORA_FAIL(Error::wrong_value, "ainncs: unknown instance " + name);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    std::vector<std::string> names;
    for (const Instance &inst : aux_instances()) names.push_back(inst.benchmark + "-" + inst.label);
    return cora::bench::run_family(argc, argv, names, aux_run);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
