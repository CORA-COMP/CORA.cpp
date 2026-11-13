// benchmark_neur_net - what the AINNCS benchmark programs share
//
// The network of a name (examples/benchmarks/data/networks/<name>.nn.json and .nn.bin, which
// matlab/benchmarks/export_network_data.m writes from the example scripts), the boxes of a
// specification and of an initial set as the sets the scripts build, the algorithm options of the
// scripts, and the names of the verdicts.
//
// Syntax:   const NeuralNetwork nn = cora::bench::load_network("quad");
//           const PolyZonotope X0 = cora::bench::box_poly({-1, 0}, {1, 0});
// See also: benchmark_data.h, benchmark_neur_net_contr_reach_unicycle.cpp
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "benchmark_data.h"
#include "benchmark_run.h"
#include "cora/cora.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::bench {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The network <name>.nn.json / .nn.bin of examples/benchmarks/data/networks.
inline NeuralNetwork load_network(const std::string &name) {
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
inline Polytope box_polytope(const std::vector<double> &lo, const std::vector<double> &hi) {
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

/// The zonotope of the box (CORA's zonotope(interval(lo, hi))).
inline Zonotope box_zonotope(const std::vector<double> &lo, const std::vector<double> &hi) {
    const auto n = int64_t(lo.size());
    std::vector<double> c, r;
    for (std::size_t i = 0; i < lo.size(); ++i) {
        c.push_back((lo[i] + hi[i]) / 2);
        r.push_back((hi[i] - lo[i]) / 2);
    }
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(r, {n, 1}).diag()};
}

/// The polynomial zonotope of the box (CORA's polyZonotope(interval(lo, hi))).
inline PolyZonotope box_poly(const std::vector<double> &lo, const std::vector<double> &hi) {
    const auto n = int64_t(lo.size());
    return PolyZonotope(Interval(Tensor::from_data(lo, {n, 1}), Tensor::from_data(hi, {n, 1})));
}

/// The options of the scripts: alg lin with the step, orders and the method of the network's
/// polynomials. CORACPP_AINNCS_OPTIONS = time_step,tensor_order,taylor,zonotope,error,intermediate
/// overrides the numbers (how the tuned settings were found).
inline ClosedLoopOptions loop_options(double time_step, int tensor_order, int taylor_terms,
                                      int zonotope_order, int error_order = 0,
                                      int intermediate_order = 0,
                                      PolyMethod method = PolyMethod::singh) {
    ClosedLoopOptions o;
    if (const char *tune = std::getenv("CORACPP_AINNCS_OPTIONS")) {
        std::sscanf(tune, "%lf,%d,%d,%d,%d,%d", &time_step, &tensor_order, &taylor_terms,
                    &zonotope_order, &error_order, &intermediate_order);
    }
    // the algorithm of the plant, then the method of the network
    o.plant = {.time_step = time_step,
               .taylor_terms = taylor_terms,
               .zonotope_order = zonotope_order,
               .tensor_order = tensor_order,
               .error_order = error_order,
               .intermediate_order = intermediate_order,
               .error_bound = cora::bench::error_bound()};
    o.nn.poly_method = method;
    return o;
}

inline const char *verdict_name(Verdict v) {
    return v == Verdict::verified ? "VERIFIED" : v == Verdict::falsified ? "FALSIFIED" : "UNKNOWN";
}

/// One instance: how to verify it, and the verdict of the ARCH-COMP 2026 report (CORA).
struct AinncsInstance {
    std::string benchmark, label;
    Verdict expected;
    double expected_time; ///< CORA's time in the report (Table 1)
    std::function<ClosedLoopVerdict(Rng &)> run;
    std::string tuning = ""; ///< what differs from the script (settings, input), if anything
};

/// verify of the closed loop of a plant and a network, with `points` random runs; the closure owns
/// copies, as the instance outlives its arguments. CORACPP_BENCHMARK_VERBOSE prints the box of
/// every set of a reach without the specification.
template <class S>
std::function<ClosedLoopVerdict(Rng &)>
make_verify(const NonlinearSys &plant, const std::string &net, double dt, const S &X0,
            const std::optional<Zonotope> &U, double t_final, const ClosedLoopOptions &options,
            const TimedSpecifications &specs, const VerifySplit &split = {}, int64_t points = 20) {
    return [=](Rng &rng) {
        const NeurNetContrSys sys(plant, load_network(net), dt);
        if (cora::bench::env_flag("CORACPP_BENCHMARK_VERBOSE")) {
            const ClosedLoopReach<S> R = sys.reach(X0, U, t_final, options);
            for (std::size_t k = 0; k < R.time_point.size(); ++k) {
                const Interval box = R.time_point[k].interval();
                std::cerr << "  t = " << R.time[k] << ": inf";
                for (const double d : box.inf().data()) std::cerr << ' ' << d;
                std::cerr << " sup";
                for (const double d : box.sup().data()) std::cerr << ' ' << d;
                std::cerr << '\n';
            }
        }
        // CORACPP_AINNCS_SPLIT = depth,threads[,rule] overrides the splitting of verify
        VerifySplit tuned = split;
        if (const char *env = std::getenv("CORACPP_AINNCS_SPLIT")) {
            int rule = int(tuned.rule);
            std::sscanf(env, "%d,%d,%d", &tuned.depth, &tuned.max_threads, &rule);
            tuned.rule = VerifySplit::Rule(rule);
        }
        // CORACPP_AINNCS_POINTS = n sets the number of random runs
        int64_t runs = points;
        if (const char *env = std::getenv("CORACPP_AINNCS_POINTS")) runs = std::atoll(env);
        return sys.verify(X0, U, t_final, options, specs, runs, rng, tuned);
    };
}

template <class S>
std::function<ClosedLoopVerdict(Rng &)>
make_verify(const NonlinearSys &plant, const std::string &net, double dt, const S &X0,
            const std::optional<Zonotope> &U, double t_final, const ClosedLoopOptions &options,
            const TimedSpecification &spec, const VerifySplit &split = {}, int64_t points = 20) {
    return make_verify(plant, net, dt, X0, U, t_final, options, TimedSpecifications{spec}, split,
                       points);
}

/// Runs the instance of the name, prints the line of the scripts ('benchmark,instance,verdict,
/// time', the time of the whole run) and, on stderr, what the run and the report add. True when
/// the verdict is the expected one.
inline bool run_instance(const std::vector<AinncsInstance> &instances, const std::string &name) {
    for (const AinncsInstance &inst : instances) {
        if (inst.benchmark + "-" + inst.label != name) continue;
        // a seeded generator, timed with the simulations as the scripts' tTotal
        Rng rng(1);
        const auto start = std::chrono::steady_clock::now();
        const ClosedLoopVerdict out = inst.run(rng);
        const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
        // the line of the scripts, then what the run adds on stderr
        std::cout << inst.benchmark << "," << inst.label << "," << verdict_name(out.verdict) << ","
                  << wall.count() << '\n'
                  << std::flush;
        if (out.verdict == Verdict::unknown)
            std::cerr << "  reach up to t = " << out.time_reached
                      << (out.exploded   ? " (sets exploded)"
                          : out.violated ? " (violation)"
                                         : "")
                      << '\n';
        // the splitting and the simulated run of a falsification
        if (out.pieces_reached > 1)
            std::cerr << "  split: " << out.pieces_reached << " pieces reached, "
                      << out.pieces_verified << " leaves verified, at most " << out.max_splits
                      << " splits deep\n";
        if (out.violating_start)
            std::cerr << "  a simulated run from " << out.violating_start->matrix_transpose()
                      << " violates at t = " << out.violation_time << " in "
                      << out.violating_state->matrix_transpose() << '\n';
        if (!inst.tuning.empty()) std::cerr << "  differs from the script: " << inst.tuning << '\n';
        std::cerr << "  " << name << ": ARCH26 report (CORA): " << verdict_name(inst.expected)
                  << ", " << inst.expected_time << " s\n";
        return out.verdict == inst.expected;
    }
    CORA_FAIL(Error::wrong_value, "ainncs: unknown instance " + name);
}

/// The main of a program: the instances named on the command line, else all of them; the exit code
/// counts the verdicts that differ from the report's.
inline int run_instances(int argc, char **argv,
                         const std::function<std::vector<AinncsInstance>()> &make) {
    std::vector<std::string> names;
    for (const AinncsInstance &inst : make()) names.push_back(inst.benchmark + "-" + inst.label);
    return run_family(argc, argv, names,
                      [&](const std::string &name) { return run_instance(make(), name); });
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
