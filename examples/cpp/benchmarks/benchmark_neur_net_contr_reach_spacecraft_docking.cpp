// benchmark_neur_net_contr_reach_spacecraft_docking - Spacecraft Docking
//
// Docking with CORA's script and its own check of the sets, and a second row: the constraint from
// inside by a polytope
// (|s| >= (sx + sy) / sqrt 2, the disk |v| <= R by the inscribed 32-gon, shrunk by 1e-9), verified
// with the splitting rule widest (environment: the README). The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   benchmark_neur_net_contr_reach_spacecraft_docking [instance ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_10_spacecraftDocking.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_neur_net.h"
#include "benchmark_run.h"
#include "cora/cora.h"
#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::AinncsInstance;
using cora::bench::box_poly;
using cora::bench::load_network;
using cora::bench::loop_options;

using Vars = std::vector<Expr>;

// The plant ------------------------------------------------------------------------------------

/// The Clohessy-Wiltshire equations of the docking: f(x, u) = A x + B u.
NonlinearSys aux_docking() {
    return {[](const Vars &x, const Vars &u) {
                constexpr double m = 12, n = 0.001027;
                return Vars{x[2] + 0.0 * x[0], x[3] + 0.0 * x[0],
                            3 * n * n * x[0] + 2 * n * x[3] + u[0] / m, -2 * n * x[2] + u[1] / m};
            },
            4, 2};
}

// The docking check -------------------------------------------------------------------------------

/// The smallest and the largest square of a value in [lo, hi].
std::pair<double, double> aux_square_range(double lo, double hi) {
    const double small = lo <= 0 && hi >= 0 ? 0.0 : std::min(lo * lo, hi * hi);
    return {small, std::max(lo * lo, hi * hi)};
}

/// Whether 0.2 + 2n |s| - |v| >= 0 on the set of the states (sx, sy, vx, vy), by interval
/// arithmetic on its box and, where that fails, on the halves along the longest generator, `depth`
/// times (instead of CORA's quadMap and Singh root, which did not hold the initial set here).
bool aux_docking_safe(const PolyZonotope &P, double n, int depth) {
    const Interval box = P.zonotope().interval();
    const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
    const auto [sx_small, sx_big] = aux_square_range(lo[0], hi[0]);
    const auto [sy_small, sy_big] = aux_square_range(lo[1], hi[1]);
    const auto [vx_small, vx_big] = aux_square_range(lo[2], hi[2]);
    const auto [vy_small, vy_big] = aux_square_range(lo[3], hi[3]);
    (void)sx_big, (void)sy_big, (void)vx_small, (void)vy_small;
    if (0.2 + 2 * n * std::sqrt(sx_small + sy_small) - std::sqrt(vx_big + vy_big) >= 0) return true;
    if (depth == 0 || P.num_dependent() == 0) return false;
    const auto [a, b] = P.split_longest_gen();
    return aux_docking_safe(a, n, depth - 1) && aux_docking_safe(b, n, depth - 1);
}

/// The constraint |v| <= 0.2 + 2n |s| from inside as a polytope in (sx, sy, vx, vy) (file header).
Polytope aux_docking_polytope(double n, int sides) {
    // the polytope of the file header, inside the constraint
    constexpr double pi = 3.14159265358979323846, shrink = 1.0 - 1e-9;
    const double c = std::cos(pi / sides) * shrink;
    std::vector<double> A, b;
    for (int k = 0; k < sides; ++k) {
        const double phi = 2 * pi * k / sides;
        // cos(phi_k) vx + sin(phi_k) vy - c 2n (sx + sy) / sqrt 2 <= 0.2 c
        const double w = c * 2 * n / std::sqrt(2.0) * shrink;
        A.insert(A.end(), {-w, -w, std::cos(phi), std::sin(phi)});
        b.push_back(0.2 * c);
    }
    return {Tensor::from_data(A, {sides, 4}), Tensor::from_data(b, {sides, 1})};
}

/// Whether the state (sx, sy, vx, vy) satisfies the constraint of the docking.
bool aux_docking_point_safe(const std::vector<double> &x, double n) {
    return std::sqrt(x[2] * x[2] + x[3] * x[3]) <=
           0.2 + 2 * n * std::sqrt(x[0] * x[0] + x[1] * x[1]);
}

/// example_neuralNet_reach_10_spacecraftDocking.m: random runs against the constraint, then the
/// reach of the whole set and the check of every set over a step; with polygon, verify against the
/// 32-gon polytope (visualized: https://claude.ai/artifact/65csLQFHAMLGhkGSvH4TvH).
ClosedLoopVerdict aux_docking_run(Rng &rng, bool polygon) {
    constexpr double n = 0.001027;
    const NeurNetContrSys sys(aux_docking(), load_network("docking"), 0.1);
    // the initial set, or one corner piece of it for diagnosis (CORACPP_DOCKING_PIECE)
    std::vector<double> lo{70, 70, -0.28, -0.28}, hi{106, 106, 0.28, 0.28};
    if (const char *env = std::getenv("CORACPP_DOCKING_PIECE")) {
        double f, a[4];
        std::sscanf(env, "%lf,%lf,%lf,%lf,%lf", &f, a, a + 1, a + 2, a + 3);
        for (std::size_t i = 0; i < 4; ++i) {
            const double w = hi[i] - lo[i], l = lo[i];
            lo[i] = l + (w - f * w) * a[i];
            hi[i] = lo[i] + f * w;
        }
    }
    const PolyZonotope X0 = box_poly(lo, hi);
    const ClosedLoopOptions options = loop_options(0.1, 2, 4, 20, 0, 0, PolyMethod::regression);
    ClosedLoopVerdict out;
    out.simulations = 20;
    if (const char *env = std::getenv("CORACPP_AINNCS_POINTS")) out.simulations = std::atoi(env);
    // the random runs, the constraint at every sample
    const ClosedLoopTrajectories traj = sys.simulate_random(
        X0.zonotope(), std::nullopt, 40.0, options.plant.time_step, out.simulations, 0.5, rng);
    // the smallest margin of the constraint over the samples is printed in verbose mode
    const bool verbose = std::getenv("CORACPP_BENCHMARK_VERBOSE") != nullptr;
    double worst = 1e9, worst_time = 0;
    for (std::size_t k = 0; k < traj.x.size(); ++k) {
        const std::vector<double> x = traj.x[k].data();
        const std::size_t N = std::size_t(traj.x[k].shape()[1]);
        for (std::size_t j = 0; j < N; ++j) {
            if (verbose) {
                const double margin = 0.2 + 2 * n * std::hypot(x[j], x[N + j]) -
                                      std::hypot(x[2 * N + j], x[3 * N + j]);
                if (margin < worst) worst = margin, worst_time = traj.time[k];
            }
            // a sample outside the constraint is a violation
            if (!aux_docking_point_safe({x[j], x[N + j], x[2 * N + j], x[3 * N + j]}, n)) {
                out.verdict = Verdict::falsified;
                out.violation_time = traj.time[k];
                return out;
            }
        }
    }
    if (verbose)
        std::cerr << "  smallest margin of the runs " << worst << " at t = " << worst_time << '\n';
    // the polygon: verify with the splitting, against the polytope inside the constraint
    if (polygon) {
        VerifySplit split{.depth = 11, .rule = VerifySplit::Rule::widest};
        if (const char *env = std::getenv("CORACPP_AINNCS_SPLIT"))
            std::sscanf(env, "%d,%d", &split.depth, &split.max_threads);
        int sides = 32;
        if (const char *env = std::getenv("CORACPP_DOCKING_SIDES")) sides = std::atoi(env);
        out =
            sys.verify(X0, std::nullopt, 40.0, options,
                       TimedSpecification{Specification::safe_set(aux_docking_polytope(n, sides))},
                       1, rng, split);
        out.simulations = 20;
        if (out.verdict == Verdict::falsified) out.verdict = Verdict::unknown;
        return out;
    }
    // the reach, and the check of the sets over the steps
    const ClosedLoopReach<PolyZonotope> R = sys.reach(X0, std::nullopt, 40.0, options);
    out.time_reached = R.time.back();
    out.exploded = R.exploded;
    out.verdict = Verdict::verified;
    for (std::size_t j = 0; j < R.time_interval.size() && out.verdict == Verdict::verified; ++j)
        if (!aux_docking_safe(R.time_interval[j], n, 10)) {
            std::cerr << "  unable to verify t in [" << R.time[j] << ", " << R.time[j + 1] << "]\n";
            out.verdict = Verdict::unknown;
        }
    if (!R.completed) out.verdict = Verdict::unknown;
    return out;
}

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    std::vector<AinncsInstance> v;
    // Docking: the constraint is not a polytope; the script's own check
    v.push_back({"Docking", "constraint", Verdict::unknown, 95.038,
                 [](Rng &rng) { return aux_docking_run(rng, false); }});
    v.push_back({"Docking", "constraint-polygon", Verdict::verified, 95.038,
                 [](Rng &rng) { return aux_docking_run(rng, true); },
                 "the constraint from inside by a 32-gon polytope, initial set split in its widest "
                 "dimension (depth 11); CORA's script is UNKNOWN"});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
