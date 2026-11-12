// benchmark_neur_net_contr_reach_arch26_ainncs_b - QUAD, Docking, NAV, CartPole (ARCH-COMP 2026)
//
// MATLAB CORA's benchmark_neuralNet_reach_09_QUAD.m,
// example_neuralNet_reach_10_spacecraftDocking.m, example_neuralNet_reach_11_NAV.m (standard:
// point-trained network, robust: set-trained) and example_neuralNet_reach_12_cartPole.m with CORA's
// settings (poly_method regression of order 1). QUAD, NAV and CartPole run through verify with the
// splitting of the script (NAV 9, CartPole 10) or the one that was needed. Docking is CORA's script
// with its own check of the sets, and a second row: the constraint from inside by a polytope (|s|
// >= (sx + sy) / sqrt 2, the disk |v| <= R by the inscribed 32-gon, coefficients shrunk by 1e-9),
// verified with the splitting rule `widest`. CartPole has a second row too, from the report's
// initial set (the script's has 0.4 times its radius). Each instance prints the line of the scripts
// and, on stderr, the verdict and time of the ARCH-COMP 2026 report. Environment:
// CORACPP_AINNCS_OPTIONS, CORACPP_AINNCS_SPLIT, CORACPP_AINNCS_POINTS, CORACPP_DOCKING_SIDES,
// CORACPP_DOCKING_PIECE = f,a,b,c,d (a corner).
//
// Syntax:   build/core/benchmarks/benchmark_neur_net_contr_reach_arch26_ainncs_b [QUAD-reach ...]
// Outputs:  one line per instance on stdout
// See also: benchmark_neur_net_contr_reach_arch26_ainncs.cpp, benchmark_neur_net.h
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
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
using cora::bench::box_polytope;
using cora::bench::box_zonotope;
using cora::bench::kInf;
using cora::bench::load_network;
using cora::bench::loop_options;
using cora::bench::make_verify;

using Vars = std::vector<Expr>;

// The plants -------------------------------------------------------------------------------------

/// dynamics_quad.m (tau = 0) with the report's x8', x9'; CORACPP_QUAD_SCRIPT=1 takes the
/// script's, which differ from the report (cora-differences.md).
NonlinearSys aux_quad() {
    const bool report = std::getenv("CORACPP_QUAD_SCRIPT") == nullptr;
    return {
        [=](const Vars &x, const Vars &u) {
            constexpr double g = 9.81, m = 1.4, Jx = 0.054, Jy = 0.054, Jz = 0.104;
            // the sines and cosines of the angles, then the positions, velocities and rates
            const Expr s7 = sin(x[6]), c7 = cos(x[6]), s8 = sin(x[7]), c8 = cos(x[7]),
                       s9 = sin(x[8]), c9 = cos(x[8]);
            return Vars{
                c8 * c9 * x[3] + (s7 * s8 * c9 - c7 * s9) * x[4] + (c7 * s8 * c9 + s7 * s9) * x[5],
                c8 * s9 * x[3] + (s7 * s8 * s9 + c7 * c9) * x[4] + (c7 * s8 * s9 - s7 * c9) * x[5],
                s8 * x[3] - s7 * c8 * x[4] - c7 * c8 * x[5],
                x[11] * x[4] - x[10] * x[5] - g * s8,
                x[9] * x[5] - x[11] * x[3] + g * c8 * s7,
                x[10] * x[3] - x[9] * x[4] + g * c8 * c7 - g - u[0] / m,
                x[9] + s7 * tan(x[7]) * x[10] + c7 * tan(x[7]) * x[11],
                report ? c7 * x[10] - s7 * x[11] : c7 * x[10] + s7 * x[11],
                report ? s7 / c8 * x[10] - c7 / c8 * x[11] : s7 / c8 * x[10] - s7 * x[11],
                (Jy - Jz) / Jx * x[10] * x[11] + u[1] / Jx,
                (Jz - Jx) / Jy * x[9] * x[11] + u[2] / Jy,
                (Jx - Jy) / Jz * x[9] * x[10] + 0.0 / Jz};
        },
        12, 3};
}

/// The Clohessy-Wiltshire equations of the docking: f(x, u) = A x + B u.
NonlinearSys aux_docking() {
    return {[](const Vars &x, const Vars &u) {
                constexpr double m = 12, n = 0.001027;
                return Vars{x[2] + 0.0 * x[0], x[3] + 0.0 * x[0],
                            3 * n * n * x[0] + 2 * n * x[3] + u[0] / m, -2 * n * x[2] + u[1] / m};
            },
            4, 2};
}

/// f(x, u) = (x3 cos x4, x3 sin x4, u1, u2) of the navigation task.
NonlinearSys aux_nav() {
    return {[](const Vars &x, const Vars &u) {
                return Vars{x[2] * cos(x[3]), x[2] * sin(x[3]), u[0] + 0.0 * x[0],
                            u[1] + 0.0 * x[0]};
            },
            4, 2};
}

/// The cart pole: x1' = x2, x2' = 2 u, x3' = x4, x4' = (0.08 0.41 (9.8 sin x3 - 2 u cos x3) -
/// 0.0021 x4) / 0.0105.
NonlinearSys aux_cart_pole() {
    return {[](const Vars &x, const Vars &u) {
                return Vars{
                    x[1] + 0.0 * x[0], 2.0 * u[0], x[3] + 0.0 * x[0],
                    (0.08 * 0.41 * (9.8 * sin(x[2]) - 2.0 * u[0] * cos(x[2])) - 0.0021 * x[3]) /
                        0.0105};
            },
            4, 1};
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
/// reach of the whole set and the check of every set over a step.
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
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // QUAD: x3 in [0.94, 1.06] at t = 5, from x1..x6 in [-0.4, 0.4]
    {
        std::vector<double> lo(12, 0.0), hi(12, 0.0), lo_g(12, -kInf), hi_g(12, kInf);
        for (int i = 0; i < 6; ++i) lo[std::size_t(i)] = -0.4, hi[std::size_t(i)] = 0.4;
        lo_g[2] = 0.94, hi_g[2] = 1.06;
        v.push_back(
            {"QUAD", "reach", Verdict::verified, 33.415,
             make_verify(aux_quad(), "quad", 0.1, box_poly(lo, hi), none, 5.0,
                         loop_options(0.1, 2, 1, 20, 0, 0, PolyMethod::regression),
                         TimedSpecification{Specification::safe_set(box_polytope(lo_g, hi_g)),
                                            std::make_pair(5.0, 5.0)},
                         {.depth = 4}),
             "order-1 enclosure and splitting (CORA: order 2 in one layer, no splitting)"});
    }
    // Docking: the constraint is not a polytope; the script's own check
    v.push_back({"Docking", "constraint", Verdict::unknown, 95.038,
                 [](Rng &rng) { return aux_docking_run(rng, false); }});
    v.push_back({"Docking", "constraint-polygon", Verdict::verified, 95.038,
                 [](Rng &rng) { return aux_docking_run(rng, true); },
                 "the constraint from inside by a 32-gon polytope, initial set split in its widest "
                 "dimension (depth 11); CORA's script is UNKNOWN"});
    // NAV: the goal at t = 6, the obstacle all the time
    for (const bool point : {true, false}) {
        const TimedSpecifications specs{
            {Specification::safe_set(
                 box_polytope({-0.5, -0.5, -kInf, -kInf}, {0.5, 0.5, kInf, kInf})),
             std::make_pair(6.0, 6.0)},
            {Specification::unsafe_set(box_polytope({1, 1, -kInf, -kInf}, {2, 2, kInf, kInf}))}};
        v.push_back(
            {"NAV", point ? "standard" : "robust", Verdict::verified, point ? 411.641 : 2.271,
             make_verify(aux_nav(), point ? "nav_point" : "nav_set", 0.2,
                         box_zonotope({2.9, 2.9, 0, 0}, {3.1, 3.1, 0, 0}), none, 6.0,
                         loop_options(point ? 0.05 : 0.1, 2, 1, 10, 0, 0, PolyMethod::regression),
                         specs, {.depth = 9})});
    }
    // CartPole: x1, x3, x4 in [-0.001, 0.001] for t in [8, 10]
    for (const bool report : {false, true}) {
        // the script's set has 0.4 times the radius of the report's
        const double r = report ? 1.0 : 0.4;
        const std::vector<double> lo{-0.1 * r, -0.05 * r, -0.1 * r, -0.05 * r},
            hi{0.1 * r, 0.05 * r, 0.1 * r, 0.05 * r};
        v.push_back(
            {"CartPole", report ? "reach-report" : "reach", Verdict::verified, 227.216,
             make_verify(aux_cart_pole(), "cartpole", 0.02, box_poly(lo, hi), none, 10.0,
                         loop_options(0.02, 2, 1, 50, 0, 0, PolyMethod::regression),
                         TimedSpecification{
                             Specification::safe_set(box_polytope({-0.001, -kInf, -0.001, -0.001},
                                                                  {0.001, kInf, 0.001, 0.001})),
                             std::make_pair(8.0, 10.0)},
                         {.depth = report ? 16 : 10}),
             report
                 ? "the report's initial set (no tool ran it), split to depth 16 (CORA's time is "
                   "for the smaller set)"
                 : "the smaller initial set of the script (0.4 times the radius of the report's)"});
    }
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    std::vector<std::string> names;
    for (const AinncsInstance &inst : aux_instances())
        names.push_back(inst.benchmark + "-" + inst.label);
    return cora::bench::run_family(argc, argv, names, [&](const std::string &name) {
        return cora::bench::run_instance(aux_instances(), name);
    });
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
