// benchmark_hybrid_reach_arch23_powertrain - powertrain benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_hybrid_reach_ARCH23_powerTrain_<instance>.m with the
// automaton of models/Cora/powertrain/initPowerTrain.m. The flows are affine, so their Jacobians
// (CORA computes them symbolically) are the coefficients of powertrain<dim>Eq.m, written out here
// equation by equation; the locations differ by the shaft stiffness k (0 in the backlash) and the
// sign of alpha. Two reach calls: 0.2 s braking (u1 = -5), then up to 2 s accelerating (u1 = 5)
// from the last set. The scripts check no specification: the verdict is that both calls end.
//
// Syntax:   build/core/benchmarks/benchmark_hybrid_reach_arch23_powertrain [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmark_hybrid.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <chrono>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;
using cora::bench::hyperplane;

/// powertrainParameters.m (the control gains k_P, k_I, k_D, k_KK are zero and do not enter).
struct Params {
    double alpha = 0.03, tau_eng = 0.1, b_l = 5.6, b_m = 0, b_i = 1, i = 12, k = 10e3, k_i = 10e4;
    double J_l = 140, J_m = 0.3, J_i = 0.01, k_K = 0.5, k_KD = 0.5, k_KI = 0.5;
};

/// The Jacobians A (dim, dim), B (dim, 2) and the value c at 0 of powertrain<dim>Eq.m.
struct Flow {
    std::vector<double> A, B, c;
};

Flow aux_flow(int64_t dim, const Params &p) {
    const auto n = std::size_t(dim);
    Flow f{std::vector<double>(n * n, 0.0), std::vector<double>(n * 2, 0.0),
           std::vector<double>(n, 0.0)};
    // 1-based indices, as the equations
    auto A = [&](std::size_t r, std::size_t col, double v) { f.A[(r - 1) * n + col - 1] += v; };
    auto B = [&](std::size_t r, std::size_t col, double v) { f.B[(r - 1) * 2 + col - 1] += v; };
    // f2 = (v - x2) / tau_eng, v = k_K (i x4 - x7) + k_KD (i u1 - J_m f7) + k_KI i (x3 - x1 - x8)
    const double t = p.tau_eng;
    A(2, 4, p.k_K * p.i / t);
    A(2, 7, -p.k_K / t + p.k_KD * p.b_m / p.J_m / t);
    A(2, 2, -p.k_KD / p.J_m / t - 1 / t);
    A(2, 1, p.k_KD * p.k / (p.J_m * p.i) / t - p.k_KI * p.i / t);
    A(2, 3, p.k_KI * p.i / t);
    A(2, 8, -p.k_KI * p.i / t);
    B(2, 1, p.k_KD * p.i / t);
    f.c[1] = -p.k_KD * p.k * p.alpha / (p.J_m * p.i) / t;
    // f1 = x7 / i - x9, f3 = x4, f4 = u1, f5 = x6
    A(1, 7, 1 / p.i);
    A(1, 9, -1);
    A(3, 4, 1);
    B(4, 1, 1);
    A(5, 6, 1);
    // f6 = (k_i (x_{dim-1} - x5) - u2 - b_l x6) / J_l
    A(6, n - 1, p.k_i / p.J_l);
    A(6, 5, -p.k_i / p.J_l);
    A(6, 6, -p.b_l / p.J_l);
    B(6, 2, -1 / p.J_l);
    // f7 = (x2 - k/i (x1 - alpha) - b_m x7) / J_m
    A(7, 2, 1 / p.J_m);
    A(7, 1, -p.k / (p.J_m * p.i));
    A(7, 7, -p.b_m / p.J_m);
    f.c[6] = p.k * p.alpha / (p.J_m * p.i);
    // f8 = x9, f9 = J_i (k (x1 - alpha) - k_i (x8 - x10) - b_i x9)
    A(8, 9, 1);
    A(9, 1, p.J_i * p.k);
    A(9, 8, -p.J_i * p.k_i);
    A(9, 10, p.J_i * p.k_i);
    A(9, 9, -p.J_i * p.b_i);
    f.c[8] = -p.J_i * p.k * p.alpha;
    // the inertias j = 2..K at x_{2j+6}, each coupled to its neighbours (the last to x5)
    const std::size_t K = (n - 7) / 2;
    for (std::size_t j = 2; j <= K; ++j) {
        const std::size_t pos = 2 * j + 6, prev = pos - 2, next = j == K ? 5 : pos + 2;
        A(pos, pos + 1, 1);
        A(pos + 1, prev, p.J_i * p.k_i);
        A(pos + 1, pos, -p.J_i * p.k_i - p.J_i * p.k_i);
        A(pos + 1, next, p.J_i * p.k_i);
        A(pos + 1, pos + 1, -p.J_i * p.b_i);
    }
    return f;
}

/// The model of initPowerTrain(dim) and its initial center and generator.
struct Model {
    HybridAutomaton HA;
    Tensor center, delta;
    std::vector<Flow> flows;
};

Model aux_model(int64_t dim) {
    Params p;
    // the flows of the locations: with backlash contact, in the gap (k = 0), the other contact
    Params gap = p, other = p;
    gap.k = 0;
    other.alpha = -p.alpha;
    const std::vector<Flow> flows = {aux_flow(dim, p), aux_flow(dim, gap), aux_flow(dim, other)};
    std::vector<double> nrm(std::size_t(dim), 0.0);
    nrm[0] = 1;
    const Polytope h1 = hyperplane(nrm, p.alpha), h2 = hyperplane(nrm, -p.alpha);
    const LinearReset id(Tensor::eye(dim));
    const Tensor e1 = Tensor::from_data(nrm, {1, dim});
    auto sys = [&](std::size_t l) {
        return LinearSys(Tensor::from_data(flows[l].A, {dim, dim}), Tensor::eye(dim));
    };
    Location loc1("loc1", Polytope(-e1, column({-p.alpha})), {Transition(h1, id, 1)}, sys(0));
    Location loc2("loc2", Polytope(Tensor::concat({e1, -e1}, -2), column({p.alpha, p.alpha})),
                  {Transition(h1, id, 0), Transition(h2, id, 2)}, sys(1));
    Location loc3("loc3", Polytope(e1, column({-p.alpha})), {Transition(h2, id, 1)}, sys(2));
    // powertrainParameters.m: the initial set
    const double omega_c = 30, omega_d = 10, T_l = -300;
    const double x1_0 = p.b_l / p.k * omega_c - p.alpha + T_l / p.k;
    const double dx1_0 = p.b_l / p.k * omega_d;
    const double xy_0 = p.b_l / p.k_i * omega_c + T_l / p.k_i, dxy_0 = p.b_l / p.k_i * omega_d;
    const double T_m0 = p.k / p.i * (x1_0 + p.alpha) + p.b_m * omega_c;
    const double dT_m0 = p.k / p.i * dx1_0 + p.b_m * omega_d;
    std::vector<double> c = {x1_0, T_m0, 0, omega_c, 0, omega_c, p.i * omega_c};
    std::vector<double> d = {dx1_0, dT_m0, 0, omega_d, 0, omega_d, p.i * omega_d};
    for (int64_t k = 0; k < (dim - 7) / 2; ++k) {
        c.insert(c.end(), {xy_0, omega_c});
        d.insert(d.end(), {dxy_0, omega_d});
    }
    return {HybridAutomaton("powertrain", {loc1, loc2, loc3}), column(c), column(d), flows};
}

/// The constant inputs B_i u + c_i of the locations, as points.
std::vector<Zonotope> aux_inputs(const Model &m, double u1) {
    std::vector<Zonotope> U;
    for (const Flow &f : m.flows) {
        const auto n = int64_t(f.c.size());
        const Tensor u = column({u1, 0});
        const Tensor point = Tensor::from_data(f.B, {n, 2}).matmul(u) + column(f.c);
        U.emplace_back(point, Tensor::zeros({n, 0}));
    }
    return U;
}

/// The two reach calls of the script from each initial set; true if both end for every set.
bool aux_run(const Model &m, const std::vector<Zonotope> &R0, const alg::HybridStandard &a) {
    for (const Zonotope &X0 : R0) {
        HybridReachProblem first;
        first.initial_set = X0;
        first.start_location = 2;
        first.t_final = 0.2;
        first.input_set = aux_inputs(m, -5);
        const HybridReachSet R1 = m.HA.reach(first, a);
        // the second call starts from the last set (CORA's R.timePoint.set{end})
        HybridReachProblem second = first;
        second.initial_set = R1.branches.back().time_point.back().as<Zonotope>();
        second.t_start = 0.2;
        second.t_final = 2;
        second.input_set = aux_inputs(m, 5);
        const HybridReachSet R2 = m.HA.reach(second, a);
        std::cerr << "  " << R1.branches.size() << " + " << R2.branches.size() << " branches\n";
        if (!R1.verified || !R2.verified) return false;
    }
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    const std::vector<std::string> family = {"DTN01", "DTN02", "DTN03", "DTN04", "DTN05", "DTN06"};
    const std::vector<std::string> names =
        argc < 2 ? family : std::vector<std::string>(argv + 1, argv + argc);
    int differ = 0;
    for (const std::string &name : names) {
        // dimension, the scale of the initial generator and the step sizes of each script
        const int64_t dim = name <= "DTN03" ? 11 : 51;
        const double scale = name == "DTN01" || name == "DTN04"   ? 0.05
                             : name == "DTN02" || name == "DTN05" ? 0.3
                                                                  : 1.0;
        alg::HybridStandard a;
        a.taylor_terms = 20;
        a.zonotope_order = 20;
        a.guard_intersect = alg::ZonoGirard{{alg::Enclose::pca}};
        a.time_step = name <= "DTN03"   ? std::vector<double>{5e-4}
                      : name == "DTN04" ? std::vector<double>{2e-4}
                      : name == "DTN05" ? std::vector<double>{1e-4}
                                        : std::vector<double>{2e-4, 1e-4, 1e-4};
        const Model m = aux_model(dim);
        const Tensor delta = m.delta * scale;
        // DTN06 splits the initial segment into four
        std::vector<Zonotope> R0;
        if (name == "DTN06")
            for (double s : {0.75, 0.25, -0.25, -0.75})
                R0.emplace_back(m.center + delta * s, delta * 0.25);
        else
            R0.emplace_back(m.center, delta);
        const auto start = std::chrono::steady_clock::now();
        const bool verified = aux_run(m, R0, a);
        const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
        std::cout << "Powertrain," << name << "," << verified << "," << wall.count() << '\n';
        if (!verified) ++differ; // MATLAB CORA: 1 for every instance
    }
    return differ;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
