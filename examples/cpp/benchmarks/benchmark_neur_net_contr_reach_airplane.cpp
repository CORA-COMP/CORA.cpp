// benchmark_neur_net_contr_reach_airplane - Airplane
//
// Airplane: y, phi, theta, psi stay in [-1, 1]; falsified by a simulation. The network is
// examples/benchmarks/data/networks/<name>.nn.json and .nn.bin that
// matlab/benchmarks/export_network_data.m writes from the script; the plant is written out as the
// model file writes it. The line of the scripts, 'benchmark,instance,verdict,time', goes to stdout
// (the time of verify, simulations included), the expected verdict and time of the ARCH-COMP 2026
// report (CORA, Table 1) go to stderr. The exit code counts the verdicts that differ from the
// report's.
//
// Syntax:   benchmark_neur_net_contr_reach_airplane [instance ...]
// Outputs:  one line 'benchmark,instance,verdict,time' per instance on stdout
// See also: benchmark_neur_net.h
//
// CORA: contDynamics/neurNetContrSys/example_neuralNet_reach_06_airplane.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_neur_net.h"
#include "benchmark_run.h"
#include "cora/cora.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::AinncsInstance;
using cora::bench::box_poly;
using cora::bench::box_polytope;
using cora::bench::kInf;
using cora::bench::loop_options;
using cora::bench::make_verify;

using Vars = std::vector<Expr>;

// The plant ------------------------------------------------------------------------------------

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

// The instances ----------------------------------------------------------------------------------

std::vector<AinncsInstance> aux_instances() {
    const std::optional<Zonotope> none = std::nullopt;
    std::vector<AinncsInstance> v;
    // Airplane: y, phi, theta, psi in [-1, 1]; falsified by a simulation
    std::vector<double> lo(12, -kInf), hi(12, kInf);
    for (const int i : {1, 6, 7, 8}) lo[std::size_t(i)] = -1, hi[std::size_t(i)] = 1;
    std::vector<double> a_lo(12, 0.0), a_hi(12, 0.0);
    for (const int i : {3, 4, 5, 6, 7, 8}) a_hi[std::size_t(i)] = 1;
    v.push_back({"Airplane", "continuous", Verdict::falsified, 6.809,
                 make_verify(aux_airplane(), "airplane", 0.1, box_poly(a_lo, a_hi), none, 2.0,
                             loop_options(0.1, 2, 4, 10),
                             {Specification::safe_set(box_polytope(lo, hi))})});
    return v;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) { return cora::bench::run_instances(argc, argv, aux_instances); }

// ---------------------------------------  END OF CODE  ---------------------------------------- //
