// benchmark_nonlinear_reach_arch23_robertson - the Robertson reaction (ROBE25) of ARCH 2023
//
// MATLAB CORA's benchmark_nonlinear_reach_ARCH23_Robertson.m: the three cases x' = -a x + b y z,
// y' = a x - b y z - c y^2, z' = c y^2 with a = 0.4 from x(0) = (1, 0, 0), reached until t = 40
// by alg "lin-adaptive". The verdict is the specification of the ARCH-COMP report: the width of
// the sum of the concentrations s = x + y + z at t = 40 below 1e-5 (the script prints the constant
// 1 once reach has finished). The algorithm is CORA's lin-adaptive as CORALean repairs it; its
// zeta_z is tuned per case (kCases), the script's are 0.00025, 0.0006 and 0.017. The line has the
// time of reach, as the script's tic and toc; the number of sets and the width go to stderr.
//
// Syntax:   build/core/benchmarks/benchmark_nonlinear_reach_arch23_robertson [ROBE25_1 ...]
// Outputs:  one line 'benchmark,instance,result,time' (on stderr: the number of sets, the width)
// See also: benchmark_run.h, nonlinear_sys.h
//
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <chrono>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;

/// One case: beta, gamma of the model and the tuning of the algorithm (zeta_z: the share of the
/// width a reduction may add).
struct Case {
    const char *name;
    double beta, gamma;
    alg::LinearizationAdaptive alg;
};

const Case kCases[] = {{"ROBE25_1", 1e2, 1e3, {.zeta_z = 0.0001}},
                       {"ROBE25_2", 1e3, 1e5, {.zeta_z = 0.0015}},
                       {"ROBE25_3", 1e3, 1e7, {.zeta_z = 0.01}}};

/// Robertson_case*.m: the right-hand side, written as the model file does; u is not used.
NonlinearSys aux_robertson(double beta, double gamma) {
    const double alpha = 0.4;
    return NonlinearSys(
        [=](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{-alpha * x[0] + beta * x[1] * x[2],
                                     alpha * x[0] - beta * x[1] * x[2] - gamma * pow(x[1], 2),
                                     gamma * pow(x[1], 2)};
        },
        3, 1);
}

/// The first step sizes, the first and last boxes and the step sizes, to compare with MATLAB.
void aux_print_details(const ReachSet &R) {
    std::cerr.precision(17);
    std::cerr << "    first steps:";
    for (std::size_t k = 0; k < 4 && k + 1 < R.time.size(); ++k)
        std::cerr << " " << R.time[k + 1] - R.time[k];
    // the boxes of the first and of the last set over a step
    for (const auto &[label, Z] :
         {std::pair<const char *, const Zonotope *>{"first", &R.time_interval.front()},
          {"last", &R.time_interval.back()}}) {
        const Interval I = Z->interval();
        std::cerr << "\n    " << label << " box inf:";
        for (const double v : I.inf().data()) std::cerr << " " << v;
        std::cerr << "\n    " << label << " box sup:";
        for (const double v : I.sup().data()) std::cerr << " " << v;
    }
    std::cerr << "\n    steps:";
    for (std::size_t k = 0; k < R.time.size() - 1; ++k)
        std::cerr << " " << R.time[k + 1] - R.time[k];
    std::cerr << "\n    generators of the last point set: "
              << R.time_point.back().generators().shape()[1] << '\n';
}

/// Reaches one case, prints the line of the script and reports the width of s at the end.
bool aux_run(const std::string &name) {
    // the case of the name
    const Case *c = nullptr;
    for (const Case &k : kCases)
        if (name == k.name) c = &k;
    if (!c) CORA_FAIL(Error::wrong_value, "Robertson: unknown instance " + name);

    // the start set, the (unused) input set, and reach until t = 40
    const NonlinearSys sys = aux_robertson(c->beta, c->gamma);
    const Zonotope X0(column({1, 0, 0}), Tensor::zeros({3, 0}));
    const Zonotope U(column({0}), Tensor::zeros({1, 0}));
    const auto start = std::chrono::steady_clock::now();
    const ReachSet R = sys.reach({.initial_set = X0, .input_set = U, .t_final = 40}, c->alg);
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;

    // the width of s = x + y + z: 2 rad of the interval sum of the last set over a step
    const Interval last = R.time_interval.back().interval();
    double lo = 0, hi = 0;
    for (const double v : last.inf().data()) lo += v;
    for (const double v : last.sup().data()) hi += v;
    const double width = hi - lo;
    const bool verified = width < 1e-5;
    cora::bench::print_line("Robertson", name, verified, wall.count());
    std::cerr << "  " << R.time_interval.size() << " sets, width of s at t = 40: " << width
              << (verified ? " (verified: below" : " (not verified: not below")
              << " the 1e-5 of ARCH 2026)\n";
    if (std::getenv("CORACPP_BENCHMARK_VERBOSE")) aux_print_details(R);
    return verified;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    return cora::bench::run_family(argc, argv, {"ROBE25_1", "ROBE25_2", "ROBE25_3"}, aux_run);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
