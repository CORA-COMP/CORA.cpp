// benchmark_nonlinear_reach_arch23_laubloomis - the Laub-Loomis model (LALO20) of ARCH 2023
//
// MATLAB CORA's benchmark_nonlinear_reach_ARCH23_laubLoomis.m with the model of
// models/Cora/laubLoomis/laubLoomis.m, numbers copied: seven concentrations from
// x0 = (1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45), boxes of the half-width W = 0.01, 0.05 and 0.1 around
// it, up to t = 20. The script runs W001 and W005 with CORA's alg "lin-adaptive" and W01 with
// "poly-adaptive". The specification of the ARCH-COMP report is x4 < 4.5 for W001 and W005 and
// x4 < 5 for W01, over the whole horizon; the script prints the width of x4 at t = 20.
// CORACPP_LALO20_W01_T_FINAL=2 shortens the horizon of W01 (the coverage build, several times
// slower, would exceed the time of a test); the exit code then ignores the verdict.
//
// Syntax:   build/core/benchmarks/benchmark_nonlinear_reach_arch23_laubloomis [LALO20_W001 ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, the width of x4 on stderr
// See also: benchmark_nonlinear_reach_arch23_robertson.cpp, benchmark_hybrid.h
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_hybrid.h"

#include <chrono>
#include <cstdlib>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using namespace cora;
using cora::bench::column;

/// laubLoomis.m: the right-hand side as the model file writes it; the input u is not used.
NonlinearSys aux_laub_loomis() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{1.4 * x[2] - 0.9 * x[0],        2.5 * x[4] - 1.5 * x[1],
                                     0.6 * x[6] - 0.8 * x[2] * x[1], 2.0 - 1.3 * x[3] * x[2],
                                     0.7 * x[0] - 1.0 * x[3] * x[4], 0.3 * x[0] - 3.1 * x[5],
                                     1.8 * x[5] - 1.5 * x[6] * x[1]};
        },
        7, 1);
}

/// The center of the initial sets and the zonotope of half-width W.
Tensor aux_x0() { return column({1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45}); }
Zonotope aux_start(double W) { return Zonotope(aux_x0(), Tensor::eye(7) * W); }

/// The bound of x4 (the fourth state) over the time-interval sets, and its width at the end.
struct X4 {
    double sup = -1e300, width = 0;
};

template <class Reach>
X4 aux_x4(const Reach &R) {
    X4 out;
    for (const auto &S : R.time_interval) out.sup = std::max(out.sup, S.interval().sup().data()[3]);
    const Interval last = R.time_point.back().interval();
    out.width = last.sup().data()[3] - last.inf().data()[3];
    return out;
}

/// Reaches one instance, prints the line of the script and the width of x4.
bool aux_run(const std::string &name) {
    const NonlinearSys sys = aux_laub_loomis();
    const Zonotope U(column({0}), Tensor::zeros({1, 0}));
    const auto start = std::chrono::steady_clock::now();
    X4 x4;
    double limit = 4.5;
    bool shortened = false;
    std::size_t sets = 0;
    // W001 and W005: the adaptive linearization of the script
    if (name == "LALO20_W001" || name == "LALO20_W005") {
        const ReachSet R = sys.reach({.initial_set = aux_start(name == "LALO20_W001" ? 0.01 : 0.05),
                                      .input_set = U,
                                      .t_final = 20},
                                     alg::LinearizationAdaptive{});
        x4 = aux_x4(R);
        sets = R.time_interval.size();
    } else if (name == "LALO20_W01") {
        // W01: CORA's poly-adaptive with the constants of the script
        limit = 5;
        const char *shorter = std::getenv("CORACPP_LALO20_W01_T_FINAL");
        shortened = shorter != nullptr;
        const PolyReachSet R = sys.reach({.initial_set = PolyZonotope(aux_start(0.1)),
                                          .input_set = U,
                                          .t_final = shortened ? std::atof(shorter) : 20},
                                         alg::PolynomializationAdaptive{});
        x4 = aux_x4(R);
        sets = R.time_interval.size();
    } else {
        CORA_FAIL(Error::wrong_value, "laubLoomis: unknown instance " + name);
    }
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
    const bool verified = x4.sup < limit;
    cora::bench::print_line("LaubLoomis", name, verified, wall.count());
    std::cerr << "  " << sets << " sets, width of x4 at t = 20: " << x4.width
              << ", largest x4: " << x4.sup << " (the limit is " << limit << ")\n";
    return verified || shortened;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    return cora::bench::run_family(argc, argv, {"LALO20_W001", "LALO20_W005", "LALO20_W01"},
                                   aux_run);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
