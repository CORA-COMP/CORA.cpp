// micro - micro-benchmarks of the public API, run one at a time so a profiler counts its instructions
//
// Syntax:   micro            list the benchmarks
//           micro <name>     run one: the setup, then the measured region (micro_measured_region)
// Outputs:  the name and a checksum of the result (keeps the work from being optimized away)
// See also: scripts/ci/instruction_counts.py, which counts the instructions of the region only

#include "cora/cora.h"

#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

namespace {

/// A benchmark: its setup, which returns the work to measure and gives its checksum.
using Work = std::function<double()>;
using Benchmark = std::function<Work()>;

double sum_of(const Tensor &t) {
    const std::vector<double> d = t.data();
    return std::accumulate(d.begin(), d.end(), 0.0);
}

const std::map<std::string, Benchmark> &benchmarks() {
    static const std::map<std::string, Benchmark> table = {
        {"zonotope_linear_map", [] {
             Rng rng(1);
             const Zonotope Z = Zonotope::generate_random(10, 40, rng);
             const Tensor M = Zonotope::generate_random(10, 10, rng).generators();
             return Work([=] {
                 double s = 0;
                 for (int k = 0; k < 2000; ++k) s += sum_of(Z.linear_map(M).generators());
                 return s;
             });
         }},
        {"zonotope_plus", [] {
             Rng rng(2);
             const Zonotope A = Zonotope::generate_random(10, 40, rng),
                            B = Zonotope::generate_random(10, 40, rng);
             return Work([=] {
                 double s = 0;
                 for (int k = 0; k < 20000; ++k) s += sum_of(A.plus(B).generators());
                 return s;
             });
         }},
        {"zonotope_reduce", [] {
             Rng rng(3);
             const Zonotope Z = Zonotope::generate_random(10, 100, rng);
             return Work([=] {
                 double s = 0;
                 for (int k = 0; k < 1000; ++k) s += sum_of(Z.reduce(2).generators());
                 return s;
             });
         }},
        {"linear_sys_reach_5dim", [] {
             const LinearSys sys(Tensor({{-1, -4, 0, 0, 0}, {4, -1, 0, 0, 0}, {0, 0, -3, 1, 0},
                                         {0, 0, -1, -3, 0}, {0, 0, 0, 0, -2}}));
             const Zonotope X0(Tensor({1, 1, 1, 1, 1}), 0.1 * Tensor::eye(5));
             return Work([=] {
                 double s = 0;
                 for (int k = 0; k < 20; ++k) {
                     const ReachSet R = sys.reach({.initial_set = X0, .t_final = 1.0},
                                                  alg::Standard{.time_step = 0.02, .taylor_terms = 4});
                     s += sum_of(R.time_interval.back().interval().sup());
                 }
                 return s;
             });
         }},
        {"interval_matrix_product", [] {
             Rng rng(4);
             const Tensor M = Zonotope::generate_random(20, 20, rng).generators(),
                          C = Zonotope::generate_random(20, 20, rng).generators();
             const Tensor w = 0.1 * Tensor::ones({20, 20});
             const Interval I(C - w, C + w);
             return Work([=] {
                 double s = 0;
                 for (int k = 0; k < 5000; ++k) s += sum_of(I.linear_map(M).sup());
                 return s;
             });
         }},
    };
    return table;
}

} // namespace

/// The region a profiler counts: extern "C" and not inlined, so its symbol is stable.
extern "C" __attribute__((noinline)) double micro_measured_region(const Work &work) {
    return work();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        for (const auto &entry : benchmarks()) std::cout << entry.first << "\n";
        return 0;
    }
    const auto found = benchmarks().find(argv[1]);
    if (found == benchmarks().end()) {
        std::cerr << "micro: no benchmark " << argv[1] << "\n";
        return 2;
    }
    const Work work = found->second();
    std::cout << found->first << " " << micro_measured_region(work) << "\n";
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
