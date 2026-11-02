// micro - micro-benchmarks of the public API, run one at a time so a profiler counts its instructions
//
// Syntax:   micro            list the benchmarks
//           micro <name>     run one: the setup, then the measured region (micro_measured_region)
// Outputs:  the name and a checksum of the result (keeps the work from being optimized away)
//
// The set operations build against any commit with an install tree. The tensor operations need the
// op table (cora/global/tensor/ops.def); against an older commit they are left out, and
// instruction_counts.py reports them as new.
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

#if __has_include("cora/global/tensor/ops.def")
/// `calls` calls of f, summed.
template <class F> Work aux_repeat(int calls, F f) {
    return Work([=] {
        double s = 0;
        for (int k = 0; k < calls; ++k) s += f();
        return s;
    });
}

/// The core and fused tensor operations, one benchmark each, on Eigen.
void aux_tensor_benchmarks(std::map<std::string, Benchmark> &table) {
    const auto matrix = [](int64_t n, int64_t m, uint64_t seed) {
        Rng rng(seed);
        return Zonotope::generate_random(n, m, rng).generators();
    };
    table["tensor_matmul_5x5"] = [=] {
        const Tensor A = matrix(5, 5, 5), B = matrix(5, 5, 6);
        return aux_repeat(20000, [=] { return sum_of(A.matmul(B)); });
    };
    table["tensor_matmul_100"] = [=] {
        const Tensor A = matrix(100, 100, 7), B = matrix(100, 100, 8);
        return aux_repeat(50, [=] { return sum_of(A.matmul(B)); });
    };
    table["tensor_add_sum"] = [=] {
        const Tensor A = matrix(100, 100, 9), B = matrix(100, 100, 10);
        return aux_repeat(2000, [=] { return sum_of(A.add(B).abs().sum(-1, true)); });
    };
    table["tensor_take_concat"] = [=] {
        const Tensor A = matrix(20, 200, 11);
        const Tensor i = A.sum(-2, true).argsort(-1);
        return aux_repeat(2000, [=] {
            return sum_of(Tensor::concat({A.take_along_axis(i, -1), A}, -1));
        });
    };
    table["tensor_expm_10"] = [=] {
        const Tensor A = 0.1 * matrix(10, 10, 12);
        return aux_repeat(1000, [=] { return sum_of(A.expm()); });
    };
    table["fused_zonotope_linear_map"] = [=] {
        const Tensor M = matrix(10, 10, 13), c = matrix(10, 1, 14), G = matrix(10, 40, 15);
        return aux_repeat(5000, [=] {
            return sum_of(Tensor::zonotope_linear_map(M, c, G).second);
        });
    };
    table["fused_zonotope_interval_map"] = [=] {
        const Tensor lo = matrix(10, 10, 16), hi = lo + matrix(10, 10, 17).abs();
        const Tensor c = matrix(10, 1, 18), G = matrix(10, 40, 19);
        return aux_repeat(5000, [=] {
            return sum_of(Tensor::zonotope_interval_map(lo, hi, c, G).second);
        });
    };
    table["fused_zonotope_linear_combination"] = [=] {
        const Tensor c = matrix(10, 1, 20), G = matrix(10, 40, 21), H = matrix(10, 40, 22);
        return aux_repeat(5000, [=] {
            return sum_of(Tensor::zonotope_linear_combination(c, G, c, H).second);
        });
    };
    table["fused_interval_matmul"] = [=] {
        const Tensor lo = matrix(20, 20, 23), hi = lo + matrix(20, 20, 24).abs();
        const Tensor M = matrix(20, 20, 25);
        return aux_repeat(5000, [=] { return sum_of(Tensor::interval_matmul(lo, hi, M).second); });
    };
    table["fused_matmul_interval"] = [=] {
        const Tensor lo = matrix(20, 20, 26), hi = lo + matrix(20, 20, 27).abs();
        const Tensor M = matrix(20, 20, 28);
        return aux_repeat(5000, [=] { return sum_of(Tensor::matmul_interval(M, lo, hi).second); });
    };
    table["fused_girard_split"] = [=] {
        const Tensor G = matrix(10, 100, 29);
        return aux_repeat(5000, [=] { return sum_of(G.girard_split(20).second); });
    };
    table["fused_abs_sum"] = [=] {
        const Tensor G = matrix(10, 100, 30);
        return aux_repeat(20000, [=] { return sum_of(G.abs_sum()); });
    };
}
#endif

const std::map<std::string, Benchmark> &benchmarks() {
    static const std::map<std::string, Benchmark> table = [] {
    std::map<std::string, Benchmark> sets = {
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
#if __has_include("cora/global/tensor/ops.def")
    aux_tensor_benchmarks(sets);
#endif
    return sets;
    }();
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
