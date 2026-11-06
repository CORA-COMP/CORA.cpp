// test_linear_sys_batch_torch - automatic batching ("vmap"): linear_sys is written for one set and
//
// one system, and one call over a batch of sets, of systems, or of both equals one call per member.
//
// CORA: contDynamics/@linearSys/reach.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/backend/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using test::lin::kAlgorithms;
using test::lin::name;

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// A set from a centre `(..., n)` and generators `(..., n, m)`.
Zonotope make_set(const torch::Tensor &c, const torch::Tensor &G) {
    return {from_torch(c.unsqueeze(-1)), from_torch(G)};
}

ReachSet run(const torch::Tensor &A, const torch::Tensor &c, const torch::Tensor &G,
             const std::string &a) {
    return LinearSys(from_torch(A))
        .reach({.initial_set = make_set(c, G), .t_final = 1.0}, test::lin::algorithm(a, 0.1, 8));
}

/// Whether member `b` of `batched` is `single`, for every step, centre and generators.
bool member_is(const ReachSet &batched, int64_t b, const ReachSet &single) {
    for (std::size_t k = 0; k < single.time_interval.size(); ++k) {
        if (max_diff(to_torch(batched.time_interval[k].center())[b],
                     to_torch(single.time_interval[k].center())) > 1e-12 ||
            max_diff(to_torch(batched.time_interval[k].generators())[b],
                     to_torch(single.time_interval[k].generators())) > 1e-12)
            return false;
    }
    for (std::size_t k = 0; k < single.time_point.size(); ++k)
        if (max_diff(to_torch(batched.time_point[k].generators())[b],
                     to_torch(single.time_point[k].generators())) > 1e-12)
            return false;
    return true;
}

void batches(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 3, m = 4, S = 3;
    const torch::Tensor A = torch::randn({n, n}, opts) * 0.5;
    const torch::Tensor As = torch::randn({S, n, n}, opts) * 0.5;
    const torch::Tensor cs = torch::randn({S, n}, opts), Gs = torch::randn({S, n, m}, opts);

    for (const std::string &algorithm : kAlgorithms) {
        // the algorithms run on one system, on a batch and on both
        const std::string what = where + ": " + name(algorithm);
        const ReachSet sets = run(A, cs, Gs, algorithm);
        const ReachSet systems = run(As, cs[0], Gs[0], algorithm);
        const ReachSet both = run(As, cs, Gs, algorithm);
        for (int64_t b = 0; b < S; ++b) {
            check(member_is(sets, b, run(A, cs[b], Gs[b], algorithm)),
                  what + ": a batch of sets, member " + std::to_string(b));
            check(member_is(systems, b, run(As[b], cs[0], Gs[0], algorithm)),
                  what + ": a batch of systems, member " + std::to_string(b));
            check(member_is(both, b, run(As[b], cs[b], Gs[b], algorithm)),
                  what + ": sets and systems together, member " + std::to_string(b));
        }
    }
}

/// Two batch dimensions, and a batch of one, go through unchanged.
void shapes(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 2, m = 3;
    const torch::Tensor A = torch::randn({n, n}, opts) * 0.5;
    const ReachSet grid =
        run(A, torch::randn({2, 3, n}, opts), torch::randn({2, 3, n, m}, opts), "standard");
    check(to_torch(grid.time_interval[0].center()).sizes().vec() ==
              std::vector<int64_t>({2, 3, n, 1}),
          where + ": two batch dimensions");
    const ReachSet one =
        run(A, torch::randn({1, n}, opts), torch::randn({1, n, m}, opts), "standard");
    check(to_torch(one.time_point[0].center()).size(0) == 1, where + ": a batch of one");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    std::vector<torch::TensorOptions> all{torch::TensorOptions().dtype(torch::kDouble)};
    std::vector<std::string> names{"cpu"};
    if (torch::cuda::is_available()) {
        all.push_back(torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA, 0));
        names.push_back("gpu");
    }
    const BackendScope torch_backend("torch");
    torch::manual_seed(7);
    for (std::size_t i = 0; i < all.size(); ++i) {
        batches(all[i], names[i]);
        shapes(all[i], names[i]);
    }
    return test::finish("linear_sys batch");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
