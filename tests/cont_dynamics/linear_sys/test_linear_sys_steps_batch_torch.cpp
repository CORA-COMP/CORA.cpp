// test_linear_sys_steps_batch_torch - the batched steps: one call over a batch of sets, of systems
//
// or of both equals one call per member (homogeneous_solution, particular_solution_time_varying)
//
// CORA: contDynamics/@linearSys/homogeneousSolution.m
// CORA: contDynamics/@linearSys/particularSolution_timeVarying.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/backend/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kStep = 0.1;
constexpr int kTerms = 8;

double max_diff(const Tensor &a, const Tensor &b) {
    return (to_torch(a).to(torch::kCPU) - to_torch(b).to(torch::kCPU)).abs().max().item<double>();
}

/// Whether member b of a batched set is the single set.
bool member_is(const Zonotope &batched, int64_t b, const Zonotope &single) {
    return max_diff(from_torch(to_torch(batched.center())[b]), single.center()) < 1e-12 &&
           max_diff(from_torch(to_torch(batched.generators())[b]), single.generators()) < 1e-12;
}

/// A set from a center (..., n) and generators (..., n, m).
Zonotope make_set(const torch::Tensor &c, const torch::Tensor &G) {
    return {from_torch(c.unsqueeze(-1)), from_torch(G)};
}

/// The homogeneous solution and the time-varying input solution agree with the members.
void batches(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 3, m = 4, S = 3;
    const torch::Tensor As = torch::randn({S, n, n}, opts) * 0.5;
    const torch::Tensor cs = torch::randn({S, n}, opts), Gs = torch::randn({S, n, m}, opts);
    // a batch of sets, of systems, and of both
    const LinearSys single(from_torch(As[0])), batch(from_torch(As));
    const Zonotope sets = make_set(cs, Gs), set = make_set(cs[0], Gs[0]);
    const HomogeneousSolution over_sets = single.homogeneous_solution(sets, kStep, kTerms);
    const HomogeneousSolution over_systems = batch.homogeneous_solution(set, kStep, kTerms);
    const HomogeneousSolution both = batch.homogeneous_solution(sets, kStep, kTerms);
    const Zonotope pu_sets = single.particular_solution_time_varying(sets, kStep, kTerms);
    const Zonotope pu_systems = batch.particular_solution_time_varying(set, kStep, kTerms);
    // member b of every batched result is the call on member b alone
    for (int64_t b = 0; b < S; ++b) {
        const LinearSys one(from_torch(As[b]));
        const Zonotope own = make_set(cs[b], Gs[b]);
        const std::string at = where + ": member " + std::to_string(b);
        check(member_is(over_sets.time_interval, b,
                        single.homogeneous_solution(own, kStep, kTerms).time_interval),
              at + ": a batch of sets");
        check(member_is(over_systems.time_interval, b,
                        one.homogeneous_solution(set, kStep, kTerms).time_interval),
              at + ": a batch of systems");
        check(
            member_is(both.time_point, b, one.homogeneous_solution(own, kStep, kTerms).time_point),
            at + ": sets and systems together");
        check(member_is(pu_sets, b, single.particular_solution_time_varying(own, kStep, kTerms)),
              at + ": the input solution of a batch of sets");
        check(member_is(pu_systems, b, one.particular_solution_time_varying(set, kStep, kTerms)),
              at + ": the input solution of a batch of systems");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    // the same on the CPU and, when there is one, on the GPU
    test::for_each_device(batches);
    return test::finish("linear_sys steps batch");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
