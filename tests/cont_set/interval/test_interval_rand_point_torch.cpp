// test_interval_rand_point_torch - interval rand_point over a batch of boxes
//
// CORA: contSet/@interval/randPoint_.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/backend/torch.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void batched_interval_rand_point(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t S = 4, n = 2, N = 300;
    cora::Rng rng(5);
    const torch::Tensor lo = torch::randn({S, n, 1}, opts),
                        hi = lo + torch::rand({S, n, 1}, opts) + 0.1;
    const Interval I(from_torch(lo), from_torch(hi));
    const torch::Tensor P = to_torch(I.rand_point(N, rng));
    check(P.sizes().vec() == std::vector<int64_t>({S, n, N}), where + ": batched box points");
    check((P - lo).min().item<double>() >= 0.0 && (hi - P).min().item<double>() >= 0.0,
          where + ": batched box points stay in their box");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_device([](const torch::TensorOptions &opts, const std::string &where) {
        batched_interval_rand_point(opts, where);
    });
    return test::finish("interval rand_point (libtorch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
