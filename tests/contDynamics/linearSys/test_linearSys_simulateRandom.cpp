// test_linearSys_simulateRandom - linearSys simulateRandom: seeded trajectories from any initial
// set

#include "contDynamics/linearSys/linearSysTesting.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using namespace test::lin;
using matlab_reference::System;

namespace {

/// `simulateRandom` draws from the initial set: same seed, same trajectories.
void simulateRandom_is_seeded(const std::string &b) {
    const Zonotope X0(Tensor({1.0, 0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.1, 0.1}}));
    const LinearSys sys(tensor_of(oscillator3()));
    cora::Rng first(5), second(5), other(6);
    const std::vector<Tensor> a = sys.simulateRandom(X0, 8, 0.1, 1.0, first);
    const std::vector<Tensor> c = sys.simulateRandom(X0, 8, 0.1, 1.0, second);
    const std::vector<Tensor> d = sys.simulateRandom(X0, 8, 0.1, 1.0, other);
    check(a.size() == 11 && a[0].shape() == std::vector<int64_t>({3, 8}), b + ": shapes");
    check(close(a.back(), c.back()), b + ": the same seed gives the same trajectories");
    check(!close(a.back(), d.back()), b + ": another seed gives others");
    // The initial set is an interval too: any ContSet works.
    const Interval box(Tensor({0.9, -0.1, -0.1}), Tensor({1.1, 0.1, 0.1}));
    cora::Rng rng(1);
    check(sys.simulateRandom(box, 5, 0.1, 0.5, rng).size() == 6,
          b + ": simulateRandom from an interval");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { simulateRandom_is_seeded(b); });
    return test::finish("linearSys simulateRandom");
}
