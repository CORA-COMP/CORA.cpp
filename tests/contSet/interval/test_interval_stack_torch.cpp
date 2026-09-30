// test_interval_stack_torch - interval stack: a batch member is the interval it was made from

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void stack_keeps_each_member() {
    cora::Rng rng(2);
    const std::vector<Interval> Is = {Interval::generateRandom(3, rng),
                                      Interval::generateRandom(3, rng)};
    const Interval S = Interval::stack(Is);
    check(S.inf.shape() == std::vector<int64_t>({2, 3, 1}), "stack: bounds shape");
    for (int64_t b = 0; b < 2; ++b) {
        check(torch::allclose(toTorch(S.inf)[b], toTorch(Is[b].inf)), "stack: member inf");
        check(torch::allclose(toTorch(S.sup)[b], toTorch(Is[b].sup)), "stack: member sup");
    }
}

} // namespace

int main() {
    setBackend("torch");
    stack_keeps_each_member();
    check(test::throws([] { Interval::stack({}); }), "stack: an empty list throws");
    return test::finish("interval stack (torch)");
}
