// test_interval_stack_torch - interval stack: a batch member is the interval it was made from

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "cora/global/backend/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void stack_keeps_each_member() {
    cora::Rng rng(2);
    const std::vector<Interval> Is = {Interval::generate_random(3, rng),
                                      Interval::generate_random(3, rng)};
    const Interval S = Interval::stack(Is);
    check(S.inf().shape() == std::vector<int64_t>({2, 3, 1}), "stack: bounds shape");
    for (int64_t b = 0; b < 2; ++b) {
        check(torch::allclose(to_torch(S.inf())[b], to_torch(Is[b].inf())), "stack: member inf");
        check(torch::allclose(to_torch(S.sup())[b], to_torch(Is[b].sup())), "stack: member sup");
    }
}

} // namespace

int main() {
    const BackendScope backend_scope("torch");
    stack_keeps_each_member();
    check(test::throws([] { Interval::stack({}); }), "stack: an empty list throws");
    return test::finish("interval stack (torch)");
}
