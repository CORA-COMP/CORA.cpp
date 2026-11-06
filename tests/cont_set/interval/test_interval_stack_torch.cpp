// test_interval_stack_torch - interval stack: a batch member is the interval it was made from
//
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/backend/torch.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void stack_keeps_each_member() {
    cora::Rng rng(2);
    const std::vector<Interval> Is = {Interval::generate_random(rng, 3),
                                      Interval::generate_random(rng, 3)};
    const Interval S = Interval::stack(Is);
    check(S.inf().shape() == std::vector<int64_t>({2, 3, 1}), "stack: bounds shape");
    for (int64_t b = 0; b < 2; ++b) {
        check(torch::allclose(to_torch(S.inf())[b], to_torch(Is[b].inf())), "stack: member inf");
        check(torch::allclose(to_torch(S.sup())[b], to_torch(Is[b].sup())), "stack: member sup");
    }
}

/// An empty list is refused; with the checks off the stack is the same.
void checks_and_their_switch() {
    check(test::raises([] { Interval::stack({}); }, Error::wrong_value), "stack: an empty list");
    const Interval S = Interval(test::column({0, -1}), test::column({1, 1}));
    const Tensor on = Interval::stack({S, S}).center();
    ChecksScope off(false);
    check(test::close(Interval::stack({S, S}).center(), on, 0),
          "stack: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    stack_keeps_each_member();
    check(test::throws([] { Interval::stack({}); }), "stack: an empty list throws");
    checks_and_their_switch();
    return test::finish("interval stack (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
