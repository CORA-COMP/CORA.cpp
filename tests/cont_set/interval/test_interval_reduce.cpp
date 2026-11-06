// test_interval_reduce - interval reduce: the order reduction of a box leaves it unchanged
//
// CORA: contSet/@interval/reduce.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void reduce_returns_the_box(const std::string &b) {
    const Interval I(column({-1.0, 0.5}), column({2.0, 3.0}));
    for (const int order : {1, 3}) {
        const Interval R = I.reduce(order);
        check(R.inf().data() == I.inf().data() && R.sup().data() == I.sup().data(),
              b + ": the box, order " + std::to_string(order));
    }
    check(I.reduce().sup().data() == I.sup().data(), b + ": the default order");
    const Interval S = Interval::stack({I, I}).reduce(2);
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch stays a batch");
}

/// An order below 1 is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0}), column({2.0}));
    check(test::raises([&] { I.reduce(0); }, Error::wrong_value), "reduce: order 0");
    ChecksScope off(false);
    check(I.reduce(2).sup().data() == I.sup().data(), "reduce: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(reduce_returns_the_box);
    checks_and_their_switch();
    return test::finish("interval reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
