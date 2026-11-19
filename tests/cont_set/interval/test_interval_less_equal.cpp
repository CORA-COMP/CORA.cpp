// test_interval_less_equal - interval less_equal: whether a box lies in another (CORA's le)
//
// CORA: contSet/@interval/le.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void less_equal_is_containment(const std::string &b) {
    const Interval small(vec({0.0, 1.0}), vec({1.0, 2.0})), big(vec({-1.0, 1.0}), vec({3.0, 2.0}));
    check(small.less_equal(big) && small <= big, b + ": a box lies in a larger one");
    check(small.less_equal(small), b + ": and in itself, bounds may touch");
    check(!big.less_equal(small) && !(big <= small), b + ": not the other way round");
    check(!small.less_equal(Interval(vec({0.5, 1.0}), vec({3.0, 2.0}))),
          b + ": one lower bound inside is enough to fail");
    check(Interval::empty(2).less_equal(small), b + ": an empty box has nothing outside");
    check(Interval::stack({small, small}).less_equal(Interval::stack({big, big})) &&
              !Interval::stack({small, big}).less_equal(Interval::stack({big, small})),
          b + ": a batch needs every member");
}

/// Boxes of different dimension are refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(vec({0.0, 1.0}), vec({1.0, 2.0}));
    check(test::raises([&] { I.less_equal(Interval(vec({0.0}), vec({1.0}))); },
                       Error::dimension_mismatch),
          "less_equal: dimensions 2 and 1");
    ChecksScope off(false);
    check(I.less_equal(I), "less_equal: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(less_equal_is_containment);
    checks_and_their_switch();
    return test::finish("interval less_equal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
