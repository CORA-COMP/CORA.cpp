// test_interval_less - interval less: whether a box lies strictly in another (CORA's lt)
//
// CORA: contSet/@interval/lt.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
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

void less_is_strict_containment(const std::string &b) {
    const Interval small(column({0.0, 1.0}), column({1.0, 2.0})),
        big(column({-1.0, 0.0}), column({3.0, 3.0}));
    check(small.less(big) && small < big, b + ": a box strictly inside");
    // NOLINTNEXTLINE(misc-redundant-expression): the test is that a box is not less than itself
    check(!small.less(small) && !(small < small), b + ": not in itself");
    check(!small.less(Interval(column({0.0, 0.0}), column({3.0, 3.0}))),
          b + ": a shared lower bound is not strict");
    check(!small.less(Interval(column({-1.0, 0.0}), column({1.0, 3.0}))),
          b + ": a shared upper bound is not strict");
    check(Interval::stack({small, small}).less(Interval::stack({big, big})) &&
              !Interval::stack({small, small}).less(Interval::stack({big, small})),
          b + ": a batch needs every member");
}

/// Boxes of different dimension are refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0}));
    check(test::raises([&] { I.less(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "less: dimensions 2 and 1");
    ChecksScope off(false);
    check(!I.less(I), "less: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(less_is_strict_containment);
    checks_and_their_switch();
    return test::finish("interval less");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
