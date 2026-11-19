// test_interval_less - interval less: whether a box lies strictly in another (CORA's lt)
//
// CORA: contSet/@interval/lt.m
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

void less_is_strict_containment(const std::string &b) {
    const Interval small(vec({0.0, 1.0}), vec({1.0, 2.0})), big(vec({-1.0, 0.0}), vec({3.0, 3.0}));
    check(small.less(big) && small < big, b + ": a box strictly inside");
    // NOLINTNEXTLINE(misc-redundant-expression): the test is that a box is not less than itself
    check(!small.less(small) && !(small < small), b + ": not in itself");
    check(!small.less(Interval(vec({0.0, 0.0}), vec({3.0, 3.0}))),
          b + ": a shared lower bound is not strict");
    check(!small.less(Interval(vec({-1.0, 0.0}), vec({1.0, 3.0}))),
          b + ": a shared upper bound is not strict");
    check(Interval::stack({small, small}).less(Interval::stack({big, big})) &&
              !Interval::stack({small, small}).less(Interval::stack({big, small})),
          b + ": a batch needs every member");
}

/// Boxes of different dimension are refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(vec({0.0, 1.0}), vec({1.0, 2.0}));
    check(
        test::raises([&] { I.less(Interval(vec({0.0}), vec({1.0}))); }, Error::dimension_mismatch),
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
