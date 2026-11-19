// test_interval_contains - interval contains: a point in a box
//
// CORA: contSet/@interval/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void contains_a_point(const std::string &b) {
    const Interval I(vec({0.0, -1.0}), vec({2.0, 1.0}));
    check(I.contains(vec({1.0, 0.0})), b + ": an inner point");
    check(I.contains(vec({0.0, 1.0})), b + ": a corner belongs to the box");
    check(!I.contains(vec({2.1, 0.0})), b + ": past the upper bound");
    check(!I.contains(vec({1.0, -1.5})), b + ": below the lower bound");
}

/// A point of the wrong dimension and a negative tol are refused; checks off, the same answer.
void checks_and_their_switch() {
    const Interval I = Interval(test::vec({0, -1}), test::vec({1, 1}));
    check(test::raises([&] { I.contains(test::vec({0, 0, 0})); }, Error::dimension_mismatch),
          "contains: a point of the wrong dimension");
    check(test::raises([&] { I.contains(test::vec({0, 0}), -1.0); }, Error::wrong_value),
          "contains: tol -1");
    const auto on = I.contains(test::vec({0.5, 0}));
    ChecksScope off(false);
    check(I.contains(test::vec({0.5, 0})) == on, "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { contains_a_point(b); });
    checks_and_their_switch();
    return test::finish("interval contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
