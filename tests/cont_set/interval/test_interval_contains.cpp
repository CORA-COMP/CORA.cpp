// test_interval_contains - interval contains: a point in a box

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void contains_a_point(const std::string &b) {
    const Interval I(column({0.0, -1.0}), column({2.0, 1.0}));
    check(I.contains(column({1.0, 0.0})), b + ": an inner point");
    check(I.contains(column({0.0, 1.0})), b + ": a corner belongs to the box");
    check(!I.contains(column({2.1, 0.0})), b + ": past the upper bound");
    check(!I.contains(column({1.0, -1.5})), b + ": below the lower bound");
}

/// A point of the wrong dimension and a negative tol are refused; checks off, the same answer.
void checks_and_their_switch() {
    const Interval I = Interval(test::column({0, -1}), test::column({1, 1}));
    check(test::raises([&] { I.contains(test::column({0, 0, 0})); }, Error::dimension_mismatch),
          "contains: a point of the wrong dimension");
    check(test::raises([&] { I.contains(test::column({0, 0}), -1.0); }, Error::wrong_value),
          "contains: tol -1");
    const auto on = I.contains(test::column({0.5, 0}));
    ChecksScope off(false);
    check(I.contains(test::column({0.5, 0})) == on, "contains: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { contains_a_point(b); });
    checks_and_their_switch();
    return test::finish("interval contains");
}
