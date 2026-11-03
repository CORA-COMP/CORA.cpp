// test_interval_convex_hull - interval convex_hull: the hull of two boxes
//
// CORA: contSet/@interval/convHull_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void convex_hull_takes_the_extremes(const std::string &b) {
    const Interval I1(column({-1.0, 0.0}), column({1.0, 2.0})),
        I2(column({0.0, 3.0}), column({4.0, 5.0}));
    const Interval R = I1.convex_hull(I2);
    check(close(R.inf(), std::vector<double>{-1, 0}) && close(R.sup(), std::vector<double>{4, 5}),
          b + ": the hull of two boxes");
    check(close(I2.convex_hull(I1).inf(), R.inf()) && close(I2.convex_hull(I1).sup(), R.sup()),
          b + ": the hull is symmetric");
    const Interval S = Interval::stack({I1, I2}).convex_hull(Interval::stack({I2, I2}));
    check(close(S.sup().slice(0, 0, 1).reshape({2, 1}), R.sup()), b + ": a batch member");
    const Set via_registry = convex_hull(I1, I2);
    check(close(via_registry.as<Interval>().sup(), R.sup()), b + ": through the registry");
}

/// An empty operand returns the other.
void convex_hull_of_an_empty_box(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({1.0, 2.0})), E = Interval::empty(2);
    check(close(I.convex_hull(E).inf(), I.inf()) && close(E.convex_hull(I).sup(), I.sup()),
          b + ": the empty box is neutral");
    check(E.convex_hull(E).is_empty(), b + ": two empty boxes");
}

/// Boxes of other dimensions are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0})), J(column({0.0}), column({1.0}));
    check(test::raises([&] { I.convex_hull(J); }, Error::dimension_mismatch),
          "convex_hull: 2 and 1");
    const Tensor on = I.convex_hull(I).sup();
    ChecksScope off(false);
    check(close(I.convex_hull(I).sup(), on, 0), "convex_hull: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        convex_hull_takes_the_extremes(b);
        convex_hull_of_an_empty_box(b);
    });
    checks_and_their_switch();
    return test::finish("interval convex_hull");
}
