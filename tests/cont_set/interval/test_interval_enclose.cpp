// test_interval_enclose - interval enclose: a box around two boxes
//
// CORA: contSet/@interval/enclose.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void enclose_is_the_hull(const std::string &b) {
    const Interval I1(column({-1.0, 0.0}), column({1.0, 2.0})),
        I2(column({0.0, 3.0}), column({4.0, 5.0}));
    const Interval R = I1.enclose(I2);
    check(close(R.inf(), std::vector<double>{-1, 0}) && close(R.sup(), std::vector<double>{4, 5}),
          b + ": the box around two boxes");
    check(close(R.inf(), I1.convex_hull(I2).inf()) && close(R.sup(), I1.convex_hull(I2).sup()),
          b + ": the convex hull");
    check(close(Interval::stack({I1, I2})
                    .enclose(Interval::stack({I2, I1}))
                    .sup()
                    .slice(0, 0, 1)
                    .reshape({2, 1}),
                R.sup()),
          b + ": a batch member");
}

/// A box of another dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0})), J(column({0.0}), column({1.0}));
    check(test::raises([&] { I.enclose(J); }, Error::dimension_mismatch), "enclose: 2 and 1");
    const Tensor on = I.enclose(I).sup();
    ChecksScope off(false);
    check(close(I.enclose(I).sup(), on, 0), "enclose: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(enclose_is_the_hull);
    checks_and_their_switch();
    return test::finish("interval enclose");
}
