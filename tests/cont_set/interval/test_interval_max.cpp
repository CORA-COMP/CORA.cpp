// test_interval_max - interval max: the elementwise larger of two boxes
//
// CORA: contSet/@interval/max.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void max_takes_the_larger_bounds(const std::string &b) {
    const Interval I(column({0.0, -1.0}), column({2.0, 1.0})), J(column({1.0, -2.0}), column({3.0, 0.0}));
    const Interval R = I.max(J);
    check(close(R.inf(), std::vector<double>{1, -1}) && close(R.sup(), std::vector<double>{3, 1}),
          b + ": [max(inf, inf2), max(sup, sup2)]");
    const Interval Q = I.max(column({1.0, 0.5}));
    check(close(Q.inf(), std::vector<double>{1, 0.5}) && close(Q.sup(), std::vector<double>{2, 1}),
          b + ": a column of numbers");
    check(close(I.max(Tensor::from_data({0.5}, {1, 1})).inf(), std::vector<double>{0.5, 0.5}),
          b + ": one number for every coordinate");
    const Zonotope Z(column({3.0, 3.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2}));
    check(close(I.max(Z).inf(), std::vector<double>{2, 2}) &&
              close(I.max(Z).sup(), std::vector<double>{4, 4}),
          b + ": another set by its interval hull");
    check(close(I.max(), I.sup()), b + ": without an argument the upper bound");
    const Interval S = Interval::stack({I, J}).max(J);
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}) &&
              close(S.sup().slice(0, 0, 1).reshape({2, 1}), R.sup()),
          b + ": a batch");
}

/// A set or an array of the wrong dimension is refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, -1.0}), column({2.0, 1.0}));
    check(test::raises([&] { I.max(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "max: a box of dimension 1");
    check(test::raises([&] { I.max(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "max: three numbers for two coordinates");
    const Tensor on = I.max(column({1.0, 0.5})).sup();
    ChecksScope off(false);
    check(close(I.max(column({1.0, 0.5})).sup(), on, 0), "max: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(max_takes_the_larger_bounds);
    checks_and_their_switch();
    return test::finish("interval max");
}
