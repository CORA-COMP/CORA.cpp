// test_interval_origin - interval origin: the box {0}
//
// CORA: contSet/@interval/origin.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void origin_is_the_point_zero(const std::string &b) {
    const Interval I = Interval::origin(3);
    check(I.dim() == 3 && I.inf().shape() == std::vector<int64_t>({3, 1}), b + ": a column of 3");
    check(I.inf().data() == std::vector<double>(3, 0.0) && I.sup().data() == std::vector<double>(3, 0.0),
          b + ": [0, 0] in every coordinate");
    check(I.contains(test::column({0.0, 0.0, 0.0})) && !I.contains(test::column({0.0, 0.1, 0.0})),
          b + ": it contains only the origin");
}

/// A dimension below 1 is refused; with the checks off the box is the same.
void checks_and_their_switch() {
    check(test::raises([] { Interval::origin(0); }, Error::wrong_value), "origin: dimension 0");
    check(test::raises([] { Interval::origin(-2); }, Error::wrong_value), "origin: dimension -2");
    ChecksScope off(false);
    check(Interval::origin(2).sup().data() == std::vector<double>(2, 0.0),
          "origin: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(origin_is_the_point_zero);
    checks_and_their_switch();
    return test::finish("interval origin");
}
