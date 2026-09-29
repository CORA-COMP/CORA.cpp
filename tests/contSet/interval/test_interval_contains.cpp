// test_interval_contains - interval contains: a point in a box

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
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

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { contains_a_point(b); });
    return test::finish("interval contains");
}
