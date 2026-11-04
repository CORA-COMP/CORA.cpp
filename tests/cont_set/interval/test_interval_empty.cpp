// test_interval_empty - interval empty: the empty interval
//
// CORA: contSet/@interval/empty.m
// PROPERTY: contSet/@interval/empty.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

void empty_has_no_points(const std::string &b) {
    const Interval E = Interval::empty(3);
    check(E.is_empty() && E.dim() == 3, b + ": empty, in 3 dimensions");
    check(E.inf().shape() == std::vector<int64_t>({3, 0}), b + ": bounds without columns");
    check(Interval::empty(0).is_empty() && Interval::empty(0).dim() == 0, b + ": the dimension 0");
    // random boxes of the same dimension: neutral for the hull, absorbing for the meet
    cora::Rng rng(36);
    for (int trial = 0; trial < 10; ++trial) {
        const Interval I = Interval::generate_random(rng, 3);
        check(E.convex_hull(I).sup().data() == I.sup().data() &&
                  I.convex_hull(E).inf().data() == I.inf().data(),
              b + ": neutral for the hull");
        check(I.intersect(E).is_empty() && E.intersect(I).is_empty(),
              b + ": absorbing for the meet");
        check(!I.is_empty(), b + ": a box is not empty");
    }
}

/// A negative dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    check(test::raises([] { Interval::empty(-1); }, Error::wrong_value), "empty: dimension -1");
    ChecksScope off(false);
    check(Interval::empty(2).is_empty(), "empty: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(empty_has_no_points);
    checks_and_their_switch();
    return test::finish("interval empty");
}
