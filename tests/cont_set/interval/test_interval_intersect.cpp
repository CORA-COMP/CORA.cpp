// test_interval_intersect - interval intersect: the common part of two boxes, or the empty box
//
// CORA: contSet/@interval/and_.m
// PROPERTY: contSet/@interval/and_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void intersect_of_overlapping_boxes(const std::string &b) {
    const Interval I1(column({1.0, -1.0}), column({2.0, 1.0})),
        I2(column({1.5, -2.0}), column({2.5, 0.0}));
    const Interval R = I1.intersect(I2);
    check(close(R.inf(), std::vector<double>{1.5, -1.0}) &&
              close(R.sup(), std::vector<double>{2.0, 0.0}),
          b + ": the larger infimum and the smaller supremum");
    const Set S = intersect(I1, I2);
    check(close(S.as<Interval>().inf(), R.inf()), b + ": the registry gives the same box");
}

/// Disjoint boxes give the empty interval, whose bounds have no columns; a gap of at most eps does
/// not, and its bounds are put in order as CORA does.
void intersect_of_disjoint_boxes(const std::string &b) {
    const Interval I1(column({0.0, 0.0}), column({1.0, 1.0})),
        I2(column({0.5, 2.0}), column({2.0, 3.0}));
    const Interval R = I1.intersect(I2);
    check(R.is_empty() && R.dim() == 2 && R.inf().shape() == std::vector<int64_t>({2, 0}),
          b + ": disjoint boxes are empty");
    check(I1.intersect(R).is_empty() && R.intersect(I1).is_empty(), b + ": empty stays empty");
    const double gap = std::numeric_limits<double>::epsilon() / 2;
    const Interval touching(column({1.0 + gap}), column({2.0}));
    const Interval T = Interval(column({0.0}), column({1.0})).intersect(touching);
    check(!T.is_empty() && T.inf().data()[0] <= T.sup().data()[0], b + ": a gap below eps");
}

/// A batch in which every member meets is intersected member by member.
void intersect_of_a_batch(const std::string &b) {
    const Interval A1(column({0.0}), column({2.0})), A2(column({-1.0}), column({1.0}));
    const Interval B1(column({1.0}), column({3.0})), B2(column({0.5}), column({4.0}));
    const Interval R = Interval::stack({A1, A2}).intersect(Interval::stack({B1, B2}));
    check(close(R.inf(), Interval::stack({A1.intersect(B1), A2.intersect(B2)}).inf()) &&
              close(R.sup(), Interval::stack({A1.intersect(B1), A2.intersect(B2)}).sup()),
          b + ": the members are intersected one by one");
    const Interval far(column({10.0}), column({11.0}));
    check(test::raises([&] { Interval::stack({A1, A2}).intersect(Interval::stack({B1, far})); },
                       Error::not_supported),
          b + ": a batch of meeting and disjoint members");
}

/// A point is in the intersection exactly when it is in both boxes.
void intersect_is_the_set_intersection(const std::string &b) {
    cora::Rng rng(5);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I1 = Interval::generate_random(rng, 3),
                       I2 = Interval::generate_random(rng, 3);
        const Interval R = I1.intersect(I2);
        const Tensor x = I1.rand_point(10, rng);
        for (int k = 0; k < 10; ++k) {
            const Tensor p = x.slice(-1, k, k + 1);
            const bool in_both = I2.contains(p, 0), in_result = !R.is_empty() && R.contains(p, 0);
            check(in_both == in_result, b + ": a point of I1 is in R exactly when it is in I2");
        }
    }
}

/// Boxes of different dimension are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval S(column({0, -1}), column({1, 1})), S3(column({0, -1, 0}), column({1, 1, 2}));
    check(test::raises([&] { S.intersect(S3); }, Error::dimension_mismatch),
          "intersect: dimensions 2 and 3");
    const Tensor on = S.intersect(S).sup();
    ChecksScope off(false);
    check(close(S.intersect(S).sup(), on, 0), "intersect: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        intersect_of_overlapping_boxes(b);
        intersect_of_disjoint_boxes(b);
        intersect_of_a_batch(b);
        intersect_is_the_set_intersection(b);
    });
    checks_and_their_switch();
    return test::finish("interval intersect");
}
