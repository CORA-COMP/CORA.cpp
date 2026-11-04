// test_interval_atan2 - interval atan2: the hull of the four-quadrant arctangent at the corners
//
// CORA: contSet/@interval/atan2.m
// PROPERTY: contSet/@interval/atan2.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// y in [1, 2] and x in [1, 2]: the extremes are at the corners (1, 2) and (2, 1).
void atan2_is_the_hull_of_the_corners(const std::string &b) {
    const Interval Y(column({1.0, -2.0}), column({2.0, -1.0})),
        X(column({1.0, 1.0}), column({2.0, 2.0}));
    const Interval R = Y.atan2(X);
    check(close(R.inf().slice(-2, 0, 1), std::vector<double>{0.4636476090008061}) &&
              close(R.sup().slice(-2, 0, 1), std::vector<double>{1.1071487177940904}),
          b + ": the first quadrant");
    check(close(R.inf().slice(-2, 1, 2), std::vector<double>{-1.1071487177940904}) &&
              close(R.sup().slice(-2, 1, 2), std::vector<double>{-0.4636476090008061}),
          b + ": the fourth quadrant");
    const Interval S = Interval::stack({Y, Y}).atan2(Interval::stack({X, X}));
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}) &&
              close(S.sup().slice(0, 1, 2).reshape({2, 1}), R.sup()),
          b + ": a batch");
}

/// A box across the branch cut (x < 0, y = 0) gets [-pi, pi] (CORALean's guard), not the corners'
/// hull [-2.678, 2.678] of CORA's atan2.m, which misses +-pi.
void atan2_across_the_branch_cut_is_an_enclosure(const std::string &b) {
    const Interval Y(column({-1.0}), column({1.0})), X(column({-2.0}), column({-1.0}));
    const Interval R = Y.atan2(X);
    check(R.sup().data()[0] >= M_PI && R.inf().data()[0] <= -M_PI &&
              R.sup().data()[0] < M_PI + 1e-14,
          b + ": +-pi are inside");
    cora::Rng rng(43);
    bool inside = true;
    for (int k = 0; k < 50; ++k) {
        const Tensor y = Y.rand_point(1, rng), x = X.rand_point(1, rng);
        inside = inside && R.contains(y.atan2(x));
    }
    check(inside, b + ": a sample across the cut is inside");
    const Interval P(column({0.0}), column({1.0})), Q(column({-3.0}), column({0.0}));
    check(P.atan2(Q).sup().data()[0] >= M_PI, b + ": y touching 0, x <= 0: the guard fails");
}

/// Away from the cut (x > 0) the image of every sampled point lies in the result.
void atan2_encloses_the_image_off_the_cut(const std::string &b) {
    cora::Rng rng(41);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> lo(2), hi(2);
        rng.uniform(lo.data(), 2, 0.1, 3.0);
        rng.uniform(hi.data(), 2, 3.0, 6.0);
        const Interval X(column({lo[0], lo[1]}), column({hi[0], hi[1]}));
        const Interval Y = Interval::generate_random(rng, 2);
        const Interval R = Y.atan2(X);
        const Tensor y = Y.rand_point(15, rng), x = X.rand_point(15, rng);
        bool inside = true;
        for (int k = 0; k < 15; ++k)
            inside = inside && R.contains(y.slice(-1, k, k + 1).atan2(x.slice(-1, k, k + 1)));
        check(inside, b + ": atan2 of a sample is in the result");
    }
}

/// Boxes of different shape are refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    const Interval I(column({0.5, 1.0}), column({1.0, 2.0})), I3(column({0.5}), column({1.0}));
    check(test::raises([&] { I.atan2(I3); }, Error::dimension_mismatch), "atan2: dimensions 2 and 1");
    const Tensor on = I.atan2(I).sup();
    ChecksScope off(false);
    check(close(I.atan2(I).sup(), on, 0), "atan2: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        atan2_is_the_hull_of_the_corners(b);
        atan2_across_the_branch_cut_is_an_enclosure(b);
        atan2_encloses_the_image_off_the_cut(b);
    });
    checks_and_their_switch();
    return test::finish("interval atan2");
}
