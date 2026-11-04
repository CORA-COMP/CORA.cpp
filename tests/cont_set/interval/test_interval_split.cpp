// test_interval_split - interval split: a box cut in two at the center of one coordinate
//
// CORA: contSet/@interval/split.m
// PROPERTY: contSet/@interval/split.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void split_cuts_at_the_center(const std::string &b) {
    const Interval I(column({0.0, -2.0, 1.0}), column({2.0, 4.0, 5.0}));
    const auto [I1, I2] = I.split(1);
    check(close(I1.inf(), I.inf(), 0) && close(I1.sup(), std::vector<double>{2, 1, 5}),
          b + ": the lower half");
    check(close(I2.inf(), std::vector<double>{0, 1, 1}) && close(I2.sup(), I.sup(), 0),
          b + ": the upper half");
    check(close(I.split(0).first.sup(), std::vector<double>{1, 4, 5}) &&
              close(I.split(2).second.inf(), std::vector<double>{0, -2, 3}),
          b + ": the first and the last coordinate");
    const auto [S1, S2] = Interval::stack({I, I}).split(1);
    check(S1.sup().shape() == std::vector<int64_t>({2, 3, 1}) &&
              close(S2.inf().slice(0, 1, 2).reshape({3, 1}), I2.inf(), 0),
          b + ": a batch is cut per member");
}

/// The halves cover the box, each lies in it, and a sampled point is in one of them.
void split_halves_cover_the_box(const std::string &b) {
    cora::Rng rng(71);
    for (int trial = 0; trial < 10; ++trial) {
        const Interval I = Interval::generate_random(rng, 4);
        const int64_t n = trial % 4;
        const auto [I1, I2] = I.split(n);
        check(I1.convex_hull(I2).isequal(I, 0.0), b + ": the hull of the halves is the box");
        check(I1 <= I && I2 <= I, b + ": the halves lie in the box");
        const Tensor x = I.rand_point(20, rng);
        bool covered = true;
        for (int k = 0; k < 20; ++k) {
            const Tensor p = x.slice(-1, k, k + 1);
            covered = covered && (I1.contains(p) || I2.contains(p));
        }
        check(covered, b + ": a sample is in one half");
    }
}

/// The coordinate must exist and the box be a column; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({2.0, 3.0}));
    check(test::raises([&] { I.split(2); }, Error::wrong_value), "split: coordinate 2 of 2");
    check(test::raises([&] { I.split(-1); }, Error::wrong_value), "split: coordinate -1");
    const Tensor lo = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    check(test::raises([&] { Interval(lo, lo + 1.0).split(0); }, Error::wrong_value),
          "split: a matrix");
    const Tensor on = I.split(1).first.sup();
    ChecksScope off(false);
    check(close(I.split(1).first.sup(), on, 0), "split: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        split_cuts_at_the_center(b);
        split_halves_cover_the_box(b);
    });
    checks_and_their_switch();
    return test::finish("interval split");
}
