// test_interval_matrix_volume - IntervalMatrix volume: the product of the entry widths
//
// CORA: matrixSet/@intervalMatrix/volume.m
// PROPERTY: matrixSet/@intervalMatrix/volume.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The widths of [0, 1], [0, 2], [0, 3], [0, 4] multiply to 24; a width 0 gives 0.
void volume_of_boxes(const std::string &b) {
    const IntervalMatrix box(Interval(test::matrix({0, 0, 0, 0}, 2, 2), test::matrix({1, 2, 3, 4}, 2, 2)));
    check(close(box.volume(), std::vector<double>{24}), b + ": 1 * 2 * 3 * 4");
    check(close(test::example().volume(), std::vector<double>{0}), b + ": a degenerate entry");
    check(close(IntervalMatrix(test::matrix({1, 2}, 1, 2)).volume(), std::vector<double>{0}),
          b + ": a point matrix");
    const IntervalMatrix tall(test::matrix({1, 2, 3}, 3, 1), test::matrix({0.5, 1, 1.5}, 3, 1));
    check(close(tall.volume(), std::vector<double>{6}), b + ": 1 * 2 * 3");
}

/// The volume scales by s^(entries) with the radii, and a batch is per member.
void volume_scales_and_batches(const std::string &b) {
    Rng rng(16);
    const IntervalMatrix A = test::random_matrix(rng, 3, 1.0);
    const Tensor C = A.center();
    const IntervalMatrix half(C, A.rad() * 0.5);
    check(close(half.volume() * 512.0, A.volume(), 1e-9), b + ": 2^9 for halved widths");
    const IntervalMatrix box(Interval(test::matrix({0, 0, 0, 0}, 2, 2), test::matrix({1, 2, 3, 4}, 2, 2)));
    const Tensor v = IntervalMatrix(Interval::stack({box.interval(), test::example().interval()})).volume();
    check(close(v.reshape({2}), std::vector<double>{24, 0}), b + ": batch");
}

/// volume has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const IntervalMatrix box(Interval(test::matrix({0, 0}, 1, 2), test::matrix({1, 2}, 1, 2)));
    const Tensor on = box.volume();
    ChecksScope off(false);
    check(close(box.volume(), on, 0), "volume: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        volume_of_boxes(b);
        volume_scales_and_batches(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix volume");
}
