// test_interval_matrix_rad - IntervalMatrix rad: the radius matrix
//
// CORA: matrixSet/@intervalMatrix/rad.m
// PROPERTY: matrixSet/@intervalMatrix/rad.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

void rad_of_the_example(const std::string &b) {
    check(close(test::example().rad(), std::vector<double>{0.5, 1, 1.5, 0}), b + ": (sup - inf) / 2");
    const IntervalMatrix P(test::matrix({3, -1}, 1, 2));
    check(close(P.rad(), std::vector<double>{0, 0}), b + ": a point matrix has radius 0");
}

/// Every member is within rad of the center, and C +- D has the radius |D|; a batch is per member.
void rad_bounds_the_distance_to_the_center(const std::string &b) {
    Rng rng(8);
    const Tensor C = test::matrix({1, -2, 0.5, 3}, 2, 2), D = test::matrix({0.5, -1, 0, 2}, 2, 2);
    check(close(IntervalMatrix(C, D).rad(), D.abs()), b + ": |D|");
    const IntervalMatrix A = test::random_matrix(rng, 3, 1.0);
    bool within = true;
    for (int trial = 0; trial < 20; ++trial)
        within = within &&
                 (test::member(A, rng) - A.center()).abs().less_equal(A.rad() + 1e-15).min().item() != 0;
    check(within, b + ": members are within rad of the center");
    const IntervalMatrix T = test::random_matrix(rng, 2, 1.0);
    const IntervalMatrix batch(Interval::stack({T.interval(), test::example().interval()}));
    check(close(batch.rad().slice(0, 1, 2).reshape({2, 2}), test::example().rad()), b + ": batch");
}

/// rad has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor on = test::example().rad();
    ChecksScope off(false);
    check(close(test::example().rad(), on, 0), "rad: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        rad_of_the_example(b);
        rad_bounds_the_distance_to_the_center(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix rad");
}
