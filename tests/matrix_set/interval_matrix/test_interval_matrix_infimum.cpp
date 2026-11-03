// test_interval_matrix_infimum - IntervalMatrix infimum: the lower bound matrix
//
// CORA: matrixSet/@intervalMatrix/infimum.m
// PROPERTY: matrixSet/@intervalMatrix/infimum.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

void infimum_of_the_example(const std::string &b) {
    check(close(test::example().infimum(), std::vector<double>{1, -1, 0, 2}), b + ": the lower bound");
}

/// The infimum is a member below every other one; a batch is per member.
void infimum_is_the_smallest_member(const std::string &b) {
    Rng rng(6);
    const IntervalMatrix A = test::random_matrix(rng, 3, 1.0);
    check(A.contains(A.infimum(), "exact", 0.0), b + ": the infimum is a member");
    bool below = true;
    for (int trial = 0; trial < 20; ++trial)
        below = below && A.infimum().less_equal(test::member(A, rng)).min().item() != 0;
    check(below, b + ": below every member");
    const IntervalMatrix T = test::random_matrix(rng, 2, 1.0);
    const IntervalMatrix batch(Interval::stack({T.interval(), test::example().interval()}));
    check(close(batch.infimum().slice(0, 1, 2).reshape({2, 2}), test::example().infimum()), b + ": batch");
}

/// infimum has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor on = test::example().infimum();
    ChecksScope off(false);
    check(close(test::example().infimum(), on, 0), "infimum: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        infimum_of_the_example(b);
        infimum_is_the_smallest_member(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix infimum");
}
