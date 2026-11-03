// test_interval_matrix_supremum - IntervalMatrix supremum: the upper bound matrix
//
// CORA: matrixSet/@intervalMatrix/supremum.m
// PROPERTY: matrixSet/@intervalMatrix/supremum.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

void supremum_of_the_example(const std::string &b) {
    check(close(test::example().supremum(), std::vector<double>{2, 1, 3, 2}), b + ": the upper bound");
}

/// The supremum is a member above every other one; a batch is per member.
void supremum_is_the_largest_member(const std::string &b) {
    Rng rng(6);
    const IntervalMatrix A = test::random_matrix(rng, 3, 1.0);
    check(A.contains(A.supremum(), "exact", 0.0), b + ": the supremum is a member");
    bool above = true;
    for (int trial = 0; trial < 20; ++trial)
        above = above && test::member(A, rng).less_equal(A.supremum()).min().item() != 0;
    check(above, b + ": above every member");
    const IntervalMatrix T = test::random_matrix(rng, 2, 1.0);
    const IntervalMatrix batch(Interval::stack({T.interval(), test::example().interval()}));
    check(close(batch.supremum().slice(0, 1, 2).reshape({2, 2}), test::example().supremum()), b + ": batch");
}

/// supremum has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor on = test::example().supremum();
    ChecksScope off(false);
    check(close(test::example().supremum(), on, 0), "supremum: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        supremum_of_the_example(b);
        supremum_is_the_largest_member(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix supremum");
}
