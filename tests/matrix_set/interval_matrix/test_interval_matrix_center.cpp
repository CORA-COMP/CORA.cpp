// test_interval_matrix_center - IntervalMatrix center: the middle matrix
//
// CORA: matrixSet/@intervalMatrix/center.m
// PROPERTY: matrixSet/@intervalMatrix/center.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void center_of_the_example(const std::string &b) {
    check(close(test::example().center(), std::vector<double>{1.5, 0, 1.5, 2}),
          b + ": (inf + sup) / 2");
}

/// The center is in the set, and centers a set built around it; a batch is per member.
void center_is_a_member(const std::string &b) {
    Rng rng(4);
    const IntervalMatrix A = test::random_matrix(rng, 3, 1.0);
    check(A.contains(A.center(), "exact", 0.0), b + ": the center is a member");
    const Tensor C = test::member(A, rng);
    const IntervalMatrix B(C, test::matrix({1, 0, 2, 0.5, 1, 1, 0, 2, 1}, 3, 3));
    check(close(B.center(), C, 1e-12), b + ": the center of C +- D is C");
    const IntervalMatrix T = test::random_matrix(rng, 2, 1.0);
    const IntervalMatrix batch(Interval::stack({T.interval(), test::example().interval()}));
    check(close(batch.center().slice(0, 1, 2).reshape({2, 2}), test::example().center()),
          b + ": batch");
}

/// center has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor on = test::example().center();
    ChecksScope off(false);
    check(close(test::example().center(), on, 0), "center: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        center_of_the_example(b);
        center_is_a_member(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix center");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
