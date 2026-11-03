// test_interval_matrix_representsa - IntervalMatrix representsa: the types of the stacked entries
//
// CORA: matrixSet/@intervalMatrix/representsa.m
// PROPERTY: matrixSet/@intervalMatrix/representsa.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;

namespace {

/// A point matrix is a point (and the origin if it is 0), a box is neither; types that every box
/// is, and the one that none is, answer as the interval does on the stacked vector.
void types_of_the_example(const std::string &b) {
    const IntervalMatrix A = test::example(), P(test::matrix({1, 2, 3, 4}, 2, 2)),
                         O(test::matrix({0, 0, 0, 0}, 2, 2));
    check(!A.representsa("point") && P.representsa("point") && O.representsa("point"), b + ": point");
    check(!A.representsa("origin") && !P.representsa("origin") && O.representsa("origin"), b + ": origin");
    check(!A.representsa("emptySet") && !P.representsa("emptySet"), b + ": emptySet");
    check(A.representsa("interval") && A.representsa("zonotope"), b + ": every box is an interval");
    check(!A.representsa("fullspace") && !A.representsa("halfspace"), b + ": no unbounded set");
    // at most one entry with a width: a capsule; the example has three
    const IntervalMatrix one(
        Interval(test::matrix({0, 1, 2, 3}, 2, 2), test::matrix({1, 1, 2, 3}, 2, 2)));
    check(one.representsa("capsule") && !A.representsa("capsule"), b + ": capsule");
}

/// tol is the width that counts as zero; a batch is a point if every member is.
void tolerance_and_batches(const std::string &b) {
    const IntervalMatrix thin(test::matrix({1, 2, 3, 4}, 2, 2),
                              test::matrix({1e-6, 0, 0, 1e-6}, 2, 2));
    check(!thin.representsa("point", 1e-9) && thin.representsa("point", 1e-5), b + ": tol");
    const IntervalMatrix P(test::matrix({1, 2, 3, 4}, 2, 2));
    check(IntervalMatrix(Interval::stack({P.interval(), P.interval()})).representsa("point"),
          b + ": a batch of points");
    check(!IntervalMatrix(Interval::stack({P.interval(), test::example().interval()}))
               .representsa("point"),
          b + ": a batch with a box");
}

/// Random matrices: a point matrix is a point, a set of positive widths is not, and its center is.
void random_sets(const std::string &b) {
    Rng rng(20);
    bool ok = true;
    for (int trial = 0; trial < 10; ++trial) {
        const IntervalMatrix A = test::random_matrix(rng, 3, 1.0);
        const IntervalMatrix P(A.center());
        ok = ok && !A.representsa("point", 1e-12) && P.representsa("point") && !P.representsa("origin");
    }
    check(ok, b + ": random sets");
}

/// A type CORA does not know and the polygon are refused; with the checks off a valid call gives
/// the same.
void checks_and_their_switch() {
    const IntervalMatrix A = test::example();
    check(test::raises([&] { A.representsa("nothing"); }, Error::wrong_value),
          "representsa: unknown type");
    check(test::raises([&] { A.representsa("polygon"); }, Error::not_supported),
          "representsa: polygon");
    check(test::raises([&] { A.representsa("point", -1.0); }, Error::wrong_value),
          "representsa: negative tol");
    const bool on = A.representsa("point");
    ChecksScope off(false);
    check(A.representsa("point") == on, "representsa: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        types_of_the_example(b);
        tolerance_and_batches(b);
        random_sets(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix representsa");
}
