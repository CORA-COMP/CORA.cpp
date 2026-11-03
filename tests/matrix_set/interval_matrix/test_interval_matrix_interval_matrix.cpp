// test_interval_matrix_interval_matrix - the constructors of an interval matrix
//
// CORA: matrixSet/@intervalMatrix/intervalMatrix.m
// PROPERTY: matrixSet/@intervalMatrix/intervalMatrix.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// Center and delta give [C - |D|, C + |D|]; a center alone is a point; bounds are kept as they
/// are.
void the_three_constructors(const std::string &b) {
    const Tensor C = test::matrix({1, -2, 0, 3}, 2, 2), D = test::matrix({0.5, -1, 0, 2}, 2, 2);
    const IntervalMatrix A(C, D);
    check(close(A.infimum(), std::vector<double>{0.5, -3, 0, 1}), b + ": lower bound C - |D|");
    check(close(A.interval().sup(), std::vector<double>{1.5, -1, 0, 5}), b + ": upper C + |D|");
    const IntervalMatrix P(C);
    check(close(P.infimum(), C) && close(P.interval().sup(), C), b + ": a point matrix");
    const IntervalMatrix E = test::example();
    check(E.rows() == 2 && E.cols() == 2, b + ": the size");
    check(close(IntervalMatrix(E.interval()).infimum(), E.infimum()), b + ": from an interval");
}

/// A set built from the center and the radius of a random set is that set up to rounding, and it
/// holds its center.
void rebuilt_from_center_and_radius(const std::string &b) {
    Rng rng(5);
    const IntervalMatrix A = test::random_matrix(rng, 3, 1.0);
    const IntervalMatrix B(A.center(), A.interval().rad());
    check(close(B.infimum(), A.infimum(), 1e-12) &&
              close(B.interval().sup(), A.interval().sup(), 1e-12),
          b + ": the center and the radius rebuild the set");
    check(B.contains(A.center(), "exact", 0.0), b + ": it contains its center");
}

/// A center and a delta of different shapes are refused; with the checks off a valid one is the
/// same.
void checks_and_their_switch() {
    const Tensor C = test::matrix({1, 2, 3, 4}, 2, 2), D = test::matrix({1, 2, 3}, 1, 3);
    check(test::raises([&] { IntervalMatrix(C, D); }, Error::wrong_input_in_constructor),
          "IntervalMatrix: a delta of another shape");
    const Tensor on = IntervalMatrix(C, C).infimum();
    ChecksScope off(false);
    check(close(IntervalMatrix(C, C).infimum(), on, 0),
          "IntervalMatrix: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        the_three_constructors(b);
        rebuilt_from_center_and_radius(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix constructor");
}
