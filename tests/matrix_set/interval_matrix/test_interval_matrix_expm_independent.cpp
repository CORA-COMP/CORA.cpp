// test_interval_matrix_expm_independent - IntervalMatrix expm_independent: independent powers
//
// CORA: matrixSet/@intervalMatrix/expmInd.m
// PROPERTY: matrixSet/@intervalMatrix/expmInd.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The exponential of every member is inside enclosure; the zero matrix gives I; the powers start
/// at 1.
void expm_independent_encloses(const std::string &b) {
    Rng rng(19);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.3);
    const ExpmResult R = A.expm_independent(8);
    bool inside = true;
    for (int trial = 0; trial < 30; ++trial)
        inside = inside && R.enclosure.contains(test::member(A, rng).expm(), "exact", 1e-12);
    check(inside, b + ": exp(M) of every member is inside");
    // continued from [A]^3, the result is the series from the third order: exp(M) - I - M - M^2 / 2
    const ExpmResult from_third = A.expm_independent(8, 3, A.matrix_power(3));
    bool tail_inside = true;
    for (int trial = 0; trial < 30; ++trial) {
        const Tensor M = test::member(A, rng);
        const Tensor head = M.eye_like(3) + M + M.matmul(M) * 0.5;
        tail_inside = tail_inside && from_third.enclosure.contains(M.expm() - head, "exact", 1e-12);
    }
    check(tail_inside, b + ": from the third order, the tail of the series");
    check(R.first_order == 1 && R.powers.size() == 8, b + ": the powers of order 1 to 8");
    check(close(R.powers[0].center(), A.center()), b + ": the first power is A");
    const ExpmResult zero = IntervalMatrix(test::matrix({0, 0, 0, 0}, 2, 2)).expm_independent(5);
    check(close(zero.enclosure.center(), std::vector<double>{1, 0, 0, 1}, 1e-14),
          b + ": exp(0) = I");
    const IntervalMatrix coarse = IntervalMatrix(A.center()).expm_independent(14).enclosure;
    check(close(coarse.center(), A.center().expm(), 1e-9),
          b + ": a point matrix is its exponential");
}

/// A matrix that is not square and a negative order are refused; the checks off give the same.
void checks_and_their_switch() {
    const IntervalMatrix wide(test::matrix({1, 2, 3}, 1, 3)), A = test::example();
    check(test::raises([&] { wide.expm_independent(4); }, Error::dimension_mismatch),
          "expm_independent: a non-square matrix");
    check(test::raises([&] { A.expm_independent(-1); }, Error::wrong_value),
          "expm_independent: a negative order");
    check(test::raises([&] { A.expm_independent(4, 2, wide); }, Error::dimension_mismatch),
          "expm_independent: an initial power of another size");
    const Tensor on = A.expm_independent(5).enclosure.center();
    ChecksScope off(false);
    check(close(A.expm_independent(5).enclosure.center(), on, 0),
          "expm_independent: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { expm_independent_encloses(b); });
    checks_and_their_switch();
    return test::finish("interval_matrix expm_independent");
}
