// test_interval_matrix_expm_mixed - IntervalMatrix expm_mixed: the series in two parts
//
// CORA: matrixSet/@intervalMatrix/expmMixed.m
// PROPERTY: matrixSet/@intervalMatrix/expmMixed.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The two parts add up to an enclosure of exp(M) for every member, as the single-part expm does.
void the_parts_enclose_the_exponentials(const std::string &b) {
    Rng rng(20);
    const IntervalMatrix A = test::random_matrix(rng, 3, 0.3);
    for (const auto [r, intermediate] : {std::pair{1.0, 3}, std::pair{0.5, 4}, std::pair{2.0, 2}}) {
        const ExpmMixedResult R = A.expm_mixed(r, intermediate, 9);
        const IntervalMatrix total(R.enclosure.interval() + R.enclosure_rest.interval());
        bool inside = true;
        for (int trial = 0; trial < 30; ++trial)
            inside = inside && total.contains(test::member(A, rng).expm(), "exact", 1e-12);
        const std::string order = ", intermediate order " + std::to_string(intermediate);
        check(inside, b + ": exp(M) is in the sum of the parts" + order);
        check(R.powers.size() == static_cast<std::size_t>(intermediate),
              b + ": the powers of the first part" + order);
        check(R.powers2.size() == static_cast<std::size_t>(9 - intermediate),
              b + ": the powers of the second" + order);
    }
    const ExpmMixedResult zero = IntervalMatrix(test::matrix({0, 0, 0, 0}, 2, 2)).expm_mixed(1.0, 3, 6);
    const IntervalMatrix sum(zero.enclosure.interval() + zero.enclosure_rest.interval());
    check(close(sum.center(), std::vector<double>{1, 0, 0, 1}, 1e-14), b + ": exp(0) = I");
}

/// The first part is the series to the intermediate order and the second continues it: the
/// remainder is the one of the whole order.
void the_second_part_continues_the_first(const std::string &b) {
    Rng rng(21);
    const IntervalMatrix A = test::random_matrix(rng, 2, 0.3);
    const ExpmMixedResult R = A.expm_mixed(1.0, 3, 8);
    check(close(R.remainder.interval().sup(), A.exponential_remainder(8).interval().sup(), 1e-14),
          b + ": the remainder of the order 8");
    const ExpmResult tail = A.expm(1.0, 8, 4, A * R.powers[2]);
    check(close(R.enclosure_rest.center(), tail.enclosure.center(), 1e-14),
          b + ": the second part is expm from the next power");
}

/// A matrix that is not square, a zero step and an intermediate order below 2 are refused; the
/// checks off give the same.
void checks_and_their_switch() {
    const IntervalMatrix wide(test::matrix({1, 2, 3}, 1, 3)), A = test::example();
    check(test::raises([&] { wide.expm_mixed(1.0, 3, 6); }, Error::dimension_mismatch),
          "expm_mixed: a non-square matrix");
    check(test::raises([&] { A.expm_mixed(0.0, 3, 6); }, Error::wrong_value), "expm_mixed: the step 0");
    check(test::raises([&] { A.expm_mixed(1.0, 1, 6); }, Error::wrong_value),
          "expm_mixed: an intermediate order of 1");
    const Tensor on = A.expm_mixed(1.0, 3, 6).enclosure_rest.center();
    ChecksScope off(false);
    check(close(A.expm_mixed(1.0, 3, 6).enclosure_rest.center(), on, 0),
          "expm_mixed: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        the_parts_enclose_the_exponentials(b);
        the_second_part_continues_the_first(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix expm_mixed");
}
